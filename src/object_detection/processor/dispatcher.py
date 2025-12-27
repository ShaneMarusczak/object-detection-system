"""
Event Dispatcher

Routes events from detector to consumers based on declarative event definitions.
Event definitions are the single source of truth for what events fire
and what actions they trigger.

In local mode: receives from multiprocessing.Queue
In distributed mode: receives from Redis Streams (swap Queue for RedisConsumer)
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Process, Queue

from ..models import EventDefinition
from ..utils.constants import DEFAULT_TEMP_FRAME_DIR
from .digest_scheduler import DigestScheduler
from .email_digest import generate_email_digest
from .email_immediate import ImmediateEmailHandler
from .frame_capture import frame_capture_consumer
from .json_writer import json_writer_consumer
from .pdf_report import generate_pdf_reports

logger = logging.getLogger(__name__)


def dispatch_events(data_queue: Queue, config: dict, model_names: dict[int, str]):
    """
    Central event dispatcher - routes events to consumers based on event definitions.

    Consumers are started automatically based on what actions are used in events.
    No separate "consumers" config needed - events drive everything.

    Args:
        data_queue: Queue receiving raw events from detector
        config: Configuration dictionary
        model_names: Mapping from COCO class ID to class name
    """
    try:
        # Initialize shutdown service variables
        pdf_shutdown_config = None
        start_time = datetime.now(timezone.utc)

        # Parse event definitions
        event_defs = _parse_event_definitions(config)
        logger.info(f"Loaded {len(event_defs)} event definition(s)")
        for event_def in event_defs:
            logger.info(f"  - {event_def.name}")

        # Parse zone and line lookups
        zone_lookup = _build_zone_lookup(config)
        line_lookup = _build_line_lookup(config)

        # Determine which consumers are needed based on event actions
        needed_actions = set()
        for event_def in event_defs:
            needed_actions.update(event_def.actions.keys())

        consumers = []
        consumer_queues = {}
        notification_config = config.get("notifications", {})
        output_config = config.get("output", {})
        frame_storage_config = config.get("frame_storage", {})

        # JSON Writer - start if json_log action is used
        if "json_log" in needed_actions:
            json_queue = Queue()
            consumer_queues["json_log"] = json_queue
            console_config = config.get("console_output", {})
            json_consumer_config = {
                "json_dir": output_config.get("json_dir", "data"),
                "console_enabled": console_config.get("enabled", True),
                "console_level": console_config.get("level", "detailed"),
            }
            json_process = Process(
                target=json_writer_consumer,
                args=(json_queue, json_consumer_config),
                name="JSONWriter",
            )
            json_process.start()
            consumers.append(json_process)
            logger.info("Started JSONWriter consumer")

        # Email Immediate - inline handler (fire-and-forget threads)
        email_handler = None
        if "email_immediate" in needed_actions:
            email_handler = ImmediateEmailHandler(
                notification_config,
                config.get("temp_frame_dir", DEFAULT_TEMP_FRAME_DIR),
            )

        # Email Digest - scheduled and/or on shutdown
        digest_scheduler = None
        digest_shutdown_config = None
        if "email_digest" in needed_actions:
            digest_configs = config.get("digests", [])

            # Start scheduler for digests with interval_hours
            json_dir = output_config.get("json_dir", "data")
            digest_scheduler = DigestScheduler(
                json_dir=json_dir,
                digest_configs=digest_configs,
                notification_config=notification_config,
                frame_storage_config=frame_storage_config,
            )
            digest_scheduler.start()

            # Track digests that should run on shutdown
            shutdown_digests = [d for d in digest_configs if d.get("on_shutdown", True)]
            if shutdown_digests:
                digest_shutdown_config = {
                    "digests": shutdown_digests,
                    "notification_config": notification_config,
                    "frame_service_config": {"storage": frame_storage_config},
                }

        # Frame Capture - start if frame_capture action is used
        if "frame_capture" in needed_actions:
            frame_queue = Queue()
            consumer_queues["frame_capture"] = frame_queue
            frame_consumer_config = {
                "temp_frame_dir": config.get("temp_frame_dir", DEFAULT_TEMP_FRAME_DIR),
                "storage": frame_storage_config,
                "lines": config.get("lines", []),
                "zones": config.get("zones", []),
                "roi": config.get("roi", {}),
            }
            frame_process = Process(
                target=frame_capture_consumer,
                args=(frame_queue, frame_consumer_config),
                name="FrameCapture",
            )
            frame_process.start()
            consumers.append(frame_process)
            logger.info("Started FrameCapture consumer")

        # PDF Report config - generates synchronously at shutdown
        if "pdf_report" in needed_actions:
            pdf_report_list = config.get("pdf_reports", [])
            if pdf_report_list:
                pdf_shutdown_config = {
                    "pdf_reports": pdf_report_list,
                    "frame_service_config": {"storage": frame_storage_config},
                }

        logger.info(f"Dispatcher initialized with {len(consumers)} consumer(s)")

        # Process and route events
        event_count = 0
        events_by_class = Counter()
        while True:
            raw_event = data_queue.get()

            if raw_event is None:
                logger.info("Received shutdown signal")
                break

            # Enrich raw event with descriptions (handles all event types)
            enriched_event = _enrich_event(
                raw_event, zone_lookup, line_lookup, model_names
            )

            # Match against event definitions and route to consumers
            matched = False
            for event_def in event_defs:
                if event_def.matches(enriched_event):
                    matched = True
                    event_count += 1
                    events_by_class[
                        enriched_event.get("object_class_name", "unknown")
                    ] += 1

                    # Tag event with definition name
                    enriched_event["event_definition"] = event_def.name

                    # Route to consumers based on actions
                    _route_event(
                        enriched_event,
                        event_def.actions,
                        consumer_queues,
                        email_handler,
                    )

                    # Only match first definition (events are mutually exclusive)
                    break

            if not matched:
                # Event didn't match any definition - discard it
                event_type = enriched_event.get("event_type")
                obj_class = enriched_event.get("object_class_name")
                loc = enriched_event.get("zone_description") or enriched_event.get(
                    "line_description"
                )
                logger.debug(f"No match: {event_type} {obj_class} {loc}")

    except Exception as e:
        logger.error(f"Error in dispatcher: {e}", exc_info=True)
    finally:
        # Print shutdown summary
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()
        elapsed_str = (
            f"{elapsed / 60:.1f} minutes" if elapsed >= 60 else f"{elapsed:.0f} seconds"
        )

        logger.info("=" * 50)
        logger.info(f"Session: {elapsed_str}, {event_count} events")
        if events_by_class:
            class_summary = ", ".join(
                f"{count} {cls}" for cls, count in events_by_class.most_common()
            )
            logger.info(f"  {class_summary}")
        logger.info("=" * 50)

        # Stop digest scheduler first (uses threading.Event, exits cleanly)
        if digest_scheduler:
            digest_scheduler.stop()

        # Shutdown all consumers
        logger.info("Shutting down consumers...")
        for queue_name, queue in consumer_queues.items():
            queue.put(None)

        for consumer in consumers:
            consumer.join(timeout=10)
            if consumer.is_alive():
                logger.warning(f"Consumer {consumer.name} did not shutdown gracefully")
                consumer.terminate()

        # Generate reports at shutdown (after all consumers finish)
        json_dir = config.get("output", {}).get("json_dir", "data")

        if pdf_shutdown_config:
            logger.info("Generating PDF report...")
            generate_pdf_reports(json_dir, pdf_shutdown_config, start_time)

        if digest_shutdown_config:
            logger.info("Sending shutdown email digest...")
            generate_email_digest(json_dir, digest_shutdown_config, start_time)

        logger.info(f"Dispatcher shutdown complete ({event_count} events processed)")


def _parse_event_definitions(config: dict) -> list[EventDefinition]:
    """
    Parse event definitions from config.

    Actions should already be resolved by prepare_runtime_config() -
    this just creates EventDefinition objects for matching.
    """
    event_defs = []

    for event_config in config.get("events", []):
        name = event_config.get("name", "unnamed")
        match = event_config.get("match", {})
        actions = event_config.get("actions", {})  # Already resolved

        event_defs.append(EventDefinition(name, match, actions))

    return event_defs


def _enrich_event(
    raw_event: dict, zone_lookup: dict, line_lookup: dict, model_names: dict[int, str]
) -> dict:
    """Enrich raw event with descriptions and timestamps."""
    enriched = raw_event.copy()

    # Add ISO timestamp
    enriched["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Add object class name
    obj_class = raw_event.get("object_class")
    if obj_class is not None:
        enriched["object_class_name"] = model_names.get(obj_class, f"class_{obj_class}")

    # Add zone description
    if "zone_id" in raw_event:
        zone_info = zone_lookup.get(raw_event["zone_id"])
        if zone_info:
            enriched["zone_description"] = zone_info["description"]

    # Add line description
    if "line_id" in raw_event:
        line_info = line_lookup.get(raw_event["line_id"])
        if line_info:
            enriched["line_description"] = line_info["description"]

    return enriched


def _route_event(
    event: dict,
    actions: dict,
    consumer_queues: dict[str, Queue],
    email_handler: ImmediateEmailHandler | None = None,
):
    """Route event to appropriate consumers based on action configuration."""

    # JSON logging
    if actions.get("json_log", False):
        if "json_log" in consumer_queues:
            consumer_queues["json_log"].put(event)

    # Immediate email - fire and forget via handler
    email_immediate = actions.get("email_immediate")
    if email_immediate and email_immediate.get("enabled", False):
        if email_handler:
            # Tag event with email config and send
            event["_email_immediate_config"] = email_immediate
            email_handler.handle_event(event)

    # Email digest - event logged to JSON, digest generated at shutdown
    # No action needed here - digest reads from JSON logs at shutdown

    # Frame capture
    frame_capture = actions.get("frame_capture")
    if frame_capture and frame_capture.get("enabled", False):
        if "frame_capture" in consumer_queues:
            # Tag event with frame config
            event["_frame_capture_config"] = frame_capture
            consumer_queues["frame_capture"].put(event)


def _build_zone_lookup(config: dict) -> dict[str, dict]:
    """Build lookup from zone_id to zone info."""
    lookup = {}
    for i, zone in enumerate(config.get("zones", []), 1):
        zone_id = f"Z{i}"
        lookup[zone_id] = {
            "description": zone.get("description", zone_id),
            "config": zone,
        }
    return lookup


def _build_line_lookup(config: dict) -> dict[str, dict]:
    """Build lookup from line_id to line info."""
    lookup = {}
    v_count = 0
    h_count = 0

    for line in config.get("lines", []):
        if line["type"] == "vertical":
            v_count += 1
            line_id = f"V{v_count}"
        else:
            h_count += 1
            line_id = f"H{h_count}"

        lookup[line_id] = {
            "description": line.get("description", line_id),
            "config": line,
        }

    return lookup
