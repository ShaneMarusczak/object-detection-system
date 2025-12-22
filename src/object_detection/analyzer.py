"""
Event Analyzer - Coordinator
Enriches raw detection events and broadcasts to configured consumers.
"""

import logging
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from typing import Dict

from .consumers import json_writer_consumer, email_notifier_consumer, email_digest_consumer

logger = logging.getLogger(__name__)


def analyze_events(data_queue: Queue, config: dict, model_names: Dict[int, str]) -> None:
    """
    Process detection events, enrich them, and broadcast to consumers.

    Args:
        data_queue: Queue receiving raw detection events
        config: Configuration dictionary from config.yaml
        model_names: Dictionary mapping COCO class IDs to names
    """
    # Build lookup tables for enrichment
    line_descriptions, line_configs = _build_line_lookups(config)
    zone_descriptions, zone_configs = _build_zone_lookups(config)

    speed_enabled = config.get('speed_calculation', {}).get('enabled', False)

    # Initialize consumers based on configuration
    consumers = []
    consumer_queues = []

    # JSON Writer Consumer (default enabled)
    json_config = config.get('consumers', {}).get('json_writer', {})
    if json_config.get('enabled', True):
        json_queue = Queue()
        consumer_queues.append(json_queue)

        json_consumer_config = {
            'output_dir': config.get('output', {}).get('json_dir', 'data'),
            'console_enabled': config.get('console_output', {}).get('enabled', True),
            'console_level': config.get('console_output', {}).get('level', 'detailed'),
            'prompt_save': json_config.get('prompt_save', True)
        }

        json_process = Process(
            target=json_writer_consumer,
            args=(json_queue, json_consumer_config),
            name='JSONWriter'
        )
        json_process.start()
        consumers.append(json_process)
        logger.info("Started JSON Writer consumer")

    # Email Notifier Consumer (per-event, default disabled)
    email_config = config.get('consumers', {}).get('email_notifier', {})
    if email_config.get('enabled', False):
        email_queue = Queue()
        consumer_queues.append(email_queue)

        email_consumer_config = {
            'notification_config': config.get('notifications', {}),
            'line_configs': line_configs,
            'zone_configs': zone_configs
        }

        email_process = Process(
            target=email_notifier_consumer,
            args=(email_queue, email_consumer_config),
            name='EmailNotifier'
        )
        email_process.start()
        consumers.append(email_process)
        logger.info("Started Per-Event Email Notifier consumer")

    # Email Digest Consumer (periodic summaries from JSON logs, default disabled)
    digest_config = config.get('consumers', {}).get('email_digest', {})
    if digest_config.get('enabled', False):
        json_dir = config.get('output', {}).get('json_dir', 'data')

        digest_consumer_config = {
            'notification_config': config.get('notifications', {}),
            'period_minutes': digest_config.get('period_minutes', 60),
            'period_label': digest_config.get('period_label', f"Last {digest_config.get('period_minutes', 60)} Minutes")
        }

        digest_process = Process(
            target=email_digest_consumer,
            args=(json_dir, digest_consumer_config),
            name='EmailDigest'
        )
        digest_process.start()
        consumers.append(digest_process)
        logger.info(f"Started Email Digest consumer (period: {digest_config.get('period_minutes', 60)} min)")

    if not consumers:
        logger.warning("No consumers enabled - events will be discarded!")

    logger.info(f"Analyzer initialized with {len(consumers)} consumer(s)")

    # Process and broadcast events
    start_time = datetime.now(timezone.utc)

    try:
        while True:
            event = data_queue.get()

            if event is None:  # Shutdown signal
                break

            # Enrich event
            enriched_event = _enrich_event(
                event, model_names, line_descriptions,
                zone_descriptions, start_time, speed_enabled
            )

            # Broadcast to all consumers
            for queue in consumer_queues:
                queue.put(enriched_event)

    except KeyboardInterrupt:
        logger.info("Analyzer stopped by user")
    except Exception as e:
        logger.error(f"Error in analyzer: {e}", exc_info=True)
    finally:
        # Shutdown all consumers
        logger.info("Shutting down consumers...")
        for queue in consumer_queues:
            queue.put(None)  # Send shutdown signal

        for consumer in consumers:
            consumer.join()

        logger.info("All consumers shutdown complete")


def _enrich_event(
    event: dict,
    model_names: Dict[int, str],
    line_descriptions: Dict[str, str],
    zone_descriptions: Dict[str, str],
    start_time: datetime,
    speed_enabled: bool
) -> dict:
    """
    Enrich raw detection event with semantic meaning.

    Args:
        event: Raw event from detector
        model_names: COCO class name lookup
        line_descriptions: Line ID -> description mapping
        zone_descriptions: Zone ID -> description mapping
        start_time: System start time for absolute timestamps
        speed_enabled: Whether to calculate speed

    Returns:
        Enriched event dictionary
    """
    # Calculate absolute timestamp
    relative_time = event['timestamp_relative']
    absolute_timestamp = start_time.timestamp() + relative_time
    iso_timestamp = datetime.fromtimestamp(absolute_timestamp, tz=timezone.utc).isoformat()

    # Look up object class name
    object_class = event['object_class']
    object_class_name = model_names.get(object_class, f"unknown_{object_class}")

    # Build base enriched event
    enriched = {
        'event_type': event['event_type'],
        'timestamp': iso_timestamp,
        'timestamp_relative': round(relative_time, 3),
        'track_id': event['track_id'],
        'object_class': object_class,
        'object_class_name': object_class_name
    }

    # Add event-specific fields
    event_type = event['event_type']

    if event_type == 'LINE_CROSS':
        line_id = event['line_id']
        enriched['line_id'] = line_id
        enriched['line_description'] = line_descriptions.get(line_id, 'unknown')
        enriched['direction'] = event['direction']

        # Add speed data if available
        if speed_enabled and 'distance_pixels' in event and 'time_elapsed' in event:
            distance = event['distance_pixels']
            time_elapsed = event['time_elapsed']
            speed = distance / time_elapsed if time_elapsed > 0 else 0
            enriched['distance_pixels'] = round(distance, 2)
            enriched['time_elapsed'] = round(time_elapsed, 3)
            enriched['speed_px_per_sec'] = round(speed, 2)

    elif event_type == 'ZONE_ENTER':
        zone_id = event['zone_id']
        enriched['zone_id'] = zone_id
        enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')

    elif event_type == 'ZONE_EXIT':
        zone_id = event['zone_id']
        enriched['zone_id'] = zone_id
        enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')
        enriched['dwell_time'] = round(event['dwell_time'], 3)

    return enriched


def _build_line_lookups(config: dict) -> tuple[Dict[str, str], Dict[str, Dict]]:
    """Build line_id -> description and config lookup tables."""
    descriptions, configs = {}, {}
    vertical_count = horizontal_count = 0

    for line_config in config.get('lines', []):
        if line_config['type'] == 'vertical':
            vertical_count += 1
            line_id = f"V{vertical_count}"
        else:
            horizontal_count += 1
            line_id = f"H{horizontal_count}"

        descriptions[line_id] = line_config['description']
        configs[line_id] = {
            'notify_email': line_config.get('notify_email', False),
            'cooldown_minutes': line_config.get('cooldown_minutes', 60),
            'message': line_config.get('message')
        }

    return descriptions, configs


def _build_zone_lookups(config: dict) -> tuple[Dict[str, str], Dict[str, Dict]]:
    """Build zone_id -> description and config lookup tables."""
    descriptions, configs = {}, {}

    for i, zone_config in enumerate(config.get('zones', []), 1):
        zone_id = f"Z{i}"
        descriptions[zone_id] = zone_config['description']
        configs[zone_id] = {
            'notify_email': zone_config.get('notify_email', False),
            'cooldown_minutes': zone_config.get('cooldown_minutes', 60),
            'message': zone_config.get('message')
        }

    return descriptions, configs
