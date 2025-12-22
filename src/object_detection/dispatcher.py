"""
Event Dispatcher
Routes events from detector to consumers based on declarative event definitions.
Event definitions are the single source of truth for what events fire and what actions they trigger.
"""

import logging
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Any, Set

from .coco_classes import COCO_NAME_TO_ID
from .consumers import json_writer_consumer, email_notifier_consumer, email_digest_consumer, frame_capture_consumer

logger = logging.getLogger(__name__)


class EventDefinition:
    """Declarative event definition - specifies what to match and what actions to take."""

    def __init__(self, name: str, match: Dict[str, Any], actions: Dict[str, Any], digest_configs: Dict[str, Dict] = None):
        self.name = name
        self.match = match

        # Apply implied action rules (AWS-style primitive composition)
        self.actions = self._apply_implied_actions(actions, digest_configs or {})

        # Parse match criteria
        self.event_type = match.get('event_type')
        self.zone = match.get('zone')
        self.line = match.get('line')

        # Handle single class or list of classes
        obj_class = match.get('object_class')
        if isinstance(obj_class, list):
            self.object_classes = set(obj_class)
        elif obj_class:
            self.object_classes = {obj_class}
        else:
            self.object_classes = None  # Match any class

    def _apply_implied_actions(self, actions: Dict[str, Any], digest_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Apply implied action dependencies (AWS-style primitive composition).

        Rules:
        - email_digest requires json_log (digest reads from JSON)
        - email_digest with photos: true requires frame_capture (digest shows photos)
        """
        actions = actions.copy()

        # If email_digest is specified, check for dependencies
        digest_id = actions.get('email_digest')
        if digest_id:
            # email_digest always requires json_log
            if not actions.get('json_log'):
                actions['json_log'] = True
                logger.debug(f"Auto-enabled json_log (required by email_digest)")

            # Check if this digest wants photos
            digest = digest_configs.get(digest_id, {})
            if digest.get('photos'):
                if not actions.get('frame_capture'):
                    # Auto-enable frame capture with digest's frame_config
                    frame_config = digest.get('frame_config', {})
                    actions['frame_capture'] = {
                        'enabled': True,
                        **frame_config
                    }
                    logger.debug(f"Auto-enabled frame_capture (required by photo digest '{digest_id}')")

        return actions

    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if raw event matches this definition."""
        # Check event type
        if self.event_type and event.get('event_type') != self.event_type:
            return False

        # Check zone
        if self.zone and event.get('zone_description') != self.zone:
            return False

        # Check line
        if self.line and event.get('line_description') != self.line:
            return False

        # Check object class
        if self.object_classes and event.get('object_class_name') not in self.object_classes:
            return False

        return True

    def get_object_classes(self) -> Set[str]:
        """Get all object classes this event definition matches."""
        return self.object_classes if self.object_classes else set()


def derive_track_classes(event_defs: List[EventDefinition]) -> List[int]:
    """
    Derive COCO class IDs from event definitions.
    Only track classes that appear in at least one event definition.
    """
    class_names = set()
    for event_def in event_defs:
        class_names.update(event_def.get_object_classes())

    # Convert class names to COCO IDs
    class_ids = []
    for name in class_names:
        name_lower = name.lower()
        if name_lower in COCO_NAME_TO_ID:
            class_ids.append(COCO_NAME_TO_ID[name_lower])
        else:
            logger.warning(f"Unknown object class '{name}' in event definitions (not in COCO dataset)")
            logger.warning("Available classes: person, bicycle, car, motorcycle, bus, truck, cat, dog, etc.")

    if not class_ids:
        logger.warning("No valid object classes found in event definitions - will track nothing!")

    return sorted(class_ids)


def dispatch_events(data_queue: Queue, config: dict, model_names: Dict[int, str]):
    """
    Central event dispatcher - routes events to consumers based on event definitions.

    Args:
        data_queue: Queue receiving raw events from detector
        config: Configuration dictionary
        model_names: Mapping from COCO class ID to class name
    """
    try:
        # Parse event definitions
        event_defs = _parse_event_definitions(config)
        logger.info(f"Loaded {len(event_defs)} event definition(s)")
        for event_def in event_defs:
            logger.info(f"  - {event_def.name}")

        # Parse zone and line lookups
        zone_lookup = _build_zone_lookup(config)
        line_lookup = _build_line_lookup(config)

        # Initialize consumers
        consumers = []
        consumer_queues = {}

        # JSON Writer (always enabled - logs all matched events)
        json_config = config.get('consumers', {}).get('json_writer', {})
        if json_config.get('enabled', True):
            json_queue = Queue()
            consumer_queues['json_log'] = json_queue

            json_dir = config.get('output', {}).get('json_dir', 'data')
            console_config = config.get('console_output', {})

            json_consumer_config = {
                'json_dir': json_dir,
                'console_enabled': console_config.get('enabled', True),
                'console_level': console_config.get('level', 'detailed'),
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

        # Email Notifier (always enabled - processes events with email_immediate action)
        email_queue = Queue()
        consumer_queues['email_immediate'] = email_queue

        notification_config = config.get('notifications', {})
        email_config = notification_config.get('email', {})

        email_consumer_config = {
            'notification_config': notification_config
        }

        email_process = Process(
            target=email_notifier_consumer,
            args=(email_queue, email_consumer_config),
            name='EmailNotifier'
        )
        email_process.start()
        consumers.append(email_process)
        logger.info("Started Email Notifier consumer")

        # Email Digest (always enabled - processes digests defined in config)
        digest_config = config.get('digests', [])
        if digest_config:
            json_dir = config.get('output', {}).get('json_dir', 'data')

            # Build frame service config for digest
            frame_storage_config = config.get('frame_storage', {})

            digest_consumer_config = {
                'digests': digest_config,
                'notification_config': notification_config,
                'frame_service_config': frame_storage_config
            }

            digest_process = Process(
                target=email_digest_consumer,
                args=(json_dir, digest_consumer_config),
                name='EmailDigest'
            )
            digest_process.start()
            consumers.append(digest_process)
            logger.info(f"Started Email Digest consumer with {len(digest_config)} digest(s)")

        # Frame Capture (always enabled - processes events with frame_capture action)
        frame_queue = Queue()
        consumer_queues['frame_capture'] = frame_queue

        frame_storage_config = config.get('frame_storage', {})
        frame_consumer_config = {
            'temp_frame_dir': config.get('temp_frame_dir', '/tmp/frames'),
            'storage': frame_storage_config
        }

        frame_process = Process(
            target=frame_capture_consumer,
            args=(frame_queue, frame_consumer_config),
            name='FrameCapture'
        )
        frame_process.start()
        consumers.append(frame_process)
        logger.info("Started Frame Capture consumer")

        if not consumers:
            logger.warning("No consumers enabled - events will be discarded!")

        logger.info(f"Dispatcher initialized with {len(consumers)} consumer(s)")

        # Process and route events
        event_count = 0
        while True:
            raw_event = data_queue.get()

            if raw_event is None:
                logger.info("Received shutdown signal")
                break

            # Enrich raw event with descriptions
            enriched_event = _enrich_event(raw_event, zone_lookup, line_lookup, model_names)

            # Match against event definitions and route to consumers
            matched = False
            for event_def in event_defs:
                if event_def.matches(enriched_event):
                    matched = True
                    event_count += 1

                    # Tag event with definition name
                    enriched_event['event_definition'] = event_def.name

                    # Route to consumers based on actions
                    _route_event(enriched_event, event_def.actions, consumer_queues)

                    # Only match first definition (events are mutually exclusive)
                    break

            if not matched:
                # Event didn't match any definition - discard it
                logger.debug(f"Event did not match any definition: {enriched_event.get('event_type')} "
                           f"{enriched_event.get('object_class_name')} "
                           f"{enriched_event.get('zone_description') or enriched_event.get('line_description')}")

    except Exception as e:
        logger.error(f"Error in dispatcher: {e}", exc_info=True)
    finally:
        # Shutdown all consumers
        logger.info("Shutting down consumers...")
        for queue_name, queue in consumer_queues.items():
            queue.put(None)

        for consumer in consumers:
            consumer.join(timeout=10)
            if consumer.is_alive():
                logger.warning(f"Consumer {consumer.name} did not shutdown gracefully")
                consumer.terminate()

        logger.info(f"Dispatcher shutdown complete ({event_count} events processed)")


def _parse_event_definitions(config: dict) -> List[EventDefinition]:
    """Parse event definitions from config."""
    event_defs = []

    # Parse digest configs first (for implied action rules)
    digest_configs = {}
    for digest in config.get('digests', []):
        digest_id = digest.get('id')
        if digest_id:
            digest_configs[digest_id] = digest

    # Parse events with digest context
    for event_config in config.get('events', []):
        name = event_config.get('name', 'unnamed')
        match = event_config.get('match', {})
        actions = event_config.get('actions', {})

        event_defs.append(EventDefinition(name, match, actions, digest_configs))

    return event_defs


def _enrich_event(raw_event: Dict, zone_lookup: Dict, line_lookup: Dict, model_names: Dict[int, str]) -> Dict:
    """Enrich raw event with descriptions and timestamps."""
    enriched = raw_event.copy()

    # Add ISO timestamp
    enriched['timestamp'] = datetime.now(timezone.utc).isoformat()

    # Add object class name
    obj_class = raw_event.get('object_class')
    enriched['object_class_name'] = model_names.get(obj_class, f"class_{obj_class}")

    # Add zone description
    if 'zone_id' in raw_event:
        zone_info = zone_lookup.get(raw_event['zone_id'])
        if zone_info:
            enriched['zone_description'] = zone_info['description']

    # Add line description
    if 'line_id' in raw_event:
        line_info = line_lookup.get(raw_event['line_id'])
        if line_info:
            enriched['line_description'] = line_info['description']

    return enriched


def _route_event(event: Dict, actions: Dict, consumer_queues: Dict[str, Queue]):
    """Route event to appropriate consumers based on action configuration."""

    # JSON logging
    if actions.get('json_log', False):
        if 'json_log' in consumer_queues:
            consumer_queues['json_log'].put(event)

    # Immediate email
    email_immediate = actions.get('email_immediate')
    if email_immediate and email_immediate.get('enabled', False):
        if 'email_immediate' in consumer_queues:
            # Tag event with email config
            event['_email_immediate_config'] = email_immediate
            consumer_queues['email_immediate'].put(event)

    # Email digest (just tag the event, digest consumer reads from JSON logs)
    email_digest = actions.get('email_digest')
    if email_digest:
        # Event is already logged to JSON by json_log action
        # Digest consumer will filter by event_definition name
        pass

    # Frame capture
    frame_capture = actions.get('frame_capture')
    if frame_capture and frame_capture.get('enabled', False):
        if 'frame_capture' in consumer_queues:
            # Tag event with frame config
            event['_frame_capture_config'] = frame_capture
            consumer_queues['frame_capture'].put(event)


def _build_zone_lookup(config: dict) -> Dict[str, Dict]:
    """Build lookup from zone_id to zone info."""
    lookup = {}
    for i, zone in enumerate(config.get('zones', []), 1):
        zone_id = f"Z{i}"
        lookup[zone_id] = {
            'description': zone.get('description', zone_id),
            'config': zone
        }
    return lookup


def _build_line_lookup(config: dict) -> Dict[str, Dict]:
    """Build lookup from line_id to line info."""
    lookup = {}
    v_count = 0
    h_count = 0

    for line in config.get('lines', []):
        if line['type'] == 'vertical':
            v_count += 1
            line_id = f"V{v_count}"
        else:
            h_count += 1
            line_id = f"H{h_count}"

        lookup[line_id] = {
            'description': line.get('description', line_id),
            'config': line
        }

    return lookup
