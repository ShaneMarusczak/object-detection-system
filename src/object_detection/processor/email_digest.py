"""
Email Digest Consumer
Sends periodic summary emails by reading from JSON log files.
Supports multiple digest configurations with independent filters and photos.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from collections import Counter

from .email_service import EmailService
from .frame_service import FrameService

logger = logging.getLogger(__name__)


def email_digest_consumer(json_dir: str, config: dict) -> None:
    """
    Send periodic digest emails by reading JSON log files.
    Runs independently, reading from persistent JSON logs.
    Supports multiple digest configurations with independent filters and photos.

    Args:
        json_dir: Directory containing JSON log files
        config: Consumer configuration (can contain 'digests' list or single config)
    """
    # Initialize email service
    notification_config = config.get('notification_config', {})
    email_config = notification_config.get('email', {})
    email_service = EmailService(email_config)

    # Initialize frame service (for photo linking)
    frame_service_config = config.get('frame_service_config', {})
    frame_service = FrameService(frame_service_config) if frame_service_config else None

    # Support multiple digest configurations
    digest_configs = config.get('digests', [])

    # Backward compatibility: single config mode
    if not digest_configs:
        period_minutes = config.get('period_minutes', 60)
        period_label = config.get('period_label', f"Last {period_minutes} Minutes")
        digest_configs = [{
            'period_minutes': period_minutes,
            'period_label': period_label,
            'filters': {}
        }]

    # Track last digest time for each config
    digest_states = {}
    for i, digest_config in enumerate(digest_configs):
        digest_id = digest_config.get('id', f"digest_{i}")
        digest_states[digest_id] = {
            'config': digest_config,
            'last_digest_time': datetime.now()
        }
        logger.info(f"Digest '{digest_id}': {digest_config.get('period_label', 'N/A')} "
                   f"({digest_config.get('period_minutes', 60)} min)")

    logger.info(f"Email Digest Notifier started with {len(digest_configs)} digest(s)")
    logger.info(f"Reading from: {json_dir}")
    if email_service.enabled:
        logger.info("Digest email notifications: Enabled")
    else:
        logger.info("Digest email notifications: Disabled")

    try:
        while True:
            # Sleep and check periodically
            time.sleep(60)  # Check every minute

            current_time = datetime.now()

            # Check each digest configuration independently
            for digest_id, state in digest_states.items():
                digest_config = state['config']
                last_digest_time = state['last_digest_time']
                period_minutes = digest_config.get('period_minutes', 60)
                elapsed = (current_time - last_digest_time).total_seconds() / 60

                if elapsed >= period_minutes:
                    # Time to send this digest
                    period_label = digest_config.get('period_label', f"Last {period_minutes} Minutes")
                    filters = digest_config.get('filters', {})

                    logger.info(f"Generating digest '{digest_id}' from JSON logs...")

                    # Calculate time window
                    start_time = last_digest_time
                    end_time = current_time

                    # Read and aggregate events from JSON logs with filters
                    stats = _aggregate_from_json(json_dir, start_time, end_time, filters)

                    if stats['total_events'] > 0:
                        # Get frame URLs for events (if frame service available)
                        frame_urls = {}
                        if frame_service and stats.get('events'):
                            frame_urls = frame_service.get_frames_for_events(stats['events'])
                            logger.debug(f"Retrieved {len(frame_urls)} frame URLs")

                        logger.info(f"Sending digest '{digest_id}' ({stats['total_events']} events)...")
                        if email_service.send_digest(period_label, stats, frame_urls):
                            logger.info(f"Digest '{digest_id}' sent successfully")
                        else:
                            logger.warning(f"Failed to send digest '{digest_id}'")
                    else:
                        logger.debug(f"No events for digest '{digest_id}' - skipping")

                    # Update last digest time for this config
                    state['last_digest_time'] = current_time

    except KeyboardInterrupt:
        logger.info("Email Digest Notifier stopped by user")
    except Exception as e:
        logger.error(f"Error in Email Digest Notifier: {e}", exc_info=True)
    finally:
        logger.info("Email Digest Notifier shutdown complete")


def _aggregate_from_json(json_dir: str, start_time: datetime, end_time: datetime, filters: Dict = None) -> Dict:
    """
    Aggregate statistics from JSON log files within time window.

    Args:
        json_dir: Directory containing JSON log files
        start_time: Start of time window
        end_time: End of time window
        filters: Optional filters (event_types, zone_descriptions, line_descriptions, object_classes)

    Returns:
        Dictionary with aggregated statistics and events list
    """
    if filters is None:
        filters = {}

    filter_event_types = filters.get('event_types', [])
    filter_zones = filters.get('zone_descriptions', [])
    filter_lines = filters.get('line_descriptions', [])
    filter_classes = filters.get('object_classes', [])

    total_events = 0
    events_by_type = Counter()
    events_by_class = Counter()
    events_by_zone = Counter()
    events_by_line = Counter()
    events_by_track = Counter()
    track_classes = {}
    matched_events = []  # Store actual events for frame retrieval

    first_event_time = None
    last_event_time = None

    # Find all JSONL files in directory
    json_path = Path(json_dir)
    if not json_path.exists():
        logger.warning(f"JSON directory does not exist: {json_dir}")
        return _empty_stats()

    jsonl_files = sorted(json_path.glob("events_*.jsonl"))

    if not jsonl_files:
        logger.debug(f"No JSON log files found in {json_dir}")
        return _empty_stats()

    # Read events from all files
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        # Parse event timestamp
                        event_time_str = event.get('timestamp')
                        if not event_time_str:
                            continue

                        # Parse ISO timestamp
                        event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))

                        # Remove timezone for comparison (convert to naive)
                        event_time = event_time.replace(tzinfo=None)

                        # Check if event is within time window
                        if event_time < start_time or event_time > end_time:
                            continue

                        # Apply filters
                        if filter_event_types and event.get('event_type') not in filter_event_types:
                            continue
                        if filter_zones and event.get('zone_description') not in filter_zones:
                            continue
                        if filter_lines and event.get('line_description') not in filter_lines:
                            continue
                        if filter_classes and event.get('object_class_name') not in filter_classes:
                            continue

                        # Track time range
                        if first_event_time is None or event_time < first_event_time:
                            first_event_time = event_time
                        if last_event_time is None or event_time > last_event_time:
                            last_event_time = event_time

                        # Aggregate event
                        total_events += 1
                        events_by_type[event.get('event_type', 'UNKNOWN')] += 1
                        events_by_class[event.get('object_class_name', 'unknown')] += 1

                        if 'zone_description' in event:
                            events_by_zone[event['zone_description']] += 1
                        if 'line_description' in event:
                            events_by_line[event['line_description']] += 1

                        track_id = event.get('track_id')
                        if track_id:
                            events_by_track[track_id] += 1
                            track_classes[track_id] = event.get('object_class_name', 'unknown')

                        # Store event for frame retrieval
                        matched_events.append(event)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing event: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Error reading {jsonl_file}: {e}")
            continue

    # Build top tracks
    top_tracks = [
        (track_id, track_classes.get(track_id, 'unknown'), count)
        for track_id, count in events_by_track.most_common(10)
    ]

    return {
        'total_events': total_events,
        'events_by_type': dict(events_by_type),
        'events_by_class': dict(events_by_class),
        'events_by_zone': dict(events_by_zone),
        'events_by_line': dict(events_by_line),
        'top_tracks': top_tracks,
        'start_time': first_event_time.isoformat() if first_event_time else None,
        'end_time': last_event_time.isoformat() if last_event_time else None,
        'events': matched_events  # Include events for frame retrieval
    }


def _empty_stats() -> Dict:
    """Return empty statistics dictionary."""
    return {
        'total_events': 0,
        'events_by_type': {},
        'events_by_class': {},
        'events_by_zone': {},
        'events_by_line': {},
        'top_tracks': [],
        'start_time': None,
        'end_time': None,
        'events': []
    }
