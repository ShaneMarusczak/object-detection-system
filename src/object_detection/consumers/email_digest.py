"""
Email Digest Consumer
Sends periodic summary emails by reading from JSON log files.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from collections import Counter

from .email_service import EmailService

logger = logging.getLogger(__name__)


def email_digest_consumer(json_dir: str, config: dict) -> None:
    """
    Send periodic digest emails by reading JSON log files.
    Runs independently, reading from persistent JSON logs.

    Args:
        json_dir: Directory containing JSON log files
        config: Consumer configuration
    """
    # Initialize email service
    notification_config = config.get('notification_config', {})
    email_config = notification_config.get('email', {})
    email_service = EmailService(email_config)

    # Get digest configuration
    period_minutes = config.get('period_minutes', 60)
    period_label = config.get('period_label', f"Last {period_minutes} Minutes")

    logger.info(f"Email Digest Notifier started (period: {period_minutes} minutes)")
    logger.info(f"Reading from: {json_dir}")
    if email_service.enabled:
        logger.info("Digest email notifications: Enabled")
    else:
        logger.info("Digest email notifications: Disabled")

    # Track last digest time
    last_digest_time = datetime.now()

    try:
        while True:
            # Sleep and check periodically
            time.sleep(60)  # Check every minute

            current_time = datetime.now()
            elapsed = (current_time - last_digest_time).total_seconds() / 60

            if elapsed >= period_minutes:
                # Time to send digest
                logger.info("Generating digest from JSON logs...")

                # Calculate time window
                start_time = last_digest_time
                end_time = current_time

                # Read and aggregate events from JSON logs
                stats = _aggregate_from_json(json_dir, start_time, end_time)

                if stats['total_events'] > 0:
                    logger.info(f"Sending digest email ({stats['total_events']} events)...")
                    if email_service.send_digest(period_label, stats):
                        logger.info("Digest email sent successfully")
                    else:
                        logger.warning("Failed to send digest email")
                else:
                    logger.debug(f"No events in last {period_minutes} minutes - skipping digest")

                last_digest_time = current_time

    except KeyboardInterrupt:
        logger.info("Email Digest Notifier stopped by user")
    except Exception as e:
        logger.error(f"Error in Email Digest Notifier: {e}", exc_info=True)
    finally:
        logger.info("Email Digest Notifier shutdown complete")


def _aggregate_from_json(json_dir: str, start_time: datetime, end_time: datetime) -> Dict:
    """
    Aggregate statistics from JSON log files within time window.

    Args:
        json_dir: Directory containing JSON log files
        start_time: Start of time window
        end_time: End of time window

    Returns:
        Dictionary with aggregated statistics
    """
    total_events = 0
    events_by_type = Counter()
    events_by_class = Counter()
    events_by_zone = Counter()
    events_by_line = Counter()
    events_by_track = Counter()
    track_classes = {}

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
        'end_time': last_event_time.isoformat() if last_event_time else None
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
        'end_time': None
    }
