"""
Email Notifier Consumer
Sends immediate emails for events. No filtering - if it's on the queue, send it.
Cooldown configuration comes from event metadata.
"""

import glob
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from .email_service import EmailService

logger = logging.getLogger(__name__)


class EventNotifier:
    """Tracks cooldowns for immediate email notifications."""

    def __init__(self):
        self._last_notification: Dict[str, datetime] = {}

    def should_notify(self, identifier: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last notification."""
        if identifier not in self._last_notification:
            return True

        elapsed = datetime.now() - self._last_notification[identifier]
        return elapsed >= timedelta(minutes=cooldown_minutes)

    def mark_notified(self, identifier: str):
        """Mark that notification was sent."""
        self._last_notification[identifier] = datetime.now()


def email_notifier_consumer(event_queue, config: dict) -> None:
    """
    Send immediate email notifications for events.
    No filtering - events are pre-filtered by dispatcher.

    Args:
        event_queue: Queue receiving events that need immediate emails
        config: Consumer configuration with notification settings
    """
    # Initialize email service
    notification_config = config.get('notification_config', {})
    email_config = notification_config.get('email', {})
    email_service = EmailService(email_config)

    # Temp frame directory for include_frame feature
    temp_frame_dir = config.get('temp_frame_dir', '/tmp/frames')

    # Initialize cooldown tracker
    notifier = EventNotifier()

    logger.info("Email Notifier started")
    logger.info(f"Temp frame dir: {temp_frame_dir}")
    if email_service.enabled:
        logger.info("Immediate email notifications: Enabled")
    else:
        logger.info("Immediate email notifications: Disabled (will process but not send)")

    try:
        while True:
            event = event_queue.get()

            if event is None:
                logger.info("Email Notifier received shutdown signal")
                break

            # Extract email config from event metadata
            email_action_config = event.get('_email_immediate_config', {})
            cooldown_minutes = email_action_config.get('cooldown_minutes', 60)
            custom_message = email_action_config.get('message')
            custom_subject = email_action_config.get('subject')
            include_frame = email_action_config.get('include_frame', False)

            # Build cooldown identifier from (track_id, zone/line)
            track_id = event.get('track_id')
            zone = event.get('zone_description', '')
            line = event.get('line_description', '')
            location = zone or line
            identifier = f"{track_id}:{location}"

            # Check cooldown
            if notifier.should_notify(identifier, cooldown_minutes):
                logger.info(f"Sending email for: {event.get('event_definition')}")

                # Get frame data if requested
                frame_data = None
                if include_frame:
                    frame_data = _find_and_read_temp_frame(temp_frame_dir, event.get('timestamp'))
                    if frame_data:
                        logger.debug("Including frame in email")

                if email_service.send_event_notification(event, custom_message, frame_data, custom_subject):
                    notifier.mark_notified(identifier)
                    logger.debug(f"Email sent (cooldown: {cooldown_minutes} min)")
                else:
                    logger.warning("Failed to send email notification")
            else:
                logger.debug(f"Skipping email due to cooldown: {identifier}")

    except KeyboardInterrupt:
        logger.info("Email Notifier stopped by user")
    except Exception as e:
        logger.error(f"Error in Email Notifier: {e}", exc_info=True)
    finally:
        logger.info("Email Notifier shutdown complete")


def _find_and_read_temp_frame(temp_dir: str, event_timestamp: str,
                               tolerance_seconds: int = 5) -> Optional[bytes]:
    """
    Find temp frame closest to event timestamp and read its bytes.

    Args:
        temp_dir: Directory containing temp frames
        event_timestamp: ISO timestamp of event
        tolerance_seconds: Maximum time difference to accept

    Returns:
        Frame bytes if found, None otherwise
    """
    if not event_timestamp or not os.path.exists(temp_dir):
        return None

    try:
        # Parse event timestamp
        event_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
        event_time = event_time.replace(tzinfo=None)
    except Exception:
        logger.warning(f"Could not parse event timestamp: {event_timestamp}")
        return None

    # Find all temp frames
    temp_frames = glob.glob(os.path.join(temp_dir, 'frame_*.jpg'))
    if not temp_frames:
        return None

    # Find closest frame within tolerance
    best_frame = None
    best_diff = float('inf')

    for frame_path in temp_frames:
        try:
            # Extract timestamp from filename: frame_YYYYMMDD_HHMMSS_ffffff.jpg
            filename = os.path.basename(frame_path)
            timestamp_str = filename.replace('frame_', '').replace('.jpg', '')
            frame_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')

            diff = abs((frame_time - event_time).total_seconds())
            if diff < best_diff and diff <= tolerance_seconds:
                best_diff = diff
                best_frame = frame_path
        except Exception:
            continue

    if best_frame:
        try:
            with open(best_frame, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read temp frame {best_frame}: {e}")

    return None
