"""
Email Notifier Consumer
Sends immediate emails for events. No filtering - if it's on the queue, send it.
Cooldown configuration comes from event metadata.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict

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

    # Initialize cooldown tracker
    notifier = EventNotifier()

    logger.info("Email Notifier started")
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
            email_config = event.get('_email_immediate_config', {})
            cooldown_minutes = email_config.get('cooldown_minutes', 60)
            custom_message = email_config.get('message')

            # Build cooldown identifier from (track_id, zone/line)
            track_id = event.get('track_id')
            zone = event.get('zone_description', '')
            line = event.get('line_description', '')
            location = zone or line
            identifier = f"{track_id}:{location}"

            # Check cooldown
            if notifier.should_notify(identifier, cooldown_minutes):
                logger.info(f"Sending email for: {event.get('event_definition')}")

                if email_service.send_event_notification(event, custom_message):
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
