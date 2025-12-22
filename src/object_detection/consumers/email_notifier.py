"""
Per-Event Email Notifier Consumer
Sends immediate email notifications when tagged zones/lines are crossed.
"""

import logging
from datetime import datetime, timedelta
from multiprocessing import Queue
from typing import Dict, Optional

from .email_service import EmailService

logger = logging.getLogger(__name__)


class EventNotifier:
    """Manages per-event notifications with cooldown tracking."""

    def __init__(self, email_service: EmailService):
        """Initialize event notifier.

        Args:
            email_service: Shared email service for sending emails
        """
        self.email_service = email_service
        self._last_notification: Dict[str, datetime] = {}

    def should_notify(self, identifier: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last notification.

        Args:
            identifier: Unique identifier (e.g., "Z1" or "V1")
            cooldown_minutes: Minimum minutes between notifications

        Returns:
            True if notification should be sent
        """
        if identifier not in self._last_notification:
            return True

        elapsed = datetime.now() - self._last_notification[identifier]
        return elapsed >= timedelta(minutes=cooldown_minutes)

    def notify(self, event: Dict, identifier: str, custom_message: Optional[str] = None) -> bool:
        """Send notification for an event.

        Args:
            event: Enriched event dictionary
            identifier: Unique identifier for cooldown tracking
            custom_message: Optional custom message

        Returns:
            True if notification sent successfully
        """
        if self.email_service.send_event_notification(event, custom_message):
            self._last_notification[identifier] = datetime.now()
            return True
        return False


def email_notifier_consumer(event_queue: Queue, config: dict) -> None:
    """
    Consume enriched events and send immediate email notifications.

    Args:
        event_queue: Queue receiving enriched events
        config: Consumer configuration with notification settings
    """
    # Initialize email service
    notification_config = config.get('notification_config', {})
    email_config = notification_config.get('email', {})
    email_service = EmailService(email_config)

    # Initialize notifier
    notifier = EventNotifier(email_service)

    # Get zone/line configs for notification matching
    line_configs = config.get('line_configs', {})
    zone_configs = config.get('zone_configs', {})

    logger.info("Per-Event Email Notifier started")
    if email_service.enabled:
        logger.info("Per-event email notifications: Enabled")
    else:
        logger.info("Per-event email notifications: Disabled")

    try:
        while True:
            event = event_queue.get()

            if event is None:  # Shutdown signal
                break

            if not email_service.enabled:
                continue

            # Check for notifications on zones/lines
            event_type = event['event_type']

            if event_type == 'LINE_CROSS':
                line_id = event.get('line_id')
                if line_id in line_configs:
                    config = line_configs[line_id]
                    if config.get('notify_email', False):
                        cooldown = config.get('cooldown_minutes', 60)
                        if notifier.should_notify(line_id, cooldown):
                            notifier.notify(event, line_id, config.get('message'))
                        else:
                            logger.debug(f"Skipping notification for {line_id} - cooldown active")

            elif event_type in ['ZONE_ENTER', 'ZONE_EXIT']:
                zone_id = event.get('zone_id')
                if zone_id in zone_configs:
                    config = zone_configs[zone_id]
                    if config.get('notify_email', False):
                        cooldown = config.get('cooldown_minutes', 60)
                        if notifier.should_notify(zone_id, cooldown):
                            notifier.notify(event, zone_id, config.get('message'))
                        else:
                            logger.debug(f"Skipping notification for {zone_id} - cooldown active")

    except KeyboardInterrupt:
        logger.info("Per-Event Email Notifier stopped by user")
    except Exception as e:
        logger.error(f"Error in Per-Event Email Notifier: {e}", exc_info=True)
    finally:
        logger.info("Per-Event Email Notifier shutdown complete")
