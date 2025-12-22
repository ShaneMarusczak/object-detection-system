"""
Email Notifier Consumer
Sends email notifications based on zone/line events.
"""

import logging
from multiprocessing import Queue
from typing import Dict

from ..notifier import NotificationManager

logger = logging.getLogger(__name__)


def email_notifier_consumer(event_queue: Queue, config: dict) -> None:
    """
    Consume enriched events and send email notifications.

    Args:
        event_queue: Queue receiving enriched events
        config: Consumer configuration with notification settings
    """
    # Initialize notification manager
    notification_manager = NotificationManager(config.get('notification_config', {}))

    # Get zone/line configs for notification matching
    line_configs = config.get('line_configs', {})
    zone_configs = config.get('zone_configs', {})

    logger.info("Email Notifier started")
    if notification_manager.enabled:
        logger.info("Email notifications: Enabled")
    else:
        logger.info("Email notifications: Disabled (set notifications.enabled=true)")

    try:
        while True:
            event = event_queue.get()

            if event is None:  # Shutdown signal
                break

            if not notification_manager.enabled:
                continue

            # Check for notifications on zones/lines
            event_type = event['event_type']

            if event_type == 'LINE_CROSS':
                line_id = event.get('line_id')
                if line_id in line_configs:
                    notification_manager.check_and_notify(
                        event,
                        line_id=line_id,
                        notify_config=line_configs[line_id]
                    )

            elif event_type in ['ZONE_ENTER', 'ZONE_EXIT']:
                zone_id = event.get('zone_id')
                if zone_id in zone_configs:
                    notification_manager.check_and_notify(
                        event,
                        zone_id=zone_id,
                        notify_config=zone_configs[zone_id]
                    )

    except KeyboardInterrupt:
        logger.info("Email Notifier stopped by user")
    except Exception as e:
        logger.error(f"Error in Email Notifier: {e}", exc_info=True)
    finally:
        logger.info("Email Notifier shutdown complete")
