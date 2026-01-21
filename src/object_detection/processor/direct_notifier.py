"""
Direct Notifier Consumer - Sends notifications without VLM analysis.

This consumer:
1. Receives events with direct notify actions
2. Formats the message template with event data
3. Sends to the configured notifiers
4. Optionally attaches the captured frame
"""

import logging
from multiprocessing import Queue
from typing import Any

from .notifiers import Notifier, create_notifiers, format_template

logger = logging.getLogger(__name__)


def process_notify_event(
    event: dict[str, Any],
    notifiers: dict[str, Notifier],
) -> None:
    """
    Process a single event with notify actions.

    Args:
        event: Event dictionary with _notify_config list
        notifiers: Dictionary of notifier ID -> Notifier
    """
    notify_configs = event.get("_notify_config", [])
    if not notify_configs:
        return

    frame_path = event.get("_frame_path")

    for notify_config in notify_configs:
        notifier_id = notify_config.get("notifier")
        message_template = notify_config.get("message", "{object_class} detected")
        include_image = notify_config.get("include_image", False)

        notifier = notifiers.get(notifier_id)
        if not notifier:
            logger.warning(f"Notifier not found: {notifier_id}")
            continue

        # Format the message
        message = format_template(message_template, event)

        # Determine image path
        image_path = frame_path if include_image else None

        try:
            success = notifier.send(
                analysis=message,
                image_path=image_path,
                event=event,
            )
            if success:
                logger.debug(f"Direct notification sent via {notifier_id}")
            else:
                logger.warning(f"Direct notification failed for {notifier_id}")
        except Exception as e:
            logger.error(f"Notifier {notifier_id} error: {e}")


def direct_notifier_consumer(
    event_queue: Queue,
    config: dict[str, Any],
) -> None:
    """
    Direct notifier consumer process.

    Receives events from the queue and sends direct notifications.

    Args:
        event_queue: Queue of events to process
        config: Configuration dictionary with notifiers list
    """
    logger.info("Direct Notifier consumer started")

    # Create notifiers from config
    notifier_configs = config.get("notifiers", [])
    notifiers = create_notifiers(notifier_configs)
    logger.info(f"Initialized {len(notifiers)} notifier(s)")

    try:
        while True:
            event = event_queue.get()

            # Shutdown signal
            if event is None:
                logger.info("Direct Notifier consumer received shutdown signal")
                break

            try:
                process_notify_event(event, notifiers)
            except Exception as e:
                logger.error(f"Error processing notify event: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("Direct Notifier consumer interrupted")
    finally:
        logger.info("Direct Notifier consumer shutdown")
