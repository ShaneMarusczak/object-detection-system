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

from .notifiers import Notifier, create_notifiers

logger = logging.getLogger(__name__)


def format_message(template: str, event: dict[str, Any]) -> str:
    """
    Format message template with event data.

    Available template variables:
        {event_name} - Event definition name
        {event_type} - Event type (LINE_CROSS, ZONE_ENTER, etc.)
        {object_class} - Detected object class name
        {confidence} - Detection confidence (float)
        {confidence_pct} - Detection confidence as percentage string
        {zone} - Zone description (if applicable)
        {line} - Line description (if applicable)
        {timestamp} - Event timestamp
        {track_id} - Tracking ID

    Args:
        template: Message template with {variable} placeholders
        event: Event dictionary

    Returns:
        Formatted message string
    """
    confidence = event.get("confidence", 0)

    template_vars = {
        "event_name": event.get("_event_name", "Detection"),
        "event_type": event.get("event_type", "DETECTED"),
        "object_class": event.get("object_class_name", "object"),
        "confidence": confidence,
        "confidence_pct": f"{confidence:.0%}",
        "zone": event.get("zone_description", ""),
        "line": event.get("line_description", ""),
        "timestamp": event.get("timestamp", ""),
        "track_id": event.get("track_id", ""),
    }

    try:
        return template.format(**template_vars)
    except KeyError as e:
        logger.warning(f"Message template error: missing key {e}")
        return template


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
        message = format_message(message_template, event)

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
