"""
Notifiers - Pluggable notification backends.

Provides a common interface for different notification services:
- ntfy: Push notifications via ntfy.sh
- webhook: Generic HTTP webhooks

Used by both VLM analyzer (sends analysis text) and direct notifier (sends formatted messages).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


def format_title(template: str | None, event: dict[str, Any]) -> str:
    """
    Format notification title using template and event data.

    This is a shared utility used by all notifier implementations.

    Args:
        template: Optional title template with {variable} placeholders.
                  If None, generates a default title.
        event: Event dictionary with metadata

    Returns:
        Formatted title string

    Available template variables:
        {event_name} - Event definition name
        {event_type} - Event type (LINE_CROSS, ZONE_ENTER, etc.)
        {object_class} - Detected object class name
        {confidence} - Detection confidence (float)
        {zone} - Zone description (if applicable)
        {line} - Line description (if applicable)
        {timestamp} - Event timestamp
    """
    if not template:
        # Default title based on event type
        event_name = event.get("_event_name", "Detection")
        obj_class = event.get("object_class_name", "object")
        return f"{event_name}: {obj_class}"

    try:
        template_vars = {
            "event_name": event.get("_event_name", "Detection"),
            "event_type": event.get("event_type", "DETECTED"),
            "object_class": event.get("object_class_name", "object"),
            "confidence": event.get("confidence", 0),
            "zone": event.get("zone_description", ""),
            "line": event.get("line_description", ""),
            "timestamp": event.get("timestamp", ""),
        }
        return template.format(**template_vars)
    except KeyError as e:
        logger.warning(f"Title template error: missing key {e}")
        return template


class Notifier(ABC):
    """
    Abstract base class for notification backends.

    All notifiers must implement the `send` method which receives:
    - analysis: The VLM analysis text
    - image_path: Optional path to the image file
    - event: The full event dictionary with all metadata
    """

    @abstractmethod
    def send(
        self,
        analysis: str,
        image_path: str | None,
        event: dict[str, Any],
    ) -> bool:
        """
        Send a notification.

        Args:
            analysis: VLM analysis text
            image_path: Optional path to image file to attach
            event: Full event dictionary

        Returns:
            True if notification was sent successfully
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the notifier ID from config."""
        pass


def create_notifier(config: dict[str, Any]) -> Notifier:
    """
    Factory function to create a notifier from config.

    Args:
        config: Notifier configuration dict with 'type' field

    Returns:
        Configured Notifier instance

    Raises:
        ValueError: If notifier type is unknown
    """
    notifier_type = config.get("type")

    if notifier_type == "ntfy":
        from .ntfy import NtfyNotifier

        return NtfyNotifier(config)

    elif notifier_type == "webhook":
        from .webhook import WebhookNotifier

        return WebhookNotifier(config)

    else:
        raise ValueError(f"Unknown notifier type: {notifier_type}")


def create_notifiers(configs: list[dict[str, Any]]) -> dict[str, Notifier]:
    """
    Create multiple notifiers from config list.

    Args:
        configs: List of notifier configurations

    Returns:
        Dictionary mapping notifier ID to Notifier instance
    """
    notifiers = {}
    for config in configs:
        try:
            notifier = create_notifier(config)
            notifiers[notifier.id] = notifier
            logger.debug(f"Created notifier: {notifier.id} ({config.get('type')})")
        except Exception as e:
            logger.error(f"Failed to create notifier {config.get('id')}: {e}")

    return notifiers


__all__ = ["Notifier", "create_notifier", "create_notifiers", "format_title"]
