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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_FACTOR = 1.0  # seconds


def create_retry_session(
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> requests.Session:
    """
    Create a requests Session with automatic retry and exponential backoff.

    Uses urllib3's built-in Retry mechanism for robust HTTP requests.
    Retries on:
    - Connection errors
    - Timeouts
    - Server errors (500, 502, 503, 504)

    Does NOT retry on:
    - Client errors (4xx) - those won't fix themselves

    Args:
        max_retries: Maximum number of retry attempts (default: 2)
        backoff_factor: Delay factor for exponential backoff (default: 1.0)
                       Delay = backoff_factor * (2 ** retry_number)
                       With factor=1.0: 1s, 2s, 4s...

    Returns:
        Configured Session with retry behavior
    """
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
        raise_on_status=False,  # Let us handle status codes ourselves
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def read_image_file(path: str) -> bytes | None:
    """
    Read an image file and return its contents.

    Handles common errors and logs warnings.

    Args:
        path: Path to the image file

    Returns:
        Image data as bytes, or None if read failed
    """
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {path}")
        return None
    except OSError as e:
        logger.warning(f"Failed to read image: {e}")
        return None


def build_template_vars(event: dict[str, Any]) -> dict[str, Any]:
    """
    Build standard template variables from an event.

    This is the single source of truth for template variables used
    across all notification formatting (titles, messages, etc.).

    Args:
        event: Event dictionary with metadata

    Returns:
        Dictionary of template variables

    Available variables:
        {event_name} - Event definition name
        {event_type} - Event type (LINE_CROSS, ZONE_ENTER, etc.)
        {object_class} - Detected object class name
        {confidence} - Detection confidence (float)
        {confidence_pct} - Detection confidence as percentage string
        {zone} - Zone description (if applicable)
        {line} - Line description (if applicable)
        {timestamp} - Event timestamp
        {track_id} - Tracking ID
    """
    confidence = event.get("confidence", 0)
    return {
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


def format_template(template: str, event: dict[str, Any]) -> str:
    """
    Format a template string with event data.

    Args:
        template: Template string with {variable} placeholders
        event: Event dictionary with metadata

    Returns:
        Formatted string
    """
    try:
        return template.format(**build_template_vars(event))
    except KeyError as e:
        logger.warning(f"Template error: missing key {e}")
        return template


def format_title(template: str | None, event: dict[str, Any]) -> str:
    """
    Format notification title using template and event data.

    Args:
        template: Optional title template with {variable} placeholders.
                  If None, generates a default title.
        event: Event dictionary with metadata

    Returns:
        Formatted title string
    """
    if not template:
        # Default title based on event type
        event_name = event.get("_event_name", "Detection")
        obj_class = event.get("object_class_name", "object")
        return f"{event_name}: {obj_class}"

    return format_template(template, event)


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


__all__ = [
    "Notifier",
    "build_template_vars",
    "create_notifier",
    "create_notifiers",
    "create_retry_session",
    "format_template",
    "format_title",
    "read_image_file",
]
