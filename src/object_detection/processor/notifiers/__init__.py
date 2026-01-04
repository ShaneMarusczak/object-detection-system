"""
Notifiers - Pluggable notification backends.

Provides a common interface for different notification services:
- ntfy: Push notifications via ntfy.sh
- webhook: Generic HTTP webhooks

Used by both VLM analyzer (sends analysis text) and direct notifier (sends formatted messages).
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_DELAY = 1.0  # seconds


def with_retry(
    func: Callable[[], requests.Response],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> requests.Response:
    """
    Execute a request function with exponential backoff retry.

    Retries on transient network errors (timeout, connection error).
    Does NOT retry on 4xx client errors (bad request, unauthorized, etc).

    Args:
        func: Callable that performs the request and returns Response
        max_retries: Maximum number of retry attempts (default: 2)
        base_delay: Initial delay in seconds, doubles each retry (default: 1.0)

    Returns:
        The successful Response object

    Raises:
        requests.RequestException: If all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = func()
            # Don't retry client errors (4xx) - those won't fix themselves
            if response.status_code < 500:
                return response
            # Server error (5xx) - worth retrying
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Server error {response.status_code}, retry {attempt + 1}/{max_retries} in {delay}s"
                )
                time.sleep(delay)
                continue
            return response

        except (requests.Timeout, requests.ConnectionError) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Network error, retry {attempt + 1}/{max_retries} in {delay}s: {e}"
                )
                time.sleep(delay)
            else:
                raise

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise requests.RequestException("Retry exhausted")


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


__all__ = [
    "Notifier",
    "create_notifier",
    "create_notifiers",
    "format_title",
    "with_retry",
]
