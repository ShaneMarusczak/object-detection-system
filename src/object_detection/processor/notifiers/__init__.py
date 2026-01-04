"""
Notifiers - Pluggable notification backends for VLM analysis results.

Provides a common interface for different notification services:
- ntfy: Push notifications via ntfy.sh
- webhook: Generic HTTP webhooks
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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


__all__ = ["Notifier", "create_notifier", "create_notifiers"]
