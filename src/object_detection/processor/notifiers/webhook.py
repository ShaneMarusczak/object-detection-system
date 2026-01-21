"""
Webhook Notifier - Generic HTTP webhook notifications.

Sends JSON payloads to configured webhook endpoints.
Compatible with Home Assistant, IFTTT, Zapier, and custom endpoints.
"""

import base64
import logging
from typing import Any

import requests

from . import Notifier, create_retry_session, format_title, read_image_file

logger = logging.getLogger(__name__)


class WebhookNotifier(Notifier):
    """
    Notifier that sends JSON payloads to HTTP webhooks.

    Config options:
        id: Notifier identifier
        type: "webhook"
        url: Webhook endpoint URL (required)
        priority: Priority level (included in payload)
        title_template: Optional title with {variable} placeholders
    """

    def __init__(self, config: dict[str, Any]):
        self._id = config["id"]
        self._url = config["url"]
        self._priority = config.get("priority", "default")
        self._title_template = config.get("title_template")
        self._timeout = 10  # seconds
        self._include_image = config.get("include_image", False)
        self._session = create_retry_session()

        logger.debug(f"WebhookNotifier initialized: {self._id} -> {self._url}")

    @property
    def id(self) -> str:
        return self._id

    def send(
        self,
        analysis: str,
        image_path: str | None,
        event: dict[str, Any],
    ) -> bool:
        """
        Send notification to webhook endpoint.

        Sends a JSON payload with event data and analysis.

        Args:
            analysis: VLM analysis text
            image_path: Optional path to image file
            event: Full event dictionary

        Returns:
            True if notification was sent successfully
        """
        title = format_title(self._title_template, event)

        # Build payload
        payload = {
            "title": title,
            "analysis": analysis,
            "priority": self._priority,
            "event": {
                "name": event.get("_event_name"),
                "type": event.get("event_type"),
                "object_class": event.get("object_class_name"),
                "confidence": event.get("confidence"),
                "timestamp": event.get("timestamp"),
                "zone": event.get("zone_description"),
                "line": event.get("line_description"),
                "track_id": event.get("track_id"),
            },
        }

        # Optionally include base64-encoded image
        if self._include_image and image_path:
            image_data = read_image_file(image_path)
            if image_data:
                payload["image_base64"] = base64.b64encode(image_data).decode("utf-8")

        try:
            response = self._session.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )

            if not response.ok:
                logger.warning(
                    f"Webhook failed: {response.status_code} {response.text[:100]}"
                )
                return False

            logger.debug(f"Webhook sent to {self._url}")
            return True

        except requests.RequestException as e:
            logger.error(f"Webhook error: {e}")
            return False
