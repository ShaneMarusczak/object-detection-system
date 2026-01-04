"""
Webhook Notifier - Generic HTTP webhook notifications.

Sends JSON payloads to configured webhook endpoints.
Compatible with Home Assistant, IFTTT, Zapier, and custom endpoints.
"""

import base64
import logging
from typing import Any

import requests

from . import Notifier

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

        logger.debug(f"WebhookNotifier initialized: {self._id} -> {self._url}")

    @property
    def id(self) -> str:
        return self._id

    def _format_title(self, event: dict[str, Any]) -> str:
        """Format title using template and event data."""
        if not self._title_template:
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
            return self._title_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Title template error: missing key {e}")
            return self._title_template

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
        title = self._format_title(event)

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
            try:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    payload["image_base64"] = image_data
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to read image: {e}")

        try:
            response = requests.post(
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
