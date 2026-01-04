"""
ntfy.sh Notifier - Push notifications via ntfy.sh service.

Sends notifications to ntfy.sh topics with optional image attachments.
See: https://ntfy.sh/
"""

import logging
from datetime import datetime
from typing import Any

import requests

from . import Notifier

logger = logging.getLogger(__name__)

# ntfy.sh API endpoint
NTFY_BASE_URL = "https://ntfy.sh"


class NtfyNotifier(Notifier):
    """
    Notifier that sends push notifications via ntfy.sh.

    Config options:
        id: Notifier identifier
        type: "ntfy"
        topic: ntfy topic name (required)
        priority: min/low/default/high/urgent
        title_template: Optional title with {variable} placeholders
    """

    def __init__(self, config: dict[str, Any]):
        self._id = config["id"]
        self._topic = config["topic"]
        self._priority = config.get("priority", "default")
        self._title_template = config.get("title_template")
        self._timeout = 10  # seconds

        self._url = f"{NTFY_BASE_URL}/{self._topic}"
        logger.debug(f"NtfyNotifier initialized: {self._id} -> {self._topic}")

    @property
    def id(self) -> str:
        return self._id

    def _format_title(self, event: dict[str, Any]) -> str:
        """Format title using template and event data."""
        if not self._title_template:
            # Default title based on event type
            event_name = event.get("_event_name", "Detection")
            obj_class = event.get("object_class_name", "object")
            return f"{event_name}: {obj_class}"

        try:
            # Flatten event for easy template access
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
        Send notification to ntfy.sh.

        Sends image first (if available), then analysis text.

        Args:
            analysis: VLM analysis text
            image_path: Optional path to image file
            event: Full event dictionary

        Returns:
            True if notification was sent successfully
        """
        title = self._format_title(event)
        success = True

        # Send image if available
        if image_path:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"

                with open(image_path, "rb") as f:
                    response = requests.post(
                        self._url,
                        data=f.read(),
                        headers={
                            "Title": title,
                            "Priority": self._priority,
                            "Filename": filename,
                        },
                        timeout=self._timeout,
                    )

                if not response.ok:
                    logger.warning(
                        f"ntfy image upload failed: {response.status_code} {response.text}"
                    )
                    success = False
                else:
                    logger.debug(f"ntfy image sent to {self._topic}")

            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}")
            except requests.RequestException as e:
                logger.error(f"ntfy image upload error: {e}")
                success = False

        # Send analysis text
        try:
            response = requests.post(
                self._url,
                data=analysis,
                headers={
                    "Priority": self._priority,
                },
                timeout=self._timeout,
            )

            if not response.ok:
                logger.warning(
                    f"ntfy text send failed: {response.status_code} {response.text}"
                )
                success = False
            else:
                logger.debug(f"ntfy text sent to {self._topic}")

        except requests.RequestException as e:
            logger.error(f"ntfy text send error: {e}")
            success = False

        return success
