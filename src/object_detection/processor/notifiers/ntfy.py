"""
ntfy.sh Notifier - Push notifications via ntfy.sh service.

Sends notifications to ntfy.sh topics with optional image attachments.
See: https://ntfy.sh/
"""

import logging
from datetime import datetime
from typing import Any

import requests

from . import Notifier, create_retry_session, format_title, read_image_file

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
        self._session = create_retry_session()

        logger.debug(f"NtfyNotifier initialized: {self._id} -> {self._topic}")

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
        Send notification to ntfy.sh.

        Sends image first (if available), then analysis text.

        Args:
            analysis: VLM analysis text
            image_path: Optional path to image file
            event: Full event dictionary

        Returns:
            True if notification was sent successfully
        """
        title = format_title(self._title_template, event)
        success = True

        # Send image if available
        if image_path:
            image_data = read_image_file(image_path)
            if image_data:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"

                    response = self._session.post(
                        self._url,
                        data=image_data,
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

                except requests.RequestException as e:
                    logger.error(f"ntfy image upload error: {e}")
                    success = False

        # Send analysis text
        try:
            response = self._session.post(
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
