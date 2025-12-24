"""
Immediate Email Handler

Fire-and-forget email sending for events. Handles cooldowns inline,
spawns threads for actual SMTP work to avoid blocking.
"""

import logging
import os
import threading
from datetime import datetime, timedelta

from ..utils.constants import DEFAULT_TEMP_FRAME_DIR
from .email_service import EmailService

logger = logging.getLogger(__name__)


class ImmediateEmailHandler:
    """
    Handles immediate email notifications with cooldown tracking.
    Spawns daemon threads for actual email sending (fire-and-forget).
    """

    def __init__(self, notification_config: dict, temp_frame_dir: str = None):
        email_config = notification_config.get("email", {})
        self.email_service = EmailService(email_config)
        self.temp_frame_dir = temp_frame_dir or DEFAULT_TEMP_FRAME_DIR
        self._last_notification: dict[str, datetime] = {}
        self._lock = threading.Lock()

        if self.email_service.enabled:
            logger.info("Immediate email handler: Enabled")
        else:
            logger.info("Immediate email handler: Disabled (no email config)")

    def _should_notify(self, identifier: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last notification."""
        with self._lock:
            if identifier not in self._last_notification:
                return True
            elapsed = datetime.now() - self._last_notification[identifier]
            return elapsed >= timedelta(minutes=cooldown_minutes)

    def _mark_notified(self, identifier: str):
        """Mark that notification was sent."""
        with self._lock:
            self._last_notification[identifier] = datetime.now()

    def handle_event(self, event: dict) -> None:
        """
        Handle an event that needs immediate email notification.
        Checks cooldown, then spawns a thread to send email.
        """
        # Extract email config from event metadata
        email_action_config = event.get("_email_immediate_config", {})
        cooldown_minutes = email_action_config.get("cooldown_minutes", 60)
        custom_message = email_action_config.get("message")
        custom_subject = email_action_config.get("subject")
        include_frame = email_action_config.get("include_frame", False)

        # Build cooldown identifier from (track_id, zone/line)
        track_id = event.get("track_id")
        zone = event.get("zone_description", "")
        line = event.get("line_description", "")
        location = zone or line
        identifier = f"{track_id}:{location}"

        # Check cooldown
        if not self._should_notify(identifier, cooldown_minutes):
            logger.debug(f"Skipping email due to cooldown: {identifier}")
            return

        # Mark as notified before sending (optimistic)
        self._mark_notified(identifier)

        # Get frame data if requested
        frame_data = None
        if include_frame:
            frame_id = event.get("frame_id")
            if frame_id:
                frame_path = os.path.join(self.temp_frame_dir, f"{frame_id}.jpg")
                if os.path.exists(frame_path):
                    try:
                        with open(frame_path, "rb") as f:
                            frame_data = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read frame {frame_id}: {e}")

        # Fire-and-forget: spawn daemon thread to send email
        thread = threading.Thread(
            target=self._send_email,
            args=(event, custom_message, frame_data, custom_subject),
            daemon=True,
        )
        thread.start()
        logger.info(f"Email queued for: {event.get('event_definition')}")

    def _send_email(
        self,
        event: dict,
        custom_message: str | None,
        frame_data: bytes | None,
        custom_subject: str | None,
    ) -> None:
        """Send email in background thread."""
        try:
            if self.email_service.send_event_notification(
                event, custom_message, frame_data, custom_subject
            ):
                logger.debug("Email sent successfully")
            else:
                logger.warning("Failed to send email notification")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
