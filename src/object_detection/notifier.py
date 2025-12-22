"""
Simple notification system for sending email alerts.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Handles sending email notifications via SMTP with cooldown tracking."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize email notifier with SMTP configuration.

        Args:
            config: Email configuration dictionary
        """
        self.enabled = config.get('enabled', False)
        self.smtp_server = config.get('smtp_server', '')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_address = config.get('from_address', self.username)
        self.to_addresses = config.get('to_addresses', [])
        self.use_tls = config.get('use_tls', True)

        # Track last notification time per zone/line to implement cooldown
        self._last_notification: Dict[str, datetime] = {}

    def should_notify(self, identifier: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last notification.

        Args:
            identifier: Unique identifier (e.g., "Z1" or "V1")
            cooldown_minutes: Minimum minutes between notifications

        Returns:
            True if notification should be sent
        """
        if identifier not in self._last_notification:
            return True

        elapsed = datetime.now() - self._last_notification[identifier]
        return elapsed >= timedelta(minutes=cooldown_minutes)

    def send(self, subject: str, body: str, event: Dict[str, Any], identifier: str) -> bool:
        """Send an email notification.

        Args:
            subject: Email subject
            body: Email body text
            event: Event data to include in email
            identifier: Unique identifier for cooldown tracking

        Returns:
            True if email sent successfully
        """
        if not self.enabled:
            logger.debug("Email notifications disabled")
            return False

        if not self.to_addresses:
            logger.warning("No email recipients configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = subject

            # Build body with event details
            full_body = body + "\n\n"
            full_body += "Event Details:\n"
            full_body += f"Time: {event.get('timestamp', 'N/A')}\n"
            full_body += f"Type: {event.get('event_type', 'N/A')}\n"
            full_body += f"Object: {event.get('object_class_name', 'N/A')}\n"

            # Add type-specific details
            if event.get('event_type') == 'LINE_CROSS':
                full_body += f"Line: {event.get('line_description', 'N/A')}\n"
                full_body += f"Direction: {event.get('direction', 'N/A')}\n"
            elif event.get('event_type') in ['ZONE_ENTER', 'ZONE_EXIT']:
                full_body += f"Zone: {event.get('zone_description', 'N/A')}\n"
                if 'dwell_time' in event:
                    full_body += f"Dwell Time: {event['dwell_time']:.1f}s\n"

            msg.attach(MIMEText(full_body, 'plain'))

            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            # Update last notification time
            self._last_notification[identifier] = datetime.now()

            logger.info(f"Email notification sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class NotificationManager:
    """Manages email notifications for zones and lines."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager with configuration.

        Args:
            config: Notification configuration dictionary
        """
        self.enabled = config.get('enabled', False)

        # Initialize email notifier
        email_config = config.get('email', {})
        self.email_notifier = EmailNotifier(email_config)

        if self.enabled and not self.email_notifier.enabled:
            logger.warning("Notifications enabled but email is not configured")
            self.enabled = False

    def check_and_notify(
        self,
        event: Dict[str, Any],
        zone_id: Optional[str] = None,
        line_id: Optional[str] = None,
        notify_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Check if event should trigger notification and send if needed.

        Args:
            event: Event dictionary
            zone_id: Zone ID if this is a zone event
            line_id: Line ID if this is a line event
            notify_config: Configuration dict with notify_email, cooldown_minutes, message
        """
        if not self.enabled or not notify_config:
            return

        # Check if this zone/line has email notifications enabled
        if not notify_config.get('notify_email', False):
            return

        # Get cooldown period (default 60 minutes)
        cooldown_minutes = notify_config.get('cooldown_minutes', 60)

        # Determine identifier for cooldown tracking
        identifier = zone_id or line_id
        if not identifier:
            return

        # Check cooldown
        if not self.email_notifier.should_notify(identifier, cooldown_minutes):
            logger.debug(f"Skipping notification for {identifier} - cooldown active")
            return

        # Build notification message
        event_type = event.get('event_type', '')
        obj_name = event.get('object_class_name', 'object')

        # Use custom message if provided, otherwise generate default
        if 'message' in notify_config:
            message = notify_config['message']
        else:
            # Generate default message based on event type
            if event_type == 'ZONE_ENTER':
                zone_desc = event.get('zone_description', 'zone')
                message = f"A {obj_name} entered {zone_desc}"
            elif event_type == 'ZONE_EXIT':
                zone_desc = event.get('zone_description', 'zone')
                dwell = event.get('dwell_time', 0)
                message = f"A {obj_name} exited {zone_desc} (dwell: {dwell:.1f}s)"
            elif event_type == 'LINE_CROSS':
                line_desc = event.get('line_description', 'line')
                direction = event.get('direction', '')
                message = f"A {obj_name} crossed {line_desc} ({direction})"
            else:
                message = f"Event detected: {obj_name}"

        # Subject line
        description = event.get('zone_description') or event.get('line_description', 'Detection')
        subject = f"Object Detection Alert: {description}"

        # Send notification
        self.email_notifier.send(subject, message, event, identifier)
