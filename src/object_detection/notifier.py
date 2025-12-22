"""
Notification system for sending alerts based on event patterns.
Supports email and SMS notifications with pattern matching and rate limiting.
"""

import logging
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NotificationRule:
    """Represents a single notification rule with pattern matching."""

    def __init__(self, rule_config: Dict[str, Any]):
        """Initialize notification rule from configuration.

        Args:
            rule_config: Dictionary containing rule configuration
        """
        self.name = rule_config.get('name', 'Unnamed Rule')
        self.enabled = rule_config.get('enabled', True)
        self.conditions = rule_config.get('conditions', {})
        self.message = rule_config.get('message', 'Event detected')
        self.cooldown_minutes = rule_config.get('cooldown_minutes', 60)
        self.last_triggered: Optional[datetime] = None

    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if an event matches this rule's conditions.

        Args:
            event: Event dictionary to check

        Returns:
            True if event matches all conditions
        """
        if not self.enabled:
            return False

        # Check cooldown period
        if self.last_triggered:
            elapsed = datetime.now() - self.last_triggered
            if elapsed < timedelta(minutes=self.cooldown_minutes):
                return False

        # Check all conditions
        for key, value in self.conditions.items():
            if key not in event:
                return False

            # Handle different comparison types
            if isinstance(value, dict):
                # Complex comparisons (>, <, >=, <=, !=)
                event_value = event[key]
                for op, compare_value in value.items():
                    if op == 'gt' and not (event_value > compare_value):
                        return False
                    elif op == 'gte' and not (event_value >= compare_value):
                        return False
                    elif op == 'lt' and not (event_value < compare_value):
                        return False
                    elif op == 'lte' and not (event_value <= compare_value):
                        return False
                    elif op == 'ne' and not (event_value != compare_value):
                        return False
                    elif op == 'eq' and not (event_value == compare_value):
                        return False
            elif isinstance(value, list):
                # Value must be in list
                if event[key] not in value:
                    return False
            else:
                # Simple equality check
                if event[key] != value:
                    return False

        return True

    def trigger(self) -> str:
        """Mark rule as triggered and return formatted message.

        Returns:
            Formatted notification message
        """
        self.last_triggered = datetime.now()
        return self.message


class EmailNotifier:
    """Handles sending email notifications via SMTP."""

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

    def send(self, subject: str, body: str, event: Optional[Dict[str, Any]] = None) -> bool:
        """Send an email notification.

        Args:
            subject: Email subject
            body: Email body text
            event: Optional event data to include in email

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

            # Build body with event details if provided
            full_body = body
            if event:
                full_body += "\n\nEvent Details:\n"
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

            logger.info(f"Email notification sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class SMSNotifier:
    """Handles sending SMS notifications via Twilio."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SMS notifier with Twilio configuration.

        Args:
            config: SMS configuration dictionary
        """
        self.enabled = config.get('enabled', False)
        self.account_sid = config.get('account_sid', '')
        self.auth_token = config.get('auth_token', '')
        self.from_number = config.get('from_number', '')
        self.to_numbers = config.get('to_numbers', [])
        self.client = None

        if self.enabled:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
            except ImportError:
                logger.error("Twilio library not installed. Install with: pip install twilio")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.enabled = False

    def send(self, message: str) -> bool:
        """Send an SMS notification.

        Args:
            message: SMS message text (max 160 chars recommended)

        Returns:
            True if SMS sent successfully
        """
        if not self.enabled:
            logger.debug("SMS notifications disabled")
            return False

        if not self.client:
            logger.warning("Twilio client not initialized")
            return False

        if not self.to_numbers:
            logger.warning("No SMS recipients configured")
            return False

        try:
            # Truncate message if too long
            if len(message) > 160:
                message = message[:157] + "..."

            # Send to all recipients
            for to_number in self.to_numbers:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=to_number
                )

            logger.info(f"SMS notification sent: {message[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")
            return False


class NotificationManager:
    """Manages notification rules and dispatches alerts."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager with configuration.

        Args:
            config: Notification configuration dictionary
        """
        self.enabled = config.get('enabled', False)

        # Initialize notifiers
        email_config = config.get('email', {})
        sms_config = config.get('sms', {})

        self.email_notifier = EmailNotifier(email_config)
        self.sms_notifier = SMSNotifier(sms_config)

        # Load rules
        self.rules: List[NotificationRule] = []
        for rule_config in config.get('rules', []):
            self.rules.append(NotificationRule(rule_config))

        logger.info(f"Notification manager initialized with {len(self.rules)} rules")

    def process_event(self, event: Dict[str, Any]) -> None:
        """Process an event and trigger notifications if rules match.

        Args:
            event: Event dictionary to process
        """
        if not self.enabled:
            return

        # Check each rule
        for rule in self.rules:
            if rule.matches(event):
                message = rule.trigger()
                self._send_notification(message, event, rule)

    def _send_notification(self, message: str, event: Dict[str, Any], rule: NotificationRule) -> None:
        """Send notification via configured channels.

        Args:
            message: Notification message
            event: Event that triggered notification
            rule: Rule that was triggered
        """
        subject = f"Object Detection Alert: {rule.name}"

        # Send email
        if self.email_notifier.enabled:
            self.email_notifier.send(subject, message, event)

        # Send SMS (shorter message)
        if self.sms_notifier.enabled:
            sms_message = f"{rule.name}: {message}"
            self.sms_notifier.send(sms_message)
