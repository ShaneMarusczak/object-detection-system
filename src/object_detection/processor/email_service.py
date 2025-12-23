"""
Shared Email Service
Handles email sending via SMTP for all email consumers.
Supports embedding images directly in emails.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EmailService:
    """Handles sending emails via SMTP with image attachment support."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize email service with SMTP configuration.

        Args:
            config: Email configuration dictionary
        """
        self.enabled = config.get('enabled', False)
        self.smtp_server = config.get('smtp_server', '')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')

        # Password can come from config directly or from environment variable
        self.password = config.get('password', '')
        password_env = config.get('password_env')
        if password_env and not self.password:
            self.password = os.environ.get(password_env, '')

        self.from_address = config.get('from_address', self.username)
        self.to_addresses = config.get('to_addresses', [])
        self.use_tls = config.get('use_tls', True)

    def send(self, subject: str, body: str, attachments: List[Dict] = None) -> bool:
        """Send an email with optional attachments.

        Args:
            subject: Email subject
            body: Email body text (plain text or HTML)
            attachments: Optional list of attachments, each dict with:
                - 'data': bytes of the file
                - 'filename': name for the attachment
                - 'content_type': MIME type (e.g., 'image/jpeg')

        Returns:
            True if email sent successfully
        """
        if not self.enabled:
            logger.debug("Email service disabled")
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

            msg.attach(MIMEText(body, 'plain'))

            # Add attachments
            if attachments:
                for attachment in attachments:
                    self._add_attachment(msg, attachment)

            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            attachment_count = len(attachments) if attachments else 0
            logger.info(f"Email sent: {subject} ({attachment_count} attachments)")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _add_attachment(self, msg: MIMEMultipart, attachment: Dict) -> None:
        """Add an attachment to the email message."""
        data = attachment.get('data')
        filename = attachment.get('filename', 'attachment')
        content_type = attachment.get('content_type', 'application/octet-stream')

        if not data:
            return

        maintype, subtype = content_type.split('/', 1) if '/' in content_type else ('application', 'octet-stream')

        if maintype == 'image':
            part = MIMEImage(data, _subtype=subtype, name=filename)
        else:
            part = MIMEBase(maintype, subtype)
            part.set_payload(data)
            encoders.encode_base64(part)

        part.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(part)

    def send_event_notification(self, event: Dict[str, Any], custom_message: str = None,
                                 frame_data: bytes = None, custom_subject: str = None) -> bool:
        """Send notification for a single event with optional frame attachment.

        Args:
            event: Enriched event dictionary
            custom_message: Optional custom message (otherwise auto-generate)
            frame_data: Optional JPEG bytes to attach
            custom_subject: Optional custom subject line

        Returns:
            True if email sent successfully
        """
        event_type = event.get('event_type', '')
        obj_name = event.get('object_class_name', 'object')

        # Generate message
        if custom_message:
            message = custom_message
        else:
            # Auto-generate based on event type
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

        # Build email body with event details
        body = message + "\n\n"
        body += "Event Details:\n"
        body += f"Time: {event.get('timestamp', 'N/A')}\n"
        body += f"Type: {event.get('event_type', 'N/A')}\n"
        body += f"Object: {event.get('object_class_name', 'N/A')}\n"

        # Add type-specific details
        if event_type == 'LINE_CROSS':
            body += f"Line: {event.get('line_description', 'N/A')}\n"
            body += f"Direction: {event.get('direction', 'N/A')}\n"
        elif event_type in ['ZONE_ENTER', 'ZONE_EXIT']:
            body += f"Zone: {event.get('zone_description', 'N/A')}\n"
            if 'dwell_time' in event:
                body += f"Dwell Time: {event['dwell_time']:.1f}s\n"

        if frame_data:
            body += "\nSee attached photo.\n"

        # Subject (use custom if provided)
        if custom_subject:
            subject = custom_subject
        else:
            description = event.get('zone_description') or event.get('line_description', 'Detection')
            subject = f"Object Detection Alert: {description}"

        # Build attachments
        attachments = []
        if frame_data:
            timestamp = event.get('timestamp', 'frame').replace(':', '-').replace('.', '-')
            attachments.append({
                'data': frame_data,
                'filename': f"{timestamp}_{obj_name}.jpg",
                'content_type': 'image/jpeg'
            })

        return self.send(subject, body, attachments)

    def send_digest(self, period: str, stats: Dict[str, Any],
                    frame_data_map: Dict[str, bytes] = None,
                    subject: str = None) -> bool:
        """Send a digest email with aggregated statistics and embedded photos.

        Args:
            period: Time period description (e.g., "Last Hour", "Last 24 Hours")
            stats: Dictionary with statistics to include
            frame_data_map: Optional dict mapping event_id to JPEG bytes
            subject: Optional custom subject line

        Returns:
            True if email sent successfully
        """
        if not subject:
            subject = f"Object Detection Digest: {period}"

        # Build digest body
        body = f"Activity Summary - {period}\n"
        body += "=" * 50 + "\n\n"

        # Total events
        total = stats.get('total_events', 0)
        body += f"Total Events: {total}\n\n"

        # Events by type
        by_type = stats.get('events_by_type', {})
        if by_type:
            body += "Events by Type:\n"
            for event_type, count in sorted(by_type.items()):
                body += f"  {event_type}: {count}\n"
            body += "\n"

        # Events by object class
        by_class = stats.get('events_by_class', {})
        if by_class:
            body += "Events by Object:\n"
            for obj_class, count in sorted(by_class.items(), key=lambda x: x[1], reverse=True):
                body += f"  {obj_class}: {count}\n"
            body += "\n"

        # Events by zone
        by_zone = stats.get('events_by_zone', {})
        if by_zone:
            body += "Zone Activity:\n"
            for zone, count in sorted(by_zone.items(), key=lambda x: x[1], reverse=True):
                body += f"  {zone}: {count} events\n"
            body += "\n"

        # Events by line
        by_line = stats.get('events_by_line', {})
        if by_line:
            body += "Line Crossings:\n"
            for line, count in sorted(by_line.items(), key=lambda x: x[1], reverse=True):
                body += f"  {line}: {count} crossings\n"
            body += "\n"

        # Top tracks (most active objects)
        top_tracks = stats.get('top_tracks', [])
        if top_tracks:
            body += "Most Active Objects:\n"
            for track_id, obj_class, count in top_tracks[:5]:
                body += f"  Track {track_id} ({obj_class}): {count} events\n"
            body += "\n"

        # Build attachments from frame data
        attachments = []
        if frame_data_map:
            body += f"\nAttached Photos: {len(frame_data_map)}\n"
            body += "=" * 50 + "\n"

            events = stats.get('events', [])

            # Create attachment list with context
            for event in events:
                event_id = f"{event['timestamp']}_{event['track_id']}"
                if event_id in frame_data_map:
                    frame_bytes = frame_data_map[event_id]
                    if frame_bytes:
                        location = event.get('zone_description') or event.get('line_description', 'detection')
                        obj_class = event.get('object_class_name', 'unknown')
                        timestamp = event['timestamp'].replace(':', '-').replace('.', '-')

                        # Add to body
                        body += f"\n{event['timestamp']} - {obj_class} at {location}\n"

                        attachments.append({
                            'data': frame_bytes,
                            'filename': f"{timestamp}_{obj_class}_{location}.jpg",
                            'content_type': 'image/jpeg'
                        })

        # Time range
        start_time = stats.get('start_time')
        end_time = stats.get('end_time')
        if start_time and end_time:
            body += f"\nPeriod: {start_time} to {end_time}\n"

        return self.send(subject, body, attachments)
