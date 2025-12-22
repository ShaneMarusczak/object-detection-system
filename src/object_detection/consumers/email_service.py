"""
Shared Email Service
Handles email sending via SMTP for all email consumers.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class EmailService:
    """Handles sending emails via SMTP."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize email service with SMTP configuration.

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

    def send(self, subject: str, body: str) -> bool:
        """Send an email.

        Args:
            subject: Email subject
            body: Email body text (plain text or formatted)

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

            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_event_notification(self, event: Dict[str, Any], custom_message: str = None) -> bool:
        """Send notification for a single event.

        Args:
            event: Enriched event dictionary
            custom_message: Optional custom message (otherwise auto-generate)

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

        # Subject
        description = event.get('zone_description') or event.get('line_description', 'Detection')
        subject = f"Object Detection Alert: {description}"

        return self.send(subject, body)

    def send_digest(self, period: str, stats: Dict[str, Any], frame_urls: Dict[str, str] = None) -> bool:
        """Send a digest email with aggregated statistics and photo links.

        Args:
            period: Time period description (e.g., "Last Hour", "Last 24 Hours")
            stats: Dictionary with statistics to include
            frame_urls: Optional dictionary mapping event_id to frame URL/path

        Returns:
            True if email sent successfully
        """
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

        # Photo links (if available)
        if frame_urls:
            body += "Captured Frames:\n"
            body += "=" * 50 + "\n"
            events = stats.get('events', [])

            # Group frames by zone/line for better organization
            frames_by_location = {}
            for event in events:
                event_id = f"{event['timestamp']}_{event['track_id']}"
                if event_id in frame_urls:
                    location = event.get('zone_description') or event.get('line_description', 'Unknown')
                    if location not in frames_by_location:
                        frames_by_location[location] = []

                    frames_by_location[location].append({
                        'timestamp': event['timestamp'],
                        'object': event.get('object_class_name', 'unknown'),
                        'event_type': event.get('event_type', 'N/A'),
                        'url': frame_urls[event_id]
                    })

            # Format frame links by location
            for location, frames in sorted(frames_by_location.items()):
                body += f"\n{location}:\n"
                for frame in frames:
                    body += f"  {frame['timestamp']} - {frame['object']} ({frame['event_type']})\n"
                    body += f"  Photo: {frame['url']}\n\n"

        # Time range
        start_time = stats.get('start_time')
        end_time = stats.get('end_time')
        if start_time and end_time:
            body += f"Period: {start_time} to {end_time}\n"

        return self.send(subject, body)
