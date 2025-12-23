"""
PDF Report Generator
Generates PDF reports synchronously on shutdown covering the entire run.
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from collections import Counter

from .frame_service import FrameService

logger = logging.getLogger(__name__)

# Import reportlab (optional dependency)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed - PDF reports disabled")


def generate_pdf_reports(json_dir: str, config: dict, start_time: datetime) -> None:
    """
    Generate PDF reports synchronously at shutdown.

    This is called directly by the dispatcher after all other consumers have
    finished, so it can take as long as needed without timeout.

    Args:
        json_dir: Directory containing JSON log files
        config: Consumer configuration with 'pdf_reports' list
        start_time: When the run started (for filtering events)
    """
    if not REPORTLAB_AVAILABLE:
        logger.error("reportlab not installed - cannot generate PDF reports")
        return

    # Get PDF report configurations
    pdf_report_configs = config.get('pdf_reports', [])

    if not pdf_report_configs:
        logger.warning("No PDF report configurations found")
        return

    end_time = datetime.now(timezone.utc)
    logger.info(f"Generating {len(pdf_report_configs)} PDF report(s)...")

    # Create FrameService - metadata should exist after all frames saved
    frame_service_config = config.get('frame_service_config', {})
    frame_service = FrameService(frame_service_config) if frame_service_config else None

    for i, report_config in enumerate(pdf_report_configs, 1):
        report_id = report_config.get('id', 'report')
        event_names = report_config.get('events', [])

        logger.info(f"[{i}/{len(pdf_report_configs)}] Processing report '{report_id}'...")

        # Aggregate all events from the run
        stats = _aggregate_from_json(json_dir, start_time, end_time, event_names)

        if stats['total_events'] == 0:
            logger.info(f"  No events for report '{report_id}' - skipping")
            continue

        logger.info(f"  Found {stats['total_events']} events to include")

        # Get frame data if photos enabled
        frame_data_map = {}
        if report_config.get('photos') and frame_service and stats.get('events'):
            logger.info(f"  Loading {len(stats['events'])} frames for photos...")
            frame_paths = frame_service.get_frame_paths_for_events(stats['events'])
            loaded = 0
            for event_id, path in frame_paths.items():
                frame_bytes = frame_service.read_frame_bytes(event_id)
                if frame_bytes:
                    frame_data_map[event_id] = frame_bytes
                    loaded += 1
            logger.info(f"  Loaded {loaded} frames")

        # Generate PDF
        output_dir = report_config.get('output_dir', 'reports')
        title = report_config.get('title', 'Object Detection Report')

        logger.info(f"  Building PDF ({stats['total_events']} events, {len(frame_data_map)} photos)...")
        pdf_path = _generate_pdf(
            output_dir=output_dir,
            title=title,
            stats=stats,
            frame_data_map=frame_data_map,
            start_time=start_time,
            end_time=end_time
        )

        if pdf_path:
            logger.info(f"  Report saved: {pdf_path}")
        else:
            logger.warning(f"  Failed to generate report '{report_id}'")

    logger.info("PDF report generation complete")


def _aggregate_from_json(json_dir: str, start_time: datetime, end_time: datetime,
                         event_names: List[str] = None) -> Dict:
    """
    Aggregate statistics from JSON log files within time window.

    Args:
        json_dir: Directory containing JSON log files
        start_time: Start of time window
        end_time: End of time window
        event_names: Optional list of event definition names to include

    Returns:
        Dictionary with aggregated statistics and events list
    """
    if event_names is None:
        event_names = []

    total_events = 0
    events_by_type = Counter()
    events_by_class = Counter()
    events_by_zone = Counter()
    events_by_line = Counter()
    events_by_track = Counter()
    track_classes = {}
    matched_events = []

    first_event_time = None
    last_event_time = None

    json_path = Path(json_dir)
    if not json_path.exists():
        logger.warning(f"JSON directory does not exist: {json_dir}")
        return _empty_stats()

    jsonl_files = sorted(json_path.glob("events_*.jsonl"))

    if not jsonl_files:
        logger.debug(f"No JSON log files found in {json_dir}")
        return _empty_stats()

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        event_time_str = event.get('timestamp')
                        if not event_time_str:
                            continue

                        event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                        # Ensure timezone-aware for comparison
                        if event_time.tzinfo is None:
                            event_time = event_time.replace(tzinfo=timezone.utc)

                        if event_time < start_time or event_time > end_time:
                            continue

                        if event_names and event.get('event_definition') not in event_names:
                            continue

                        if first_event_time is None or event_time < first_event_time:
                            first_event_time = event_time
                        if last_event_time is None or event_time > last_event_time:
                            last_event_time = event_time

                        total_events += 1
                        events_by_type[event.get('event_type', 'UNKNOWN')] += 1
                        events_by_class[event.get('object_class_name', 'unknown')] += 1

                        if 'zone_description' in event:
                            events_by_zone[event['zone_description']] += 1
                        if 'line_description' in event:
                            events_by_line[event['line_description']] += 1

                        track_id = event.get('track_id')
                        if track_id:
                            events_by_track[track_id] += 1
                            track_classes[track_id] = event.get('object_class_name', 'unknown')

                        matched_events.append(event)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing event: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Error reading {jsonl_file}: {e}")
            continue

    top_tracks = [
        (track_id, track_classes.get(track_id, 'unknown'), count)
        for track_id, count in events_by_track.most_common(10)
    ]

    return {
        'total_events': total_events,
        'events_by_type': dict(events_by_type),
        'events_by_class': dict(events_by_class),
        'events_by_zone': dict(events_by_zone),
        'events_by_line': dict(events_by_line),
        'top_tracks': top_tracks,
        'start_time': first_event_time.isoformat() if first_event_time else None,
        'end_time': last_event_time.isoformat() if last_event_time else None,
        'events': matched_events
    }


def _empty_stats() -> Dict:
    """Return empty statistics dictionary."""
    return {
        'total_events': 0,
        'events_by_type': {},
        'events_by_class': {},
        'events_by_zone': {},
        'events_by_line': {},
        'top_tracks': [],
        'start_time': None,
        'end_time': None,
        'events': []
    }


def _generate_pdf(output_dir: str, title: str, stats: Dict,
                  frame_data_map: Dict[str, bytes],
                  start_time: datetime, end_time: datetime) -> str:
    """
    Generate a PDF report with statistics and embedded images.

    Args:
        output_dir: Directory for PDF output
        title: Report title
        stats: Aggregated statistics
        frame_data_map: Dictionary mapping event_id to JPEG bytes
        start_time: Report period start
        end_time: Report period end

    Returns:
        Path to generated PDF, or None on failure
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.pdf"
        pdf_path = os.path.join(output_dir, filename)

        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
        normal_style = styles['Normal']

        story = []

        # Title
        story.append(Paragraph(title, title_style))

        # Period
        period_text = f"Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
        story.append(Paragraph(period_text, normal_style))
        story.append(Spacer(1, 20))

        # Summary
        story.append(Paragraph("Summary", heading_style))
        story.append(Paragraph(f"Total Events: {stats['total_events']}", normal_style))
        story.append(Spacer(1, 10))

        # Events by type table
        by_type = stats.get('events_by_type', {})
        if by_type:
            story.append(Paragraph("Events by Type", heading_style))
            table_data = [['Event Type', 'Count']]
            for event_type, count in sorted(by_type.items()):
                table_data.append([event_type, str(count)])
            table = Table(table_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))

        # Events by object class table
        by_class = stats.get('events_by_class', {})
        if by_class:
            story.append(Paragraph("Events by Object", heading_style))
            table_data = [['Object Class', 'Count']]
            for obj_class, count in sorted(by_class.items(), key=lambda x: x[1], reverse=True):
                table_data.append([obj_class, str(count)])
            table = Table(table_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))

        # Zone activity
        by_zone = stats.get('events_by_zone', {})
        if by_zone:
            story.append(Paragraph("Zone Activity", heading_style))
            table_data = [['Zone', 'Events']]
            for zone, count in sorted(by_zone.items(), key=lambda x: x[1], reverse=True):
                table_data.append([zone, str(count)])
            table = Table(table_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))

        # Line crossings
        by_line = stats.get('events_by_line', {})
        if by_line:
            story.append(Paragraph("Line Crossings", heading_style))
            table_data = [['Line', 'Crossings']]
            for line, count in sorted(by_line.items(), key=lambda x: x[1], reverse=True):
                table_data.append([line, str(count)])
            table = Table(table_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))

        # Event timeline with inline photos
        events = stats.get('events', [])
        if events:
            total_events = len(events)
            story.append(Paragraph(f"Event Timeline ({total_events} events)", heading_style))

            # Style for events without photos (smaller, more compact)
            event_style = ParagraphStyle('Event', parent=normal_style, fontSize=9, leading=12)
            photo_caption_style = ParagraphStyle('PhotoCaption', parent=normal_style, fontSize=10, fontName='Helvetica-Bold')

            for i, event in enumerate(events, 1):
                location = event.get('zone_description') or event.get('line_description', 'detection')
                obj_class = event.get('object_class_name', 'unknown')
                direction = event.get('direction', '')
                direction_str = f" ({direction})" if direction else ""
                timestamp = event.get('timestamp', '')
                # Format timestamp more readably
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp

                event_id = f"{event['timestamp']}_{event['track_id']}"
                event_num = f"#{i}"

                if event_id in frame_data_map:
                    # Event with photo - more prominent
                    frame_bytes = frame_data_map[event_id]
                    if frame_bytes:
                        caption = f"{event_num} | {time_str} - {obj_class} at {location}{direction_str}"
                        story.append(Paragraph(caption, photo_caption_style))

                        try:
                            img = Image(io.BytesIO(frame_bytes))
                            img.drawWidth = 4 * inch
                            img.drawHeight = 3 * inch
                            story.append(img)
                            story.append(Spacer(1, 15))
                        except Exception as e:
                            logger.warning(f"Failed to embed image: {e}")
                else:
                    # Event without photo - compact text line
                    line = f"{event_num} | {time_str} - {obj_class} at {location}{direction_str}"
                    story.append(Paragraph(line, event_style))

        # Build PDF
        doc.build(story)
        return pdf_path

    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}", exc_info=True)
        return None
