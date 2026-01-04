"""
HTML Report Generator
Generates HTML reports synchronously on shutdown covering the entire run.

No external dependencies needed. View in browser or print if needed.
"""

import base64
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .frame_service import FrameService

logger = logging.getLogger(__name__)

# HTML template with embedded CSS
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --header-color: #2C3E50;
  --border-color: #DEE2E6;
  --alt-row: #F8F9FA;
  --text-muted: #666;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 900px;
  margin: 0 auto;
  padding: 40px 20px;
  line-height: 1.5;
  color: #333;
}}
h1 {{ color: var(--header-color); margin-bottom: 5px; }}
.subtitle {{ color: var(--text-muted); margin-bottom: 20px; }}
.summary {{
  display: flex;
  gap: 20px;
  background: var(--alt-row);
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
  text-align: center;
}}
.summary-item {{ flex: 1; }}
.summary-value {{ font-size: 24px; font-weight: bold; color: var(--header-color); }}
.summary-label {{ font-size: 12px; color: var(--text-muted); }}
h2 {{
  color: var(--header-color);
  border-bottom: 2px solid var(--border-color);
  padding-bottom: 8px;
  margin-top: 30px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
}}
th {{
  background: var(--header-color);
  color: white;
  padding: 10px 12px;
  text-align: left;
  font-weight: 500;
}}
th:last-child {{ text-align: right; }}
td {{
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color);
}}
td:last-child {{ text-align: right; }}
tr:nth-child(even) {{ background: var(--alt-row); }}
.timeline {{ margin-top: 20px; }}
.event {{
  padding: 8px 0;
  border-bottom: 1px solid var(--border-color);
  font-size: 14px;
  color: var(--text-muted);
}}
.event-photo {{
  margin: 20px 0;
  padding: 15px;
  background: var(--alt-row);
  border-radius: 8px;
}}
.event-photo-caption {{
  font-weight: bold;
  color: var(--header-color);
  margin-bottom: 10px;
}}
.event-photo img {{
  max-width: 100%;
  border-radius: 4px;
  display: block;
}}
.no-data {{ color: var(--text-muted); font-style: italic; }}
@media print {{
  body {{ padding: 20px; }}
  .event-photo {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def generate_html_reports(json_dir: str, config: dict, start_time: datetime) -> None:
    """
    Generate HTML reports synchronously at shutdown.

    Args:
        json_dir: Directory containing JSON log files
        config: Consumer configuration with 'reports' list
        start_time: When the run started (for filtering events)
    """
    report_configs = config.get("reports", [])

    if not report_configs:
        return

    end_time = datetime.now(timezone.utc)
    logger.info(f"Generating {len(report_configs)} HTML report(s)...")

    # Create FrameService for photos
    frame_service_config = config.get("frame_service_config", {})
    frame_service = FrameService(frame_service_config) if frame_service_config else None

    for i, report_config in enumerate(report_configs, 1):
        report_id = report_config.get("id", "report")
        event_names = report_config.get("events", [])

        logger.info(f"[{i}/{len(report_configs)}] Processing report '{report_id}'...")

        # Aggregate events from JSON logs
        stats = _aggregate_from_json(json_dir, start_time, end_time, event_names)

        if stats["total_events"] == 0:
            logger.info(f"  No events for report '{report_id}' - skipping")
            continue

        logger.info(f"  Found {stats['total_events']} events to include")

        # Load frame data if photos enabled
        frame_data_map = {}
        if report_config.get("photos") and frame_service and stats.get("events"):
            logger.info("  Loading frames for photos...")
            frame_paths = frame_service.get_frame_paths_for_events(stats["events"])
            for event_id, path in frame_paths.items():
                frame_bytes = frame_service.read_frame_bytes(event_id)
                if frame_bytes:
                    frame_data_map[event_id] = frame_bytes
            logger.info(f"  Loaded {len(frame_data_map)} frames")

        # Generate HTML
        output_dir = report_config.get("output_dir", "reports")
        title = report_config.get("title", "Object Detection Report")

        html_path = _generate_html(
            output_dir=output_dir,
            title=title,
            stats=stats,
            frame_data_map=frame_data_map,
            start_time=start_time,
            end_time=end_time,
        )

        if html_path:
            logger.info(f"  Report saved: {html_path}")
        else:
            logger.warning(f"  Failed to generate report '{report_id}'")

    logger.info("HTML report generation complete")


def _aggregate_from_json(
    json_dir: str,
    start_time: datetime,
    end_time: datetime,
    event_names: list[str] | None = None,
) -> dict:
    """Aggregate statistics from JSON log files within time window."""
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
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        event_time_str = event.get("timestamp")
                        if not event_time_str:
                            continue

                        event_time = datetime.fromisoformat(
                            event_time_str.replace("Z", "+00:00")
                        )
                        if event_time.tzinfo is None:
                            event_time = event_time.replace(tzinfo=timezone.utc)

                        if event_time < start_time or event_time > end_time:
                            continue

                        if (
                            event_names
                            and event.get("event_definition") not in event_names
                        ):
                            continue

                        if first_event_time is None or event_time < first_event_time:
                            first_event_time = event_time
                        if last_event_time is None or event_time > last_event_time:
                            last_event_time = event_time

                        total_events += 1
                        events_by_type[event.get("event_type", "UNKNOWN")] += 1
                        events_by_class[event.get("object_class_name", "unknown")] += 1

                        if "zone_description" in event:
                            events_by_zone[event["zone_description"]] += 1
                        if "line_description" in event:
                            events_by_line[event["line_description"]] += 1

                        track_id = event.get("track_id")
                        if track_id:
                            events_by_track[track_id] += 1
                            track_classes[track_id] = event.get(
                                "object_class_name", "unknown"
                            )

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
        (track_id, track_classes.get(track_id, "unknown"), count)
        for track_id, count in events_by_track.most_common(10)
    ]

    return {
        "total_events": total_events,
        "events_by_type": dict(events_by_type),
        "events_by_class": dict(events_by_class),
        "events_by_zone": dict(events_by_zone),
        "events_by_line": dict(events_by_line),
        "top_tracks": top_tracks,
        "start_time": first_event_time.isoformat() if first_event_time else None,
        "end_time": last_event_time.isoformat() if last_event_time else None,
        "events": matched_events,
    }


def _empty_stats() -> dict:
    """Return empty statistics dictionary."""
    return {
        "total_events": 0,
        "events_by_type": {},
        "events_by_class": {},
        "events_by_zone": {},
        "events_by_line": {},
        "top_tracks": [],
        "start_time": None,
        "end_time": None,
        "events": [],
    }


def _generate_html(
    output_dir: str,
    title: str,
    stats: dict,
    frame_data_map: dict[str, bytes],
    start_time: datetime,
    end_time: datetime,
) -> str | None:
    """Generate an HTML report with statistics and embedded images."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.html"
        html_path = os.path.join(output_dir, filename)

        # Build content
        content = []

        # Title and subtitle
        duration = (end_time - start_time).total_seconds()
        if duration >= 3600:
            duration_str = f"{duration / 3600:.1f} hours"
        else:
            duration_str = f"{duration / 60:.0f} minutes"

        period = f"{start_time.strftime('%Y-%m-%d %H:%M')} — {end_time.strftime('%H:%M')} ({duration_str})"

        content.append(f"<h1>{_escape(title)}</h1>")
        content.append(f'<p class="subtitle">{period}</p>')

        # Summary boxes
        content.append('<div class="summary">')
        content.append(f"""<div class="summary-item">
            <div class="summary-value">{stats["total_events"]}</div>
            <div class="summary-label">Total Events</div>
        </div>""")
        content.append(f"""<div class="summary-item">
            <div class="summary-value">{len(stats.get("events_by_class", {}))}</div>
            <div class="summary-label">Object Types</div>
        </div>""")
        content.append(f"""<div class="summary-item">
            <div class="summary-value">{duration_str}</div>
            <div class="summary-label">Duration</div>
        </div>""")
        content.append("</div>")

        # Events by class table
        by_class = stats.get("events_by_class", {})
        if by_class:
            content.append("<h2>Events by Object Class</h2>")
            content.append(
                _make_table(
                    ["Object Class", "Count"],
                    [
                        (cls.capitalize(), str(cnt))
                        for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1])
                    ],
                )
            )

        # Line crossings table
        by_line = stats.get("events_by_line", {})
        if by_line:
            content.append("<h2>Line Crossings</h2>")
            content.append(
                _make_table(
                    ["Line", "Crossings"],
                    [
                        (line, str(cnt))
                        for line, cnt in sorted(by_line.items(), key=lambda x: -x[1])
                    ],
                )
            )

        # Zone activity table
        by_zone = stats.get("events_by_zone", {})
        if by_zone:
            content.append("<h2>Zone Activity</h2>")
            content.append(
                _make_table(
                    ["Zone", "Events"],
                    [
                        (zone, str(cnt))
                        for zone, cnt in sorted(by_zone.items(), key=lambda x: -x[1])
                    ],
                )
            )

        # Event timeline
        events = stats.get("events", [])
        if events:
            num_photos = len(frame_data_map)
            content.append("<h2>Event Timeline</h2>")
            content.append(
                f'<p class="subtitle">{len(events)} events, {num_photos} photos captured</p>'
            )
            content.append('<div class="timeline">')

            for i, event in enumerate(events, 1):
                location = event.get("zone_description") or event.get(
                    "line_description", "detection"
                )
                obj_class = event.get("object_class_name", "unknown").capitalize()
                direction = event.get("direction", "")
                direction_str = f" {direction}" if direction else ""
                timestamp_str = event.get("timestamp", "")

                try:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = timestamp_str

                track_id = event.get("track_id", "unknown")
                event_id = f"{event['timestamp']}_{track_id}"

                if event_id in frame_data_map:
                    # Event with photo
                    frame_bytes = frame_data_map[event_id]
                    b64 = base64.b64encode(frame_bytes).decode("ascii")
                    caption = f"#{i} — {time_str} — {_escape(obj_class)} at {_escape(location)}{_escape(direction_str)}"
                    content.append(f"""<div class="event-photo">
                        <div class="event-photo-caption">{caption}</div>
                        <img src="data:image/jpeg;base64,{b64}" alt="Event {i}">
                    </div>""")
                else:
                    # Event without photo
                    line = f"#{i}  {time_str}  {_escape(obj_class)} at {_escape(location)}{_escape(direction_str)}"
                    content.append(f'<div class="event">{line}</div>')

            content.append("</div>")

        # Build final HTML
        html = HTML_TEMPLATE.format(title=_escape(title), content="\n".join(content))

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html_path

    except Exception as e:
        logger.error(f"Failed to generate HTML: {e}", exc_info=True)
        return None


def _make_table(headers: list[str], rows: list[tuple]) -> str:
    """Generate an HTML table."""
    lines = ["<table>", "<tr>"]
    for h in headers:
        lines.append(f"<th>{_escape(h)}</th>")
    lines.append("</tr>")

    for row in rows:
        lines.append("<tr>")
        for cell in row:
            lines.append(f"<td>{_escape(str(cell))}</td>")
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
