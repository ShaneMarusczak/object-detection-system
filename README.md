# Object Detection System

GPU-accelerated object detection for tracking movement across lines and through zones. YOLO + ByteTrack with headlight detection for nighttime coverage.

## Features

- **Event-driven**: Define what you care about, system routes automatically
- **Headlight detection**: Blob detection fallback when YOLO can't see (darkness)
- **PDF reports**: Generated on shutdown with event summaries and photos
- **Email digests**: Periodic summaries with optional frame captures
- **Terraform workflow**: `--validate`, `--plan`, `--dry-run` before running
- **42+ FPS** on Jetson Orin Nano

## Quick Start

```bash
pip install -r requirements.txt
./run.sh
```

The run script is the preferred entry point. It offers two options:

1. **Run with existing config** - Validates, plans, dry-runs, then executes
2. **Build new config** - Interactive wizard that guides you through setup

### Config Builder

The config builder is a TUI wizard for first-time setup or creating new configurations:

```bash
./run.sh        # Choose option 2
# or directly:
python -m object_detection --build-config
```

Features:
- Connects to camera and serves preview frames via HTTP (for visual feedback)
- Guides through lines, zones, events, reports, and email setup
- Validates inputs (zone bounds, COCO classes)
- Saves to `configs/` and optionally runs immediately

### Manual Commands

For scripting or advanced use:

```bash
python -m object_detection --validate   # Check config
python -m object_detection --plan       # Preview routing
python -m object_detection --dry-run    # Simulate events
python -m object_detection 1            # Run for 1 hour
```

## Configuration

The main `config.yaml` uses a pointer to the active config:

```yaml
use: configs/traffic.yaml
```

Switch configs by changing the `use:` path, or let the builder set it for you.

### Config Format

```yaml
lines:
  - type: vertical
    position_pct: 30
    description: "entrance gate"

zones:
  - x1_pct: 0
    y1_pct: 60
    x2_pct: 25
    y2_pct: 100
    description: "loading dock"

events:
  - name: "vehicle_entering"
    match:
      event_type: "LINE_CROSS"
      line: "entrance gate"
      object_class: ["car", "truck", "headlight"]
    actions:
      json_log: true
      pdf_report: "traffic_report"

pdf_reports:
  - id: "traffic_report"
    title: "Traffic Report"
    photos: true
```

Headlight is just another class - use it anywhere you'd use `car` or `truck`.

## Project Structure

```
src/object_detection/
├── cli.py                 # Entry point
├── core/
│   ├── detector.py        # YOLO + headlight detection
│   ├── nighttime.py       # HeadlightDetector (blob detection)
│   └── models.py          # TrackedObject, LineConfig, ZoneConfig
├── processor/
│   ├── dispatcher.py      # Event routing
│   ├── json_writer.py     # JSONL logging + console output
│   ├── frame_capture.py   # Annotated frame saves
│   ├── pdf_report.py      # PDF generation
│   ├── email_digest.py    # Periodic email summaries
│   └── coco_classes.py    # Class ID mappings (includes headlight)
├── config/
│   ├── planner.py         # validate/plan/dry-run
│   └── schemas.py         # Config validation
└── edge/
    └── detector.py        # Jetson edge deployment
```

## Output

**JSONL** (`data/events_*.jsonl`):
```json
{"event_type":"LINE_CROSS","track_id":42,"object_class_name":"car","line_description":"entrance gate","direction":"LTR","timestamp":"2025-12-23T04:11:17Z"}
```

**PDF**: Generated on shutdown with event tables and captured frames.

**Console**:
```
od.processor.json_writer INFO #   1 | Track 42 (car) crossed V1 (entrance gate) LTR
od.processor.json_writer INFO #   2 | Track 10005 (headlight) crossed V1 (entrance gate) LTR
```

## License

MIT
