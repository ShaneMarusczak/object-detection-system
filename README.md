# Object Detection System

GPU-accelerated object detection for tracking movement across lines and through zones. YOLO + ByteTrack with blob-based nighttime detection.

## Features

- **Event-driven**: Define events declaratively, system routes automatically
- **Nighttime detection**: Headlight/taillight blob scoring when YOLO can't see
- **PDF reports**: Generated on shutdown with event summaries and photos
- **Email digests**: Periodic summaries with optional frame captures
- **Terraform workflow**: `--validate`, `--plan`, `--dry-run` before running
- **42+ FPS** on Jetson Orin Nano

## Quick Start

```bash
pip install -r requirements.txt
./run.sh
```

The run script is the preferred entry point. It offers four options:

1. **Run** - Validates, plans, dry-runs, then executes with current config
2. **Pick a config** - Choose from saved configs in `configs/`
3. **Build new config** - Interactive wizard that guides you through setup
4. **Edit a config** - Modify an existing config file

### Config Builder

The config builder is a TUI wizard for first-time setup or creating new configurations:

```bash
./run.sh        # Choose option 3
# or directly:
python -m object_detection --build-config
```

Features:
- Connects to camera and serves preview frames via HTTP (for visual feedback)
- Guides through lines, zones, events, reports, and email setup
- Validates inputs (zone bounds, model classes)
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
    description: "driveway"

events:
  # Daytime: YOLO detects cars
  - name: "vehicle_entering"
    match:
      event_type: LINE_CROSS
      line: "entrance gate"
      object_class: [car, truck]
    actions:
      json_log: true
      pdf_report: "traffic_report"

  # Nighttime: Blob detection for headlights/taillights
  - name: "nighttime_car"
    match:
      event_type: NIGHTTIME_CAR
      zone: "driveway"
      nighttime_detection:
        score_threshold: 85
    actions:
      json_log: true
      pdf_report: "traffic_report"

pdf_reports:
  - id: "traffic_report"
    title: "Traffic Report"
    photos: true
```

## Project Structure

```
src/object_detection/
├── cli.py                 # Entry point
├── models/                # Consolidated data models
│   ├── tracking.py        # TrackedObject, LineConfig, ZoneConfig
│   ├── events.py          # EventDefinition
│   └── detector.py        # Detector protocol interface
├── core/
│   ├── detector.py        # YOLO detection loop
│   ├── camera.py          # Camera init with retry
│   ├── frame_saver.py     # Temp + annotated frame saving
│   └── nighttime_zone.py  # Blob-based nighttime detection
├── processor/
│   ├── dispatcher.py      # Event routing to consumers
│   ├── json_writer.py     # JSONL logging + console output
│   ├── pdf_report.py      # PDF generation on shutdown
│   └── email_digest.py    # Periodic email summaries
├── config/
│   ├── planner.py         # validate/plan/dry-run logic
│   ├── schemas.py         # Pydantic config validation
│   └── builder.py         # Interactive config wizard
├── utils/
│   ├── event_schema.py    # Event format documentation
│   └── queue_protocol.py  # Queue abstraction for distributed mode
└── edge/                  # Self-contained for Jetson deployment
    ├── detector.py        # Minimal edge detector
    └── config.py          # Edge-specific config parsing
```

## Output

**JSONL** (`data/events_*.jsonl`):
```json
{"event_type":"LINE_CROSS","track_id":42,"object_class_name":"car","line_description":"entrance gate","direction":"LTR","timestamp":"2025-12-23T04:11:17Z"}
{"event_type":"NIGHTTIME_CAR","track_id":"nc_1","zone_description":"driveway","score":92,"timestamp":"2025-12-23T22:45:03Z"}
```

**PDF**: Generated on shutdown with event tables and captured frames.

**Console**:
```
json_writer INFO #   1 | Track 42 (car) crossed V1 (entrance gate) LTR
json_writer INFO #   2 | nighttime_car detected in driveway (score=92)
```

## License

MIT
