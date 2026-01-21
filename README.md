# Object Detection System

GPU-accelerated object detection for tracking movement across lines and through zones. YOLO + ByteTrack with blob-based nighttime detection.

## Features

- **Event-driven**: Define events declaratively, system routes automatically
- **Pluggable emitters**: Each event type (LINE_CROSS, ZONE_ENTER, DETECTED, etc.) is handled by a dedicated emitter
- **DETECTED event**: Raw detection → event for maximum sensitivity (no tracking required)
- **Nighttime detection**: Headlight/taillight blob scoring when YOLO can't see
- **HTML reports**: Generated on shutdown with event summaries and photos
- **Command actions**: Run shell scripts on events (webhooks, notifications, etc.)
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
- Guides through lines, zones, events, and report setup
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
  # Daytime: YOLO detects cars crossing line
  - name: "vehicle_entering"
    cooldown_seconds: 300  # Rate limit: max 1 event per 5 minutes
    match:
      event_type: LINE_CROSS
      line: "entrance gate"
      object_class: [car, truck]
    actions:
      json_log: true
      report: "traffic_report"

  # Raw detection: Fire on every detection (no tracking)
  - name: "print_failure"
    match:
      event_type: DETECTED
      object_class: spaghetti
    actions:
      json_log: true
      command:
        exec: "./scripts/notify.sh"

  # Nighttime: Blob detection for headlights/taillights
  - name: "nighttime_car"
    cooldown_seconds: 180  # Rate limit: max 1 event per 3 minutes
    match:
      event_type: NIGHTTIME_CAR
      zone: "driveway"
      nighttime_detection:
        score_threshold: 85
    actions:
      json_log: true
      report: "traffic_report"

reports:
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
│   ├── detector.py        # Main detection loop (uses emitters)
│   ├── camera.py          # Camera init with retry
│   └── frame_saver.py     # Temp + annotated frame saving
├── emitters/              # Pluggable event emitters
│   ├── registry.py        # Emitter registration and dispatch
│   ├── tracking_state.py  # Shared tracking state for emitters
│   ├── detected.py        # DETECTED emitter (raw detection)
│   ├── line_cross.py      # LINE_CROSS emitter
│   ├── zone_enter.py      # ZONE_ENTER emitter
│   ├── zone_exit.py       # ZONE_EXIT emitter
│   └── nighttime_car.py   # NIGHTTIME_CAR emitter (blob scoring)
├── processor/
│   ├── dispatcher.py      # Event routing to consumers
│   ├── json_writer.py     # JSONL logging + console output
│   ├── html_report.py     # HTML report generation on shutdown
│   ├── command_runner.py  # Execute shell commands on events
│   └── frame_capture.py   # Save frames for reports
├── config/
│   ├── planner.py         # validate/plan/dry-run logic
│   ├── schemas.py         # Pydantic config validation
│   └── builder.py         # Interactive config wizard
└── utils/
    ├── constants.py       # Shared constants
    ├── snapshot_server.py # Live preview server
    └── event_schema.py    # Event format documentation
```

## Output

**JSONL** (`data/events_*.jsonl`):
```json
{"event_type":"LINE_CROSS","track_id":42,"object_class_name":"car","line_description":"entrance gate","direction":"LTR","timestamp":"2025-12-23T04:11:17Z"}
{"event_type":"DETECTED","object_class_name":"spaghetti","confidence":0.87,"timestamp":"2025-12-23T15:22:41Z"}
{"event_type":"NIGHTTIME_CAR","track_id":"nc_1","zone_description":"driveway","score":92,"timestamp":"2025-12-23T22:45:03Z"}
```

**HTML**: Generated on shutdown with event tables and captured frames. Open in browser or print-to-PDF.

**Console**:
```
json_writer INFO #   1 | Track 42 (car) crossed V1 (entrance gate) LTR
json_writer INFO #   2 | spaghetti detected (conf=87%)
json_writer INFO #   3 | nighttime_car detected in driveway (score=92)
```

## License

MIT
