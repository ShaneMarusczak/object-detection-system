# Object Detection System

Production-quality object detection for tracking movement across boundaries and through zones. Built on YOLO and ByteTrack with GPU acceleration.

## Features

- **Event-driven architecture**: Define events declaratively, system handles routing
- **Terraform-like workflow**: `--validate`, `--plan`, `--dry-run` before running
- **Email digests**: Daily/hourly summaries with optional photos
- **Multi-line detection**: Vertical/horizontal counting lines
- **Zone monitoring**: Entry/exit with automatic dwell time
- **GPU-accelerated**: 40+ FPS on Jetson Orin Nano

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Validate config
python -m object_detection --validate

# Preview what will happen
python -m object_detection --plan

# Simulate with sample events
python -m object_detection --dry-run

# Run for 1 hour
python -m object_detection 1
```

## Configuration

Edit `config.yaml`:

```yaml
# Define geometry
lines:
  - type: vertical
    position_pct: 25
    description: "driveway"
    allowed_classes: [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Define what you care about
events:
  - name: "vehicle-crossing"
    match:
      event_type: "LINE_CROSS"
      line: "driveway"
      object_class: ["car", "truck"]
    actions:
      json_log: true          # Log to JSON
      email_digest: "daily"   # Include in daily summary

# Configure digests
digests:
  - id: "daily"
    period_minutes: 1440
    period_label: "Daily Traffic"
    photos: true  # Auto-enables frame capture
```

### AWS-Style Composition

Actions have automatic dependencies:
- `email_digest` → auto-enables `json_log`
- `email_digest` with `photos: true` → auto-enables `frame_capture`

Just say what you want, the system figures out the rest.

## Terraform-like Workflow

```bash
# Check config validity
python -m object_detection --validate
# ✓ Configuration is valid
# Derived: track_classes: car (2), truck (7)
# Active consumers: json_writer, email_digest, frame_capture

# Preview event routing
python -m object_detection --plan
# Event: vehicle-crossing
#   Match: LINE_CROSS + driveway + [car, truck]
#   Actions:
#     -> json_writer
#     -> email_digest (daily)
#     -> frame_capture (implied by photos: true)

# Simulate without running
python -m object_detection --dry-run
# [1] LINE_CROSS: car @ driveway
#     -> Matched: vehicle-crossing
#        -> Write to JSON log
#        -> Queue for digest: daily
```

## Project Structure

```
src/object_detection/
├── cli.py              # CLI & orchestration
├── core/
│   ├── detector.py     # YOLO detection
│   ├── dispatcher.py   # Event routing
│   └── models.py       # Data classes
├── config/
│   ├── validator.py    # Config validation
│   └── planner.py      # validate/plan/dry-run
├── consumers/
│   ├── json_writer.py
│   ├── email_notifier.py
│   ├── email_digest.py
│   └── frame_capture.py
└── utils/
    ├── constants.py
    └── coco_classes.py
```

## Output

Events logged to `data/events_YYYYMMDD_HHMMSS.jsonl`:

```json
{"event_type":"LINE_CROSS","timestamp":"2025-12-22T14:35:47Z","track_id":47,"object_class_name":"car","line_description":"driveway","direction":"LTR","event_definition":"vehicle-crossing"}
```

## License

MIT License - see [LICENSE](LICENSE).
