# Object Detection System

A generic, production-quality object detection system for tracking movement across boundaries and through zones. Built on YOLO and ByteTrack for real-time object tracking with GPU acceleration.

## Overview

This system detects objects (vehicles, people, pets, etc.) and logs boundary crossing events. It's domain-agnostic - configure it for traffic analysis, pet monitoring, doorway counting, or any scenario where you need to track object movement.

### Key Features

- **Multi-line detection**: Define vertical and horizontal counting lines
- **Zone monitoring**: Track entry/exit with automatic dwell time calculation
- **Class filtering**: Different boundaries can track different object types
- **GPU-accelerated**: 40+ FPS on Jetson Orin Nano with YOLO11n
- **Streaming output**: Events written to JSONL in real-time
- **ROI cropping**: Focus processing on specific frame regions
- **Speed calculation**: Optional for linear motion analysis (traffic)
- **Multiprocessing architecture**: Detection and analysis run in parallel

## Architecture

```
┌─────────────┐     Queue      ┌──────────────┐
│  Detector   │ ─────events───> │   Analyzer   │
│  (GPU)      │                 │   (CPU)      │
│             │                 │              │
│ - YOLO      │                 │ - Enrichment │
│ - ByteTrack │                 │ - JSONL out  │
│ - Boundary  │                 │ - Console    │
│   checking  │                 │              │
└─────────────┘                 └──────────────┘
```

**Detector** (Producer):
- Runs YOLO object detection on GPU
- Tracks objects with ByteTrack
- Checks boundaries (lines, zones)
- Emits minimal events to queue

**Analyzer** (Consumer):
- Receives events from queue
- Enriches with semantic meaning
- Writes JSONL output
- Handles console logging

## Installation

### Requirements

- Python 3.7+
- CUDA-capable GPU (optional but recommended)
- OpenCV
- PyTorch
- Ultralytics YOLO

### Setup

```bash
# Install dependencies
pip install ultralytics opencv-python pyyaml torch torchvision

# Clone/download this system
git clone <repository> object_detection_system
cd object_detection_system

# Download a YOLO model
# Example: YOLO11n (nano - fastest)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# Or use other models:
# yolo11s.pt (small - more accurate, slower)
# yolov8n.pt (older but proven)
```

### Verify Installation

```bash
# Test configuration validation
python config_validator.py

# Should output: "✓ Configuration is valid"
```

## Quick Start

### 1. Configure Your Use Case

Edit `config.yaml`:

```yaml
detection:
  model_file: "yolo11n.pt"
  track_classes: [15, 16]  # cats and dogs
  confidence_threshold: 0.25

lines:
  - type: vertical
    position_pct: 50
    description: "room divider"

camera:
  url: "http://192.168.1.100:4747/video"  # Your camera URL
```

### 2. Run the System

```bash
# Run for 1 hour
python run_system.py 1

# Run for 30 minutes
python run_system.py 0.5

# Run indefinitely (Ctrl+C to stop)
python run_system.py 999
```

### 3. Check Output

Events are logged to `data/events_YYYYMMDD_HHMMSS.jsonl`:

```json
{"event_type":"LINE_CROSS","timestamp":"2025-12-22T14:35:47.123Z","track_id":47,"object_class":15,"object_class_name":"cat","line_id":"V1","line_description":"room divider","direction":"LTR"}
{"event_type":"ZONE_ENTER","timestamp":"2025-12-22T14:35:50.234Z","track_id":47,"object_class":15,"object_class_name":"cat","zone_id":"Z1","zone_description":"food bowl"}
{"event_type":"ZONE_EXIT","timestamp":"2025-12-22T14:36:02.654Z","track_id":47,"object_class":15,"object_class_name":"cat","zone_id":"Z1","zone_description":"food bowl","dwell_time":12.42}
```

## Configuration Guide

### Object Classes

COCO dataset has 80 classes. Common ones:

| ID | Class | Use Case |
|----|-------|----------|
| 0 | person | Doorway counting, occupancy |
| 2 | car | Traffic analysis |
| 3 | motorcycle | Traffic analysis |
| 5 | bus | Traffic analysis |
| 7 | truck | Traffic analysis |
| 15 | cat | Pet monitoring |
| 16 | dog | Pet monitoring |
| 17 | horse | Livestock tracking |

See [full COCO class list](https://docs.ultralytics.com/datasets/detect/coco/).

### Lines

Lines detect boundary crossings:

```yaml
lines:
  # Vertical line at 30% from left edge
  - type: vertical
    position_pct: 30
    description: "entrance"
    allowed_classes: [0]  # People only

  # Horizontal line at 60% from top
  - type: horizontal
    position_pct: 60
    description: "floor boundary"
    allowed_classes: [15, 16]  # Cats and dogs
```

**Directions:**
- Vertical: `LTR` (left-to-right), `RTL` (right-to-left)
- Horizontal: `TTB` (top-to-bottom), `BTT` (bottom-to-top)

### Zones

Zones detect presence within rectangular areas:

```yaml
zones:
  # Food bowl area
  - x1_pct: 10   # Left edge
    y1_pct: 20   # Top edge
    x2_pct: 30   # Right edge
    y2_pct: 40   # Bottom edge
    description: "food bowl"
    allowed_classes: [15]  # Cats only
```

Events: `ZONE_ENTER`, `ZONE_EXIT` (includes dwell time)

### ROI Cropping

Crop frame to region of interest for performance:

```yaml
roi:
  horizontal:
    enabled: true
    crop_from_left_pct: 0
    crop_to_right_pct: 50  # Left half only

  vertical:
    enabled: false
    crop_from_top_pct: 0
    crop_to_bottom_pct: 100
```

### Speed Calculation

Enable for traffic/linear motion:

```yaml
speed_calculation:
  enabled: true  # Adds distance, time, speed to events
```

Tracks distance traveled from first detection to line crossing.

### Console Output

Control verbosity:

```yaml
console_output:
  enabled: true
  level: "detailed"  # or "summary" or "silent"
```

- `detailed`: Every event printed
- `summary`: Periodic statistics only
- `silent`: No output (data still logged)

## Use Cases

### Pet Monitoring

Track when cats use food bowl and litterbox:

```yaml
detection:
  track_classes: [15]  # cats

zones:
  - x1_pct: 10
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"

  - x1_pct: 70
    y1_pct: 60
    x2_pct: 90
    y2_pct: 85
    description: "litterbox"
```

**Analysis queries:**
- How many times did cat eat today?
- Average litterbox dwell time
- Eating frequency by hour

### Traffic Analysis

Count vehicles and measure speeds:

```yaml
detection:
  track_classes: [2, 3, 5, 7]  # vehicles

lines:
  - type: vertical
    position_pct: 50
    description: "counting line"

speed_calculation:
  enabled: true
```

**Analysis queries:**
- Vehicles per hour
- Speed distribution
- Peak traffic times

### Doorway Counting

Track people entering/exiting:

```yaml
detection:
  track_classes: [0]  # person

lines:
  - type: vertical
    position_pct: 40
    description: "entrance"

  - type: vertical
    position_pct: 60
    description: "exit"
```

**Analysis queries:**
- Net occupancy (entries - exits)
- Dwell time between lines
- Busiest hours

## Performance Tuning

### Model Selection

| Model | Speed (FPS) | Accuracy | Use Case |
|-------|-------------|----------|----------|
| yolo11n | 40+ | Good | Real-time, multiple cameras |
| yolo11s | 15-20 | Better | Single camera, higher accuracy |
| yolo11m | 8-12 | Best | Offline processing |

### GPU Utilization

Check GPU usage:
```bash
nvidia-smi
```

System automatically uses GPU if available. CPU fallback is slow (1-3 FPS).

### ROI Cropping

Significant performance boost if objects are in consistent region:

```yaml
roi:
  horizontal:
    enabled: true
    crop_from_left_pct: 0
    crop_to_right_pct: 50  # 2x speedup
```

### Frame Resolution

Lower resolution = higher FPS. Configure in camera source if possible.

## Data Analysis

Events are in JSONL (JSON Lines) format - one JSON object per line.

### Python Analysis

```python
import json

# Load all events
events = []
with open('data/events_20251222_143547.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Filter line crossings
crossings = [e for e in events if e['event_type'] == 'LINE_CROSS']

# Count by class
from collections import Counter
class_counts = Counter(e['object_class_name'] for e in crossings)
print(class_counts)
```

### Command Line

```bash
# Count events by type
cat data/events_*.jsonl | jq -r '.event_type' | sort | uniq -c

# Find long dwell times
cat data/events_*.jsonl | jq 'select(.event_type=="ZONE_EXIT" and .dwell_time > 30)'

# Extract speeds (if enabled)
cat data/events_*.jsonl | jq -r 'select(.speed_px_per_sec) | .speed_px_per_sec'
```

## Troubleshooting

### "Cannot connect to camera"

- Verify camera URL is correct
- Check camera is on same network
- Test URL in browser first
- For DroidCam: ensure app is running

### "Model file not found"

```bash
# Download model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# Verify file exists
ls -lh yolo11n.pt
```

### Low FPS (<10)

- Check if GPU is being used (should see "Model on CUDA: True")
- Reduce frame resolution at camera source
- Enable ROI cropping
- Switch to lighter model (yolo11n)

### No events logged

- Check `allowed_classes` includes detected objects
- Lower `confidence_threshold` (try 0.15)
- Verify lines/zones are positioned correctly
- Enable frame saving to see what's detected

### Memory issues

- Reduce queue size in `run_system.py` (default 1000)
- Enable ROI cropping to reduce frame size
- Use lighter model

## File Structure

```
object_detection_system/
├── run_system.py          # Main entry point
├── config.yaml            # Configuration
├── config_validator.py    # Config validation
├── detect_objects.py      # Detection process
├── analyze_events.py      # Analysis process
├── yolo11n.pt            # YOLO model
├── data/                  # Output directory
│   └── events_*.jsonl    # Event logs
└── output_frames/         # Saved frames (if enabled)
    └── frame_*.jpg
```

## Advanced Usage

### Custom Analysis

After collecting data, process with custom scripts:

```python
import json
import pandas as pd

# Load events into DataFrame
events = []
with open('data/events_20251222_143547.jsonl', 'r') as f:
    events = [json.loads(line) for line in f]

df = pd.DataFrame(events)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Hourly activity
df['hour'] = df['timestamp'].dt.hour
hourly_counts = df.groupby('hour').size()
print(hourly_counts)
```

### Integration with MongoDB

```python
from pymongo import MongoClient
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['object_detection']
collection = db['events']

# Import events
with open('data/events_20251222_143547.jsonl', 'r') as f:
    for line in f:
        event = json.loads(line)
        collection.insert_one(event)

# Query
recent_exits = collection.find({
    'event_type': 'ZONE_EXIT',
    'zone_description': 'food bowl'
}).limit(10)
```

### LLM-Powered Reporting

```python
import anthropic

# Load events
with open('data/events_20251222_143547.jsonl', 'r') as f:
    events_text = f.read()

# Generate report
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[{
        "role": "user",
        "content": f"Analyze these pet monitoring events and summarize behavior patterns:\n\n{events_text}"
    }]
)

print(response.content[0].text)
```

## License

MIT License - use freely for any purpose.

## Credits

Built with:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
