# Object Detection System

Production-quality object detection for tracking movement across boundaries and through zones. Built on YOLO and ByteTrack with GPU acceleration.

## Features

- **Multi-line detection**: Define vertical/horizontal counting lines
- **Zone monitoring**: Track entry/exit with automatic dwell time calculation
- **Class filtering**: Different boundaries for different object types
- **GPU-accelerated**: 40+ FPS on Jetson Orin Nano with YOLO11n
- **Real-time JSONL output**: Events streamed as they happen
- **ROI cropping**: Focus processing on specific frame regions
- **Multiprocessing**: Detection (GPU) and analysis (CPU) run in parallel

## Quick Start

### Installation (Jetson Orin Nano)

```bash
# 1. Clone repository
git clone <repository> object-detection-system
cd object-detection-system

# 2. Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for Jetson (see requirements.txt for details)
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### Configuration

Copy the example config and customize for your use case:

```bash
cp examples/config.yaml config.yaml
nano config.yaml
```

Key settings:

```yaml
detection:
  model_file: "yolo11n.pt"
  track_classes: [15, 16]  # cats, dogs
  confidence_threshold: 0.25

lines:
  - type: vertical
    position_pct: 50      # Line at 50% from left
    description: "doorway"

camera:
  url: "http://192.168.1.100:4747/video"  # Or use CAMERA_URL env var
```

### Run

```bash
# Run for 1 hour
python -m object_detection 1

# Run for 30 minutes
python -m object_detection 0.5

# Run indefinitely (Ctrl+C to stop)
python -m object_detection 999
```

## Output

Events are logged to `data/events_YYYYMMDD_HHMMSS.jsonl`:

```json
{"event_type":"LINE_CROSS","timestamp":"2025-12-22T14:35:47.123Z","track_id":47,"object_class":15,"object_class_name":"cat","line_id":"V1","line_description":"doorway","direction":"LTR"}
{"event_type":"ZONE_ENTER","timestamp":"2025-12-22T14:35:50.234Z","track_id":47,"object_class":15,"object_class_name":"cat","zone_id":"Z1","zone_description":"food bowl"}
{"event_type":"ZONE_EXIT","timestamp":"2025-12-22T14:36:02.654Z","track_id":47,"object_class":15,"object_class_name":"cat","zone_id":"Z1","zone_description":"food bowl","dwell_time":12.42}
```

## Configuration Guide

### COCO Classes

Common object classes:

| ID | Class | Use Case |
|----|-------|----------|
| 0 | person | Doorway counting, occupancy |
| 2 | car | Traffic analysis |
| 3 | motorcycle | Traffic analysis |
| 15 | cat | Pet monitoring |
| 16 | dog | Pet monitoring |

[Full list](https://docs.ultralytics.com/datasets/detect/coco/)

### Lines

```yaml
lines:
  - type: vertical           # or 'horizontal'
    position_pct: 30         # 30% from left (or top)
    description: "entrance"
    allowed_classes: [0]     # Optional: filter by class
```

**Directions:**
- Vertical: `LTR` (left-to-right), `RTL` (right-to-left)
- Horizontal: `TTB` (top-to-bottom), `BTT` (bottom-to-top)

### Zones

```yaml
zones:
  - x1_pct: 10              # Rectangle boundaries (%)
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"
    allowed_classes: [15, 16]  # Optional: filter by class
```

Events: `ZONE_ENTER`, `ZONE_EXIT` (with dwell time)

### ROI Cropping

Crop frame for better performance:

```yaml
roi:
  horizontal:
    enabled: true
    crop_from_left_pct: 0
    crop_to_right_pct: 50   # Process left half only
  vertical:
    enabled: false
```

### Environment Variables

Override camera URL without editing config:

```bash
export CAMERA_URL="http://192.168.1.200:4747/video"
python -m object_detection 1
```

## Use Cases

### Pet Monitoring

Track food/water bowl usage:

```yaml
detection:
  track_classes: [15]  # cats

zones:
  - {x1_pct: 10, y1_pct: 20, x2_pct: 30, y2_pct: 40, description: "food bowl"}
  - {x1_pct: 70, y1_pct: 60, x2_pct: 90, y2_pct: 85, description: "water"}
```

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

### Doorway Counting

Track occupancy:

```yaml
detection:
  track_classes: [0]  # person

lines:
  - type: vertical
    position_pct: 45
    description: "entrance"
  - type: vertical
    position_pct: 55
    description: "exit"
```

## Development

### Project Structure

```
object-detection-system/
├── src/object_detection/
│   ├── __init__.py
│   ├── __main__.py         # Entry point
│   ├── cli.py              # Command-line interface
│   ├── detector.py         # YOLO detection (GPU)
│   ├── analyzer.py         # Event enrichment (CPU)
│   ├── config.py           # Configuration validation
│   ├── models.py           # Data models
│   └── constants.py        # Constants
├── tests/
│   ├── test_models.py
│   ├── test_config.py
│   └── test_detector_logic.py
├── examples/
│   └── config.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

### Running Tests

```bash
python -m unittest discover tests
```

### Key Improvements (v2.0)

- **Type hints** throughout codebase
- **Logging** framework (replaces print statements)
- **TrackedObject dataclass** for cleaner state management
- **Named constants** instead of magic numbers
- **Modular functions** (broken down from 300+ line functions)
- **Camera reconnection** logic (2 retries)
- **Environment variable** support for sensitive config
- **Configurable queue size** for performance tuning
- **Comprehensive unit tests**

## Performance Tuning

### Model Selection

| Model | FPS | Accuracy | Use Case |
|-------|-----|----------|----------|
| yolo11n | 40+ | Good | Real-time, multiple cameras |
| yolo11s | 15-20 | Better | Single camera |
| yolo11m | 8-12 | Best | Offline processing |

### GPU Utilization

```bash
# Check GPU usage
nvidia-smi

# System auto-detects GPU
# CPU fallback available but slow (1-3 FPS)
```

### Optimize Performance

1. **ROI Cropping**: 2x speedup if objects in consistent region
2. **Lower resolution**: Configure at camera source
3. **Lighter model**: Use yolo11n instead of yolo11s/m
4. **Queue size**: Increase if analyzer falls behind

## Data Analysis

### Python

```python
import json

# Load events
with open('data/events_20251222_143547.jsonl', 'r') as f:
    events = [json.loads(line) for line in f]

# Filter and analyze
crossings = [e for e in events if e['event_type'] == 'LINE_CROSS']
print(f"Total crossings: {len(crossings)}")

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

# Extract speeds
cat data/events_*.jsonl | jq -r 'select(.speed_px_per_sec) | .speed_px_per_sec'
```

## Troubleshooting

### Camera Connection Issues

- Verify URL is correct
- Test URL in browser first
- Check network connectivity
- For DroidCam: ensure app is running

### Low FPS

- Check GPU is being used (logs show "Model on CUDA: True")
- Reduce camera resolution
- Enable ROI cropping
- Use lighter model (yolo11n)

### No Events Logged

- Check `allowed_classes` matches detected objects
- Lower `confidence_threshold` (try 0.15)
- Verify line/zone positions
- Enable frame saving to visualize detections

### PyTorch Import Errors (Jetson)

- Don't install torch from PyPI
- Use NVIDIA-provided wheels only
- See `requirements.txt` for installation instructions

## License

MIT License - use freely for any purpose. See [LICENSE](LICENSE).

## Credits

Built with:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
