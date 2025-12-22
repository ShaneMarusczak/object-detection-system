# Distributed Architecture Design

## Overview

This document describes how to split the object detection system into two components:

1. **Edge Detector (Jetson)** - Runs on NVIDIA Jetson, performs detection only
2. **Event Processor (Remote)** - Runs on any server, handles all event logic

```
┌─────────────────────┐          ┌─────────────────────────────────────────┐
│   Jetson Orin Nano  │          │           Remote Server                 │
│                     │          │                                         │
│  ┌───────────────┐  │          │  ┌────────────┐     ┌───────────────┐  │
│  │  RTSP Camera  │  │          │  │  Message   │────▶│  Dispatcher   │  │
│  │    Feed       │  │          │  │  Consumer  │     │               │  │
│  └───────┬───────┘  │          │  └────────────┘     └───────┬───────┘  │
│          │          │          │                             │          │
│  ┌───────▼───────┐  │   Queue  │                    ┌────────┴────────┐ │
│  │    YOLO +     │  │          │  ┌─────────────────┴───────────────┐ │ │
│  │   Tracking    │──┼─────────▶│  │                                 │ │ │
│  │   (GPU)       │  │          │  │  ┌──────────┐  ┌──────────────┐ │ │ │
│  └───────────────┘  │          │  │  │JSON Log  │  │Email Digest  │ │ │ │
│                     │          │  │  └──────────┘  └──────────────┘ │ │ │
│   ~8GB RAM          │          │  │  ┌──────────┐  ┌──────────────┐ │ │ │
│   15W power         │          │  │  │Email     │  │Frame Capture │ │ │ │
│                     │          │  │  │Immediate │  │              │ │ │ │
│                     │          │  │  └──────────┘  └──────────────┘ │ │ │
│                     │          │  │            Consumers            │ │ │
│                     │          │  └─────────────────────────────────┘ │ │
│                     │          │                                      │ │
│                     │          │  ┌─────────────────────────────────┐ │ │
│                     │          │  │       State Persistence         │ │ │
│                     │          │  │    (Digest checkpoints, GC)     │ │ │
│                     │          │  └─────────────────────────────────┘ │ │
└─────────────────────┘          └─────────────────────────────────────────┘
```

## Message Queue Options

### Option 1: Redis Streams (Recommended for simplicity)

```yaml
# Jetson: detector-only config
detector:
  model_file: yolo11n.pt
  confidence_threshold: 0.25

camera:
  url: rtsp://camera:8554/stream

output:
  type: redis
  redis_url: redis://processor-host:6379
  stream: detections

# No events, digests, or consumers - just raw detections
```

**Pros:**
- Simple setup, single Redis instance
- Built-in persistence (AOF/RDB)
- Consumer groups for scaling
- Low latency

**Cons:**
- Memory-bound for high throughput
- No message ordering guarantees across consumers

### Option 2: MQTT (Recommended for IoT deployments)

```yaml
# Jetson config
output:
  type: mqtt
  broker: mqtt://processor-host:1883
  topic: detections/{camera_id}
  qos: 1  # At least once delivery
```

**Pros:**
- Purpose-built for IoT
- Very low bandwidth
- Works over unreliable networks
- Many edge devices supported

**Cons:**
- Requires separate broker (Mosquitto)
- QoS 2 adds latency

### Option 3: ZeroMQ (Recommended for high performance)

```yaml
output:
  type: zmq
  endpoint: tcp://processor-host:5555
  pattern: push  # PUSH/PULL for load balancing
```

**Pros:**
- No broker required
- Highest throughput
- Built-in load balancing

**Cons:**
- No persistence (messages lost if consumer down)
- More complex failure handling

## Message Format

All messages use a standard envelope:

```json
{
  "version": 1,
  "source": {
    "device_id": "jetson-01",
    "camera_id": "driveway",
    "timestamp": "2025-01-15T14:30:22.123Z"
  },
  "frame": {
    "number": 12345,
    "width": 1920,
    "height": 1080,
    "roi_applied": {
      "x1": 0, "y1": 0,
      "x2": 1920, "y2": 1080
    }
  },
  "detections": [
    {
      "track_id": 42,
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.87,
      "bbox": {
        "x1": 100, "y1": 200,
        "x2": 400, "y2": 500
      },
      "center": {"x": 250, "y": 350},
      "previous_center": {"x": 240, "y": 340}
    }
  ]
}
```

### Optional Frame Data

For digests with photos, frame data can be included:

```json
{
  "...": "...",
  "frame_data": {
    "encoding": "jpeg",
    "quality": 85,
    "base64": "..."
  }
}
```

Or frames can be uploaded separately with a reference:

```json
{
  "...": "...",
  "frame_ref": {
    "type": "s3",
    "bucket": "detection-frames",
    "key": "jetson-01/2025-01-15/frame-12345.jpg"
  }
}
```

## Implementation

### Edge Detector (Jetson)

New minimal detector that only:
1. Reads camera stream
2. Runs YOLO inference
3. Runs tracker (ByteTrack)
4. Publishes detection messages

```python
# src/object_detection/edge/detector.py

class EdgeDetector:
    """Minimal detector for edge deployment."""

    def __init__(self, config: EdgeConfig, publisher: MessagePublisher):
        self.model = YOLO(config.model_file)
        self.publisher = publisher
        self.device_id = config.device_id

    def run(self):
        for frame, frame_num in self.camera.stream():
            # Detection + tracking
            results = self.model.track(
                frame,
                persist=True,
                conf=self.config.confidence_threshold,
                classes=self.config.track_classes,
            )

            # Build message
            message = self._build_message(frame_num, results)

            # Publish (non-blocking)
            self.publisher.publish(message)
```

### Event Processor (Remote)

New component that:
1. Subscribes to detection stream
2. Reconstructs tracking state
3. Runs line/zone analysis
4. Dispatches to consumers

```python
# src/object_detection/processor/consumer.py

class DetectionConsumer:
    """Processes detection messages from edge devices."""

    def __init__(self, config: Config):
        self.config = config
        self.dispatcher = EventDispatcher(config)
        self.state_manager = DigestStateManager()

        # Tracking state per device
        self.trackers: Dict[str, RemoteTracker] = {}

    def process(self, message: DetectionMessage):
        device_id = message.source.device_id

        # Get or create tracker for this device
        if device_id not in self.trackers:
            self.trackers[device_id] = RemoteTracker(self.config)

        tracker = self.trackers[device_id]

        # Update tracking state
        for detection in message.detections:
            tracker.update(detection)

        # Check for events
        events = tracker.check_events()

        # Dispatch events
        for event in events:
            enriched = self.dispatcher.enrich(event, message)
            self.dispatcher.dispatch(enriched)
```

### Remote Tracker

Reconstructs tracking state from detection messages:

```python
# src/object_detection/processor/tracker.py

class RemoteTracker:
    """Reconstructs tracking state from detection messages."""

    def __init__(self, config: Config):
        self.config = config
        self.objects: Dict[int, TrackedObject] = {}
        self.lines = self._build_lines(config)
        self.zones = self._build_zones(config)

    def update(self, detection: Detection):
        track_id = detection.track_id

        if track_id not in self.objects:
            self.objects[track_id] = TrackedObject(
                track_id=track_id,
                object_class=detection.class_id,
                current_pos=detection.center,
            )
        else:
            obj = self.objects[track_id]
            obj.previous_pos = obj.current_pos
            obj.current_pos = detection.center

    def check_events(self) -> List[Event]:
        """Check all tracked objects for line/zone events."""
        events = []

        for obj in self.objects.values():
            if obj.previous_pos is None:
                continue

            # Check lines
            for line in self.lines:
                if self._crossed_line(obj, line):
                    events.append(LineCrossEvent(obj, line))

            # Check zones
            for zone in self.zones:
                in_zone = self._in_zone(obj.current_pos, zone)
                was_in_zone = zone.id in obj.active_zones

                if in_zone and not was_in_zone:
                    events.append(ZoneEnterEvent(obj, zone))
                    obj.active_zones[zone.id] = time.time()
                elif not in_zone and was_in_zone:
                    events.append(ZoneExitEvent(obj, zone))
                    del obj.active_zones[zone.id]

        return events
```

## Deployment Configurations

### Single Jetson + Local Processor

For testing or small deployments, run both on Jetson:

```yaml
# jetson-local.yaml
mode: local

detector:
  model_file: yolo11n.pt

# Full config with events, digests, etc.
```

### Multiple Jetsons + Central Processor

Production deployment with multiple cameras:

```yaml
# jetson-01.yaml (on Jetson)
mode: edge
device_id: jetson-01

detector:
  model_file: yolo11n.pt

output:
  type: redis
  redis_url: redis://central-server:6379
  stream: detections

camera:
  url: rtsp://driveway-cam:8554/stream
```

```yaml
# processor.yaml (on central server)
mode: processor

input:
  type: redis
  redis_url: redis://localhost:6379
  stream: detections
  consumer_group: event-processor

# Device-specific configs
devices:
  jetson-01:
    camera_id: driveway
    lines:
      - type: vertical
        position_pct: 50
        description: driveway entry
    events:
      - name: car-detected
        match:
          event_type: LINE_CROSS
          line: driveway entry
        actions:
          email_immediate: true

  jetson-02:
    camera_id: backyard
    zones:
      - x1_pct: 20
        y1_pct: 30
        x2_pct: 80
        y2_pct: 90
        description: garden
    events:
      - name: animal-in-garden
        match:
          event_type: ZONE_ENTER
          zone: garden
        actions:
          json_log: true
```

## Benefits

| Aspect | Before (Monolith) | After (Distributed) |
|--------|------------------|---------------------|
| Jetson Load | 100% (detection + processing) | ~60% (detection only) |
| Scalability | 1 camera per Jetson | N cameras, 1 processor |
| Recovery | Full restart | Edge continues, processor recovers |
| Updates | Redeploy everything | Update processor only |
| Monitoring | Local only | Centralized dashboard |

## Migration Path

### Phase 1: Abstract Message Layer

Add message abstraction to current code:

```python
# Today: direct call
dispatcher.dispatch(event)

# Tomorrow: through message layer
publisher.publish("events", event)
```

### Phase 2: Redis Integration

Add Redis as message transport:

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:alpine

# Run detector
python -m object_detection.edge --config edge.yaml

# Run processor
python -m object_detection.processor --config processor.yaml
```

### Phase 3: Multi-device Support

Add device routing and per-device config:

```python
# Processor routes by device_id
message = queue.get()
config = device_configs[message.source.device_id]
processor = get_or_create_processor(message.source.device_id, config)
processor.handle(message)
```

## Files to Create

```
src/object_detection/
├── edge/
│   ├── __init__.py
│   ├── detector.py      # Minimal edge detector
│   ├── publisher.py     # Message publishing abstraction
│   └── config.py        # Edge-specific config
│
├── processor/
│   ├── __init__.py
│   ├── consumer.py      # Message consumer
│   ├── tracker.py       # Remote tracking state
│   ├── router.py        # Device routing
│   └── config.py        # Processor config
│
└── transport/
    ├── __init__.py
    ├── base.py          # Abstract transport
    ├── redis.py         # Redis Streams
    ├── mqtt.py          # MQTT
    └── zmq.py           # ZeroMQ
```

## Next Steps

1. **Implement MessagePublisher interface** - Abstract transport layer
2. **Create EdgeDetector class** - Minimal detection-only component
3. **Create DetectionConsumer class** - Process messages, reconstruct state
4. **Add Redis transport** - Start with simplest option
5. **Test with single Jetson** - Validate architecture
6. **Add multi-device support** - Per-device configs and routing
