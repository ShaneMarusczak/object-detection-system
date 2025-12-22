"""
Edge Detector (Jetson)

Minimal detection-only component for edge deployment.
Runs YOLO + ByteTrack and publishes raw events.

No:
- COCO class name enrichment (processor does this)
- Speed calculation
- Temp frame saving
- Frame annotation
- Event routing/dispatching
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import cv2
import torch
from ultralytics import YOLO

from .config import EdgeConfig, LineConfig, ZoneConfig, ROIConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Minimal tracking state for an object."""
    track_id: int
    object_class: int
    current_pos: tuple
    previous_pos: Optional[tuple] = None
    crossed_lines: set = field(default_factory=set)
    active_zones: Dict[str, float] = field(default_factory=dict)

    def update_position(self, x: float, y: float) -> None:
        self.previous_pos = self.current_pos
        self.current_pos = (x, y)

    def is_new(self) -> bool:
        return self.previous_pos is None


class EdgeDetector:
    """
    Minimal detector for Jetson edge deployment.

    Only responsibilities:
    1. Read camera stream
    2. Run YOLO inference + tracking
    3. Detect line crossings and zone events
    4. Publish raw events (class ID only, no names)
    """

    def __init__(self, config: EdgeConfig, publisher: Callable[[dict], None]):
        """
        Initialize edge detector.

        Args:
            config: Edge configuration
            publisher: Function to call with each event dict
        """
        self.config = config
        self.publisher = publisher
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.frame_count = 0
        self.event_count = 0

    def initialize(self) -> None:
        """Initialize model and camera."""
        self._init_model()
        self._init_camera()

    def _init_model(self) -> None:
        """Initialize YOLO model."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.config.model_file)
        self.model.to(device)

        logger.info(f"Model: {self.config.model_file}")
        logger.info(f"Device: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _init_camera(self) -> None:
        """Initialize camera with retry."""
        for attempt in range(self.config.max_reconnect_attempts + 1):
            logger.info(f"Connecting to camera (attempt {attempt + 1})")
            self.cap = cv2.VideoCapture(self.config.camera_url)

            if self.cap.isOpened():
                logger.info("Camera connected")
                return

            if attempt < self.config.max_reconnect_attempts:
                time.sleep(self.config.reconnect_delay)

        raise RuntimeError(f"Cannot connect to camera: {self.config.camera_url}")

    def run(self) -> None:
        """Main detection loop."""
        if not self.model or not self.cap:
            self.initialize()

        start_time = time.time()
        logger.info("Detection started")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                self.frame_count += 1
                self._process_frame(frame, time.time() - start_time)

                # Status update every 500 frames
                if self.frame_count % 500 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"[{elapsed/60:.1f}min] Frame {self.frame_count} | Events: {self.event_count}")

        except KeyboardInterrupt:
            logger.info("Detection stopped")
        finally:
            self.cap.release()
            logger.info(f"Complete: {self.frame_count} frames, {self.event_count} events")

    def _process_frame(self, frame, relative_time: float) -> None:
        """Process a single frame."""
        # Apply ROI crop
        roi_frame, roi_dims = self._apply_roi(frame)

        # Run inference
        results = self.model.track(
            source=roi_frame,
            tracker='bytetrack.yaml',
            conf=self.config.confidence_threshold,
            classes=self.config.track_classes,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            persist=True,
            verbose=False
        )

        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            self._process_detections(results[0].boxes, roi_dims, relative_time)

    def _apply_roi(self, frame) -> tuple:
        """Apply ROI cropping."""
        h, w = frame.shape[:2]
        roi = self.config.roi

        if roi.enabled:
            x1 = int(w * roi.h_from / 100)
            x2 = int(w * roi.h_to / 100)
            y1 = int(h * roi.v_from / 100)
            y2 = int(h * roi.v_to / 100)
            return frame[y1:y2, x1:x2], (x2 - x1, y2 - y1)

        return frame, (w, h)

    def _process_detections(self, boxes, roi_dims: tuple, relative_time: float) -> None:
        """Process all detections in frame."""
        current_time = time.time()
        roi_w, roi_h = roi_dims

        track_ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()

        for track_id, box, obj_class in zip(track_ids, xyxy, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Get or create tracked object
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = TrackedObject(
                    track_id=track_id,
                    object_class=obj_class,
                    current_pos=(cx, cy),
                )
            else:
                self.tracked_objects[track_id].update_position(cx, cy)

            obj = self.tracked_objects[track_id]

            # Check line crossings
            if not obj.is_new():
                self._check_lines(obj, roi_dims, relative_time)

            # Check zone events
            self._check_zones(obj, roi_dims, relative_time, current_time)

    def _check_lines(self, obj: TrackedObject, roi_dims: tuple, relative_time: float) -> None:
        """Check for line crossing events."""
        roi_w, roi_h = roi_dims
        prev_x, prev_y = obj.previous_pos
        curr_x, curr_y = obj.current_pos

        for line in self.config.lines:
            if obj.object_class not in line.allowed_classes:
                continue
            if line.line_id in obj.crossed_lines:
                continue

            crossed, direction = self._detect_crossing(
                prev_x, prev_y, curr_x, curr_y,
                line, roi_w, roi_h
            )

            if crossed:
                obj.crossed_lines.add(line.line_id)
                self.event_count += 1

                self.publisher({
                    'event_type': 'LINE_CROSS',
                    'device_id': self.config.device_id,
                    'track_id': obj.track_id,
                    'object_class': obj.object_class,  # ID only, no name
                    'line_id': line.line_id,
                    'direction': direction,
                    'timestamp_relative': relative_time,
                })

    def _detect_crossing(
        self, prev_x: float, prev_y: float, curr_x: float, curr_y: float,
        line: LineConfig, roi_w: int, roi_h: int
    ) -> tuple:
        """Detect if movement crossed a line."""
        if line.type == 'vertical':
            pos = roi_w * line.position_pct / 100
            if prev_x < pos <= curr_x:
                return True, 'LTR'
            elif prev_x > pos >= curr_x:
                return True, 'RTL'
        else:
            pos = roi_h * line.position_pct / 100
            if prev_y < pos <= curr_y:
                return True, 'TTB'
            elif prev_y > pos >= curr_y:
                return True, 'BTT'

        return False, None

    def _check_zones(
        self, obj: TrackedObject, roi_dims: tuple,
        relative_time: float, current_time: float
    ) -> None:
        """Check for zone entry/exit events."""
        roi_w, roi_h = roi_dims
        cx, cy = obj.current_pos

        for zone in self.config.zones:
            if obj.object_class not in zone.allowed_classes:
                continue

            # Zone boundaries
            zx1 = roi_w * zone.x1_pct / 100
            zx2 = roi_w * zone.x2_pct / 100
            zy1 = roi_h * zone.y1_pct / 100
            zy2 = roi_h * zone.y2_pct / 100

            inside = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            was_inside = zone.zone_id in obj.active_zones

            if inside and not was_inside:
                obj.active_zones[zone.zone_id] = current_time
                self.event_count += 1

                self.publisher({
                    'event_type': 'ZONE_ENTER',
                    'device_id': self.config.device_id,
                    'track_id': obj.track_id,
                    'object_class': obj.object_class,
                    'zone_id': zone.zone_id,
                    'timestamp_relative': relative_time,
                })

            elif not inside and was_inside:
                entry_time = obj.active_zones[zone.zone_id]
                dwell_time = current_time - entry_time
                del obj.active_zones[zone.zone_id]
                self.event_count += 1

                self.publisher({
                    'event_type': 'ZONE_EXIT',
                    'device_id': self.config.device_id,
                    'track_id': obj.track_id,
                    'object_class': obj.object_class,
                    'zone_id': zone.zone_id,
                    'timestamp_relative': relative_time,
                    'dwell_time': dwell_time,
                })
