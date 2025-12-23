"""
Object Detection - Producer
Detects objects crossing counting lines and entering/exiting zones.
GPU-accelerated YOLO detection with ByteTrack tracking.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import logging
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Queue, Event
from typing import Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO

from ..utils.constants import (
    FPS_REPORT_INTERVAL,
    FPS_WINDOW_SIZE,
    MIN_TRACKING_TIME,
    MAX_CAMERA_RECONNECT_ATTEMPTS,
    CAMERA_RECONNECT_DELAY,
)
from .models import TrackedObject, LineConfig, ZoneConfig, ROIConfig

logger = logging.getLogger(__name__)


def run_detection(data_queue: Queue, config: dict, shutdown_event: Event = None) -> None:
    """
    Main detection loop. Tracks objects and emits boundary crossing events.

    Args:
        data_queue: Queue for sending events to analyzer
        config: Configuration dictionary from config.yaml
        shutdown_event: Event to signal graceful shutdown
    """
    try:
        # Initialize components
        model = _initialize_model(config)
        cap = _initialize_camera(config)

        # Parse configuration
        lines = _parse_lines(config)
        zones = _parse_zones(config)
        roi_config = _parse_roi(config)
        speed_enabled = config.get('speed_calculation', {}).get('enabled', False)
        frame_config = config.get('frame_saving', {})

        _log_detection_config(lines, zones, roi_config, speed_enabled, frame_config)

        # Run main detection loop
        _detection_loop(
            cap, model, data_queue, config, lines, zones,
            roi_config, speed_enabled, frame_config, shutdown_event
        )

    except Exception as e:
        logger.error(f"Fatal error in detection: {e}", exc_info=True)
        data_queue.put(None)  # Signal analyzer if init failed before loop
        raise
    finally:
        if 'cap' in locals():
            cap.release()


def _initialize_model(config: dict) -> YOLO:
    """Initialize YOLO model with GPU if available."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(config['detection']['model_file'])
    model.to(device)

    logger.info(f"Model initialized: {config['detection']['model_file']}")
    logger.info(f"Device: {device}")

    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Model on CUDA: {next(model.model.parameters()).is_cuda}")
    else:
        logger.warning("Running on CPU - performance will be slow")

    return model


def _initialize_camera(config: dict) -> cv2.VideoCapture:
    """
    Initialize camera with retry logic.

    Returns:
        OpenCV VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened after retries
    """
    camera_url = config['camera']['url']

    for attempt in range(MAX_CAMERA_RECONNECT_ATTEMPTS + 1):
        logger.info(f"Connecting to camera: {camera_url} (attempt {attempt + 1})")
        cap = cv2.VideoCapture(camera_url)

        if cap.isOpened():
            logger.info("Camera connected successfully")
            return cap

        if attempt < MAX_CAMERA_RECONNECT_ATTEMPTS:
            logger.warning(f"Failed to connect, retrying in {CAMERA_RECONNECT_DELAY}s...")
            time.sleep(CAMERA_RECONNECT_DELAY)
        else:
            logger.error(f"Failed to connect to camera after {MAX_CAMERA_RECONNECT_ATTEMPTS + 1} attempts")
            raise RuntimeError(f"Cannot connect to camera: {camera_url}")

    raise RuntimeError(f"Cannot connect to camera: {camera_url}")


def _detection_loop(
    cap: cv2.VideoCapture,
    model: YOLO,
    data_queue: Queue,
    config: dict,
    lines: List[LineConfig],
    zones: List[ZoneConfig],
    roi_config: ROIConfig,
    speed_enabled: bool,
    frame_config: dict,
    shutdown_event: Event = None
) -> None:
    """Main detection loop processing frames."""

    # State tracking with TrackedObject dataclass
    tracked_objects: Dict[int, TrackedObject] = {}

    # Performance tracking
    fps_list: List[float] = []
    frame_count = 0
    event_count = 0
    start_time = time.time()

    # Temp frame saving configuration
    temp_frame_dir = config.get('temp_frame_dir', '/tmp/frames')
    temp_frame_enabled = config.get('temp_frames_enabled', True)
    temp_frame_interval = config.get('temp_frame_interval', 5)  # Save every N frames
    temp_frame_max_age = config.get('temp_frame_max_age_seconds', 30)  # Keep last 30s

    if temp_frame_enabled:
        os.makedirs(temp_frame_dir, exist_ok=True)
        logger.info(f"Temp frames: {temp_frame_dir} (every {temp_frame_interval} frames, {temp_frame_max_age}s retention)")

    logger.info("Detection started")

    try:
        while True:
            # Check for shutdown signal
            if shutdown_event and shutdown_event.is_set():
                logger.info("Shutdown signal received")
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame_count += 1
            frame_height, frame_width = frame.shape[:2]

            # Apply ROI cropping
            roi_frame, roi_dims = _apply_roi_crop(frame, roi_config)

            # Run YOLO detection + tracking
            results = _run_yolo_inference(model, roi_frame, config)

            # Track FPS
            inference_time = results[0].speed['inference']
            fps_list.append(1000 / inference_time if inference_time > 0 else 0)

            current_time = time.time()
            relative_time = current_time - start_time

            # Process detections (frame saved on-demand when events occur)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                event_count += _process_detections(
                    results[0].boxes,
                    tracked_objects,
                    lines,
                    zones,
                    roi_dims,
                    data_queue,
                    current_time,
                    relative_time,
                    speed_enabled,
                    frame if temp_frame_enabled else None,
                    temp_frame_dir,
                    temp_frame_max_age
                )

            # Save frames if enabled
            if frame_config.get('enabled') and frame_count % frame_config.get('interval', 500) == 0:
                _save_annotated_frame(
                    frame, lines, zones, roi_config,
                    (frame_width, frame_height), event_count,
                    fps_list[-1] if fps_list else 0, frame_count, frame_config
                )

            # Periodic status update
            if frame_count % FPS_REPORT_INTERVAL == 0:
                _log_status(frame_count, fps_list, event_count, start_time)

    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    finally:
        data_queue.put(None)  # Signal end to analyzer
        _log_final_stats(frame_count, fps_list, event_count, start_time)


def _run_yolo_inference(model: YOLO, frame, config: dict):
    """Run YOLO inference with tracking."""
    return model.track(
        source=frame,
        tracker='bytetrack.yaml',
        conf=config['detection']['confidence_threshold'],
        classes=config['detection']['track_classes'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        persist=True,
        verbose=False
    )


def _process_detections(
    boxes,
    tracked_objects: Dict[int, TrackedObject],
    lines: List[LineConfig],
    zones: List[ZoneConfig],
    roi_dims: tuple,
    data_queue: Queue,
    current_time: float,
    relative_time: float,
    speed_enabled: bool,
    frame=None,
    temp_frame_dir: str = None,
    temp_frame_max_age: int = 30
) -> int:
    """
    Process all detections in current frame.

    Returns:
        Number of events generated
    """
    event_count = 0
    roi_width, roi_height = roi_dims

    track_ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.int().cpu().tolist()

    for track_id, box, obj_class in zip(track_ids, xyxy, classes):
        # Calculate center point and bbox
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox = (int(x1), int(y1), int(x2), int(y2))

        # Get or create tracked object
        if track_id not in tracked_objects:
            tracked_objects[track_id] = TrackedObject(
                track_id=track_id,
                object_class=obj_class,
                current_pos=(center_x, center_y),
                bbox=bbox,
                first_pos=(center_x, center_y),
                first_seen_time=current_time
            )
        else:
            tracked_objects[track_id].update_position(center_x, center_y, bbox)

        tracked_obj = tracked_objects[track_id]

        # Check line crossings (only if we have previous position)
        if not tracked_obj.is_new():
            event_count += _check_line_crossings(
                tracked_obj, lines, roi_dims, data_queue,
                relative_time, speed_enabled, current_time,
                frame, temp_frame_dir, temp_frame_max_age
            )

        # Check zone entry/exit
        event_count += _check_zone_events(
            tracked_obj, zones, roi_dims, data_queue,
            relative_time, current_time,
            frame, temp_frame_dir, temp_frame_max_age
        )

    return event_count


def _check_line_crossings(
    tracked_obj: TrackedObject,
    lines: List[LineConfig],
    roi_dims: tuple,
    data_queue: Queue,
    relative_time: float,
    speed_enabled: bool,
    current_time: float,
    frame=None,
    temp_frame_dir: str = None,
    temp_frame_max_age: int = 30
) -> int:
    """Check if object crossed any lines."""
    event_count = 0
    roi_width, roi_height = roi_dims

    prev_x, prev_y = tracked_obj.previous_pos
    curr_x, curr_y = tracked_obj.current_pos

    for line in lines:
        # Check class permission
        if tracked_obj.object_class not in line.allowed_classes:
            continue

        # Check if already crossed
        if line.line_id in tracked_obj.crossed_lines:
            continue

        # Check for crossing
        crossed, direction = _detect_line_crossing(
            prev_x, prev_y, curr_x, curr_y,
            line, roi_width, roi_height
        )

        if crossed:
            tracked_obj.crossed_lines.add(line.line_id)
            event_count += 1

            # Save frame on-demand for this event
            frame_id = None
            if frame is not None and temp_frame_dir:
                frame_id = _save_temp_frame(frame, temp_frame_dir, temp_frame_max_age)

            event = {
                'event_type': 'LINE_CROSS',
                'track_id': tracked_obj.track_id,
                'object_class': tracked_obj.object_class,
                'bbox': tracked_obj.bbox,
                'frame_id': frame_id,
                'line_id': line.line_id,
                'direction': direction,
                'timestamp_relative': relative_time
            }

            # Add speed data if enabled
            if speed_enabled and tracked_obj.first_pos and tracked_obj.first_seen_time:
                _add_speed_data(event, tracked_obj, line, current_time)

            data_queue.put(event)

    return event_count


def _detect_line_crossing(
    prev_x: float, prev_y: float,
    curr_x: float, curr_y: float,
    line: LineConfig,
    roi_width: int,
    roi_height: int
) -> tuple[bool, Optional[str]]:
    """
    Detect if movement crossed a line.

    Returns:
        (crossed, direction) tuple
    """
    if line.type == 'vertical':
        line_pos = roi_width * line.position_pct / 100

        if prev_x < line_pos <= curr_x:
            return True, 'LTR'
        elif prev_x > line_pos >= curr_x:
            return True, 'RTL'
    else:  # horizontal
        line_pos = roi_height * line.position_pct / 100

        if prev_y < line_pos <= curr_y:
            return True, 'TTB'
        elif prev_y > line_pos >= curr_y:
            return True, 'BTT'

    return False, None


def _add_speed_data(
    event: dict,
    tracked_obj: TrackedObject,
    line: LineConfig,
    current_time: float
) -> None:
    """Add speed calculation data to event."""
    first_x, first_y = tracked_obj.first_pos
    curr_x, curr_y = tracked_obj.current_pos

    if line.type == 'vertical':
        distance = abs(curr_x - first_x)
    else:
        distance = abs(curr_y - first_y)

    time_elapsed = current_time - tracked_obj.first_seen_time

    if time_elapsed > MIN_TRACKING_TIME:
        event['distance_pixels'] = distance
        event['time_elapsed'] = time_elapsed


def _check_zone_events(
    tracked_obj: TrackedObject,
    zones: List[ZoneConfig],
    roi_dims: tuple,
    data_queue: Queue,
    relative_time: float,
    current_time: float,
    frame=None,
    temp_frame_dir: str = None,
    temp_frame_max_age: int = 30
) -> int:
    """Check for zone entry/exit events."""
    event_count = 0
    roi_width, roi_height = roi_dims
    curr_x, curr_y = tracked_obj.current_pos

    for zone in zones:
        # Check class permission
        if tracked_obj.object_class not in zone.allowed_classes:
            continue

        # Calculate zone boundaries
        zone_x1 = roi_width * zone.x1_pct / 100
        zone_x2 = roi_width * zone.x2_pct / 100
        zone_y1 = roi_height * zone.y1_pct / 100
        zone_y2 = roi_height * zone.y2_pct / 100

        # Check if inside zone
        inside = (zone_x1 <= curr_x <= zone_x2 and zone_y1 <= curr_y <= zone_y2)
        was_inside = zone.zone_id in tracked_obj.active_zones

        if inside and not was_inside:
            # ZONE_ENTER
            tracked_obj.active_zones[zone.zone_id] = current_time
            event_count += 1

            # Save frame on-demand for this event
            frame_id = None
            if frame is not None and temp_frame_dir:
                frame_id = _save_temp_frame(frame, temp_frame_dir, temp_frame_max_age)

            data_queue.put({
                'event_type': 'ZONE_ENTER',
                'track_id': tracked_obj.track_id,
                'object_class': tracked_obj.object_class,
                'bbox': tracked_obj.bbox,
                'frame_id': frame_id,
                'zone_id': zone.zone_id,
                'timestamp_relative': relative_time
            })

        elif not inside and was_inside:
            # ZONE_EXIT
            entry_time = tracked_obj.active_zones[zone.zone_id]
            dwell_time = current_time - entry_time
            del tracked_obj.active_zones[zone.zone_id]
            event_count += 1

            # Save frame on-demand for this event
            frame_id = None
            if frame is not None and temp_frame_dir:
                frame_id = _save_temp_frame(frame, temp_frame_dir, temp_frame_max_age)

            data_queue.put({
                'event_type': 'ZONE_EXIT',
                'track_id': tracked_obj.track_id,
                'object_class': tracked_obj.object_class,
                'bbox': tracked_obj.bbox,
                'frame_id': frame_id,
                'zone_id': zone.zone_id,
                'timestamp_relative': relative_time,
                'dwell_time': dwell_time
            })

    return event_count


def _apply_roi_crop(frame, roi_config: ROIConfig) -> tuple:
    """Apply ROI cropping to frame. Returns (cropped_frame, (width, height))."""
    frame_height, frame_width = frame.shape[:2]

    if roi_config.enabled:
        x1 = int(frame_width * roi_config.h_from / 100)
        x2 = int(frame_width * roi_config.h_to / 100)
        y1 = int(frame_height * roi_config.v_from / 100)
        y2 = int(frame_height * roi_config.v_to / 100)

        roi_frame = frame[y1:y2, x1:x2]
        return roi_frame, (x2 - x1, y2 - y1)
    else:
        return frame, (frame_width, frame_height)


def _parse_lines(config: dict) -> List[LineConfig]:
    """Parse line configurations from config."""
    lines = []
    vertical_count = 0
    horizontal_count = 0

    for line_config in config.get('lines', []):
        if line_config['type'] == 'vertical':
            vertical_count += 1
            line_id = f"V{vertical_count}"
        else:
            horizontal_count += 1
            line_id = f"H{horizontal_count}"

        allowed_classes = line_config.get(
            'allowed_classes',
            config['detection']['track_classes']
        )

        lines.append(LineConfig(
            line_id=line_id,
            type=line_config['type'],
            position_pct=line_config['position_pct'],
            description=line_config['description'],
            allowed_classes=allowed_classes
        ))

    return lines


def _parse_zones(config: dict) -> List[ZoneConfig]:
    """Parse zone configurations from config."""
    zones = []

    for i, zone_config in enumerate(config.get('zones', []), 1):
        allowed_classes = zone_config.get(
            'allowed_classes',
            config['detection']['track_classes']
        )

        zones.append(ZoneConfig(
            zone_id=f"Z{i}",
            x1_pct=zone_config['x1_pct'],
            y1_pct=zone_config['y1_pct'],
            x2_pct=zone_config['x2_pct'],
            y2_pct=zone_config['y2_pct'],
            description=zone_config['description'],
            allowed_classes=allowed_classes
        ))

    return zones


def _parse_roi(config: dict) -> ROIConfig:
    """Parse ROI configuration from config."""
    roi = config.get('roi', {})
    h_roi, v_roi = roi.get('horizontal', {}), roi.get('vertical', {})

    return ROIConfig(
        enabled=h_roi.get('enabled', False) or v_roi.get('enabled', False),
        h_from=h_roi.get('crop_from_left_pct', 0),
        h_to=h_roi.get('crop_to_right_pct', 100),
        v_from=v_roi.get('crop_from_top_pct', 0),
        v_to=v_roi.get('crop_to_bottom_pct', 100)
    )


def _save_temp_frame(frame, temp_dir: str, max_age_seconds: int) -> Optional[str]:
    """
    Save temporary frame for event capture with UUID filename.
    Cleans up old frames beyond max_age_seconds.

    Args:
        frame: Raw frame to save
        temp_dir: Directory for temp frames
        max_age_seconds: Maximum age of temp frames to retain

    Returns:
        Frame ID (UUID) if saved successfully, None otherwise
    """
    try:
        # Generate UUID-based filename
        frame_id = str(uuid.uuid4())
        filename = f"{frame_id}.jpg"
        filepath = os.path.join(temp_dir, filename)

        # Save frame
        cv2.imwrite(filepath, frame)

        # Cleanup old frames (UUID-based filenames)
        import glob
        temp_frames = glob.glob(os.path.join(temp_dir, '*.jpg'))
        current_time = time.time()

        for temp_frame_path in temp_frames:
            try:
                file_age = current_time - os.path.getmtime(temp_frame_path)
                if file_age > max_age_seconds:
                    os.remove(temp_frame_path)
            except Exception as e:
                logger.debug(f"Error cleaning up temp frame {temp_frame_path}: {e}")

        return frame_id

    except Exception as e:
        logger.debug(f"Error saving temp frame: {e}")
        return None


def _save_annotated_frame(
    frame, lines: List[LineConfig], zones: List[ZoneConfig],
    roi_config: ROIConfig, frame_size: tuple, event_count: int,
    fps: float, frame_count: int, frame_config: dict
) -> None:
    """Save annotated frame to disk."""
    annotated_frame = _annotate_frame(
        frame.copy(), lines, zones, roi_config,
        frame_size, event_count, fps
    )

    timestamp = datetime.now().strftime("%H%M%S")
    output_dir = frame_config.get('output_dir', 'output_frames')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/frame_{frame_count:06d}_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_frame)


def _annotate_frame(
    frame, lines: List[LineConfig], zones: List[ZoneConfig],
    roi_config: ROIConfig, frame_size: tuple,
    event_count: int, fps: float
):
    """Add visual annotations to frame."""
    frame_width, frame_height = frame_size

    # Calculate ROI boundaries
    if roi_config.enabled:
        roi_x1 = int(frame_width * roi_config.h_from / 100)
        roi_x2 = int(frame_width * roi_config.h_to / 100)
        roi_y1 = int(frame_height * roi_config.v_from / 100)
        roi_y2 = int(frame_height * roi_config.v_to / 100)
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1

        # Draw ROI boundary
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    else:
        roi_x1, roi_y1 = 0, 0
        roi_width, roi_height = frame_width, frame_height

    # Draw lines
    for line in lines:
        if line.type == 'vertical':
            line_x = roi_x1 + int(roi_width * line.position_pct / 100)
            cv2.line(frame, (line_x, roi_y1), (line_x, roi_y2), (0, 255, 0), 2)
            cv2.putText(frame, line.line_id, (line_x + 5, roi_y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            line_y = roi_y1 + int(roi_height * line.position_pct / 100)
            cv2.line(frame, (roi_x1, line_y), (roi_x2, line_y), (0, 255, 0), 2)
            cv2.putText(frame, line.line_id, (roi_x1 + 5, line_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw zones
    for zone in zones:
        zone_x1 = roi_x1 + int(roi_width * zone.x1_pct / 100)
        zone_x2 = roi_x1 + int(roi_width * zone.x2_pct / 100)
        zone_y1 = roi_y1 + int(roi_height * zone.y1_pct / 100)
        zone_y2 = roi_y1 + int(roi_height * zone.y2_pct / 100)

        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
        cv2.putText(frame, zone.zone_id, (zone_x1 + 5, zone_y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Overlay stats
    cv2.putText(frame, f"Events: {event_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def _log_detection_config(
    lines: List[LineConfig],
    zones: List[ZoneConfig],
    roi_config: ROIConfig,
    speed_enabled: bool,
    frame_config: dict
) -> None:
    """Log detection configuration."""
    logger.info(f"Lines configured: {len(lines)}")
    logger.info(f"Zones configured: {len(zones)}")

    if roi_config.enabled:
        logger.info(f"ROI: H: {roi_config.h_from}-{roi_config.h_to}%, "
                   f"V: {roi_config.v_from}-{roi_config.v_to}%")

    if speed_enabled:
        logger.info("Speed calculation: Enabled")

    if frame_config.get('enabled'):
        logger.info(f"Frame saving: Every {frame_config.get('interval')} frames "
                   f"-> {frame_config.get('output_dir')}/")


def _log_status(frame_count: int, fps_list: List[float], event_count: int, start_time: float) -> None:
    """Log periodic status update."""
    avg_fps = sum(fps_list[-FPS_WINDOW_SIZE:]) / min(len(fps_list), FPS_WINDOW_SIZE)
    elapsed = time.time() - start_time
    logger.info(f"[{elapsed/60:.1f}min] Frame {frame_count} | FPS: {avg_fps:.1f} | Events: {event_count}")


def _log_final_stats(frame_count: int, fps_list: List[float], event_count: int, start_time: float) -> None:
    """Log final detection statistics."""
    elapsed = time.time() - start_time
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    logger.info("Detection complete")
    logger.info(f"Runtime: {elapsed/60:.1f} minutes")
    logger.info(f"Frames: {frame_count}")
    logger.info(f"Avg FPS: {avg_fps:.1f}")
    logger.info(f"Events: {event_count}")
