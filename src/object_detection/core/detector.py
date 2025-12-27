"""
Object Detection - Producer
Event-driven detection using pluggable emitters.
Each emitter handles a single event type. Active emitters determined from config.
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import logging
import time
from multiprocessing import Event, Queue

import cv2
import torch
from ultralytics import YOLO

from ..emitters import TrackingState, build_active_emitters
from ..models import ROIConfig
from ..utils.constants import (
    DEFAULT_TEMP_FRAME_DIR,
    FPS_REPORT_INTERVAL,
    FPS_WINDOW_SIZE,
)
from .camera import initialize_camera
from .frame_saver import save_temp_frame

logger = logging.getLogger(__name__)


def run_detection(
    data_queue: Queue, config: dict, shutdown_event: Event = None
) -> None:
    """
    Main detection loop using emitter architecture.

    Args:
        data_queue: Queue for sending events to dispatcher
        config: Configuration dictionary
        shutdown_event: Event to signal graceful shutdown
    """
    cap = None
    try:
        # Initialize camera first to get frame dimensions
        cap = initialize_camera(config["camera"]["url"])
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from camera")

        frame_height, frame_width = first_frame.shape[:2]
        frame_dims = (frame_width, frame_height)

        # Build active emitters from config
        active_emitters = build_active_emitters(config, frame_dims)
        if not active_emitters:
            logger.warning("No emitters activated - check event definitions")

        # Determine what's needed based on emitter requirements
        needs_yolo = any(e.needs_yolo for e in active_emitters)
        needs_tracking = any(e.needs_tracking for e in active_emitters)

        logger.info(
            f"Emitter requirements: YOLO={needs_yolo}, tracking={needs_tracking}"
        )

        # Initialize model only if needed
        model = None
        if needs_yolo:
            model = _initialize_model(config)

        # Parse ROI config
        roi_config = _parse_roi(config)

        # Run detection loop
        _detection_loop(
            cap,
            model,
            data_queue,
            config,
            active_emitters,
            roi_config,
            needs_yolo,
            needs_tracking,
            shutdown_event,
            first_frame,
        )

    except Exception as e:
        logger.error(f"Fatal error in detection: {e}", exc_info=True)
        data_queue.put(None)
        raise
    finally:
        if cap is not None:
            cap.release()


def _initialize_model(config: dict) -> YOLO:
    """Initialize YOLO model with GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config["detection"]["model_file"])
    model.to(device)

    logger.info(f"Model initialized: {config['detection']['model_file']}")
    logger.info(f"Device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Model on CUDA: {next(model.model.parameters()).is_cuda}")
    else:
        logger.warning("Running on CPU - performance will be slow")

    return model


def _detection_loop(
    cap: cv2.VideoCapture,
    model: YOLO | None,
    data_queue: Queue,
    config: dict,
    active_emitters: list,
    roi_config: ROIConfig,
    needs_yolo: bool,
    needs_tracking: bool,
    shutdown_event: Event = None,
    first_frame=None,
) -> None:
    """Main detection loop using emitters."""

    # Tracking state (only if needed)
    tracking_state = TrackingState() if needs_tracking else None

    # Performance tracking
    fps_list: list[float] = []
    frame_count = 0
    event_count = 0
    start_time = time.time()

    # Temp frame configuration
    temp_frame_dir = config.get("temp_frame_dir", DEFAULT_TEMP_FRAME_DIR)
    temp_frame_enabled = config.get("temp_frames_enabled", True)
    temp_frame_max_age = config.get("temp_frame_max_age_seconds", 30)

    if temp_frame_enabled:
        os.makedirs(temp_frame_dir, exist_ok=True)
        logger.info(f"Temp frames: {temp_frame_dir} ({temp_frame_max_age}s retention)")

    logger.info("Detection started")

    try:
        # Process first frame if we have it
        frame = first_frame

        while True:
            # Check for shutdown
            if shutdown_event and shutdown_event.is_set():
                logger.info("Shutdown signal received")
                break

            # Read next frame (skip first iteration if we already have first_frame)
            if frame_count > 0 or first_frame is None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

            frame_count += 1
            current_time = time.time()
            relative_time = current_time - start_time

            # Apply ROI cropping
            roi_frame, roi_dims = _apply_roi_crop(frame, roi_config)

            # Run YOLO inference if needed
            yolo_results = None
            if needs_yolo and model is not None:
                yolo_results = _run_inference(model, roi_frame, config, needs_tracking)

                # Track FPS from YOLO
                inference_time = yolo_results[0].speed["inference"]
                fps_list.append(1000 / inference_time if inference_time > 0 else 0)

                # Update tracking state if needed
                if needs_tracking and tracking_state is not None:
                    tracking_state.update(yolo_results, current_time)

            # Process each emitter
            all_events = []
            for emitter in active_emitters:
                events = emitter.process(
                    frame=frame,
                    yolo_results=yolo_results,
                    timestamp=relative_time,
                    tracking_state=tracking_state,
                    frame_id=None,
                )
                all_events.extend(events)

            # Only save temp frame if there are events (avoid I/O overhead)
            if all_events and temp_frame_enabled:
                frame_id = save_temp_frame(frame, temp_frame_dir, temp_frame_max_age)
                for event in all_events:
                    event["frame_id"] = frame_id

            # Queue events
            for event in all_events:
                data_queue.put(event)
                event_count += 1

            # Periodic status
            if frame_count % FPS_REPORT_INTERVAL == 0:
                _log_status(frame_count, fps_list, start_time, event_count)

    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    finally:
        data_queue.put(None)  # Signal end to dispatcher
        _log_final_stats(frame_count, fps_list, start_time, event_count)


def _run_inference(model: YOLO, frame, config: dict, use_tracking: bool):
    """Run YOLO inference with or without tracking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_tracking:
        return model.track(
            source=frame,
            tracker="bytetrack.yaml",
            conf=config["detection"]["confidence_threshold"],
            classes=config["detection"].get("track_classes"),
            device=device,
            persist=True,
            verbose=False,
        )
    else:
        return model.predict(
            source=frame,
            conf=config["detection"]["confidence_threshold"],
            classes=config["detection"].get("track_classes"),
            device=device,
            verbose=False,
        )


def _apply_roi_crop(frame, roi_config: ROIConfig) -> tuple:
    """Apply ROI cropping to frame."""
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


def _parse_roi(config: dict) -> ROIConfig:
    """Parse ROI configuration."""
    roi = config.get("roi", {})
    h_roi, v_roi = roi.get("horizontal", {}), roi.get("vertical", {})

    return ROIConfig(
        enabled=h_roi.get("enabled", False) or v_roi.get("enabled", False),
        h_from=h_roi.get("crop_from_left_pct", 0),
        h_to=h_roi.get("crop_to_right_pct", 100),
        v_from=v_roi.get("crop_from_top_pct", 0),
        v_to=v_roi.get("crop_to_bottom_pct", 100),
    )


def _log_status(
    frame_count: int, fps_list: list[float], start_time: float, event_count: int
) -> None:
    """Log periodic status."""
    avg_fps = (
        sum(fps_list[-FPS_WINDOW_SIZE:]) / min(len(fps_list), FPS_WINDOW_SIZE)
        if fps_list
        else 0
    )
    elapsed = time.time() - start_time
    logger.info(
        f"[{elapsed / 60:.1f}min] Frame {frame_count} | FPS: {avg_fps:.1f} | Events: {event_count}"
    )


def _log_final_stats(
    frame_count: int, fps_list: list[float], start_time: float, event_count: int
) -> None:
    """Log final statistics."""
    elapsed = time.time() - start_time
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    logger.info("Detection complete")
    logger.info(f"Runtime: {elapsed / 60:.1f} minutes")
    logger.info(f"Frames: {frame_count}")
    logger.info(f"Avg FPS: {avg_fps:.1f}")
    logger.info(f"Events: {event_count}")
