"""
Frame Capture Consumer
Captures and saves frames for events. No filtering - if it's on the queue, capture it.
Cooldown configuration comes from event metadata.
"""

import glob
import logging
import os
import time
from datetime import datetime
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple

import cv2

from .frame_service import FrameService

logger = logging.getLogger(__name__)


def frame_capture_consumer(event_queue: Queue, config: dict) -> None:
    """
    Capture frames for events. No filtering - events are pre-filtered by dispatcher.

    Args:
        event_queue: Queue receiving events that need frame capture
        config: Consumer configuration with frame settings
    """
    # Initialize frame service
    frame_service = FrameService(config)

    # Temp frame directory
    temp_frame_dir = config.get('temp_frame_dir', '/tmp/frames')

    # Lines/zones/ROI config for annotation
    lines_config = config.get('lines', [])
    zones_config = config.get('zones', [])
    roi_config = config.get('roi', {})

    # Track cooldowns per (track_id, zone/line)
    cooldowns: Dict[Tuple[int, str], float] = {}

    logger.info("Frame Capture consumer started")
    logger.info(f"Temp frame dir: {temp_frame_dir}")
    logger.info(f"Storage: {frame_service.storage_type}")

    try:
        while True:
            event = event_queue.get()

            if event is None:
                logger.info("Frame Capture received shutdown signal")
                break

            # Extract frame config from event metadata
            frame_config = event.get('_frame_capture_config', {})
            cooldown_seconds = frame_config.get('cooldown_seconds', 180)

            # Build cooldown key
            track_id = event.get('track_id')
            zone = event.get('zone_description', '')
            line = event.get('line_description', '')
            location = zone or line
            cooldown_key = (track_id, location)

            # Check cooldown
            current_time = time.time()
            if cooldown_key in cooldowns:
                if current_time - cooldowns[cooldown_key] < cooldown_seconds:
                    logger.debug(f"Skipping frame capture due to cooldown: {cooldown_key}")
                    continue

            # Find matching temp frame
            temp_frame = _find_temp_frame(temp_frame_dir, event['timestamp'])

            if temp_frame:
                # Annotate if requested
                should_annotate = frame_config.get('annotate', False)
                frame_to_save = temp_frame

                if should_annotate:
                    annotated = _annotate_frame(
                        temp_frame, event, lines_config, zones_config, roi_config
                    )
                    if annotated:
                        frame_to_save = annotated

                # Save frame permanently
                saved_path = frame_service.save_event_frame(event, frame_to_save)

                # Clean up temp annotated file if created
                if should_annotate and annotated and annotated != temp_frame:
                    try:
                        os.remove(annotated)
                    except:
                        pass

                if saved_path:
                    logger.info(f"Captured frame for: {event.get('event_definition')}")
                    cooldowns[cooldown_key] = current_time
                else:
                    logger.warning("Failed to save frame")
            else:
                logger.warning(f"No temp frame found near {event['timestamp']}")

    except KeyboardInterrupt:
        logger.info("Frame Capture stopped by user")
    except Exception as e:
        logger.error(f"Error in Frame Capture: {e}", exc_info=True)
    finally:
        logger.info("Frame Capture shutdown complete")


def _find_temp_frame(temp_dir: str, event_timestamp: str, tolerance_seconds: int = 5) -> Optional[str]:
    """
    Find temp frame closest to event timestamp.

    Args:
        temp_dir: Directory containing temp frames
        event_timestamp: ISO timestamp of event
        tolerance_seconds: Maximum time difference to accept

    Returns:
        Path to matching frame, or None
    """
    if not os.path.exists(temp_dir):
        return None

    # Parse event timestamp
    try:
        event_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
        event_time = event_time.replace(tzinfo=None)  # Make naive for comparison
    except Exception:
        logger.warning(f"Could not parse event timestamp: {event_timestamp}")
        return None

    # Find all temp frames
    temp_frames = glob.glob(os.path.join(temp_dir, 'frame_*.jpg'))

    if not temp_frames:
        return None

    # Find closest frame within tolerance
    best_frame = None
    best_diff = float('inf')

    for frame_path in temp_frames:
        try:
            # Extract timestamp from filename: frame_YYYYMMDD_HHMMSS_ffffff.jpg
            filename = os.path.basename(frame_path)
            timestamp_str = filename.replace('frame_', '').replace('.jpg', '')

            # Parse: YYYYMMDD_HHMMSS_ffffff
            frame_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')

            # Calculate time difference
            diff = abs((frame_time - event_time).total_seconds())

            if diff < best_diff and diff <= tolerance_seconds:
                best_diff = diff
                best_frame = frame_path

        except Exception as e:
            logger.debug(f"Error parsing frame timestamp from {frame_path}: {e}")
            continue

    return best_frame


def _annotate_frame(
    frame_path: str,
    event: dict,
    lines_config: List[dict],
    zones_config: List[dict],
    roi_config: dict
) -> Optional[str]:
    """
    Annotate frame with bounding box, lines, and zones.

    Args:
        frame_path: Path to source frame
        event: Event data containing bbox and object info
        lines_config: List of line configurations
        zones_config: List of zone configurations
        roi_config: ROI configuration for coordinate mapping

    Returns:
        Path to annotated frame, or None on error
    """
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return None

        height, width = frame.shape[:2]

        # Calculate ROI offset for coordinate mapping
        roi_h = roi_config.get('horizontal', {})
        roi_v = roi_config.get('vertical', {})
        h_from = roi_h.get('crop_from_left_pct', 0) if roi_h.get('enabled', False) else 0
        v_from = roi_v.get('crop_from_top_pct', 0) if roi_v.get('enabled', False) else 0

        # Draw all configured lines (yellow)
        for line in lines_config:
            line_type = line.get('type', 'vertical')
            position_pct = line.get('position_pct', 50)
            description = line.get('description', '')

            if line_type == 'vertical':
                x = int(width * position_pct / 100)
                cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
                cv2.putText(frame, description, (x + 5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:  # horizontal
                y = int(height * position_pct / 100)
                cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
                cv2.putText(frame, description, (5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw all configured zones (cyan rectangles)
        for zone in zones_config:
            x1 = int(width * zone.get('x1_pct', 0) / 100)
            y1 = int(height * zone.get('y1_pct', 0) / 100)
            x2 = int(width * zone.get('x2_pct', 100) / 100)
            y2 = int(height * zone.get('y2_pct', 100) / 100)
            description = zone.get('description', '')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, description, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw triggering object's bounding box (green with label)
        bbox = event.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox

            # Adjust bbox coordinates if ROI was applied
            # The bbox is in ROI-relative coords, but the saved frame is full-size
            if h_from > 0 or v_from > 0:
                offset_x = int(width * h_from / 100)
                offset_y = int(height * v_from / 100)
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add label
            obj_name = event.get('object_class_name', 'object')
            track_id = event.get('track_id', '')
            label = f"{obj_name} #{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save annotated frame to temp location
        annotated_path = frame_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(annotated_path, frame)

        return annotated_path

    except Exception as e:
        logger.error(f"Error annotating frame: {e}", exc_info=True)
        return None
