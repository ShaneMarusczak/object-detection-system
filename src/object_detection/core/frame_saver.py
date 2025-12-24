"""
Frame saving utilities - temp frames and annotated frames.
"""

import glob
import logging
import os
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

from ..models import LineConfig, ZoneConfig, ROIConfig

logger = logging.getLogger(__name__)


def save_temp_frame(
    frame: np.ndarray, temp_dir: str, max_age_seconds: int
) -> str | None:
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
        temp_frames = glob.glob(os.path.join(temp_dir, "*.jpg"))
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


def save_annotated_frame(
    frame: np.ndarray,
    lines: list[LineConfig],
    zones: list[ZoneConfig],
    roi_config: ROIConfig,
    frame_size: tuple[int, int],
    event_count: int,
    fps: float,
    frame_count: int,
    output_dir: str = "output_frames",
) -> None:
    """
    Save annotated frame to disk.

    Args:
        frame: Raw frame to annotate and save
        lines: Line configurations for drawing
        zones: Zone configurations for drawing
        roi_config: ROI configuration for drawing
        frame_size: (width, height) of frame
        event_count: Current event count for overlay
        fps: Current FPS for overlay
        frame_count: Current frame number for filename
        output_dir: Directory to save frames
    """
    annotated_frame = annotate_frame(
        frame.copy(), lines, zones, roi_config, frame_size, event_count, fps
    )

    timestamp = datetime.now().strftime("%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/frame_{frame_count:06d}_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_frame)


def annotate_frame(
    frame: np.ndarray,
    lines: list[LineConfig],
    zones: list[ZoneConfig],
    roi_config: ROIConfig,
    frame_size: tuple[int, int],
    event_count: int,
    fps: float,
) -> np.ndarray:
    """
    Add visual annotations to frame.

    Args:
        frame: Frame to annotate (modified in place)
        lines: Line configurations
        zones: Zone configurations
        roi_config: ROI configuration
        frame_size: (width, height) of frame
        event_count: Event count to display
        fps: FPS to display

    Returns:
        Annotated frame
    """
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
        if line.type == "vertical":
            line_x = roi_x1 + int(roi_width * line.position_pct / 100)
            cv2.line(frame, (line_x, roi_y1), (line_x, roi_y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                line.line_id,
                (line_x + 5, roi_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            line_y = roi_y1 + int(roi_height * line.position_pct / 100)
            cv2.line(frame, (roi_x1, line_y), (roi_x2, line_y), (0, 255, 0), 2)
            cv2.putText(
                frame,
                line.line_id,
                (roi_x1 + 5, line_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Draw zones
    for zone in zones:
        zone_x1 = roi_x1 + int(roi_width * zone.x1_pct / 100)
        zone_x2 = roi_x1 + int(roi_width * zone.x2_pct / 100)
        zone_y1 = roi_y1 + int(roi_height * zone.y1_pct / 100)
        zone_y2 = roi_y1 + int(roi_height * zone.y2_pct / 100)

        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            zone.zone_id,
            (zone_x1 + 5, zone_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # Overlay stats
    cv2.putText(
        frame,
        f"Events: {event_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return frame
