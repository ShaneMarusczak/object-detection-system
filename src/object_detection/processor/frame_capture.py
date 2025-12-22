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
from typing import Dict, Optional, Tuple

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
                # Save frame permanently
                saved_path = frame_service.save_event_frame(event, temp_frame)

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
    except:
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
