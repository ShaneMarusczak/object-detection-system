"""
Frame Capture Consumer
Captures and saves frames when events match configured filters.
"""

import glob
import logging
import os
import time
from datetime import datetime, timedelta
from multiprocessing import Queue
from typing import Dict, Set, Tuple

from .frame_service import FrameService

logger = logging.getLogger(__name__)


def frame_capture_consumer(event_queue: Queue, config: dict) -> None:
    """
    Consume enriched events and capture frames for matching events.

    Args:
        event_queue: Queue receiving enriched events
        config: Consumer configuration
    """
    # Initialize frame service
    frame_service = FrameService(config)

    # Get filters
    filters = config.get('filters', {})
    event_types = filters.get('event_types', ['ZONE_ENTER', 'LINE_CROSS'])
    zone_descriptions = filters.get('zone_descriptions', [])
    line_descriptions = filters.get('line_descriptions', [])
    object_classes = filters.get('object_classes', [])

    # Cooldown configuration
    cooldown_seconds = config.get('cooldown_seconds', 180)  # 3 minutes default

    # Temp frame directory
    temp_frame_dir = config.get('temp_frame_dir', '/tmp/frames')

    # Track cooldowns per (track_id, zone/line identifier)
    cooldowns: Dict[Tuple[int, str], float] = {}

    logger.info("Frame Capture Consumer started")
    logger.info(f"Filters: event_types={event_types}, zones={zone_descriptions or 'all'}, "
                f"lines={line_descriptions or 'all'}, classes={object_classes or 'all'}")
    logger.info(f"Cooldown: {cooldown_seconds}s per track per zone/line")
    logger.info(f"Temp frames: {temp_frame_dir}")

    try:
        while True:
            event = event_queue.get()

            if event is None:  # Shutdown signal
                break

            # Filter by event type
            if event['event_type'] not in event_types:
                continue

            # Filter by object class
            if object_classes and event.get('object_class_name') not in object_classes:
                continue

            # Determine zone/line identifier for cooldown
            cooldown_key = None

            if event['event_type'] in ['ZONE_ENTER', 'ZONE_EXIT']:
                zone_desc = event.get('zone_description')

                # Filter by zone if specified
                if zone_descriptions and zone_desc not in zone_descriptions:
                    continue

                cooldown_key = (event['track_id'], f"zone:{zone_desc}")

            elif event['event_type'] == 'LINE_CROSS':
                line_desc = event.get('line_description')

                # Filter by line if specified
                if line_descriptions and line_desc not in line_descriptions:
                    continue

                cooldown_key = (event['track_id'], f"line:{line_desc}")

            if not cooldown_key:
                continue

            # Check cooldown
            current_time = time.time()
            if cooldown_key in cooldowns:
                last_capture = cooldowns[cooldown_key]
                if current_time - last_capture < cooldown_seconds:
                    logger.debug(f"Skipping frame capture - cooldown active for {cooldown_key}")
                    continue

            # Find matching temp frame
            temp_frame = _find_temp_frame(temp_frame_dir, event['timestamp'])

            if temp_frame:
                # Save frame permanently
                result = frame_service.save_event_frame(event, temp_frame)

                if result:
                    # Update cooldown
                    cooldowns[cooldown_key] = current_time
                    logger.info(f"Captured frame for {event['object_class_name']} "
                               f"{event['event_type']} (track {event['track_id']})")
                else:
                    logger.warning(f"Failed to save frame for event: {event['track_id']}")
            else:
                logger.warning(f"No temp frame found for timestamp: {event['timestamp']}")

    except KeyboardInterrupt:
        logger.info("Frame Capture Consumer stopped by user")
    except Exception as e:
        logger.error(f"Error in Frame Capture Consumer: {e}", exc_info=True)
    finally:
        logger.info("Frame Capture Consumer shutdown complete")


def _find_temp_frame(temp_dir: str, event_timestamp: str) -> str:
    """
    Find temporary frame file closest to event timestamp.

    Args:
        temp_dir: Directory containing temp frames
        event_timestamp: ISO timestamp of event

    Returns:
        Path to temp frame file, or None if not found
    """
    if not os.path.exists(temp_dir):
        return None

    # Parse event timestamp
    try:
        event_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
        event_time = event_time.replace(tzinfo=None)
    except:
        logger.error(f"Failed to parse timestamp: {event_timestamp}")
        return None

    # Find all temp frames
    frame_files = glob.glob(os.path.join(temp_dir, 'frame_*.jpg'))

    if not frame_files:
        return None

    # Find frame closest to event time (within ±5 seconds)
    best_match = None
    min_diff = float('inf')

    for frame_path in frame_files:
        try:
            # Extract timestamp from filename: frame_YYYYMMDD_HHMMSS_ffffff.jpg
            basename = os.path.basename(frame_path)
            parts = basename.replace('.jpg', '').split('_')

            if len(parts) >= 4:
                # frame_20251222_143547_123456.jpg
                date_str = parts[1]  # YYYYMMDD
                time_str = parts[2]  # HHMMSS
                micro_str = parts[3] if len(parts) > 3 else '000000'  # microseconds

                # Parse frame timestamp
                frame_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

                # Calculate time difference
                diff = abs((frame_time - event_time).total_seconds())

                # Only consider frames within 5 seconds
                if diff < 5 and diff < min_diff:
                    min_diff = diff
                    best_match = frame_path

        except Exception as e:
            logger.debug(f"Error parsing frame filename {frame_path}: {e}")
            continue

    return best_match
