"""
JSON Writer Consumer
Writes enriched events to JSONL file and optionally prints to console.
"""

import json
import logging
import os
from datetime import datetime
from multiprocessing import Queue
from typing import Dict

from ..utils import SUMMARY_EVENT_INTERVAL

logger = logging.getLogger(__name__)


def json_writer_consumer(event_queue: Queue, config: dict) -> None:
    """
    Consume enriched events and write to JSONL file.

    Args:
        event_queue: Queue receiving enriched events
        config: Consumer configuration
    """
    # Setup output
    output_dir = config.get('output_dir', 'data')
    os.makedirs(output_dir, exist_ok=True)
    json_filename = _generate_output_filename(output_dir)

    # Console settings
    console_enabled = config.get('console_enabled', True)
    console_level = config.get('console_level', 'detailed')

    logger.info(f"JSON Writer started: {json_filename}")
    logger.info(f"Console: {console_level if console_enabled else 'disabled'}")

    # Process events
    event_count = 0
    event_counts_by_type = {'LINE_CROSS': 0, 'ZONE_ENTER': 0, 'ZONE_EXIT': 0}
    start_time = datetime.now()

    try:
        with open(json_filename, 'w') as json_file:
            while True:
                event = event_queue.get()

                if event is None:  # Shutdown signal
                    break

                event_count += 1
                event_type = event['event_type']
                event_counts_by_type[event_type] = event_counts_by_type.get(event_type, 0) + 1

                # Write to JSONL
                json_file.write(json.dumps(event) + '\n')
                json_file.flush()

                # Console output
                if console_enabled:
                    _print_event(event, console_level, event_count)

                # Periodic summary
                if (console_enabled and console_level == 'summary' and
                        event_count % SUMMARY_EVENT_INTERVAL == 0):
                    _print_summary(event_count, event_counts_by_type, start_time)

    except KeyboardInterrupt:
        logger.info("JSON Writer stopped by user")
    except Exception as e:
        logger.error(f"Error in JSON Writer: {e}", exc_info=True)
    finally:
        _log_final_summary(event_count, event_counts_by_type, json_filename)


def _print_event(event: dict, level: str, event_count: int) -> None:
    """Print event to console based on verbosity level."""
    if level == 'silent' or level == 'summary':
        return

    # Detailed mode
    event_type = event['event_type']
    track_id = event['track_id']
    obj_name = event['object_class_name']

    if event_type == 'LINE_CROSS':
        line_id = event['line_id']
        line_desc = event['line_description']
        direction = event['direction']

        if 'speed_px_per_sec' in event:
            speed = event['speed_px_per_sec']
            logger.info(f"#{event_count:4d} | Track {track_id:3d} ({obj_name}) "
                       f"crossed {line_id} ({line_desc}) {direction} @ {speed:.1f} px/s")
        else:
            logger.info(f"#{event_count:4d} | Track {track_id:3d} ({obj_name}) "
                       f"crossed {line_id} ({line_desc}) {direction}")

    elif event_type == 'ZONE_ENTER':
        zone_id = event['zone_id']
        zone_desc = event['zone_description']
        logger.info(f"#{event_count:4d} | Track {track_id:3d} ({obj_name}) "
                   f"entered {zone_id} ({zone_desc})")

    elif event_type == 'ZONE_EXIT':
        zone_id = event['zone_id']
        zone_desc = event['zone_description']
        dwell = event['dwell_time']
        logger.info(f"#{event_count:4d} | Track {track_id:3d} ({obj_name}) "
                   f"exited {zone_id} ({zone_desc}) - {dwell:.1f}s dwell")


def _print_summary(event_count: int, event_counts_by_type: Dict[str, int], start_time: datetime) -> None:
    """Print periodic summary for 'summary' console mode."""
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"\n[{elapsed/60:.1f}min] Events logged: {event_count}")
    for event_type, count in event_counts_by_type.items():
        if count > 0:
            logger.info(f"  {event_type}: {count}")


def _log_final_summary(event_count: int, event_counts_by_type: Dict[str, int], json_filename: str) -> None:
    """Log final summary statistics."""
    logger.info("JSON Writer complete")
    logger.info(f"Total events: {event_count}")
    for event_type, count in event_counts_by_type.items():
        if count > 0:
            logger.info(f"  {event_type}: {count}")
    logger.info(f"Output: {json_filename}")


def _generate_output_filename(output_dir: str) -> str:
    """Generate timestamped output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{output_dir}/events_{timestamp}.jsonl"
