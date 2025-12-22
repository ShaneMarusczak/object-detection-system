"""
Event Analysis - Consumer
Receives raw detection events, enriches with semantic meaning, and writes to JSONL.
"""

import json
import logging
import os
from datetime import datetime, timezone
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, Optional

from .constants import SUMMARY_EVENT_INTERVAL

logger = logging.getLogger(__name__)


def analyze_events(data_queue: Queue, config: dict, model_names: Dict[int, str]) -> None:
    """
    Process detection events from queue and write enriched data to JSONL.

    Args:
        data_queue: Queue receiving detection events
        config: Configuration dictionary from config.yaml
        model_names: Dictionary mapping COCO class IDs to names
    """
    # Build lookup tables
    line_descriptions = _build_line_lookup(config)
    zone_descriptions = _build_zone_lookup(config)

    # Setup output
    os.makedirs(config['output']['json_dir'], exist_ok=True)
    json_filename = _generate_output_filename(config['output']['json_dir'])

    # Console settings
    console_config = config.get('console_output', {})
    console_enabled = console_config.get('enabled', True)
    console_level = console_config.get('level', 'detailed')

    speed_enabled = config.get('speed_calculation', {}).get('enabled', False)

    logger.info(f"Analyzer initialized")
    logger.info(f"Output: {json_filename}")
    logger.info(f"Console: {console_level if console_enabled else 'disabled'}")
    if speed_enabled:
        logger.info("Speed calculation: Enabled")

    # Process events
    _process_event_stream(
        data_queue, json_filename, model_names,
        line_descriptions, zone_descriptions,
        console_enabled, console_level, speed_enabled
    )


def _process_event_stream(
    data_queue: Queue,
    json_filename: str,
    model_names: Dict[int, str],
    line_descriptions: Dict[str, str],
    zone_descriptions: Dict[str, str],
    console_enabled: bool,
    console_level: str,
    speed_enabled: bool
) -> None:
    """Process event stream from queue."""

    event_count = 0
    event_counts_by_type = {
        'LINE_CROSS': 0,
        'ZONE_ENTER': 0,
        'ZONE_EXIT': 0
    }
    start_time = datetime.now(timezone.utc)

    try:
        with open(json_filename, 'w') as json_file:
            while True:
                event = data_queue.get()

                if event is None:  # End signal
                    break

                event_count += 1
                event_type = event['event_type']
                event_counts_by_type[event_type] = event_counts_by_type.get(event_type, 0) + 1

                # Enrich event
                enriched_event = _enrich_event(
                    event, model_names, line_descriptions,
                    zone_descriptions, start_time, speed_enabled
                )

                # Write to JSONL
                json_file.write(json.dumps(enriched_event) + '\n')
                json_file.flush()

                # Console output
                if console_enabled:
                    _print_event(enriched_event, console_level, event_count)

                # Periodic summary
                if (console_enabled and console_level == 'summary' and
                        event_count % SUMMARY_EVENT_INTERVAL == 0):
                    _print_summary(event_count, event_counts_by_type, start_time)

    except KeyboardInterrupt:
        logger.info("Analysis stopped by user")
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
    finally:
        _log_final_summary(event_count, event_counts_by_type, json_filename)
        # Always prompt user, even if no events (file still exists)
        _handle_data_saving(json_filename)


def _enrich_event(
    event: dict,
    model_names: Dict[int, str],
    line_descriptions: Dict[str, str],
    zone_descriptions: Dict[str, str],
    start_time: datetime,
    speed_enabled: bool
) -> dict:
    """
    Enrich raw detection event with semantic meaning.

    Args:
        event: Raw event from detector
        model_names: COCO class name lookup
        line_descriptions: Line ID -> description mapping
        zone_descriptions: Zone ID -> description mapping
        start_time: System start time for absolute timestamps
        speed_enabled: Whether to calculate speed

    Returns:
        Enriched event dictionary ready for JSONL output
    """
    # Calculate absolute timestamp
    relative_time = event['timestamp_relative']
    absolute_timestamp = start_time.timestamp() + relative_time
    iso_timestamp = datetime.fromtimestamp(absolute_timestamp, tz=timezone.utc).isoformat()

    # Look up object class name
    object_class = event['object_class']
    object_class_name = model_names.get(object_class, f"unknown_{object_class}")

    # Build base enriched event
    enriched = {
        'event_type': event['event_type'],
        'timestamp': iso_timestamp,
        'timestamp_relative': round(relative_time, 3),
        'track_id': event['track_id'],
        'object_class': object_class,
        'object_class_name': object_class_name
    }

    # Add event-specific fields
    event_type = event['event_type']

    if event_type == 'LINE_CROSS':
        _enrich_line_cross_event(enriched, event, line_descriptions, speed_enabled)
    elif event_type == 'ZONE_ENTER':
        _enrich_zone_enter_event(enriched, event, zone_descriptions)
    elif event_type == 'ZONE_EXIT':
        _enrich_zone_exit_event(enriched, event, zone_descriptions)

    return enriched


def _enrich_line_cross_event(
    enriched: dict,
    event: dict,
    line_descriptions: Dict[str, str],
    speed_enabled: bool
) -> None:
    """Add line crossing specific fields."""
    line_id = event['line_id']
    enriched['line_id'] = line_id
    enriched['line_description'] = line_descriptions.get(line_id, 'unknown')
    enriched['direction'] = event['direction']

    # Add speed data if available
    if speed_enabled and 'distance_pixels' in event and 'time_elapsed' in event:
        distance = event['distance_pixels']
        time_elapsed = event['time_elapsed']
        speed = distance / time_elapsed if time_elapsed > 0 else 0

        enriched['distance_pixels'] = round(distance, 2)
        enriched['time_elapsed'] = round(time_elapsed, 3)
        enriched['speed_px_per_sec'] = round(speed, 2)


def _enrich_zone_enter_event(
    enriched: dict,
    event: dict,
    zone_descriptions: Dict[str, str]
) -> None:
    """Add zone enter specific fields."""
    zone_id = event['zone_id']
    enriched['zone_id'] = zone_id
    enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')


def _enrich_zone_exit_event(
    enriched: dict,
    event: dict,
    zone_descriptions: Dict[str, str]
) -> None:
    """Add zone exit specific fields."""
    zone_id = event['zone_id']
    enriched['zone_id'] = zone_id
    enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')
    enriched['dwell_time'] = round(event['dwell_time'], 3)


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


def _print_summary(
    event_count: int,
    event_counts_by_type: Dict[str, int],
    start_time: datetime
) -> None:
    """Print periodic summary for 'summary' console mode."""
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info(f"\n[{elapsed/60:.1f}min] Events logged: {event_count}")
    for event_type, count in event_counts_by_type.items():
        if count > 0:
            logger.info(f"  {event_type}: {count}")


def _log_final_summary(
    event_count: int,
    event_counts_by_type: Dict[str, int],
    json_filename: str
) -> None:
    """Log final summary statistics."""
    logger.info("Analysis complete")
    logger.info(f"Total events: {event_count}")
    for event_type, count in event_counts_by_type.items():
        if count > 0:
            logger.info(f"  {event_type}: {count}")
    logger.info(f"Output: {json_filename}")


def _build_line_lookup(config: dict) -> Dict[str, str]:
    """Build line_id -> description lookup table."""
    lookup = {}
    vertical_count = 0
    horizontal_count = 0

    for line_config in config.get('lines', []):
        if line_config['type'] == 'vertical':
            vertical_count += 1
            line_id = f"V{vertical_count}"
        else:
            horizontal_count += 1
            line_id = f"H{horizontal_count}"

        lookup[line_id] = line_config['description']

    return lookup


def _build_zone_lookup(config: dict) -> Dict[str, str]:
    """Build zone_id -> description lookup table."""
    lookup = {}

    for i, zone_config in enumerate(config.get('zones', []), 1):
        zone_id = f"Z{i}"
        lookup[zone_id] = zone_config['description']

    return lookup


def _generate_output_filename(output_dir: str) -> str:
    """Generate timestamped output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{output_dir}/events_{timestamp}.jsonl"


def _handle_data_saving(json_filename: str) -> None:
    """
    Prompt user to save or delete data file.

    This function will wait indefinitely for user input.
    Default action is to save the data.
    """
    import sys

    print("\n" + "="*70)
    print("Data Collection Complete")
    print("="*70)
    print(f"\nFile location: {json_filename}")

    # Flush output to ensure user sees the prompt
    sys.stdout.flush()

    # Keep trying until we get a valid response
    while True:
        try:
            response = input("\nSave this data? (y/n) [default: y]: ").strip().lower()

            # Default to 'yes' if user just hits enter
            if not response:
                response = 'y'

            if response in ['y', 'yes']:
                print(f"✓ Data saved: {json_filename}")
                logger.info(f"Data saved: {json_filename}")
                break
            elif response in ['n', 'no']:
                try:
                    os.remove(json_filename)
                    print(f"✓ Data deleted: {json_filename}")
                    logger.info(f"Data deleted: {json_filename}")
                    break
                except OSError as e:
                    print(f"✗ Failed to delete file: {e}")
                    logger.error(f"Failed to delete file: {e}")
                    print(f"✓ Data will remain at: {json_filename}")
                    break
            else:
                print("Please enter 'y' or 'n' (or just press Enter to save)")
                continue

        except (KeyboardInterrupt, EOFError):
            # User interrupted - default to keeping the data
            print(f"\n✓ Data saved: {json_filename}")
            logger.info(f"Data saved (interrupted): {json_filename}")
            break

    print("="*70)
    sys.stdout.flush()
