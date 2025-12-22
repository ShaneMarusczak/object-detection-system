"""
Event Analysis - Consumer
Receives raw detection events, enriches with semantic meaning, and writes to JSONL.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def analyze_events(data_queue, config, model_names):
    """
    Process detection events from queue and write enriched data to JSONL.
    
    Args:
        data_queue: multiprocessing.Queue receiving detection events
        config: Configuration dictionary from config.yaml
        model_names: Dictionary mapping COCO class IDs to names (from YOLO model)
    """
    
    # Build lookup tables from config
    line_descriptions = _build_line_lookup(config)
    zone_descriptions = _build_zone_lookup(config)
    
    # Create output directory
    os.makedirs(config['output']['json_dir'], exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{config['output']['json_dir']}/events_{timestamp}.jsonl"
    
    # Console output settings
    console_enabled = config.get('console_output', {}).get('enabled', True)
    console_level = config.get('console_output', {}).get('level', 'detailed')
    
    # Speed calculation flag
    speed_enabled = config.get('speed_calculation', {}).get('enabled', False)
    
    print(f"Analyzer initialized")
    print(f"  Output: {json_filename}")
    print(f"  Console: {console_level if console_enabled else 'disabled'}")
    if speed_enabled:
        print(f"  Speed calculation: Enabled")
    print(f"\nAnalysis started\n")
    
    # Open JSONL file for streaming writes
    json_file = open(json_filename, 'w')
    
    # Statistics tracking
    event_count = 0
    event_counts_by_type = {
        'LINE_CROSS': 0,
        'ZONE_ENTER': 0,
        'ZONE_EXIT': 0
    }
    start_time = datetime.now(timezone.utc)
    
    try:
        while True:
            # Block waiting for event
            event = data_queue.get()
            
            # None signals end of stream
            if event is None:
                break
            
            event_count += 1
            event_type = event['event_type']
            event_counts_by_type[event_type] = event_counts_by_type.get(event_type, 0) + 1
            
            # Enrich event with semantic meaning
            enriched_event = _enrich_event(
                event,
                model_names,
                line_descriptions,
                zone_descriptions,
                start_time,
                speed_enabled
            )
            
            # Write to JSONL immediately (one JSON object per line)
            json_file.write(json.dumps(enriched_event) + '\n')
            json_file.flush()  # Ensure immediate write
            
            # Console output based on verbosity level
            if console_enabled:
                _print_event(enriched_event, console_level, event_count)
            
            # Periodic summary for 'summary' mode
            if console_enabled and console_level == 'summary' and event_count % 50 == 0:
                _print_summary(event_count, event_counts_by_type, start_time)
    
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nERROR in analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        json_file.close()
        
        # Final summary
        print(f"\nAnalysis complete")
        print(f"  Total events: {event_count}")
        for event_type, count in event_counts_by_type.items():
            if count > 0:
                print(f"    {event_type}: {count}")
        print(f"  Output: {json_filename}")
        
        # Prompt to delete data
        if event_count > 0:
            _handle_data_deletion(json_filename)
        else:
            print(f"  No events recorded - file will be empty")


def _enrich_event(event, model_names, line_descriptions, zone_descriptions, start_time, speed_enabled):
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
    if event['event_type'] == 'LINE_CROSS':
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
    
    elif event['event_type'] == 'ZONE_ENTER':
        zone_id = event['zone_id']
        enriched['zone_id'] = zone_id
        enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')
    
    elif event['event_type'] == 'ZONE_EXIT':
        zone_id = event['zone_id']
        enriched['zone_id'] = zone_id
        enriched['zone_description'] = zone_descriptions.get(zone_id, 'unknown')
        enriched['dwell_time'] = round(event['dwell_time'], 3)
    
    return enriched


def _print_event(event, level, event_count):
    """
    Print event to console based on verbosity level.
    
    Args:
        event: Enriched event dictionary
        level: Console output level ('detailed', 'summary', 'silent')
        event_count: Total events so far
    """
    
    if level == 'silent':
        return
    
    if level == 'summary':
        # Summary mode prints in _print_summary periodically
        return
    
    # Detailed mode: print every event
    event_type = event['event_type']
    track_id = event['track_id']
    obj_name = event['object_class_name']
    
    if event_type == 'LINE_CROSS':
        line_id = event['line_id']
        line_desc = event['line_description']
        direction = event['direction']
        
        if 'speed_px_per_sec' in event:
            speed = event['speed_px_per_sec']
            print(f"#{event_count:4d} | Track {track_id:3d} ({obj_name:12s}) crossed {line_id} ({line_desc:20s}) {direction:3s} @ {speed:6.1f} px/s")
        else:
            print(f"#{event_count:4d} | Track {track_id:3d} ({obj_name:12s}) crossed {line_id} ({line_desc:20s}) {direction:3s}")
    
    elif event_type == 'ZONE_ENTER':
        zone_id = event['zone_id']
        zone_desc = event['zone_description']
        print(f"#{event_count:4d} | Track {track_id:3d} ({obj_name:12s}) entered {zone_id} ({zone_desc})")
    
    elif event_type == 'ZONE_EXIT':
        zone_id = event['zone_id']
        zone_desc = event['zone_description']
        dwell = event['dwell_time']
        print(f"#{event_count:4d} | Track {track_id:3d} ({obj_name:12s}) exited  {zone_id} ({zone_desc}) - {dwell:.1f}s dwell")


def _print_summary(event_count, event_counts_by_type, start_time):
    """Print periodic summary for 'summary' console mode"""
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    print(f"\n[{elapsed/60:.1f}min] Events logged: {event_count}")
    for event_type, count in event_counts_by_type.items():
        if count > 0:
            print(f"  {event_type}: {count}")
    print()


def _build_line_lookup(config):
    """Build line_id -> description lookup table"""
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


def _build_zone_lookup(config):
    """Build zone_id -> description lookup table"""
    lookup = {}
    
    for i, zone_config in enumerate(config.get('zones', []), 1):
        zone_id = f"Z{i}"
        lookup[zone_id] = zone_config['description']
    
    return lookup


def _handle_data_deletion(json_filename):
    """
    Prompt user to delete data file.
    Useful for test runs that produce unwanted data.
    """
    print("\n" + "="*70)
    
    try:
        response = input("Delete this data? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            try:
                os.remove(json_filename)
                print(f"✓ Data deleted: {json_filename}")
            except OSError as e:
                print(f"✗ Failed to delete file: {e}")
        else:
            print(f"✓ Data saved: {json_filename}")
    
    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C or EOF during input
        print(f"\n✓ Data saved: {json_filename}")
    
    print("="*70)


if __name__ == "__main__":
    # Standalone testing
    import yaml
    from multiprocessing import Queue
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock model names for testing
    model_names = {
        0: 'person',
        15: 'cat',
        16: 'dog'
    }
    
    queue = Queue()
    analyze_events(queue, config, model_names)

