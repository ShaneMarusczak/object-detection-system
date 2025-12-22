#!/usr/bin/env python3
"""
Object Detection System
Runs detection and analysis as parallel processes with shared queue.

Usage:
    python run_system.py [hours]
    
Examples:
    python run_system.py 1      # Run for 1 hour
    python run_system.py 0.5    # Run for 30 minutes
    python run_system.py 3      # Run for 3 hours
"""

import sys
import time
import yaml
from pathlib import Path

# Validate Python version
if sys.version_info < (3, 7):
    print("ERROR: Python 3.7 or higher required")
    sys.exit(1)

# Load configuration
try:
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("ERROR: config.yaml not found")
        print("Please create config.yaml before running the system")
        sys.exit(1)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded from config.yaml")

except yaml.YAMLError as e:
    print(f"ERROR: Invalid YAML in config.yaml: {e}")
    sys.exit(1)

# Validate configuration before importing heavy dependencies
try:
    from config_validator import validate_config, print_validation_summary, ConfigValidationError
    
    validate_config(config)
    print("✓ Configuration validated")
    
except ConfigValidationError as e:
    print(f"\n{e}")
    print("\nPlease fix config.yaml and try again")
    sys.exit(1)
except ImportError:
    print("ERROR: config_validator.py not found")
    sys.exit(1)

# Import modules after config validation
try:
    from detect_objects import run_detection
    from analyze_events import analyze_events
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print("Make sure detect_objects.py and analyze_events.py are in the same directory")
    sys.exit(1)

from multiprocessing import Process, Queue

# Import YOLO to get class names (after validation to avoid unnecessary loading)
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics package not installed")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def get_model_class_names(model_path):
    """
    Load YOLO model and extract class names.
    
    Args:
        model_path: Path to .pt model file
        
    Returns:
        Dictionary mapping class IDs to names
    """
    try:
        print(f"Loading model to extract class names...")
        model = YOLO(model_path)
        class_names = model.names  # Dict of {id: name}
        print(f"✓ Model loaded: {len(class_names)} classes available")
        return class_names
    
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)


def run_detector(queue, config):
    """Wrapper for detection process"""
    try:
        run_detection(queue, config)
    except Exception as e:
        print(f"FATAL ERROR in detector: {e}")
        import traceback
        traceback.print_exc()
        queue.put(None)  # Signal analyzer to stop


def run_analyzer(queue, config, model_names):
    """Wrapper for analysis process"""
    try:
        analyze_events(queue, config, model_names)
    except Exception as e:
        print(f"FATAL ERROR in analyzer: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main orchestrator function"""
    
    # Parse duration argument
    if len(sys.argv) > 1:
        try:
            duration_hours = float(sys.argv[1])
            if duration_hours <= 0:
                raise ValueError("Duration must be positive")
        except ValueError as e:
            print(f"ERROR: Invalid duration '{sys.argv[1]}' - {e}")
            print(f"Usage: python run_system.py [hours]")
            sys.exit(1)
    else:
        duration_hours = config['runtime']['default_duration_hours']
    
    duration_seconds = int(duration_hours * 3600)
    
    # Print system banner
    print("\n" + "="*70)
    print("OBJECT DETECTION SYSTEM")
    print("="*70)
    
    # Print configuration summary
    print_validation_summary(config)
    
    print(f"Runtime Configuration:")
    print(f"  Duration: {duration_hours} hour(s) ({duration_seconds/60:.0f} minutes)")
    print(f"  Camera: {config['camera']['url']}")
    print(f"  Press Ctrl+C to stop early")
    print("="*70)
    print()
    
    # Get model class names before spawning processes
    model_names = get_model_class_names(config['detection']['model_file'])
    
    # Create shared queue
    queue = Queue(maxsize=1000)  # Prevent memory overflow if analyzer falls behind
    
    # Start analyzer first (consumer must be ready)
    print("Starting analyzer process...")
    analyzer = Process(
        target=run_analyzer,
        args=(queue, config, model_names),
        name="Analyzer"
    )
    analyzer.start()
    
    # Brief delay to ensure analyzer is ready
    time.sleep(config['runtime']['analyzer_startup_delay'])
    
    # Start detector
    print("Starting detector process...")
    detector = Process(
        target=run_detector,
        args=(queue, config),
        name="Detector"
    )
    detector.start()
    
    print("\n" + "="*70)
    print("SYSTEM RUNNING")
    print("="*70 + "\n")
    
    # Monitor processes and handle duration/interruption
    try:
        start = time.time()
        
        while True:
            elapsed = time.time() - start
            
            # Check if duration reached
            if elapsed >= duration_seconds:
                print(f"\n{'='*70}")
                print(f"Duration reached - stopping after {elapsed/60:.1f} minutes")
                print(f"{'='*70}")
                break
            
            # Check if detector died unexpectedly
            if not detector.is_alive():
                print(f"\n{'='*70}")
                print(f"Detector process ended")
                print(f"{'='*70}")
                break
            
            # Check if analyzer died unexpectedly
            if not analyzer.is_alive():
                print(f"\n{'='*70}")
                print(f"Analyzer process ended unexpectedly")
                print(f"{'='*70}")
                break
            
            # Sleep briefly to avoid busy waiting
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n{'='*70}")
        print(f"Interrupted by user")
        print(f"{'='*70}")
    
    # Graceful shutdown sequence
    print(f"\nShutting down...")
    
    # Stop detector first (stops producing events)
    if detector.is_alive():
        print(f"  Stopping detector...")
        detector.terminate()
        detector.join(timeout=config['runtime']['detector_shutdown_timeout'])
        
        if detector.is_alive():
            print(f"    Detector not responding - forcing shutdown...")
            detector.kill()
            detector.join()
        else:
            print(f"    ✓ Detector stopped")
    
    # Wait for analyzer to finish processing remaining events
    if analyzer.is_alive():
        print(f"  Waiting for analyzer to complete...")
        analyzer.join(timeout=config['runtime']['analyzer_shutdown_timeout'])
        
        if analyzer.is_alive():
            print(f"    Analyzer timeout - forcing shutdown...")
            analyzer.terminate()
            analyzer.join()
        else:
            print(f"    ✓ Analyzer completed")
    
    # Final status
    print(f"\n{'='*70}")
    print(f"SYSTEM SHUTDOWN COMPLETE")
    print(f"{'='*70}")
    
    # Check process exit codes
    if detector.exitcode != 0 and detector.exitcode is not None:
        print(f"WARNING: Detector exited with code {detector.exitcode}")
    
    if analyzer.exitcode != 0 and analyzer.exitcode is not None:
        print(f"WARNING: Analyzer exited with code {analyzer.exitcode}")
    
    print(f"\nCheck output files in:")
    print(f"  {config['output']['json_dir']}/")
    
    if config.get('frame_saving', {}).get('enabled', False):
        print(f"  {config['frame_saving']['output_dir']}/")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

