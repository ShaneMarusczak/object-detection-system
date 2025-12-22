"""
Object Detection System CLI
Main entry point for running the detection system.

Supports Terraform-like workflow:
  --validate  Check configuration validity
  --plan      Show event routing plan
  --dry-run   Simulate with sample events
"""

import argparse
import json
import logging
import sys
import time
import yaml
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .config import (
    load_config_with_env,
    prepare_runtime_config,
    validate_config_full,
    build_plan,
    print_validation_result,
    print_plan,
    simulate_dry_run,
    generate_sample_events,
    load_sample_events,
)
from .utils import DEFAULT_QUEUE_SIZE
from .core import run_detection
from .processor import dispatch_events

logger = logging.getLogger(__name__)


def find_config_file(config_path: str) -> Path:
    """
    Find config file in standard locations.

    Search order:
    1. Specified path (if provided and not default)
    2. Current directory (config.yaml)
    3. ~/.config/object-detection/config.yaml
    4. Package default config

    Args:
        config_path: User-specified config path

    Returns:
        Path to config file

    Raises:
        SystemExit: If no config file found
    """
    # If user specified a non-default path, use only that
    if config_path != 'config.yaml':
        specified = Path(config_path)
        if specified.exists():
            return specified
        else:
            logger.error(f"Specified config file not found: {config_path}")
            sys.exit(1)

    # Search standard locations
    search_paths = [
        Path.cwd() / 'config.yaml',  # Current directory
        Path.home() / '.config' / 'object-detection' / 'config.yaml',  # User config
        Path(__file__).parent / 'default_config.yaml',  # Package default
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Using config: {path}")
            return path

    logger.error("No config file found in any of these locations:")
    for path in search_paths:
        logger.error(f"  - {path}")
    logger.error("\nTo create a config file:")
    logger.error(f"  mkdir -p ~/.config/object-detection")
    logger.error(f"  cp {Path(__file__).parent / 'default_config.yaml'} ~/.config/object-detection/config.yaml")
    sys.exit(1)


def load_config(config_path: str = 'config.yaml', skip_validation: bool = False) -> dict:
    """
    Load and optionally validate configuration file.

    Args:
        config_path: Path to config.yaml
        skip_validation: If True, skip validation (for --validate/--plan modes)

    Returns:
        Configuration dictionary

    Raises:
        SystemExit: If config cannot be loaded or is invalid
    """
    config_file = find_config_file(config_path)

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")

    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        sys.exit(1)

    # Apply environment variable overrides
    config = load_config_with_env(config)

    if not skip_validation:
        # Use new comprehensive validation
        result = validate_config_full(config)
        if not result.valid:
            print_validation_result(result)
            sys.exit(1)
        logger.info("Configuration validated")

    return config


def get_model_class_names(model_path: str) -> dict:
    """
    Load YOLO model and extract class names.

    Args:
        model_path: Path to .pt model file

    Returns:
        Dictionary mapping class IDs to names

    Raises:
        SystemExit: If model cannot be loaded
    """
    try:
        logger.info(f"Loading model to extract class names...")
        model = YOLO(model_path)
        class_names = model.names
        logger.info(f"Model loaded: {len(class_names)} classes available")
        return class_names

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def run_detector_process(queue: Queue, config: dict) -> None:
    """Wrapper for detection process with error handling."""
    try:
        run_detection(queue, config)
    except Exception as e:
        logger.error(f"Fatal error in detector: {e}", exc_info=True)
        queue.put(None)  # Signal analyzer to stop


def run_dispatcher_process(queue: Queue, config: dict, model_names: dict) -> None:
    """Wrapper for dispatcher process with error handling."""
    try:
        dispatch_events(queue, config, model_names)
    except Exception as e:
        logger.error(f"Fatal error in dispatcher: {e}", exc_info=True)


def setup_logging(quiet: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        quiet: If True, only show warnings and errors
    """
    level = logging.WARNING if quiet else logging.INFO

    # Custom formatter with shorter module names
    class ShortNameFormatter(logging.Formatter):
        def format(self, record):
            record.name = record.name.replace('object_detection.', 'od.')
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ShortNameFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    ))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Object Detection System - Track movement across boundaries and zones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m object_detection 1             # Run for 1 hour
  python -m object_detection 0.5           # Run for 30 minutes
  python -m object_detection 1 --quiet     # Run for 1 hour with minimal logs

Terraform-like Commands:
  python -m object_detection --validate    # Check config validity
  python -m object_detection --plan        # Show event routing plan
  python -m object_detection --dry-run     # Simulate with auto-generated events
  python -m object_detection --dry-run events.json  # Simulate with custom events

Environment Variables:
  CAMERA_URL - Override camera URL from config
        """
    )

    parser.add_argument(
        'duration',
        type=float,
        nargs='?',
        help='Duration in hours (default: from config.yaml)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - only show warnings and errors (events still logged to file)'
    )

    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    # Terraform-like commands
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and show derived settings'
    )

    parser.add_argument(
        '--plan',
        action='store_true',
        help='Show event routing plan without running'
    )

    parser.add_argument(
        '--dry-run',
        nargs='?',
        const='auto',
        metavar='EVENTS_FILE',
        help='Simulate event processing (optionally with JSON events file)'
    )

    return parser.parse_args()


def parse_duration(duration_arg: Optional[float], config: dict) -> float:
    """
    Parse duration from command line argument or config.

    Args:
        duration_arg: Command line duration argument (float or None)
        config: Configuration dictionary

    Returns:
        Duration in hours

    Raises:
        SystemExit: If duration is invalid
    """
    if duration_arg is not None:
        if duration_arg <= 0:
            logger.error(f"Invalid duration '{duration_arg}' - must be positive")
            logger.error("Usage: python -m object_detection [hours]")
            sys.exit(1)
        return duration_arg
    else:
        return config['runtime']['default_duration_hours']


def print_banner(config: dict, duration_hours: float) -> None:
    """Print system startup banner."""
    duration_seconds = int(duration_hours * 3600)

    print("\n" + "="*70)
    print("OBJECT DETECTION SYSTEM v2.1")
    print("="*70)

    # Quick config summary
    detection = config.get('detection', {})
    events = config.get('events', [])
    lines = config.get('lines', [])
    zones = config.get('zones', [])

    print(f"\nModel: {detection.get('model_file', 'N/A')}")
    print(f"Events: {len(events)} defined")
    print(f"Geometry: {len(lines)} line(s), {len(zones)} zone(s)")

    print(f"\nRuntime:")
    print(f"  Duration: {duration_hours} hour(s) ({duration_seconds/60:.0f} minutes)")
    print(f"  Camera: {config['camera']['url']}")
    print(f"  Press Ctrl+C to stop early")
    print("="*70)
    print()


def monitor_processes(
    detector: Process,
    analyzer: Process,
    duration_seconds: int,
    start_time: float
) -> str:
    """
    Monitor processes and return reason for stopping.

    Args:
        detector: Detector process
        analyzer: Analyzer process
        duration_seconds: Maximum runtime in seconds
        start_time: Start timestamp

    Returns:
        Reason for stopping ('duration', 'detector_died', 'analyzer_died', 'interrupted')
    """
    try:
        while True:
            elapsed = time.time() - start_time

            # Check if duration reached
            if elapsed >= duration_seconds:
                return 'duration'

            # Check if detector died unexpectedly
            if not detector.is_alive():
                return 'detector_died'

            # Check if analyzer died unexpectedly
            if not analyzer.is_alive():
                return 'analyzer_died'

            # Sleep briefly to avoid busy waiting
            time.sleep(1)

    except KeyboardInterrupt:
        return 'interrupted'


def shutdown_processes(detector: Process, analyzer: Process, queue: Queue, config: dict) -> None:
    """Gracefully shutdown detector and analyzer processes."""
    logger.info("Shutting down...")

    # Stop detector first (stops producing events)
    if detector.is_alive():
        logger.info("Stopping detector...")
        detector.terminate()
        timeout = config['runtime'].get('detector_shutdown_timeout', 5)
        detector.join(timeout=timeout)

        if detector.is_alive():
            logger.warning("Detector not responding - forcing shutdown...")
            detector.kill()
            detector.join()
        else:
            logger.info("Detector stopped")

    # Signal analyzer to stop (detector was terminated, so it didn't send None)
    queue.put(None)

    # Wait for analyzer to finish processing remaining events
    if analyzer.is_alive():
        logger.info("Waiting for analyzer to complete...")
        analyzer.join(timeout=10)

        if analyzer.is_alive():
            logger.warning("Analyzer not responding - forcing shutdown...")
            analyzer.terminate()
            analyzer.join()
        else:
            logger.info("Analyzer completed")


def print_final_status(
    detector: Process,
    analyzer: Process,
    config: dict,
    reason: str,
    elapsed: float
) -> None:
    """Print final status and output file locations."""
    print(f"\n{'='*70}")

    if reason == 'duration':
        print(f"Duration reached - stopped after {elapsed/60:.1f} minutes")
    elif reason == 'detector_died':
        print(f"Detector process ended")
    elif reason == 'analyzer_died':
        print(f"Analyzer process ended unexpectedly")
    elif reason == 'interrupted':
        print(f"Interrupted by user")

    print("="*70)
    print("SYSTEM SHUTDOWN COMPLETE")
    print("="*70)

    # Check process exit codes
    if detector.exitcode != 0 and detector.exitcode is not None:
        logger.warning(f"Detector exited with code {detector.exitcode}")

    if analyzer.exitcode != 0 and analyzer.exitcode is not None:
        logger.warning(f"Analyzer exited with code {analyzer.exitcode}")

    print(f"\nCheck output files in:")
    print(f"  {config['output']['json_dir']}/")

    if config.get('frame_saving', {}).get('enabled', False):
        print(f"  {config['frame_saving']['output_dir']}/")

    print(f"{'='*70}\n")


def run_validate(config_path: str) -> None:
    """Run validation mode."""
    config = load_config(config_path, skip_validation=True)
    result = validate_config_full(config)
    print_validation_result(result)
    sys.exit(0 if result.valid else 1)


def run_plan(config_path: str) -> None:
    """Run plan mode."""
    config = load_config(config_path, skip_validation=True)

    # First validate
    result = validate_config_full(config)
    if not result.valid:
        print_validation_result(result)
        sys.exit(1)

    # Then show plan
    plan = build_plan(config)
    print_plan(plan)
    sys.exit(0)


def run_dry_run(config_path: str, events_file: Optional[str]) -> None:
    """Run dry-run simulation mode."""
    config = load_config(config_path, skip_validation=True)

    # First validate
    result = validate_config_full(config)
    if not result.valid:
        print_validation_result(result)
        sys.exit(1)

    # Load or generate sample events
    if events_file and events_file != 'auto':
        try:
            sample_events = load_sample_events(events_file)
            print(f"Loaded {len(sample_events)} events from {events_file}")
        except FileNotFoundError:
            print(f"Error: Events file not found: {events_file}")
            sys.exit(1)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Invalid events file: {e}")
            sys.exit(1)
    else:
        sample_events = generate_sample_events(config)
        print(f"Generated {len(sample_events)} sample events from config")

    # Run simulation
    simulate_dry_run(config, sample_events)
    sys.exit(0)


def main() -> None:
    """Main orchestrator function."""
    # Parse command line arguments
    args = parse_args()

    # Setup logging (quiet for terraform-like commands)
    is_terraform_mode = args.validate or args.plan or args.dry_run
    setup_logging(quiet=args.quiet or is_terraform_mode)

    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher required")
        sys.exit(1)

    # Handle terraform-like commands
    if args.validate:
        run_validate(args.config)
        return

    if args.plan:
        run_plan(args.config)
        return

    if args.dry_run:
        run_dry_run(args.config, args.dry_run if args.dry_run != 'auto' else None)
        return

    # Normal execution mode
    # Load configuration
    config = load_config(args.config)

    # Derive track_classes from events (event-driven wiring)
    config = prepare_runtime_config(config)

    # Parse duration
    duration_hours = parse_duration(args.duration, config)
    duration_seconds = int(duration_hours * 3600)

    # Print banner
    print_banner(config, duration_hours)

    # Get model class names before spawning processes
    model_names = get_model_class_names(config['detection']['model_file'])

    # Create shared queue
    queue_size = config['runtime'].get('queue_size', DEFAULT_QUEUE_SIZE)
    queue = Queue(maxsize=queue_size)

    # Start dispatcher first (consumer must be ready)
    logger.info("Starting dispatcher process...")
    analyzer = Process(
        target=run_dispatcher_process,
        args=(queue, config, model_names),
        name="Dispatcher"
    )
    analyzer.start()

    # Brief delay to ensure dispatcher is ready
    startup_delay = config['runtime'].get('analyzer_startup_delay', 1)
    time.sleep(startup_delay)

    # Start detector
    logger.info("Starting detector process...")
    detector = Process(
        target=run_detector_process,
        args=(queue, config),
        name="Detector"
    )
    detector.start()

    print("\n" + "="*70)
    print("SYSTEM RUNNING")
    print("="*70 + "\n")

    # Monitor processes
    start_time = time.time()
    reason = monitor_processes(detector, analyzer, duration_seconds, start_time)
    elapsed = time.time() - start_time

    # Graceful shutdown
    shutdown_processes(detector, analyzer, queue, config)

    # Print final status
    print_final_status(detector, analyzer, config, reason, elapsed)


if __name__ == "__main__":
    main()
