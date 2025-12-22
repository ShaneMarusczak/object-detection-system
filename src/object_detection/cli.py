"""
Object Detection System CLI
Main entry point for running the detection system.
"""

import argparse
import logging
import sys
import time
import yaml
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .config import validate_config, load_config_with_env, print_validation_summary, ConfigValidationError
from .constants import DEFAULT_QUEUE_SIZE
from .detector import run_detection
from .dispatcher import dispatch_events

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


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load and validate configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Validated configuration dictionary

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

    # Validate configuration
    try:
        validate_config(config)
        logger.info("Configuration validated")
    except ConfigValidationError as e:
        logger.error(f"\n{e}")
        logger.error("\nPlease fix config.yaml and try again")
        sys.exit(1)

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
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Object Detection System - Track movement across boundaries and zones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m object_detection 1           # Run for 1 hour
  python -m object_detection 0.5         # Run for 30 minutes
  python -m object_detection 1 --quiet   # Run for 1 hour with minimal logs

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
    print("OBJECT DETECTION SYSTEM v2.0")
    print("="*70)

    print_validation_summary(config)

    print(f"Runtime Configuration:")
    print(f"  Duration: {duration_hours} hour(s) ({duration_seconds/60:.0f} minutes)")
    print(f"  Camera: {config['camera']['url']}")
    print(f"  Queue size: {config['runtime'].get('queue_size', DEFAULT_QUEUE_SIZE)}")
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


def shutdown_processes(detector: Process, analyzer: Process, config: dict) -> None:
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

    # Wait for analyzer to finish processing remaining events and prompt user
    if analyzer.is_alive():
        logger.info("Waiting for analyzer to complete...")

        # Wait indefinitely for analyzer to finish and user to respond to prompt
        # No timeout - let the user take as long as they need
        analyzer.join()
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


def main() -> None:
    """Main orchestrator function."""
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging(quiet=args.quiet)

    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher required")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

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
    shutdown_processes(detector, analyzer, config)

    # Print final status
    print_final_status(detector, analyzer, config, reason, elapsed)


if __name__ == "__main__":
    main()
