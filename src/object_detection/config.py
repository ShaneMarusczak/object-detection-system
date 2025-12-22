"""
Configuration Validator
Validates config.yaml before system startup to catch errors early.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from .constants import ENV_CAMERA_URL, DEFAULT_QUEUE_SIZE

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_config(config: dict) -> bool:
    """
    Validate all configuration parameters.

    Args:
        config: Dictionary loaded from config.yaml

    Returns:
        True if valid

    Raises:
        ConfigValidationError: If any validation fails
    """
    errors = []

    errors.extend(_validate_detection(config))
    errors.extend(_validate_roi(config))
    errors.extend(_validate_lines(config))
    errors.extend(_validate_zones(config))
    errors.extend(_validate_output(config))
    errors.extend(_validate_camera(config))
    errors.extend(_validate_runtime(config))

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ConfigValidationError(error_msg)

    return True


def load_config_with_env(config: dict) -> dict:
    """
    Load config and apply environment variable overrides.

    Args:
        config: Base configuration dictionary

    Returns:
        Configuration with environment variables applied
    """
    # Override camera URL from environment if set
    if ENV_CAMERA_URL in os.environ:
        camera_url = os.environ[ENV_CAMERA_URL]
        logger.info(f"Using camera URL from environment: {ENV_CAMERA_URL}")
        if 'camera' not in config:
            config['camera'] = {}
        config['camera']['url'] = camera_url

    # Set default queue size if not specified
    if 'runtime' not in config:
        config['runtime'] = {}
    if 'queue_size' not in config['runtime']:
        config['runtime']['queue_size'] = DEFAULT_QUEUE_SIZE

    return config


def print_validation_summary(config: dict) -> None:
    """Print a summary of validated configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    # Detection
    detection = config.get('detection', {})
    print(f"\nDetection:")
    print(f"  Model: {detection.get('model_file', 'N/A')}")
    print(f"  Classes: {detection.get('track_classes', [])}")
    print(f"  Confidence: {detection.get('confidence_threshold', 'N/A')}")

    # ROI
    roi = config.get('roi', {})
    h_roi = roi.get('horizontal', {})
    v_roi = roi.get('vertical', {})

    if h_roi.get('enabled') or v_roi.get('enabled'):
        print(f"\nROI Cropping:")
        if h_roi.get('enabled'):
            print(f"  Horizontal: {h_roi.get('crop_from_left_pct')}% - "
                  f"{h_roi.get('crop_to_right_pct')}% from left")
        if v_roi.get('enabled'):
            print(f"  Vertical: {v_roi.get('crop_from_top_pct')}% - "
                  f"{v_roi.get('crop_to_bottom_pct')}% from top")

    # Lines
    lines = config.get('lines', [])
    if lines:
        print(f"\nCounting Lines ({len(lines)}):")
        for i, line in enumerate(lines, 1):
            allowed = line.get('allowed_classes', detection.get('track_classes', []))
            line_type = line['type'][0].upper()
            idx = i if line['type'] == 'vertical' else sum(1 for l in lines[:i] if l['type'] == 'horizontal')
            print(f"  {line_type}{idx}: {line['position_pct']}% - "
                  f"\"{line['description']}\" (classes: {allowed})")

    # Zones
    zones = config.get('zones', [])
    if zones:
        print(f"\nZones ({len(zones)}):")
        for i, zone in enumerate(zones, 1):
            allowed = zone.get('allowed_classes', detection.get('track_classes', []))
            print(f"  Z{i}: ({zone['x1_pct']},{zone['y1_pct']})-"
                  f"({zone['x2_pct']},{zone['y2_pct']})% - "
                  f"\"{zone['description']}\" (classes: {allowed})")

    # Speed calculation
    speed = config.get('speed_calculation', {})
    if speed.get('enabled'):
        print(f"\nSpeed Calculation: Enabled")

    # Frame saving
    frames = config.get('frame_saving', {})
    if frames.get('enabled'):
        print(f"\nFrame Saving: Every {frames.get('interval', 'N/A')} frames")

    # Runtime
    runtime = config.get('runtime', {})
    queue_size = runtime.get('queue_size', DEFAULT_QUEUE_SIZE)
    print(f"\nRuntime:")
    print(f"  Queue size: {queue_size}")
    print(f"  Default duration: {runtime.get('default_duration_hours', 1.0)} hours")

    # Output
    output = config.get('output', {})
    print(f"\nOutput:")
    print(f"  JSON: {output.get('json_dir', 'N/A')}/events_YYYYMMDD_HHMMSS.jsonl")

    # Console
    console = config.get('console_output', {})
    if console.get('enabled'):
        print(f"  Console: {console.get('level', 'detailed')}")

    print("\n" + "="*70 + "\n")


def _validate_detection(config: dict) -> List[str]:
    """Validate detection settings."""
    errors = []

    if 'detection' not in config:
        errors.append("Missing 'detection' section")
        return errors

    detection = config['detection']

    # Model file
    if 'model_file' not in detection:
        errors.append("detection.model_file is required")
    else:
        model_path = Path(detection['model_file'])
        if not model_path.exists():
            errors.append(f"Model file not found: {detection['model_file']}")
        elif not str(model_path).endswith('.pt'):
            errors.append(f"Model file must be .pt format: {detection['model_file']}")

    # Track classes
    if 'track_classes' not in detection:
        errors.append("detection.track_classes is required")
    else:
        classes = detection['track_classes']
        if not isinstance(classes, list):
            errors.append("detection.track_classes must be a list")
        elif len(classes) == 0:
            errors.append("detection.track_classes must contain at least one class")
        elif not all(isinstance(c, int) and 0 <= c <= 79 for c in classes):
            errors.append("detection.track_classes must contain COCO class IDs (0-79)")

    # Confidence threshold
    if 'confidence_threshold' not in detection:
        errors.append("detection.confidence_threshold is required")
    else:
        conf = detection['confidence_threshold']
        if not isinstance(conf, (int, float)):
            errors.append("detection.confidence_threshold must be a number")
        elif not 0.0 <= conf <= 1.0:
            errors.append("detection.confidence_threshold must be between 0.0 and 1.0")

    return errors


def _validate_roi(config: dict) -> List[str]:
    """Validate ROI settings."""
    errors = []

    if 'roi' not in config:
        return errors  # ROI is optional

    roi = config['roi']

    # Validate horizontal ROI
    if 'horizontal' in roi:
        h_roi = roi['horizontal']

        if 'enabled' not in h_roi:
            errors.append("roi.horizontal.enabled is required")

        if h_roi.get('enabled', False):
            if 'crop_from_left_pct' not in h_roi:
                errors.append("roi.horizontal.crop_from_left_pct is required when enabled")
            else:
                left = h_roi['crop_from_left_pct']
                if not isinstance(left, (int, float)) or not 0 <= left <= 100:
                    errors.append("roi.horizontal.crop_from_left_pct must be 0-100")

            if 'crop_to_right_pct' not in h_roi:
                errors.append("roi.horizontal.crop_to_right_pct is required when enabled")
            else:
                right = h_roi['crop_to_right_pct']
                if not isinstance(right, (int, float)) or not 0 <= right <= 100:
                    errors.append("roi.horizontal.crop_to_right_pct must be 0-100")

            # Check logical ordering
            if 'crop_from_left_pct' in h_roi and 'crop_to_right_pct' in h_roi:
                if h_roi['crop_from_left_pct'] >= h_roi['crop_to_right_pct']:
                    errors.append("roi.horizontal.crop_to_right_pct must be > crop_from_left_pct")

    # Validate vertical ROI
    if 'vertical' in roi:
        v_roi = roi['vertical']

        if 'enabled' not in v_roi:
            errors.append("roi.vertical.enabled is required")

        if v_roi.get('enabled', False):
            if 'crop_from_top_pct' not in v_roi:
                errors.append("roi.vertical.crop_from_top_pct is required when enabled")
            else:
                top = v_roi['crop_from_top_pct']
                if not isinstance(top, (int, float)) or not 0 <= top <= 100:
                    errors.append("roi.vertical.crop_from_top_pct must be 0-100")

            if 'crop_to_bottom_pct' not in v_roi:
                errors.append("roi.vertical.crop_to_bottom_pct is required when enabled")
            else:
                bottom = v_roi['crop_to_bottom_pct']
                if not isinstance(bottom, (int, float)) or not 0 <= bottom <= 100:
                    errors.append("roi.vertical.crop_to_bottom_pct must be 0-100")

            # Check logical ordering
            if 'crop_from_top_pct' in v_roi and 'crop_to_bottom_pct' in v_roi:
                if v_roi['crop_from_top_pct'] >= v_roi['crop_to_bottom_pct']:
                    errors.append("roi.vertical.crop_to_bottom_pct must be > crop_from_top_pct")

    return errors


def _validate_lines(config: dict) -> List[str]:
    """Validate line definitions."""
    errors = []

    if 'lines' not in config:
        return errors  # Lines are optional

    lines = config['lines']

    if not isinstance(lines, list):
        errors.append("'lines' must be a list")
        return errors

    track_classes = set(config.get('detection', {}).get('track_classes', []))

    for i, line in enumerate(lines):
        line_id = f"lines[{i}]"

        # Type
        if 'type' not in line:
            errors.append(f"{line_id}.type is required")
        elif line['type'] not in ['vertical', 'horizontal']:
            errors.append(f"{line_id}.type must be 'vertical' or 'horizontal'")

        # Position
        if 'position_pct' not in line:
            errors.append(f"{line_id}.position_pct is required")
        else:
            pos = line['position_pct']
            if not isinstance(pos, (int, float)):
                errors.append(f"{line_id}.position_pct must be a number")
            elif not 0 <= pos <= 100:
                errors.append(f"{line_id}.position_pct must be 0-100")

        # Description
        if 'description' not in line:
            errors.append(f"{line_id}.description is required")
        elif not isinstance(line['description'], str) or not line['description'].strip():
            errors.append(f"{line_id}.description must be a non-empty string")

        # Allowed classes (optional)
        if 'allowed_classes' in line:
            allowed = line['allowed_classes']
            if not isinstance(allowed, list):
                errors.append(f"{line_id}.allowed_classes must be a list")
            elif not all(isinstance(c, int) for c in allowed):
                errors.append(f"{line_id}.allowed_classes must contain integers")
            elif not set(allowed).issubset(track_classes):
                missing = set(allowed) - track_classes
                errors.append(
                    f"{line_id}.allowed_classes contains classes not in "
                    f"detection.track_classes: {missing}"
                )

    return errors


def _validate_zones(config: dict) -> List[str]:
    """Validate zone definitions."""
    errors = []

    if 'zones' not in config:
        return errors  # Zones are optional

    zones = config['zones']

    if not isinstance(zones, list):
        errors.append("'zones' must be a list")
        return errors

    track_classes = set(config.get('detection', {}).get('track_classes', []))

    for i, zone in enumerate(zones):
        zone_id = f"zones[{i}]"

        # Coordinates
        required_coords = ['x1_pct', 'y1_pct', 'x2_pct', 'y2_pct']
        for coord in required_coords:
            if coord not in zone:
                errors.append(f"{zone_id}.{coord} is required")
            else:
                val = zone[coord]
                if not isinstance(val, (int, float)):
                    errors.append(f"{zone_id}.{coord} must be a number")
                elif not 0 <= val <= 100:
                    errors.append(f"{zone_id}.{coord} must be 0-100")

        # Check logical ordering
        if all(coord in zone for coord in required_coords):
            if zone['x2_pct'] <= zone['x1_pct']:
                errors.append(f"{zone_id}.x2_pct must be > x1_pct")
            if zone['y2_pct'] <= zone['y1_pct']:
                errors.append(f"{zone_id}.y2_pct must be > y1_pct")

        # Description
        if 'description' not in zone:
            errors.append(f"{zone_id}.description is required")
        elif not isinstance(zone['description'], str) or not zone['description'].strip():
            errors.append(f"{zone_id}.description must be a non-empty string")

        # Allowed classes (optional)
        if 'allowed_classes' in zone:
            allowed = zone['allowed_classes']
            if not isinstance(allowed, list):
                errors.append(f"{zone_id}.allowed_classes must be a list")
            elif not all(isinstance(c, int) for c in allowed):
                errors.append(f"{zone_id}.allowed_classes must contain integers")
            elif not set(allowed).issubset(track_classes):
                missing = set(allowed) - track_classes
                errors.append(
                    f"{zone_id}.allowed_classes contains classes not in "
                    f"detection.track_classes: {missing}"
                )

    return errors


def _validate_output(config: dict) -> List[str]:
    """Validate output settings."""
    errors = []

    if 'output' not in config:
        errors.append("Missing 'output' section")
        return errors

    output = config['output']

    if 'json_dir' not in output:
        errors.append("output.json_dir is required")
    elif not isinstance(output['json_dir'], str):
        errors.append("output.json_dir must be a string")

    return errors


def _validate_camera(config: dict) -> List[str]:
    """Validate camera settings."""
    errors = []

    if 'camera' not in config:
        # Check if environment variable is set
        if ENV_CAMERA_URL not in os.environ:
            errors.append(f"Missing 'camera' section and {ENV_CAMERA_URL} environment variable not set")
        return errors

    camera = config['camera']

    if 'url' not in camera:
        # Check if environment variable is set
        if ENV_CAMERA_URL not in os.environ:
            errors.append(f"camera.url is required or set {ENV_CAMERA_URL} environment variable")
    elif not isinstance(camera['url'], str):
        errors.append("camera.url must be a string")

    return errors


def _validate_runtime(config: dict) -> List[str]:
    """Validate runtime settings."""
    errors = []

    if 'runtime' not in config:
        errors.append("Missing 'runtime' section")
        return errors

    runtime = config['runtime']

    if 'default_duration_hours' not in runtime:
        errors.append("runtime.default_duration_hours is required")
    else:
        duration = runtime['default_duration_hours']
        if not isinstance(duration, (int, float)):
            errors.append("runtime.default_duration_hours must be a number")
        elif duration <= 0:
            errors.append("runtime.default_duration_hours must be positive")

    # Queue size is optional with default
    if 'queue_size' in runtime:
        queue_size = runtime['queue_size']
        if not isinstance(queue_size, int):
            errors.append("runtime.queue_size must be an integer")
        elif queue_size <= 0:
            errors.append("runtime.queue_size must be positive")

    return errors
