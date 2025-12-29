"""
Configuration Validator - Validates config syntax and semantic correctness.

Provides comprehensive validation with detailed error messages.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of config validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)


def validate_config_full(
    config: dict, model_names: dict[int, str] | None = None
) -> ValidationResult:
    """
    Comprehensive config validation with detailed error messages.

    Args:
        config: Configuration dictionary to validate
        model_names: Optional mapping of class ID -> class name from loaded model.
                     If provided, event class names are validated against the model.
                     If None, class validation is skipped (useful for --validate without model).

    Returns:
        ValidationResult with errors, warnings, and derived configuration.
    """
    result = ValidationResult(valid=True)

    # Build name->id mapping from model if provided
    model_name_to_id: dict[str, int] | None = None
    if model_names is not None:
        model_name_to_id = {name.lower(): id for id, name in model_names.items()}
        result.derived["model_classes"] = list(model_names.values())

    # Basic structure validation
    _validate_required_sections(config, result)
    if result.errors:
        result.valid = False
        return result

    # Detection settings
    _validate_detection_settings(config, result)

    # ROI settings
    _validate_roi(config, result)

    # Geometry (lines and zones)
    zone_descriptions = _validate_zones(config, result)
    line_descriptions = _validate_lines(config, result)

    # Events
    _validate_events(
        config,
        result,
        zone_descriptions,
        line_descriptions,
        model_name_to_id,
    )

    # Frame storage (if events use frame_capture)
    _validate_frame_storage(config, result)

    # Derive track_classes from events
    track_classes = _derive_track_classes_from_events(config, result, model_name_to_id)
    result.derived["track_classes"] = track_classes
    result.derived["consumers"] = _derive_consumers_for_validation(config)

    if result.errors:
        result.valid = False

    return result


def _validate_required_sections(config: dict, result: ValidationResult) -> None:
    """Validate required top-level sections exist."""
    required = ["detection", "output", "camera", "runtime"]
    for section in required:
        if section not in config:
            result.errors.append(f"Missing required section: '{section}'")


def _validate_detection_settings(config: dict, result: ValidationResult) -> None:
    """Validate detection configuration."""
    detection = config.get("detection", {})

    # Model file
    model_file = detection.get("model_file")
    if not model_file:
        result.errors.append("detection.model_file is required")
    else:
        model_path = Path(model_file)
        if not model_path.exists():
            result.warnings.append(
                f"Model file not found: {model_file} (will be downloaded if valid)"
            )
        elif not str(model_path).endswith(".pt"):
            result.errors.append(f"Model file must be .pt format: {model_file}")

    # Confidence threshold
    conf = detection.get("confidence_threshold")
    if conf is None:
        result.errors.append("detection.confidence_threshold is required")
    elif not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        result.errors.append(
            "detection.confidence_threshold must be between 0.0 and 1.0"
        )


def _validate_roi(config: dict, result: ValidationResult) -> None:
    """Validate ROI (region of interest) settings."""
    roi = config.get("roi")
    if not roi:
        return  # ROI is optional

    # Validate horizontal ROI
    h_roi = roi.get("horizontal", {})
    if h_roi.get("enabled"):
        left = h_roi.get("crop_from_left_pct")
        right = h_roi.get("crop_to_right_pct")

        if left is None:
            result.errors.append(
                "roi.horizontal.crop_from_left_pct required when enabled"
            )
        elif not isinstance(left, (int, float)) or not 0 <= left <= 100:
            result.errors.append("roi.horizontal.crop_from_left_pct must be 0-100")

        if right is None:
            result.errors.append(
                "roi.horizontal.crop_to_right_pct required when enabled"
            )
        elif not isinstance(right, (int, float)) or not 0 <= right <= 100:
            result.errors.append("roi.horizontal.crop_to_right_pct must be 0-100")

        if left is not None and right is not None and left >= right:
            result.errors.append(
                "roi.horizontal: crop_to_right_pct must be > crop_from_left_pct"
            )

    # Validate vertical ROI
    v_roi = roi.get("vertical", {})
    if v_roi.get("enabled"):
        top = v_roi.get("crop_from_top_pct")
        bottom = v_roi.get("crop_to_bottom_pct")

        if top is None:
            result.errors.append("roi.vertical.crop_from_top_pct required when enabled")
        elif not isinstance(top, (int, float)) or not 0 <= top <= 100:
            result.errors.append("roi.vertical.crop_from_top_pct must be 0-100")

        if bottom is None:
            result.errors.append(
                "roi.vertical.crop_to_bottom_pct required when enabled"
            )
        elif not isinstance(bottom, (int, float)) or not 0 <= bottom <= 100:
            result.errors.append("roi.vertical.crop_to_bottom_pct must be 0-100")

        if top is not None and bottom is not None and top >= bottom:
            result.errors.append(
                "roi.vertical: crop_to_bottom_pct must be > crop_from_top_pct"
            )


def _validate_zones(config: dict, result: ValidationResult) -> set[str]:
    """Validate zone definitions. Returns set of zone descriptions."""
    descriptions = set()
    zones = config.get("zones", [])

    if not isinstance(zones, list):
        result.errors.append("'zones' must be a list")
        return descriptions

    for i, zone in enumerate(zones):
        zone_ref = f"zones[{i}]"

        # Required coordinates
        for coord in ["x1_pct", "y1_pct", "x2_pct", "y2_pct"]:
            val = zone.get(coord)
            if val is None:
                result.errors.append(f"{zone_ref}.{coord} is required")
            elif not isinstance(val, (int, float)) or not 0 <= val <= 100:
                result.errors.append(f"{zone_ref}.{coord} must be 0-100")

        # Coordinate ordering
        if all(
            zone.get(c) is not None for c in ["x1_pct", "x2_pct", "y1_pct", "y2_pct"]
        ):
            if zone["x2_pct"] <= zone["x1_pct"]:
                result.errors.append(f"{zone_ref}: x2_pct must be > x1_pct")
            if zone["y2_pct"] <= zone["y1_pct"]:
                result.errors.append(f"{zone_ref}: y2_pct must be > y1_pct")

        # Description
        desc = zone.get("description")
        if not desc or not isinstance(desc, str):
            result.errors.append(
                f"{zone_ref}.description is required and must be a string"
            )
        else:
            if desc in descriptions:
                result.errors.append(f"{zone_ref}: duplicate description '{desc}'")
            descriptions.add(desc)

    return descriptions


def _validate_lines(config: dict, result: ValidationResult) -> set[str]:
    """Validate line definitions. Returns set of line descriptions."""
    descriptions = set()
    lines = config.get("lines", [])

    if not isinstance(lines, list):
        result.errors.append("'lines' must be a list")
        return descriptions

    for i, line in enumerate(lines):
        line_ref = f"lines[{i}]"

        # Type
        line_type = line.get("type")
        if line_type not in ["vertical", "horizontal"]:
            result.errors.append(f"{line_ref}.type must be 'vertical' or 'horizontal'")

        # Position
        pos = line.get("position_pct")
        if pos is None:
            result.errors.append(f"{line_ref}.position_pct is required")
        elif not isinstance(pos, (int, float)) or not 0 <= pos <= 100:
            result.errors.append(f"{line_ref}.position_pct must be 0-100")

        # Description
        desc = line.get("description")
        if not desc or not isinstance(desc, str):
            result.errors.append(
                f"{line_ref}.description is required and must be a string"
            )
        else:
            if desc in descriptions:
                result.errors.append(f"{line_ref}: duplicate description '{desc}'")
            descriptions.add(desc)

    return descriptions


def _validate_events(
    config: dict,
    result: ValidationResult,
    zone_descriptions: set[str],
    line_descriptions: set[str],
    model_name_to_id: dict[str, int] | None = None,
) -> None:
    """Validate event definitions."""
    events = config.get("events", [])

    if not isinstance(events, list):
        result.errors.append("'events' must be a list")
        return

    if not events:
        result.warnings.append("No events defined - nothing will be tracked")
        return

    event_names = set()
    for i, event in enumerate(events):
        event_ref = f"events[{i}]"

        # Name
        name = event.get("name")
        if not name or not isinstance(name, str):
            result.errors.append(f"{event_ref}.name is required and must be a string")
        else:
            if name in event_names:
                result.errors.append(f"{event_ref}: duplicate event name '{name}'")
            event_names.add(name)

        # Match criteria
        match = event.get("match", {})
        if not match:
            result.errors.append(f"{event_ref}.match is required")
            continue

        # Event type
        event_type = match.get("event_type")
        valid_types = [
            "LINE_CROSS",
            "ZONE_ENTER",
            "ZONE_EXIT",
            "NIGHTTIME_CAR",
            "DETECTED",
        ]
        if event_type and event_type not in valid_types:
            result.errors.append(
                f"{event_ref}.match.event_type must be one of: {valid_types}"
            )

        # Zone reference
        zone = match.get("zone")
        if zone and zone not in zone_descriptions:
            result.errors.append(f"{event_ref}.match.zone '{zone}' does not exist")

        # Line reference
        line = match.get("line")
        if line and line not in line_descriptions:
            result.errors.append(f"{event_ref}.match.line '{line}' does not exist")

        # Object classes (only validate for non-NIGHTTIME_CAR events)
        obj_class = match.get("object_class")
        if obj_class and event_type != "NIGHTTIME_CAR":
            classes = [obj_class] if isinstance(obj_class, str) else obj_class
            for cls in classes:
                # Validate against loaded model if available
                if model_name_to_id is not None:
                    if cls.lower() not in model_name_to_id:
                        available = ", ".join(sorted(model_name_to_id.keys())[:10])
                        more = (
                            f" (and {len(model_name_to_id) - 10} more)"
                            if len(model_name_to_id) > 10
                            else ""
                        )
                        result.errors.append(
                            f"{event_ref}.match.object_class '{cls}' not found in model. "
                            f"Available: {available}{more}"
                        )

        # Actions
        actions = event.get("actions", {})
        if not actions:
            result.errors.append(f"{event_ref}.actions is required")
            continue

        # Validate command action
        command = actions.get("command")
        if command and isinstance(command, dict):
            exec_path = command.get("exec")
            if not exec_path:
                result.errors.append(f"{event_ref}.actions.command.exec is required")
            timeout = command.get("timeout_seconds")
            if timeout is not None and (
                not isinstance(timeout, (int, float)) or timeout <= 0
            ):
                result.errors.append(
                    f"{event_ref}.actions.command.timeout_seconds must be positive"
                )


def _validate_frame_storage(config: dict, _result: ValidationResult) -> None:
    """Validate frame storage if frame capture is needed."""
    events = config.get("events", [])

    # Check if any event needs frame capture
    needs_frames = False
    for event in events:
        actions = event.get("actions", {})
        if actions.get("frame_capture"):
            needs_frames = True
            break

    if not needs_frames:
        return

    # temp_frames is implicitly enabled if any event needs frames
    # No warning needed - the event owns the frame capture decision


def _derive_consumers_for_validation(config: dict) -> list[str]:
    """Derive which consumers will be active (for validation display only)."""
    consumers = set()
    events = config.get("events", [])
    reports = {r["id"]: r for r in config.get("reports", []) if r.get("id")}

    for event in events:
        actions = event.get("actions", {})

        if actions.get("json_log", False):
            consumers.add("json_writer")

        if actions.get("command"):
            consumers.add("command_runner")

        report_id = actions.get("report")
        if report_id:
            consumers.add("report")
            if reports.get(report_id, {}).get("photos"):
                consumers.add("frame_capture")

        if actions.get("frame_capture"):
            consumers.add("frame_capture")

    return sorted(consumers)


def _derive_track_classes_from_events(
    config: dict,
    _result: ValidationResult,
    model_name_to_id: dict[str, int] | None = None,
) -> list[tuple[int, str]]:
    """Derive class IDs from event definitions using loaded model mapping."""
    class_names = set()
    for event in config.get("events", []):
        match = event.get("match", {})
        # Skip NIGHTTIME_CAR events - they don't need YOLO classes
        if match.get("event_type") == "NIGHTTIME_CAR":
            continue
        obj_class = match.get("object_class")
        if obj_class:
            if isinstance(obj_class, list):
                class_names.update(obj_class)
            else:
                class_names.add(obj_class)

    # Convert names to (id, name) pairs using model mapping
    pairs = []
    for name in class_names:
        name_lower = name.lower()
        if model_name_to_id is not None and name_lower in model_name_to_id:
            pairs.append((model_name_to_id[name_lower], name))
        elif model_name_to_id is None:
            # No model loaded - can't derive IDs, just use name with placeholder ID
            # This happens during --validate without model
            pairs.append((-1, name))

    return sorted(pairs, key=lambda x: x[0])
