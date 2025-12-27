"""
Configuration Planner - Terraform-like validate, plan, and dry-run features.

Provides:
- validate: Check config syntax and semantic correctness
- plan: Show event routing graph and dependency resolution
- dry-run: Simulate event processing with sample events
- load_config_with_env: Apply environment variable overrides
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from ..utils.constants import DEFAULT_QUEUE_SIZE, ENV_CAMERA_URL

# Import from new modules
from .validator import (
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = ""
        cls.CYAN = cls.GRAY = cls.BOLD = cls.RESET = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


@dataclass
class EventPlan:
    """Plan for a single event definition."""

    name: str
    match_criteria: dict[str, Any]
    actions: dict[str, Any]
    implied_actions: list[str]
    consumers: list[str]
    pdf_report_id: str | None = None
    has_shutdown: bool = False


@dataclass
class ConfigPlan:
    """Complete configuration plan."""

    events: list[EventPlan]
    pdf_reports: dict[str, dict]
    track_classes: list[tuple[int, str]]  # (id, name) pairs
    consumers: list[str]
    geometry: dict[str, list[str]]  # lines/zones descriptions


def load_config_with_env(config: dict) -> dict:
    """
    Apply environment variable overrides to config.

    Args:
        config: Base configuration dictionary

    Returns:
        Configuration with environment variables applied
    """
    # Override camera URL from environment if set
    if ENV_CAMERA_URL in os.environ:
        camera_url = os.environ[ENV_CAMERA_URL]
        logger.info(f"Using camera URL from environment: {ENV_CAMERA_URL}")
        if "camera" not in config:
            config["camera"] = {}
        config["camera"]["url"] = camera_url

    # Set default queue size if not specified
    if "runtime" not in config:
        config["runtime"] = {}
    if "queue_size" not in config["runtime"]:
        config["runtime"]["queue_size"] = DEFAULT_QUEUE_SIZE

    return config


def build_plan(config: dict, model_names: dict[int, str] | None = None) -> ConfigPlan:
    """Build a complete configuration plan from config.

    Args:
        config: Configuration dictionary
        model_names: Optional mapping of class ID -> class name from loaded model
    """
    events = []
    pdf_reports = {r["id"]: r for r in config.get("pdf_reports", []) if r.get("id")}

    # Build reverse mapping (name -> id) from model
    name_to_id: dict[str, int] = {}
    if model_names is not None:
        name_to_id = {name.lower(): id for id, name in model_names.items()}

    for event_config in config.get("events", []):
        name = event_config.get("name", "unnamed")
        match = event_config.get("match", {})
        actions = event_config.get("actions", {}).copy()

        # Track implied actions
        implied = []
        consumers = []
        pdf_report_id = actions.get("pdf_report")
        has_shutdown = actions.get("shutdown", False)

        # Apply implied action rules for pdf_report
        if pdf_report_id:
            if not actions.get("json_log"):
                actions["json_log"] = True
                implied.append("json_log (required by pdf_report)")

            pdf_report = pdf_reports.get(pdf_report_id, {})
            if pdf_report.get("photos"):
                if not actions.get("frame_capture"):
                    actions["frame_capture"] = {
                        "enabled": True,
                        "annotate": pdf_report.get("annotate", False),
                    }
                    implied.append(
                        f"frame_capture (required by {pdf_report_id} with photos=true)"
                    )
                elif pdf_report.get("annotate") and isinstance(
                    actions["frame_capture"], dict
                ):
                    # Merge annotate flag into existing frame_capture
                    actions["frame_capture"]["annotate"] = True
                    implied.append(f"annotate (from {pdf_report_id})")

        # Determine consumers/handlers
        if actions.get("json_log"):
            consumers.append("json_writer")
        if actions.get("command"):
            consumers.append("command_runner")
        if pdf_report_id:
            consumers.append(f"pdf_report:{pdf_report_id}")
        if actions.get("frame_capture", {}).get(
            "enabled", actions.get("frame_capture") is True
        ):
            consumers.append("frame_capture")

        events.append(
            EventPlan(
                name=name,
                match_criteria=match,
                actions=actions,
                implied_actions=implied,
                consumers=consumers,
                pdf_report_id=pdf_report_id,
                has_shutdown=has_shutdown,
            )
        )

    # Derive track classes
    track_classes = []
    for event in events:
        # Skip events that don't need YOLO classes
        event_type = event.match_criteria.get("event_type")
        if event_type == "NIGHTTIME_CAR":
            continue
        # DETECTED events may not specify object_class (fires for any detection)
        if event_type == "DETECTED" and not event.match_criteria.get("object_class"):
            continue
        obj_class = event.match_criteria.get("object_class")
        if obj_class:
            classes = [obj_class] if isinstance(obj_class, str) else obj_class
            for cls in classes:
                cls_lower = cls.lower()
                if cls_lower in name_to_id:
                    pair = (name_to_id[cls_lower], cls_lower)
                    if pair not in track_classes:
                        track_classes.append(pair)
                elif model_names is None:
                    # No model loaded - use placeholder ID
                    pair = (-1, cls_lower)
                    if pair not in track_classes:
                        track_classes.append(pair)

    # Geometry summary
    geometry = {
        "zones": [
            z.get("description", f"zone_{i}")
            for i, z in enumerate(config.get("zones", []))
        ],
        "lines": [
            ln.get("description", f"line_{i}")
            for i, ln in enumerate(config.get("lines", []))
        ],
    }

    # Active consumers
    all_consumers = set()
    for e in events:
        all_consumers.update(c.split(" ")[0] for c in e.consumers)

    return ConfigPlan(
        events=events,
        pdf_reports=pdf_reports,
        track_classes=sorted(track_classes),
        consumers=sorted(all_consumers),
        geometry=geometry,
    )


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in Terraform-like format."""
    print()
    print(f"{Colors.BOLD}Configuration Validation{Colors.RESET}")
    print("=" * 60)

    if result.valid:
        print(f"\n{Colors.GREEN}✓ Configuration is valid{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}✗ Configuration has errors{Colors.RESET}")

    # Print errors
    if result.errors:
        print(f"\n{Colors.RED}Errors:{Colors.RESET}")
        for error in result.errors:
            print(f"  {Colors.RED}✗{Colors.RESET} {error}")

    # Print warnings
    if result.warnings:
        print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
        for warning in result.warnings:
            print(f"  {Colors.YELLOW}!{Colors.RESET} {warning}")

    # Print derived configuration
    if result.valid and result.derived:
        print(f"\n{Colors.CYAN}Derived Configuration:{Colors.RESET}")

        track_classes = result.derived.get("track_classes", [])
        if track_classes:
            class_str = ", ".join(f"{name} ({id})" for id, name in track_classes)
            print(f"  Track classes: {class_str}")

        consumers = result.derived.get("consumers", [])
        if consumers:
            print(f"  Active consumers: {', '.join(consumers)}")

    print()


def print_plan(plan: ConfigPlan) -> None:
    """Print configuration plan in Terraform-like format."""
    print()
    print(f"{Colors.BOLD}Event Routing Plan{Colors.RESET}")
    print("=" * 60)

    # Geometry summary
    if plan.geometry["zones"] or plan.geometry["lines"]:
        print(f"\n{Colors.CYAN}Geometry:{Colors.RESET}")
        if plan.geometry["zones"]:
            print(f"  Zones: {', '.join(plan.geometry['zones'])}")
        if plan.geometry["lines"]:
            print(f"  Lines: {', '.join(plan.geometry['lines'])}")

    # Track classes
    if plan.track_classes:
        print(f"\n{Colors.CYAN}Track Classes (derived from events):{Colors.RESET}")
        for class_id, class_name in plan.track_classes:
            if class_id >= 0:
                print(f"  {Colors.GREEN}+{Colors.RESET} {class_name} (ID: {class_id})")
            else:
                print(
                    f"  {Colors.YELLOW}?{Colors.RESET} {class_name} (ID: unknown - model not loaded)"
                )

    # Events
    print(f"\n{Colors.CYAN}Events:{Colors.RESET}")
    for event in plan.events:
        print(f"\n  {Colors.BOLD}{event.name}{Colors.RESET}")

        # Match criteria
        print(f"    {Colors.GRAY}Match:{Colors.RESET}")
        if event.match_criteria.get("event_type"):
            print(f"      event_type: {event.match_criteria['event_type']}")
        if event.match_criteria.get("zone"):
            print(f'      zone: "{event.match_criteria["zone"]}"')
        if event.match_criteria.get("line"):
            print(f'      line: "{event.match_criteria["line"]}"')
        if event.match_criteria.get("object_class"):
            obj = event.match_criteria["object_class"]
            if isinstance(obj, list):
                print(f"      object_class: [{', '.join(obj)}]")
            else:
                print(f"      object_class: {obj}")

        # Actions (resolved)
        print(f"    {Colors.GRAY}Actions (resolved):{Colors.RESET}")
        for consumer in event.consumers:
            print(f"      {Colors.GREEN}->{Colors.RESET} {consumer}")

        # Show resolved action details
        if event.actions.get("frame_capture"):
            fc = event.actions["frame_capture"]
            if isinstance(fc, dict):
                details = []
                if fc.get("annotate"):
                    details.append("annotate=true")
                if fc.get("max_photos"):
                    details.append(f"max_photos={fc['max_photos']}")
                if fc.get("expected_duration_seconds"):
                    hrs = fc["expected_duration_seconds"] / 3600
                    details.append(f"duration={hrs:.1f}h")
                if details:
                    print(
                        f"         {Colors.GRAY}frame_capture: {', '.join(details)}{Colors.RESET}"
                    )

        # Implied actions
        if event.implied_actions:
            print(f"    {Colors.GRAY}Implied (auto-enabled):{Colors.RESET}")
            for implied in event.implied_actions:
                print(f"      {Colors.YELLOW}+{Colors.RESET} {implied}")

        # Shutdown flag
        if event.has_shutdown:
            print(f"    {Colors.RED}SHUTDOWN: Detector stops after this event{Colors.RESET}")

    # Consumer summary
    print(f"\n{Colors.CYAN}Active Consumers:{Colors.RESET}")
    for consumer in plan.consumers:
        print(f"  {Colors.GREEN}+{Colors.RESET} {consumer}")

    print(f"\n{Colors.GREEN}No issues found. Ready to run.{Colors.RESET}")
    print()


def simulate_dry_run(
    config: dict,
    sample_events: list[dict],
    model_names: dict[int, str] | None = None,
) -> None:
    """Simulate event processing with sample events."""
    print()
    print(f"{Colors.BOLD}Dry Run Simulation{Colors.RESET}")
    print("=" * 60)

    plan = build_plan(config, model_names)

    print(
        f"\n{Colors.CYAN}Processing {len(sample_events)} sample event(s):{Colors.RESET}\n"
    )

    matched_count = 0
    unmatched_count = 0
    actions_taken = {
        "json_log": 0,
        "command": 0,
        "pdf_report": 0,
        "frame_capture": 0,
    }
    pdf_report_counts = {}
    shutdown_triggered = False

    for i, sample_event in enumerate(sample_events, 1):
        event_type = sample_event.get("event_type", "UNKNOWN")
        obj_class = sample_event.get(
            "object_class_name", sample_event.get("object_class", "unknown")
        )
        zone = sample_event.get("zone_description", sample_event.get("zone"))
        line = sample_event.get("line_description", sample_event.get("line"))
        location = zone or line or "unknown"

        print(f"  [{i}] {event_type}: {obj_class} @ {location}")

        # Find matching event definition
        matched_event = None
        for event_plan in plan.events:
            if _matches_event(sample_event, event_plan):
                matched_event = event_plan
                break

        if matched_event:
            matched_count += 1
            print(f"      {Colors.GREEN}-> Matched: {matched_event.name}{Colors.RESET}")

            # Track actions
            for consumer in matched_event.consumers:
                if "json_writer" in consumer:
                    actions_taken["json_log"] += 1
                    print(f"         {Colors.GRAY}-> Write to JSON log{Colors.RESET}")
                elif "command_runner" in consumer:
                    actions_taken["command"] += 1
                    print(f"         {Colors.GRAY}-> Run command{Colors.RESET}")
                elif "pdf_report" in consumer:
                    actions_taken["pdf_report"] += 1
                    report_id = matched_event.pdf_report_id
                    pdf_report_counts[report_id] = (
                        pdf_report_counts.get(report_id, 0) + 1
                    )
                    print(
                        f"         {Colors.GRAY}-> Include in PDF: {report_id}{Colors.RESET}"
                    )
                elif "frame_capture" in consumer:
                    actions_taken["frame_capture"] += 1
                    print(f"         {Colors.GRAY}-> Capture frame{Colors.RESET}")

            # Check for shutdown
            if matched_event.has_shutdown:
                shutdown_triggered = True
                print(f"         {Colors.RED}-> SHUTDOWN triggered{Colors.RESET}")
        else:
            unmatched_count += 1
            print(f"      {Colors.YELLOW}-> No match (discarded){Colors.RESET}")

    # Summary
    print(f"\n{Colors.CYAN}Simulation Summary:{Colors.RESET}")
    print(f"  Events processed: {len(sample_events)}")
    print(f"  Matched: {Colors.GREEN}{matched_count}{Colors.RESET}")
    print(f"  Unmatched: {Colors.YELLOW}{unmatched_count}{Colors.RESET}")

    print(f"\n{Colors.CYAN}Actions that would be taken:{Colors.RESET}")
    print(f"  JSON log writes: {actions_taken['json_log']}")
    print(f"  Commands executed: {actions_taken['command']}")
    print(f"  PDF report queue adds: {actions_taken['pdf_report']}")
    print(f"  Frame captures: {actions_taken['frame_capture']}")

    if pdf_report_counts:
        pdf_reports = {r["id"]: r for r in config.get("pdf_reports", []) if r.get("id")}
        print(
            f"\n{Colors.CYAN}PDF report contents (what would be generated):{Colors.RESET}"
        )
        for report_id, count in pdf_report_counts.items():
            report = pdf_reports.get(report_id, {})
            photos = " + photos" if report.get("photos") else ""
            print(f"  {report_id}: {count} event(s){photos}")

    if shutdown_triggered:
        print(f"\n{Colors.RED}SHUTDOWN would be triggered - detector would stop{Colors.RESET}")

    print()


def _matches_event(sample: dict, event_plan: EventPlan) -> bool:
    """Check if sample event matches event plan criteria."""
    match = event_plan.match_criteria

    # Check event type
    if match.get("event_type"):
        if sample.get("event_type") != match["event_type"]:
            return False

    # Check zone
    if match.get("zone"):
        sample_zone = sample.get("zone_description") or sample.get("zone")
        if sample_zone != match["zone"]:
            return False

    # Check line
    if match.get("line"):
        sample_line = sample.get("line_description") or sample.get("line")
        if sample_line != match["line"]:
            return False

    # Check object class
    if match.get("object_class"):
        sample_class = (
            sample.get("object_class_name") or sample.get("object_class", "")
        ).lower()
        match_classes = match["object_class"]
        if isinstance(match_classes, str):
            match_classes = [match_classes]
        if sample_class not in [c.lower() for c in match_classes]:
            return False

    return True


def generate_sample_events(config: dict) -> list[dict]:
    """Generate sample events based on config for dry-run testing."""
    samples = []

    events = config.get("events", [])

    # Generate samples based on event definitions
    for event in events:
        match = event.get("match", {})
        event_type = match.get("event_type", "LINE_CROSS")

        # Skip NIGHTTIME_CAR for sample generation (needs special handling)
        if event_type == "NIGHTTIME_CAR":
            samples.append(
                {
                    "event_type": "NIGHTTIME_CAR",
                    "object_class_name": "nighttime_car",
                    "zone_description": match.get("zone", "unknown"),
                    "track_id": f"nc_{len(samples) + 1}",
                }
            )
            continue

        # DETECTED events - raw detection to event
        if event_type == "DETECTED":
            obj_classes = match.get("object_class", ["detection"])
            if isinstance(obj_classes, str):
                obj_classes = [obj_classes]
            for obj_class in obj_classes[:2]:
                samples.append(
                    {
                        "event_type": "DETECTED",
                        "object_class_name": obj_class,
                        "track_id": len(samples) + 1,
                        "confidence": 0.85,
                    }
                )
            continue

        obj_classes = match.get("object_class", [])
        if isinstance(obj_classes, str):
            obj_classes = [obj_classes]
        if not obj_classes:
            obj_classes = ["unknown"]

        for obj_class in obj_classes[:2]:  # Limit to 2 per class
            sample = {
                "event_type": event_type,
                "object_class_name": obj_class,
                "track_id": len(samples) + 1,
            }

            if match.get("zone"):
                sample["zone_description"] = match["zone"]
            elif match.get("line"):
                sample["line_description"] = match["line"]

            samples.append(sample)

    # Add some unmatched events for realism
    samples.append(
        {
            "event_type": "LINE_CROSS",
            "object_class_name": "person",
            "line_description": "unknown line",
            "track_id": 999,
        }
    )

    return samples


def load_sample_events(path: str) -> list[dict]:
    """Load sample events from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both array and object with 'events' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "events" in data:
        return data["events"]
    else:
        raise ValueError(
            "Sample events file must contain an array or object with 'events' key"
        )
