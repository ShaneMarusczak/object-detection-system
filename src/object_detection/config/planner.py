"""
Configuration Planner - Terraform-like validate, plan, and dry-run features.

Provides:
- validate: Check config syntax and semantic correctness
- plan: Show event routing graph and dependency resolution
- dry-run: Simulate event processing with sample events
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

from ..utils.coco_classes import COCO_NAME_TO_ID, COCO_CLASSES


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = ''
        cls.CYAN = cls.GRAY = cls.BOLD = cls.RESET = ''


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


@dataclass
class ValidationResult:
    """Result of config validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    derived: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventPlan:
    """Plan for a single event definition."""
    name: str
    match_criteria: Dict[str, Any]
    actions: Dict[str, Any]
    implied_actions: List[str]
    consumers: List[str]
    digest_id: Optional[str] = None


@dataclass
class ConfigPlan:
    """Complete configuration plan."""
    events: List[EventPlan]
    digests: Dict[str, Dict]
    track_classes: List[Tuple[int, str]]  # (id, name) pairs
    consumers: List[str]
    geometry: Dict[str, List[str]]  # lines/zones descriptions


def validate_config_full(config: dict) -> ValidationResult:
    """
    Comprehensive config validation with detailed error messages.

    Returns ValidationResult with errors, warnings, and derived configuration.
    """
    result = ValidationResult(valid=True)

    # Basic structure validation
    _validate_required_sections(config, result)
    if result.errors:
        result.valid = False
        return result

    # Detection settings
    _validate_detection_settings(config, result)

    # Geometry (lines and zones)
    zone_descriptions = _validate_zones(config, result)
    line_descriptions = _validate_lines(config, result)

    # Events and digests
    digest_ids = _validate_digests(config, result)
    _validate_events(config, result, zone_descriptions, line_descriptions, digest_ids)

    # Notifications (if events use email)
    _validate_notifications(config, result)

    # Frame storage (if events use frame_capture or photo digests)
    _validate_frame_storage(config, result)

    # Derive track_classes from events
    track_classes = _derive_track_classes_from_events(config, result)
    result.derived['track_classes'] = track_classes
    result.derived['consumers'] = _derive_consumers(config)

    if result.errors:
        result.valid = False

    return result


def _validate_required_sections(config: dict, result: ValidationResult) -> None:
    """Validate required top-level sections exist."""
    required = ['detection', 'output', 'camera', 'runtime']
    for section in required:
        if section not in config:
            result.errors.append(f"Missing required section: '{section}'")


def _validate_detection_settings(config: dict, result: ValidationResult) -> None:
    """Validate detection configuration."""
    detection = config.get('detection', {})

    # Model file
    model_file = detection.get('model_file')
    if not model_file:
        result.errors.append("detection.model_file is required")
    else:
        model_path = Path(model_file)
        if not model_path.exists():
            result.warnings.append(f"Model file not found: {model_file} (will be downloaded if valid)")
        elif not str(model_path).endswith('.pt'):
            result.errors.append(f"Model file must be .pt format: {model_file}")

    # Confidence threshold
    conf = detection.get('confidence_threshold')
    if conf is None:
        result.errors.append("detection.confidence_threshold is required")
    elif not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        result.errors.append("detection.confidence_threshold must be between 0.0 and 1.0")


def _validate_zones(config: dict, result: ValidationResult) -> Set[str]:
    """Validate zone definitions. Returns set of zone descriptions."""
    descriptions = set()
    zones = config.get('zones', [])

    if not isinstance(zones, list):
        result.errors.append("'zones' must be a list")
        return descriptions

    for i, zone in enumerate(zones):
        zone_ref = f"zones[{i}]"

        # Required coordinates
        for coord in ['x1_pct', 'y1_pct', 'x2_pct', 'y2_pct']:
            val = zone.get(coord)
            if val is None:
                result.errors.append(f"{zone_ref}.{coord} is required")
            elif not isinstance(val, (int, float)) or not 0 <= val <= 100:
                result.errors.append(f"{zone_ref}.{coord} must be 0-100")

        # Coordinate ordering
        if all(zone.get(c) is not None for c in ['x1_pct', 'x2_pct', 'y1_pct', 'y2_pct']):
            if zone['x2_pct'] <= zone['x1_pct']:
                result.errors.append(f"{zone_ref}: x2_pct must be > x1_pct")
            if zone['y2_pct'] <= zone['y1_pct']:
                result.errors.append(f"{zone_ref}: y2_pct must be > y1_pct")

        # Description
        desc = zone.get('description')
        if not desc or not isinstance(desc, str):
            result.errors.append(f"{zone_ref}.description is required and must be a string")
        else:
            if desc in descriptions:
                result.errors.append(f"{zone_ref}: duplicate description '{desc}'")
            descriptions.add(desc)

    return descriptions


def _validate_lines(config: dict, result: ValidationResult) -> Set[str]:
    """Validate line definitions. Returns set of line descriptions."""
    descriptions = set()
    lines = config.get('lines', [])

    if not isinstance(lines, list):
        result.errors.append("'lines' must be a list")
        return descriptions

    for i, line in enumerate(lines):
        line_ref = f"lines[{i}]"

        # Type
        line_type = line.get('type')
        if line_type not in ['vertical', 'horizontal']:
            result.errors.append(f"{line_ref}.type must be 'vertical' or 'horizontal'")

        # Position
        pos = line.get('position_pct')
        if pos is None:
            result.errors.append(f"{line_ref}.position_pct is required")
        elif not isinstance(pos, (int, float)) or not 0 <= pos <= 100:
            result.errors.append(f"{line_ref}.position_pct must be 0-100")

        # Description
        desc = line.get('description')
        if not desc or not isinstance(desc, str):
            result.errors.append(f"{line_ref}.description is required and must be a string")
        else:
            if desc in descriptions:
                result.errors.append(f"{line_ref}: duplicate description '{desc}'")
            descriptions.add(desc)

    return descriptions


def _validate_digests(config: dict, result: ValidationResult) -> Set[str]:
    """Validate digest definitions. Returns set of digest IDs."""
    digest_ids = set()
    digests = config.get('digests', [])

    if not isinstance(digests, list):
        result.errors.append("'digests' must be a list")
        return digest_ids

    for i, digest in enumerate(digests):
        digest_ref = f"digests[{i}]"

        # ID
        digest_id = digest.get('id')
        if not digest_id or not isinstance(digest_id, str):
            result.errors.append(f"{digest_ref}.id is required and must be a string")
        else:
            if digest_id in digest_ids:
                result.errors.append(f"{digest_ref}: duplicate digest id '{digest_id}'")
            digest_ids.add(digest_id)

        # Period
        period = digest.get('period_minutes')
        if period is None:
            result.errors.append(f"{digest_ref}.period_minutes is required")
        elif not isinstance(period, (int, float)) or period <= 0:
            result.errors.append(f"{digest_ref}.period_minutes must be positive")

        # Photos flag
        photos = digest.get('photos')
        if photos is not None and not isinstance(photos, bool):
            result.errors.append(f"{digest_ref}.photos must be a boolean")

        # Frame config (required if photos: true)
        if digest.get('photos') and not digest.get('frame_config'):
            result.warnings.append(f"{digest_ref}: photos=true but no frame_config specified (using defaults)")

    return digest_ids


def _validate_events(config: dict, result: ValidationResult,
                     zone_descriptions: Set[str], line_descriptions: Set[str],
                     digest_ids: Set[str]) -> None:
    """Validate event definitions."""
    events = config.get('events', [])

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
        name = event.get('name')
        if not name or not isinstance(name, str):
            result.errors.append(f"{event_ref}.name is required and must be a string")
        else:
            if name in event_names:
                result.errors.append(f"{event_ref}: duplicate event name '{name}'")
            event_names.add(name)

        # Match criteria
        match = event.get('match', {})
        if not match:
            result.errors.append(f"{event_ref}.match is required")
            continue

        # Event type
        event_type = match.get('event_type')
        valid_types = ['LINE_CROSS', 'ZONE_ENTER', 'ZONE_EXIT', 'ZONE_DWELL']
        if event_type and event_type not in valid_types:
            result.errors.append(f"{event_ref}.match.event_type must be one of: {valid_types}")

        # Zone reference
        zone = match.get('zone')
        if zone and zone not in zone_descriptions:
            result.errors.append(f"{event_ref}.match.zone '{zone}' does not exist")

        # Line reference
        line = match.get('line')
        if line and line not in line_descriptions:
            result.errors.append(f"{event_ref}.match.line '{line}' does not exist")

        # Object classes
        obj_class = match.get('object_class')
        if obj_class:
            classes = [obj_class] if isinstance(obj_class, str) else obj_class
            for cls in classes:
                if cls.lower() not in COCO_NAME_TO_ID:
                    result.errors.append(
                        f"{event_ref}.match.object_class '{cls}' is not a valid COCO class. "
                        f"Valid: person, car, cat, dog, truck, bus, motorcycle, bird, etc."
                    )

        # Actions
        actions = event.get('actions', {})
        if not actions:
            result.errors.append(f"{event_ref}.actions is required")
            continue

        # Validate digest reference
        digest_ref = actions.get('email_digest')
        if digest_ref and digest_ref not in digest_ids:
            result.errors.append(f"{event_ref}.actions.email_digest '{digest_ref}' does not exist")

        # Validate email_immediate
        email_immediate = actions.get('email_immediate')
        if email_immediate and isinstance(email_immediate, dict):
            cooldown = email_immediate.get('cooldown_minutes')
            if cooldown is not None and (not isinstance(cooldown, (int, float)) or cooldown < 0):
                result.errors.append(f"{event_ref}.actions.email_immediate.cooldown_minutes must be non-negative")


def _validate_notifications(config: dict, result: ValidationResult) -> None:
    """Validate notification settings if email actions are used."""
    events = config.get('events', [])

    # Check if any event uses email
    uses_email = False
    for event in events:
        actions = event.get('actions', {})
        if actions.get('email_digest') or actions.get('email_immediate'):
            uses_email = True
            break

    if not uses_email:
        return

    notifications = config.get('notifications', {})
    if not notifications.get('enabled'):
        result.errors.append("Events use email actions but notifications.enabled is false")
        return

    email = notifications.get('email', {})
    if not email.get('enabled'):
        result.errors.append("Events use email actions but notifications.email.enabled is false")
        return

    # Required email fields
    required_fields = ['smtp_server', 'smtp_port', 'username', 'password', 'from_address', 'to_addresses']
    for field in required_fields:
        if not email.get(field):
            result.errors.append(f"notifications.email.{field} is required for email actions")


def _validate_frame_storage(config: dict, result: ValidationResult) -> None:
    """Validate frame storage if frame capture is needed."""
    events = config.get('events', [])
    digests = {d['id']: d for d in config.get('digests', []) if d.get('id')}

    # Check if any event needs frame capture
    needs_frames = False
    for event in events:
        actions = event.get('actions', {})
        if actions.get('frame_capture'):
            needs_frames = True
            break
        digest_id = actions.get('email_digest')
        if digest_id and digests.get(digest_id, {}).get('photos'):
            needs_frames = True
            break

    if not needs_frames:
        return

    # Check temp_frames_enabled
    if not config.get('temp_frames_enabled', False):
        result.warnings.append(
            "Frame capture is used but temp_frames_enabled is false - "
            "frames will only be captured if temp frame buffer is running"
        )


def _derive_track_classes_from_events(config: dict, result: ValidationResult) -> List[Tuple[int, str]]:
    """Derive COCO class IDs from event definitions."""
    class_names = set()

    for event in config.get('events', []):
        obj_class = event.get('match', {}).get('object_class')
        if obj_class:
            if isinstance(obj_class, list):
                class_names.update(c.lower() for c in obj_class)
            else:
                class_names.add(obj_class.lower())

    # Convert to (id, name) pairs
    track_classes = []
    for name in sorted(class_names):
        if name in COCO_NAME_TO_ID:
            track_classes.append((COCO_NAME_TO_ID[name], name))

    return track_classes


def _derive_consumers(config: dict) -> List[str]:
    """Derive which consumers will be active."""
    consumers = []
    events = config.get('events', [])
    digests = {d['id']: d for d in config.get('digests', []) if d.get('id')}

    has_json = False
    has_email_immediate = False
    has_email_digest = False
    has_frame_capture = False

    for event in events:
        actions = event.get('actions', {})

        if actions.get('json_log'):
            has_json = True
        if actions.get('email_immediate'):
            has_email_immediate = True

        digest_id = actions.get('email_digest')
        if digest_id:
            has_json = True  # Implied
            has_email_digest = True
            if digests.get(digest_id, {}).get('photos'):
                has_frame_capture = True  # Implied

        if actions.get('frame_capture'):
            has_frame_capture = True

    if has_json:
        consumers.append('json_writer')
    if has_email_immediate:
        consumers.append('email_notifier')
    if has_email_digest:
        consumers.append('email_digest')
    if has_frame_capture:
        consumers.append('frame_capture')

    return consumers


def build_plan(config: dict) -> ConfigPlan:
    """Build a complete configuration plan from config."""
    events = []
    digests = {d['id']: d for d in config.get('digests', []) if d.get('id')}

    for event_config in config.get('events', []):
        name = event_config.get('name', 'unnamed')
        match = event_config.get('match', {})
        actions = event_config.get('actions', {}).copy()

        # Track implied actions
        implied = []
        consumers = []
        digest_id = actions.get('email_digest')

        # Apply implied action rules
        if digest_id:
            if not actions.get('json_log'):
                actions['json_log'] = True
                implied.append('json_log (required by email_digest)')

            digest = digests.get(digest_id, {})
            if digest.get('photos') and not actions.get('frame_capture'):
                actions['frame_capture'] = {'enabled': True}
                implied.append(f"frame_capture (required by {digest_id} with photos=true)")

        # Determine consumers
        if actions.get('json_log'):
            consumers.append('json_writer')
        if actions.get('email_immediate', {}).get('enabled'):
            consumers.append('email_notifier')
        if digest_id:
            consumers.append(f'email_digest ({digest_id})')
        if actions.get('frame_capture', {}).get('enabled', actions.get('frame_capture') == True):
            consumers.append('frame_capture')

        events.append(EventPlan(
            name=name,
            match_criteria=match,
            actions=actions,
            implied_actions=implied,
            consumers=consumers,
            digest_id=digest_id
        ))

    # Derive track classes
    track_classes = []
    for event in events:
        obj_class = event.match_criteria.get('object_class')
        if obj_class:
            classes = [obj_class] if isinstance(obj_class, str) else obj_class
            for cls in classes:
                cls_lower = cls.lower()
                if cls_lower in COCO_NAME_TO_ID:
                    pair = (COCO_NAME_TO_ID[cls_lower], cls_lower)
                    if pair not in track_classes:
                        track_classes.append(pair)

    # Geometry summary
    geometry = {
        'zones': [z.get('description', f"zone_{i}") for i, z in enumerate(config.get('zones', []))],
        'lines': [l.get('description', f"line_{i}") for i, l in enumerate(config.get('lines', []))]
    }

    # Active consumers
    all_consumers = set()
    for e in events:
        all_consumers.update(c.split(' ')[0] for c in e.consumers)

    return ConfigPlan(
        events=events,
        digests=digests,
        track_classes=sorted(track_classes),
        consumers=sorted(all_consumers),
        geometry=geometry
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

        track_classes = result.derived.get('track_classes', [])
        if track_classes:
            class_str = ', '.join(f"{name} ({id})" for id, name in track_classes)
            print(f"  Track classes: {class_str}")

        consumers = result.derived.get('consumers', [])
        if consumers:
            print(f"  Active consumers: {', '.join(consumers)}")

    print()


def print_plan(plan: ConfigPlan) -> None:
    """Print configuration plan in Terraform-like format."""
    print()
    print(f"{Colors.BOLD}Event Routing Plan{Colors.RESET}")
    print("=" * 60)

    # Geometry summary
    if plan.geometry['zones'] or plan.geometry['lines']:
        print(f"\n{Colors.CYAN}Geometry:{Colors.RESET}")
        if plan.geometry['zones']:
            print(f"  Zones: {', '.join(plan.geometry['zones'])}")
        if plan.geometry['lines']:
            print(f"  Lines: {', '.join(plan.geometry['lines'])}")

    # Track classes
    if plan.track_classes:
        print(f"\n{Colors.CYAN}Track Classes (derived from events):{Colors.RESET}")
        for class_id, class_name in plan.track_classes:
            print(f"  {Colors.GREEN}+{Colors.RESET} {class_name} (COCO ID: {class_id})")

    # Events
    print(f"\n{Colors.CYAN}Events:{Colors.RESET}")
    for event in plan.events:
        print(f"\n  {Colors.BOLD}{event.name}{Colors.RESET}")

        # Match criteria
        print(f"    {Colors.GRAY}Match:{Colors.RESET}")
        if event.match_criteria.get('event_type'):
            print(f"      event_type: {event.match_criteria['event_type']}")
        if event.match_criteria.get('zone'):
            print(f"      zone: \"{event.match_criteria['zone']}\"")
        if event.match_criteria.get('line'):
            print(f"      line: \"{event.match_criteria['line']}\"")
        if event.match_criteria.get('object_class'):
            obj = event.match_criteria['object_class']
            if isinstance(obj, list):
                print(f"      object_class: [{', '.join(obj)}]")
            else:
                print(f"      object_class: {obj}")

        # Actions
        print(f"    {Colors.GRAY}Actions:{Colors.RESET}")
        for consumer in event.consumers:
            print(f"      {Colors.GREEN}->{Colors.RESET} {consumer}")

        # Implied actions
        if event.implied_actions:
            print(f"    {Colors.GRAY}Implied (auto-enabled):{Colors.RESET}")
            for implied in event.implied_actions:
                print(f"      {Colors.YELLOW}+{Colors.RESET} {implied}")

    # Digest schedule
    if plan.digests:
        print(f"\n{Colors.CYAN}Digest Schedule:{Colors.RESET}")
        for digest_id, digest in plan.digests.items():
            period = digest.get('period_minutes', 0)
            label = digest.get('period_label', digest_id)
            photos = "with photos" if digest.get('photos') else "counts only"

            if period >= 1440:
                period_str = f"{period // 1440} day(s)"
            elif period >= 60:
                period_str = f"{period // 60} hour(s)"
            else:
                period_str = f"{period} minute(s)"

            print(f"  {Colors.BOLD}{digest_id}{Colors.RESET}: every {period_str} ({photos})")
            print(f"    Label: \"{label}\"")

    # Consumer summary
    print(f"\n{Colors.CYAN}Active Consumers:{Colors.RESET}")
    for consumer in plan.consumers:
        print(f"  {Colors.GREEN}+{Colors.RESET} {consumer}")

    print(f"\n{Colors.GREEN}No issues found. Ready to run.{Colors.RESET}")
    print()


def simulate_dry_run(config: dict, sample_events: List[Dict]) -> None:
    """Simulate event processing with sample events."""
    print()
    print(f"{Colors.BOLD}Dry Run Simulation{Colors.RESET}")
    print("=" * 60)

    plan = build_plan(config)

    # Build lookup tables
    digests = {d['id']: d for d in config.get('digests', []) if d.get('id')}

    print(f"\n{Colors.CYAN}Processing {len(sample_events)} sample event(s):{Colors.RESET}\n")

    matched_count = 0
    unmatched_count = 0
    actions_taken = {
        'json_log': 0,
        'email_immediate': 0,
        'email_digest': 0,
        'frame_capture': 0
    }
    digest_counts = {}

    for i, sample_event in enumerate(sample_events, 1):
        event_type = sample_event.get('event_type', 'UNKNOWN')
        obj_class = sample_event.get('object_class_name', sample_event.get('object_class', 'unknown'))
        zone = sample_event.get('zone_description', sample_event.get('zone'))
        line = sample_event.get('line_description', sample_event.get('line'))
        location = zone or line or 'unknown'

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
                if 'json_writer' in consumer:
                    actions_taken['json_log'] += 1
                    print(f"         {Colors.GRAY}-> Write to JSON log{Colors.RESET}")
                elif 'email_notifier' in consumer:
                    actions_taken['email_immediate'] += 1
                    print(f"         {Colors.GRAY}-> Send immediate email{Colors.RESET}")
                elif 'email_digest' in consumer:
                    actions_taken['email_digest'] += 1
                    digest_id = matched_event.digest_id
                    digest_counts[digest_id] = digest_counts.get(digest_id, 0) + 1
                    print(f"         {Colors.GRAY}-> Queue for digest: {digest_id}{Colors.RESET}")
                elif 'frame_capture' in consumer:
                    actions_taken['frame_capture'] += 1
                    print(f"         {Colors.GRAY}-> Capture frame{Colors.RESET}")
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
    print(f"  Immediate emails: {actions_taken['email_immediate']}")
    print(f"  Digest queue adds: {actions_taken['email_digest']}")
    print(f"  Frame captures: {actions_taken['frame_capture']}")

    if digest_counts:
        print(f"\n{Colors.CYAN}Digest contents (what would be sent):{Colors.RESET}")
        for digest_id, count in digest_counts.items():
            digest = digests.get(digest_id, {})
            photos = " + photos" if digest.get('photos') else ""
            print(f"  {digest_id}: {count} event(s){photos}")

    print()


def _matches_event(sample: Dict, event_plan: EventPlan) -> bool:
    """Check if sample event matches event plan criteria."""
    match = event_plan.match_criteria

    # Check event type
    if match.get('event_type'):
        if sample.get('event_type') != match['event_type']:
            return False

    # Check zone
    if match.get('zone'):
        sample_zone = sample.get('zone_description') or sample.get('zone')
        if sample_zone != match['zone']:
            return False

    # Check line
    if match.get('line'):
        sample_line = sample.get('line_description') or sample.get('line')
        if sample_line != match['line']:
            return False

    # Check object class
    if match.get('object_class'):
        sample_class = (sample.get('object_class_name') or
                        sample.get('object_class', '')).lower()
        match_classes = match['object_class']
        if isinstance(match_classes, str):
            match_classes = [match_classes]
        if sample_class not in [c.lower() for c in match_classes]:
            return False

    return True


def generate_sample_events(config: dict) -> List[Dict]:
    """Generate sample events based on config for dry-run testing."""
    samples = []

    zones = config.get('zones', [])
    lines = config.get('lines', [])
    events = config.get('events', [])

    # Generate samples based on event definitions
    for event in events:
        match = event.get('match', {})
        event_type = match.get('event_type', 'LINE_CROSS')

        obj_classes = match.get('object_class', [])
        if isinstance(obj_classes, str):
            obj_classes = [obj_classes]
        if not obj_classes:
            obj_classes = ['unknown']

        for obj_class in obj_classes[:2]:  # Limit to 2 per class
            sample = {
                'event_type': event_type,
                'object_class_name': obj_class,
                'track_id': len(samples) + 1
            }

            if match.get('zone'):
                sample['zone_description'] = match['zone']
            elif match.get('line'):
                sample['line_description'] = match['line']

            samples.append(sample)

    # Add some unmatched events for realism
    samples.append({
        'event_type': 'LINE_CROSS',
        'object_class_name': 'person',
        'line_description': 'unknown line',
        'track_id': 999
    })

    return samples


def load_sample_events(path: str) -> List[Dict]:
    """Load sample events from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both array and object with 'events' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'events' in data:
        return data['events']
    else:
        raise ValueError("Sample events file must contain an array or object with 'events' key")
