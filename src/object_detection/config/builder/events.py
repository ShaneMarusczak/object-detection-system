"""
Event setup for the config builder.

Handles the complex event configuration workflow including:
- Event type selection (LINE_CROSS, ZONE_ENTER, etc.)
- Object class selection with validation
- Action configuration (command, report, VLM, notify)
"""

import questionary
from questionary import Choice

from ...processor.coco_classes import COCO_CLASSES
from .prompts import COMMON_CLASSES, PROMPT_STYLE, Colors


def setup_events(
    config: dict,
    model_classes: list[str],
) -> None:
    """
    Setup event definitions interactively.

    Args:
        config: Configuration dict to modify in-place
        model_classes: List of class names from loaded model
    """
    print(f"\n{Colors.BOLD}--- Events Setup ---{Colors.RESET}")
    print(
        f"{Colors.GRAY}Events define what to detect and what actions to take{Colors.RESET}"
    )

    events = list(config.get("events", []))
    lines = config.get("lines", [])
    zones = config.get("zones", [])

    while True:
        add = questionary.confirm(
            "Add an event?",
            default=True,
            style=PROMPT_STYLE,
        ).ask()

        if not add:
            break

        event = _build_event(config, lines, zones, events, model_classes)
        if event:
            events.append(event)
            print(f"  {Colors.GREEN}Event '{event['name']}' added{Colors.RESET}")

            # Inline validation warning
            if not event.get("actions"):
                print(
                    f"  {Colors.YELLOW}Warning: No actions defined - event won't do anything{Colors.RESET}"
                )

    if events:
        config["events"] = events


def _build_event(
    config: dict,
    lines: list[dict],
    zones: list[dict],
    existing_events: list[dict],
    model_classes: list[str],
) -> dict | None:
    """Build a single event definition."""
    event = {}

    # Event name
    default_name = f"event_{len(existing_events) + 1}"
    name = (
        questionary.text(
            "Event name:",
            default=default_name,
            style=PROMPT_STYLE,
        ).ask()
        or default_name
    )
    event["name"] = name

    # Match criteria
    print("  Match criteria:")
    match = _build_match_criteria(config, lines, zones, model_classes)
    if match is None:
        return None
    event["match"] = match

    # Actions
    actions = _build_actions(config, model_classes)
    event["actions"] = actions

    return event


def _build_match_criteria(
    config: dict,
    lines: list[dict],
    zones: list[dict],
    model_classes: list[str],
) -> dict | None:
    """Build event match criteria."""
    match = {}

    # Build dynamic event type menu based on available lines/zones
    event_options = []
    if lines:
        event_options.append(
            Choice(title="LINE_CROSS (object crosses a line)", value="LINE_CROSS")
        )
    if zones:
        event_options.append(
            Choice(title="ZONE_ENTER (object enters a zone)", value="ZONE_ENTER")
        )
        event_options.append(
            Choice(title="ZONE_EXIT (object exits a zone)", value="ZONE_EXIT")
        )
        event_options.append(
            Choice(
                title="NIGHTTIME_CAR (headlight blob detection in zone)",
                value="NIGHTTIME_CAR",
            )
        )
    event_options.append(
        Choice(
            title="DETECTED (any detection, no geometry required)",
            value="DETECTED",
        )
    )

    selected_type = questionary.select(
        "Event type:",
        choices=event_options,
        style=PROMPT_STYLE,
    ).ask()

    if selected_type is None:
        return None

    match["event_type"] = selected_type

    # Type-specific configuration
    if selected_type == "LINE_CROSS":
        line_choices = [
            Choice(title=ln["description"], value=ln["description"]) for ln in lines
        ]
        selected_line = questionary.select(
            "Which line?",
            choices=line_choices,
            style=PROMPT_STYLE,
        ).ask()
        match["line"] = selected_line
        _add_object_classes(match, model_classes, required=True)

    elif selected_type in ("ZONE_ENTER", "ZONE_EXIT"):
        zone_choices = [
            Choice(title=z["description"], value=z["description"]) for z in zones
        ]
        selected_zone = questionary.select(
            "Which zone?",
            choices=zone_choices,
            style=PROMPT_STYLE,
        ).ask()
        match["zone"] = selected_zone
        _add_object_classes(match, model_classes, required=True)

    elif selected_type == "DETECTED":
        print(
            f"    {Colors.GRAY}DETECTED fires for every detection - no tracking needed{Colors.RESET}"
        )
        _add_object_classes(match, model_classes, required=False)

    elif selected_type == "NIGHTTIME_CAR":
        zone_choices = [
            Choice(title=z["description"], value=z["description"]) for z in zones
        ]
        selected_zone = questionary.select(
            "Which zone to monitor for headlights?",
            choices=zone_choices,
            style=PROMPT_STYLE,
        ).ask()
        match["zone"] = selected_zone
        _add_nighttime_params(match)

    return match


def _add_object_classes(
    match: dict,
    model_classes: list[str],
    required: bool,
) -> None:
    """Add object class filter to match criteria."""
    # Use model classes if loaded, otherwise fall back to COCO
    if model_classes:
        valid_classes = set(model_classes)
        display_classes = model_classes
        default_classes = (
            model_classes[0] if len(model_classes) == 1 else "car, truck, bus"
        )
    else:
        valid_classes = set(COCO_CLASSES.values())
        display_classes = COMMON_CLASSES
        default_classes = "car, truck, bus"

    if required:
        while True:
            print(
                f"    {Colors.CYAN}Available:{Colors.RESET} {', '.join(display_classes)}"
            )
            classes_str = (
                questionary.text(
                    "Object classes (comma-separated):",
                    default=default_classes,
                    style=PROMPT_STYLE,
                ).ask()
                or default_classes
            )
            classes = [c.strip() for c in classes_str.split(",") if c.strip()]

            # Validate all classes
            invalid = [c for c in classes if c not in valid_classes]
            if invalid:
                print(
                    f"    {Colors.RED}Invalid class(es): {', '.join(invalid)}{Colors.RESET}"
                )
                print(
                    f"    {Colors.GRAY}Valid classes: {', '.join(sorted(valid_classes))}{Colors.RESET}"
                )
                continue
            break

        match["object_class"] = classes if len(classes) > 1 else classes[0]

    else:
        # Optional - for DETECTED events
        if model_classes and len(model_classes) == 1:
            single_class = model_classes[0]
            match["object_class"] = single_class
            print(
                f"    {Colors.CYAN}Single-class model: using '{single_class}'{Colors.RESET}"
            )
        else:
            filter_class = questionary.confirm(
                "Filter by object class?",
                default=False,
                style=PROMPT_STYLE,
            ).ask()

            if filter_class:
                print(
                    f"    {Colors.CYAN}Available:{Colors.RESET} {', '.join(display_classes)}"
                )
                class_str = questionary.text(
                    "Class:",
                    default=model_classes[0] if model_classes else "car",
                    style=PROMPT_STYLE,
                ).ask()

                if class_str and class_str in valid_classes:
                    match["object_class"] = class_str
                else:
                    print(
                        f"    {Colors.YELLOW}Unknown class, skipping filter{Colors.RESET}"
                    )


def _add_nighttime_params(match: dict) -> None:
    """Add nighttime detection parameters."""
    print("    Nighttime detection settings (press Enter for defaults):")

    brightness = (
        questionary.text(
            "Brightness threshold:",
            default="30",
            style=PROMPT_STYLE,
        ).ask()
        or "30"
    )

    min_blob = (
        questionary.text(
            "Min blob size:",
            default="100",
            style=PROMPT_STYLE,
        ).ask()
        or "100"
    )

    max_blob = (
        questionary.text(
            "Max blob size:",
            default="10000",
            style=PROMPT_STYLE,
        ).ask()
        or "10000"
    )

    score = (
        questionary.text(
            "Score threshold:",
            default="85",
            style=PROMPT_STYLE,
        ).ask()
        or "85"
    )

    taillight = questionary.confirm(
        "Require taillight match?",
        default=True,
        style=PROMPT_STYLE,
    ).ask()

    match["nighttime_detection"] = {
        "brightness_threshold": int(brightness),
        "min_blob_size": int(min_blob),
        "max_blob_size": int(max_blob),
        "score_threshold": int(score),
        "taillight_color_match": taillight,
    }


def _build_actions(config: dict, model_classes: list[str]) -> dict:
    """Build event actions configuration."""
    action_choices = [
        Choice(title="Run command/script", value="1"),
        Choice(title="Report with photos", value="2"),
        Choice(title="Report stats only", value="3"),
        Choice(title="JSON log only", value="4"),
        Choice(title="VLM analyze + notify", value="5"),
        Choice(title="Direct notify (no VLM)", value="6"),
    ]

    selected_actions = questionary.checkbox(
        "Select actions:",
        choices=action_choices,
        style=PROMPT_STYLE,
    ).ask() or ["1"]

    choices = set(selected_actions)
    actions = {}

    want_command = "1" in choices
    want_report_photos = "2" in choices
    want_report_only = "3" in choices
    want_json_only = "4" in choices
    want_vlm_analyze = "5" in choices
    want_direct_notify = "6" in choices

    # Build enabled list for display
    enabled = []
    if want_command:
        enabled.append("command")
    if want_report_photos:
        enabled.extend(["report", "frame_capture", "json_log"])
    if want_report_only:
        if "report" not in enabled:
            enabled.append("report")
        if "json_log" not in enabled:
            enabled.append("json_log")
    if want_json_only:
        if "json_log" not in enabled:
            enabled.append("json_log")
    if want_vlm_analyze:
        enabled.append("vlm_analyze")
        if "json_log" not in enabled:
            enabled.append("json_log")
    if want_direct_notify:
        enabled.append("notify")
        if "json_log" not in enabled:
            enabled.append("json_log")

    if enabled:
        print(f"    {Colors.GRAY}→ Enables: {', '.join(enabled)}{Colors.RESET}")

    # Configure each action type
    if want_command:
        _configure_command_action(actions)

    if want_report_photos or want_report_only:
        _configure_report_action(actions, want_report_photos)

    if want_report_photos or want_report_only or want_json_only:
        actions["json_log"] = True

    if want_vlm_analyze:
        _configure_vlm_action(config, actions)

    if want_direct_notify:
        _configure_notify_action(config, actions)

    return actions


def _configure_command_action(actions: dict) -> None:
    """Configure command action."""
    exec_path = questionary.text(
        "Command/script path:",
        style=PROMPT_STYLE,
    ).ask()

    if exec_path:
        timeout = (
            questionary.text(
                "Timeout seconds:",
                default="30",
                style=PROMPT_STYLE,
            ).ask()
            or "30"
        )

        actions["command"] = {
            "exec": exec_path,
            "timeout_seconds": int(timeout),
        }

        shutdown = questionary.confirm(
            "Stop detector after this event?",
            default=False,
            style=PROMPT_STYLE,
        ).ask()

        if shutdown:
            actions["shutdown"] = True
            print(
                f"    {Colors.RED}SHUTDOWN enabled - detector will stop after this event{Colors.RESET}"
            )


def _configure_report_action(actions: dict, with_photos: bool) -> None:
    """Configure report action."""
    report_id = (
        questionary.text(
            "Report ID:",
            default="traffic_report",
            style=PROMPT_STYLE,
        ).ask()
        or "traffic_report"
    )

    actions["report"] = report_id

    if with_photos:
        max_photos_str = (
            questionary.text(
                "Max photos:",
                default="100",
                style=PROMPT_STYLE,
            ).ask()
            or "100"
        )
        actions["frame_capture"] = {"max_photos": int(max_photos_str)}


def _configure_vlm_action(config: dict, actions: dict) -> None:
    """Configure VLM analyze action."""
    print(f"\n    {Colors.BOLD}VLM Analyze Setup:{Colors.RESET}")
    print(f"    {Colors.GRAY}Sends frame to remote VLM for analysis{Colors.RESET}")

    analyzer_id = (
        questionary.text(
            "Analyzer ID:",
            default="orin2",
            style=PROMPT_STYLE,
        ).ask()
        or "orin2"
    )

    analyzer_url = (
        questionary.text(
            "Analyzer URL:",
            default="http://orin-nvme:8080/analyze",
            style=PROMPT_STYLE,
        ).ask()
        or "http://orin-nvme:8080/analyze"
    )

    # Ensure analyzer exists in config
    if "analyzers" not in config:
        config["analyzers"] = []

    analyzer_exists = any(a.get("id") == analyzer_id for a in config["analyzers"])
    if not analyzer_exists:
        config["analyzers"].append(
            {"id": analyzer_id, "url": analyzer_url, "timeout_seconds": 60}
        )
        print(f"      {Colors.GREEN}Added analyzer: {analyzer_id}{Colors.RESET}")

    default_prompt = "Describe what you see. Be concise."
    prompt = (
        questionary.text(
            "Prompt:",
            default=default_prompt,
            style=PROMPT_STYLE,
        ).ask()
        or default_prompt
    )

    # Notifier setup
    notifier_ids = []
    add_notifier = questionary.confirm(
        "Add notification?",
        default=True,
        style=PROMPT_STYLE,
    ).ask()

    if add_notifier:
        notifier_id = _add_notifier_to_config(config)
        if notifier_id:
            notifier_ids.append(notifier_id)

    actions["vlm_analyze"] = {
        "analyzer": analyzer_id,
        "prompt": prompt,
        "notify": notifier_ids,
    }


def _configure_notify_action(config: dict, actions: dict) -> None:
    """Configure direct notify action."""
    print(f"\n    {Colors.BOLD}Direct Notify Setup:{Colors.RESET}")
    print(f"    {Colors.GRAY}Send notifications without VLM analysis{Colors.RESET}")

    notify_items = []
    while True:
        add_more = (
            True
            if not notify_items
            else questionary.confirm(
                "Add another notifier?",
                default=False,
                style=PROMPT_STYLE,
            ).ask()
        )

        if not add_more:
            break

        notifier_id = _add_notifier_to_config(config)
        if notifier_id:
            # Message template
            default_msg = "{object_class} detected in {zone} ({confidence_pct})"
            message = (
                questionary.text(
                    "Message template:",
                    default=default_msg,
                    style=PROMPT_STYLE,
                ).ask()
                or default_msg
            )

            include_image = questionary.confirm(
                "Include image?",
                default=False,
                style=PROMPT_STYLE,
            ).ask()

            notify_items.append(
                {
                    "notifier": notifier_id,
                    "message": message,
                    "include_image": include_image,
                }
            )
            print(f"      {Colors.GREEN}Added notify action{Colors.RESET}")

    if notify_items:
        actions["notify"] = notify_items


def _add_notifier_to_config(config: dict) -> str | None:
    """Add a notifier to config and return its ID."""
    notifier_type = questionary.select(
        "Notifier type:",
        choices=[
            Choice(title="ntfy (push notification)", value="ntfy"),
            Choice(title="webhook (HTTP POST)", value="webhook"),
        ],
        style=PROMPT_STYLE,
    ).ask()

    if notifier_type == "ntfy":
        topic = (
            questionary.text(
                "ntfy topic:",
                default="alerts",
                style=PROMPT_STYLE,
            ).ask()
            or "alerts"
        )

        notifier_id = f"ntfy_{topic}"
        if "notifiers" not in config:
            config["notifiers"] = []

        notifier_exists = any(n.get("id") == notifier_id for n in config["notifiers"])
        if not notifier_exists:
            config["notifiers"].append(
                {"id": notifier_id, "type": "ntfy", "topic": topic}
            )
            print(f"      {Colors.GREEN}Added notifier: {notifier_id}{Colors.RESET}")

        return notifier_id

    elif notifier_type == "webhook":
        webhook_url = questionary.text(
            "Webhook URL:",
            style=PROMPT_STYLE,
        ).ask()

        if webhook_url:
            if "notifiers" not in config:
                config["notifiers"] = []

            # Generate unique webhook ID
            existing_webhooks = [
                n for n in config["notifiers"] if n.get("type") == "webhook"
            ]
            notifier_id = f"webhook_{len(existing_webhooks) + 1}"

            config["notifiers"].append(
                {"id": notifier_id, "type": "webhook", "url": webhook_url}
            )
            print(f"      {Colors.GREEN}Added notifier: {notifier_id}{Colors.RESET}")

            return notifier_id

    return None
