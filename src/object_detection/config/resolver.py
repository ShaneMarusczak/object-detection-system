"""
Configuration Resolver - Prepares config for runtime.

Handles:
- Deriving track_classes from events
- Resolving implied actions (cascading rules)
- Normalizing action configurations
"""

import logging

logger = logging.getLogger(__name__)


def derive_track_classes(
    config: dict, model_names: dict[int, str] | None = None
) -> list[int]:
    """
    Derive class IDs from event definitions using the loaded model.

    This is the public API for getting track_classes from events.
    Returns just the IDs (not name pairs) for use by detector.

    Args:
        config: Configuration dictionary
        model_names: Mapping of class ID -> class name from loaded model

    Returns:
        List of class IDs to track
    """
    class_names = set()

    for event in config.get("events", []):
        match = event.get("match", {})
        # Skip NIGHTTIME_CAR events - they don't need YOLO classes
        if match.get("event_type") == "NIGHTTIME_CAR":
            continue
        obj_class = match.get("object_class")
        if obj_class:
            if isinstance(obj_class, list):
                class_names.update(c.lower() for c in obj_class)
            else:
                class_names.add(obj_class.lower())

    # Build reverse mapping (name -> id) from model
    name_to_id: dict[str, int] = {}
    if model_names is not None:
        name_to_id = {name.lower(): id for id, name in model_names.items()}

    # Convert to IDs
    class_ids = []
    for name in class_names:
        if name in name_to_id:
            class_ids.append(name_to_id[name])
        else:
            logger.warning(f"Unknown class '{name}' in events (not in loaded model)")

    return sorted(class_ids)


def prepare_runtime_config(
    config: dict, model_names: dict[int, str] | None = None
) -> dict:
    """
    Prepare config for runtime - resolve all implied actions statically.

    This is the single place where all inference happens:
    - Derive track_classes from events
    - Resolve implied actions (json_log, frame_capture, annotate cascade)
    - Determine which consumers are needed

    After this function, the config is fully resolved and the dispatcher
    just routes events - no inference at runtime.

    Args:
        config: Validated configuration
        model_names: Mapping of class ID -> class name from loaded model

    Returns:
        Config with all implied actions resolved
    """
    # Derive track_classes from events (the only way to specify them)
    derived_classes = derive_track_classes(config, model_names)

    if derived_classes:
        config["detection"]["track_classes"] = derived_classes
    else:
        # Check if there are events that need YOLO but don't specify classes
        # (e.g., DETECTED events that want all detections)
        has_yolo_events = any(
            e.get("match", {}).get("event_type") in ("DETECTED", "LINE_CROSS", "ZONE_ENTER", "ZONE_EXIT")
            for e in config.get("events", [])
        )
        has_nighttime_events = any(
            e.get("match", {}).get("event_type") == "NIGHTTIME_CAR"
            for e in config.get("events", [])
        )

        if has_yolo_events:
            # Events exist but no specific classes - detect all classes (None = no filter)
            config["detection"]["track_classes"] = None
        elif has_nighttime_events:
            # Only nighttime events - no YOLO classes needed
            config["detection"]["track_classes"] = []
        else:
            logger.warning("No events defined - nothing will be tracked!")
            config["detection"]["track_classes"] = []

    # Resolve all implied actions (e.g., report → json_log)
    _resolve_implied_actions(config)

    # Always enable temp_frames - consumers are always available
    config["temp_frames_enabled"] = True

    return config


def _resolve_implied_actions(config: dict) -> None:
    """
    Resolve all implied actions and modify event configs in place.

    This applies the cascading rules:
    - json_log defaults to True (opt-out, not opt-in)
    - report with photos=true → frame_capture enabled
    - report with annotate=true → frame_capture.annotate=true
    - vlm_analyze → json_log=true, temp_frames required
    """
    events = config.get("events", [])
    reports = {r["id"]: r for r in config.get("reports", []) if r.get("id")}
    analyzers = {a["id"]: a for a in config.get("analyzers", []) if a.get("id")}

    for event in events:
        actions = event.get("actions", {})

        # --- Handle report implied actions ---
        report_id = actions.get("report")
        if report_id:
            # report always requires json_log
            if not actions.get("json_log"):
                actions["json_log"] = True
                logger.debug(
                    f"Auto-enabled json_log for '{event.get('name')}' (required by report)"
                )

            # Check if report wants photos
            report = reports.get(report_id, {})
            if report.get("photos"):
                if not actions.get("frame_capture"):
                    # Auto-enable frame_capture with annotate from report
                    frame_config = report.get("frame_config", {})
                    actions["frame_capture"] = {
                        "enabled": True,
                        "annotate": report.get("annotate", False),
                        **frame_config,
                    }
                    logger.debug(
                        f"Auto-enabled frame_capture for '{event.get('name')}' (required by report photos)"
                    )
                elif report.get("annotate") and isinstance(
                    actions["frame_capture"], dict
                ):
                    # Merge annotate flag into existing frame_capture
                    actions["frame_capture"]["annotate"] = True
                    logger.debug(
                        f"Auto-enabled annotate for '{event.get('name')}' (from report)"
                    )

        # --- Normalize frame_capture config ---
        frame_capture = actions.get("frame_capture")
        if frame_capture:
            if isinstance(frame_capture, bool):
                actions["frame_capture"] = {"enabled": frame_capture}
            elif isinstance(frame_capture, dict) and "enabled" not in frame_capture:
                actions["frame_capture"]["enabled"] = True

        # --- Normalize command config ---
        command = actions.get("command")
        if command:
            if isinstance(command, str):
                actions["command"] = {"exec": command, "timeout_seconds": 30}

        # --- Normalize vlm_analyze config ---
        vlm_analyze = actions.get("vlm_analyze")
        if vlm_analyze:
            # Resolve analyzer timeout from config
            if isinstance(vlm_analyze, dict):
                analyzer_id = vlm_analyze.get("analyzer")
                if analyzer_id and analyzer_id in analyzers:
                    analyzer_config = analyzers[analyzer_id]
                    if "timeout_seconds" not in vlm_analyze:
                        vlm_analyze["_analyzer_timeout"] = analyzer_config.get(
                            "timeout_seconds", 60
                        )
                    vlm_analyze["_analyzer_url"] = analyzer_config.get("url")
                # Ensure notify is a list
                if "notify" not in vlm_analyze:
                    vlm_analyze["notify"] = []

        # --- JSON logging is opt-out (default True) ---
        # Only set to True if not explicitly set to False
        if actions.get("json_log") is None:
            actions["json_log"] = True
            logger.debug(
                f"Auto-enabled json_log for '{event.get('name')}' (default opt-out)"
            )
