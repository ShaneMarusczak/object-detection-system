"""
Configuration Resolver - Prepares config for runtime.

Handles:
- Deriving track_classes from events
- Resolving implied actions (cascading rules)
- Normalizing action configurations
"""

import logging

from ..processor.coco_classes import COCO_NAME_TO_ID

logger = logging.getLogger(__name__)


def derive_track_classes(config: dict) -> list[int]:
    """
    Derive COCO class IDs from event definitions.

    This is the public API for getting track_classes from events.
    Returns just the IDs (not name pairs) for use by detector.
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

    # Convert to IDs
    class_ids = []
    for name in class_names:
        if name in COCO_NAME_TO_ID:
            class_ids.append(COCO_NAME_TO_ID[name])
        else:
            logger.warning(f"Unknown class '{name}' in events (not in COCO)")

    return sorted(class_ids)


def prepare_runtime_config(config: dict) -> dict:
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

    Returns:
        Config with all implied actions resolved
    """
    # Derive track_classes from events (the only way to specify them)
    derived_classes = derive_track_classes(config)

    if derived_classes:
        config["detection"]["track_classes"] = derived_classes
    else:
        # Check if there are NIGHTTIME_CAR events (don't need YOLO classes)
        has_nighttime_events = any(
            e.get("match", {}).get("event_type") == "NIGHTTIME_CAR"
            for e in config.get("events", [])
        )
        if not has_nighttime_events:
            logger.warning("No events defined - nothing will be tracked!")
        config["detection"]["track_classes"] = []

    # Resolve all implied actions (e.g., pdf_report → json_log)
    _resolve_implied_actions(config)

    # Always enable temp_frames - consumers are always available
    config["temp_frames_enabled"] = True

    return config


def _resolve_implied_actions(config: dict) -> None:
    """
    Resolve all implied actions and modify event configs in place.

    This applies the cascading rules:
    - pdf_report with photos=true → frame_capture enabled
    - pdf_report with annotate=true → frame_capture.annotate=true
    - email_digest with photos=true → frame_capture enabled
    - pdf_report/email_digest → json_log=true
    """
    events = config.get("events", [])
    digests = {d["id"]: d for d in config.get("digests", []) if d.get("id")}
    pdf_reports = {r["id"]: r for r in config.get("pdf_reports", []) if r.get("id")}

    for event in events:
        actions = event.get("actions", {})

        # --- Handle email_digest implied actions ---
        digest_id = actions.get("email_digest")
        if digest_id:
            # email_digest always requires json_log
            if not actions.get("json_log"):
                actions["json_log"] = True
                logger.debug(
                    f"Auto-enabled json_log for '{event.get('name')}' (required by email_digest)"
                )

            # Check if digest wants photos
            digest = digests.get(digest_id, {})
            if digest.get("photos"):
                if not actions.get("frame_capture"):
                    frame_config = digest.get("frame_config", {})
                    actions["frame_capture"] = {"enabled": True, **frame_config}
                    logger.debug(
                        f"Auto-enabled frame_capture for '{event.get('name')}' (required by digest photos)"
                    )

        # --- Handle pdf_report implied actions ---
        pdf_report_id = actions.get("pdf_report")
        if pdf_report_id:
            # pdf_report always requires json_log
            if not actions.get("json_log"):
                actions["json_log"] = True
                logger.debug(
                    f"Auto-enabled json_log for '{event.get('name')}' (required by pdf_report)"
                )

            # Check if report wants photos
            report = pdf_reports.get(pdf_report_id, {})
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
                        f"Auto-enabled annotate for '{event.get('name')}' (from pdf_report)"
                    )

        # --- Normalize frame_capture config ---
        frame_capture = actions.get("frame_capture")
        if frame_capture:
            if isinstance(frame_capture, bool):
                actions["frame_capture"] = {"enabled": frame_capture}
            elif isinstance(frame_capture, dict) and "enabled" not in frame_capture:
                actions["frame_capture"]["enabled"] = True
