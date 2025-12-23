"""
Frame Capture Consumer
Captures and saves frames for events. No filtering - if it's on the queue, capture it.
Cooldown configuration comes from event metadata.
"""

import logging
import os
import time
from multiprocessing import Queue

import cv2

from .frame_service import FrameService

logger = logging.getLogger(__name__)


def frame_capture_consumer(event_queue: Queue, config: dict) -> None:
    """
    Capture frames for events. No filtering - events are pre-filtered by dispatcher.

    Args:
        event_queue: Queue receiving events that need frame capture
        config: Consumer configuration with frame settings
    """
    # Initialize frame service
    frame_service = FrameService(config)

    # Temp frame directory
    temp_frame_dir = config.get("temp_frame_dir", "/tmp/frames")

    # Lines/zones/ROI config for annotation
    lines_config = config.get("lines", [])
    zones_config = config.get("zones", [])
    roi_config = config.get("roi", {})

    # Track cooldowns per (track_id, zone/line)
    cooldowns: dict[tuple[int, str], float] = {}

    # Per-event photo budgets - each event definition can have its own max_photos
    # Budget state tracked separately per event_definition name
    start_time = time.time()
    default_duration = config.get("expected_duration_seconds", 3600)  # Default 1 hour
    photo_budgets: dict[str, dict] = {}  # event_def -> budget state

    logger.info("Frame Capture consumer started")
    logger.info(f"Temp frame dir: {temp_frame_dir}")
    logger.info(f"Storage: {frame_service.storage_type}")

    try:
        while True:
            event = event_queue.get()

            if event is None:
                logger.info("Frame Capture received shutdown signal")
                break

            # Extract frame config from event metadata
            frame_config = event.get("_frame_capture_config", {})
            cooldown_seconds = frame_config.get("cooldown_seconds", 180)

            # Build cooldown key
            track_id = event.get("track_id")
            zone = event.get("zone_description", "")
            line = event.get("line_description", "")
            location = zone or line
            cooldown_key = (track_id, location)

            # Check cooldown
            current_time = time.time()
            if cooldown_key in cooldowns:
                if current_time - cooldowns[cooldown_key] < cooldown_seconds:
                    logger.debug(
                        f"Skipping frame capture due to cooldown: {cooldown_key}"
                    )
                    continue

            # Check photo budget (per event definition)
            max_photos = frame_config.get("max_photos")
            if max_photos:
                event_def = event.get("event_definition", "default")

                # Initialize budget state for this event definition if needed
                if event_def not in photo_budgets:
                    expected_duration = frame_config.get(
                        "expected_duration_seconds", default_duration
                    )
                    photo_budgets[event_def] = {
                        "pending_slots": max_photos,
                        "in_time_mode": False,
                        "time_mode_start": None,
                        "time_mode_photos": 0,
                        "slot_interval": None,
                        "max_photos": max_photos,
                        "expected_duration": expected_duration,
                        "saved_frames": [],  # List of saved frame paths
                        "delete_pointer": None,  # Index for next deletion (starts at end, moves backward)
                    }
                    logger.info(
                        f"Photo budget for '{event_def}': {max_photos} over {expected_duration / 3600:.1f}h"
                    )

                budget = photo_budgets[event_def]

                # If in time mode, calculate earned slots
                if budget["in_time_mode"]:
                    elapsed = current_time - budget["time_mode_start"]
                    slots_earned = int(elapsed / budget["slot_interval"])
                    budget["pending_slots"] = max(
                        0, slots_earned - budget["time_mode_photos"]
                    )

                # Check if we have a slot available
                if budget["pending_slots"] <= 0:
                    continue  # No slot available, skip this event

                # We'll consume a slot after successful save (below)

            # Find temp frame by ID (direct lookup)
            frame_id = event.get("frame_id")
            temp_frame = None
            if frame_id:
                temp_frame_path = os.path.join(temp_frame_dir, f"{frame_id}.jpg")
                if os.path.exists(temp_frame_path):
                    temp_frame = temp_frame_path

            if temp_frame:
                # Annotate if requested
                should_annotate = frame_config.get("annotate", False)
                frame_to_save = temp_frame

                if should_annotate:
                    annotated = _annotate_frame(
                        temp_frame, event, lines_config, zones_config, roi_config
                    )
                    if annotated:
                        frame_to_save = annotated

                # Save frame permanently
                saved_path = frame_service.save_event_frame(event, frame_to_save)

                # Clean up temp annotated file if created
                if should_annotate and annotated and annotated != temp_frame:
                    try:
                        os.remove(annotated)
                    except Exception:
                        pass

                if saved_path:
                    logger.info(f"Captured frame for: {event.get('event_definition')}")
                    cooldowns[cooldown_key] = current_time

                    # Update photo budget if applicable
                    if max_photos and event.get("event_definition") in photo_budgets:
                        budget = photo_budgets[event.get("event_definition")]
                        budget["pending_slots"] -= 1

                        # In time mode, delete oldest frame before adding new one
                        if budget["in_time_mode"]:
                            budget["time_mode_photos"] += 1

                            # Delete frame at delete_pointer
                            if (
                                budget["saved_frames"]
                                and budget["delete_pointer"] is not None
                            ):
                                to_delete = budget["saved_frames"][
                                    budget["delete_pointer"]
                                ]
                                try:
                                    if os.path.exists(to_delete):
                                        os.remove(to_delete)
                                        logger.debug(
                                            f"Deleted old frame: {os.path.basename(to_delete)}"
                                        )
                                except Exception as e:
                                    logger.warning(f"Failed to delete frame: {e}")

                                # Remove from list
                                budget["saved_frames"].pop(budget["delete_pointer"])

                                # Move pointer backward, wrap if needed
                                budget["delete_pointer"] -= 1
                                if budget["delete_pointer"] < 0:
                                    budget["delete_pointer"] = (
                                        len(budget["saved_frames"]) - 1
                                    )

                        # Track saved frame
                        budget["saved_frames"].append(saved_path)

                        # Check if eager phase just ended - switch to time mode
                        if not budget["in_time_mode"] and budget["pending_slots"] == 0:
                            budget["in_time_mode"] = True
                            budget["time_mode_start"] = current_time
                            budget["delete_pointer"] = (
                                len(budget["saved_frames"]) - 1
                            )  # Start at newest
                            elapsed = current_time - start_time
                            remaining = budget["expected_duration"] - elapsed
                            if remaining > 0:
                                budget["slot_interval"] = (
                                    remaining / budget["max_photos"]
                                )
                                logger.info(
                                    f"'{event.get('event_definition')}' switching to time mode: "
                                    f"1 photo per {budget['slot_interval']:.0f}s (delete from back)"
                                )
                            else:
                                # Past expected duration, use small interval
                                budget["slot_interval"] = 60
                            budget["time_mode_photos"] = 0
                else:
                    logger.warning("Failed to save frame")
            else:
                if not frame_id:
                    logger.debug("Event has no frame_id - no temp frame available")
                else:
                    logger.warning(f"Temp frame not found: {frame_id}")

    except KeyboardInterrupt:
        logger.info("Frame Capture stopped by user")
    except Exception as e:
        logger.error(f"Error in Frame Capture: {e}", exc_info=True)
    finally:
        # Log photo budget stats
        for event_def, budget in photo_budgets.items():
            on_disk = len(budget["saved_frames"])
            logger.info(
                f"Photo budget '{event_def}': {on_disk} frames on disk (max {budget['max_photos']})"
            )
        logger.info("Frame Capture shutdown complete")


def _annotate_frame(
    frame_path: str,
    event: dict,
    lines_config: list[dict],
    zones_config: list[dict],
    roi_config: dict,
) -> str | None:
    """
    Annotate frame with bounding box, lines, and zones.

    Args:
        frame_path: Path to source frame
        event: Event data containing bbox and object info
        lines_config: List of line configurations
        zones_config: List of zone configurations
        roi_config: ROI configuration for coordinate mapping

    Returns:
        Path to annotated frame, or None on error
    """
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return None

        height, width = frame.shape[:2]

        # Calculate ROI offset for coordinate mapping
        roi_h = roi_config.get("horizontal", {})
        roi_v = roi_config.get("vertical", {})
        h_from = (
            roi_h.get("crop_from_left_pct", 0) if roi_h.get("enabled", False) else 0
        )
        v_from = roi_v.get("crop_from_top_pct", 0) if roi_v.get("enabled", False) else 0

        # Draw all configured lines (yellow)
        for line in lines_config:
            line_type = line.get("type", "vertical")
            position_pct = line.get("position_pct", 50)
            description = line.get("description", "")

            if line_type == "vertical":
                x = int(width * position_pct / 100)
                cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    description,
                    (x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            else:  # horizontal
                y = int(height * position_pct / 100)
                cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    description,
                    (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        # Draw all configured zones (cyan rectangles)
        for zone in zones_config:
            x1 = int(width * zone.get("x1_pct", 0) / 100)
            y1 = int(height * zone.get("y1_pct", 0) / 100)
            x2 = int(width * zone.get("x2_pct", 100) / 100)
            y2 = int(height * zone.get("y2_pct", 100) / 100)
            description = zone.get("description", "")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                description,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # Draw triggering object's bounding box (green with label)
        bbox = event.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox

            # Adjust bbox coordinates if ROI was applied
            # The bbox is in ROI-relative coords, but the saved frame is full-size
            if h_from > 0 or v_from > 0:
                offset_x = int(width * h_from / 100)
                offset_y = int(height * v_from / 100)
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add label
            obj_name = event.get("object_class_name", "object")
            track_id = event.get("track_id", "")
            label = f"{obj_name} #{track_id}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Save annotated frame to temp location
        annotated_path = frame_path.replace(".jpg", "_annotated.jpg")
        cv2.imwrite(annotated_path, frame)

        return annotated_path

    except Exception as e:
        logger.error(f"Error annotating frame: {e}", exc_info=True)
        return None
