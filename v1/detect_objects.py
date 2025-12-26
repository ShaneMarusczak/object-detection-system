"""
Object Detection - Producer
Detects objects crossing counting lines and entering/exiting zones.
GPU-accelerated YOLO detection with ByteTrack tracking.
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import torch
from ultralytics import YOLO
import time
from datetime import datetime


def run_detection(data_queue, config):
    """
    Main detection loop. Tracks objects and emits boundary crossing events.

    Args:
        data_queue: multiprocessing.Queue for sending events
        config: Configuration dictionary from config.yaml
    """

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config["detection"]["model_file"])
    model.to(device)

    # Verify GPU usage
    print("Detection initialized")
    print(f"  Device: {device}")
    print(f"  Model: {config['detection']['model_file']}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Model on CUDA: {next(model.model.parameters()).is_cuda}")
    else:
        print("  WARNING: Running on CPU - will be slow")

    # Camera setup
    cap = cv2.VideoCapture(config["camera"]["url"])
    if not cap.isOpened():
        print(f"ERROR: Cannot connect to {config['camera']['url']}")
        data_queue.put(None)
        return

    print(f"  Camera: {config['camera']['url']}")
    print(f"  Confidence: {config['detection']['confidence_threshold']}")
    print(f"  Tracking classes: {config['detection']['track_classes']}")

    # Parse configuration
    lines = _parse_lines(config)
    zones = _parse_zones(config)
    roi_config = _parse_roi(config)
    speed_enabled = config.get("speed_calculation", {}).get("enabled", False)
    frame_saving = config.get("frame_saving", {}).get("enabled", False)

    print(f"  Lines: {len(lines)} configured")
    print(f"  Zones: {len(zones)} configured")

    if roi_config["enabled"]:
        print(
            f"  ROI: Enabled (H: {roi_config['h_from']}-{roi_config['h_to']}%, "
            f"V: {roi_config['v_from']}-{roi_config['v_to']}%)"
        )

    if speed_enabled:
        print("  Speed calculation: Enabled")

    if frame_saving:
        save_interval = config["frame_saving"]["interval"]
        save_dir = config["frame_saving"]["output_dir"]
        os.makedirs(save_dir, exist_ok=True)
        print(f"  Frame saving: Every {save_interval} frames -> {save_dir}/")

    print("\nDetection started\n")

    # State tracking
    previous_positions = {}  # track_id -> (x, y)
    first_seen_positions = {}  # track_id -> (x, y, time) - for speed only
    first_seen_times = {}  # track_id -> time - for speed only
    crossed_lines = {}  # track_id -> set of line_ids already crossed
    active_zones = {}  # track_id -> {zone_id: entry_time}

    # Performance tracking
    fps_list = []
    frame_count = 0
    event_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                break

            frame_count += 1
            frame_height, frame_width = frame.shape[:2]

            # Apply ROI cropping if configured
            if roi_config["enabled"]:
                x1 = int(frame_width * roi_config["h_from"] / 100)
                x2 = int(frame_width * roi_config["h_to"] / 100)
                y1 = int(frame_height * roi_config["v_from"] / 100)
                y2 = int(frame_height * roi_config["v_to"] / 100)

                roi_frame = frame[y1:y2, x1:x2]
                roi_width = x2 - x1
                roi_height = y2 - y1
            else:
                roi_frame = frame
                roi_width = frame_width
                roi_height = frame_height

            # YOLO detection + ByteTrack tracking
            results = model.track(
                source=roi_frame,
                tracker="bytetrack.yaml",
                conf=config["detection"]["confidence_threshold"],
                classes=config["detection"]["track_classes"],
                device=device,
                persist=True,
                verbose=False,
            )

            # FPS tracking
            inference_time = results[0].speed["inference"]
            current_fps = 1000 / inference_time if inference_time > 0 else 0
            fps_list.append(current_fps)

            current_time = time.time()
            relative_time = current_time - start_time

            # Process detections
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                track_ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.int().cpu().tolist()

                for track_id, box, obj_class in zip(track_ids, xyxy, classes):
                    # Calculate center point in ROI coordinates
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # First detection of this object
                    if track_id not in previous_positions:
                        if speed_enabled:
                            first_seen_positions[track_id] = (center_x, center_y)
                            first_seen_times[track_id] = current_time
                        crossed_lines[track_id] = set()
                        active_zones[track_id] = {}

                    # Check line crossings
                    if track_id in previous_positions:
                        prev_x, prev_y = previous_positions[track_id]

                        for line in lines:
                            # Check if this class is allowed for this line
                            if obj_class not in line["allowed_classes"]:
                                continue

                            # Check if already crossed (prevent duplicate events)
                            line_key = f"{line['line_id']}"
                            if line_key in crossed_lines[track_id]:
                                continue

                            crossed = False
                            direction = None

                            if line["type"] == "vertical":
                                line_pos = roi_width * line["position_pct"] / 100

                                if prev_x < line_pos <= center_x:
                                    crossed = True
                                    direction = "LTR"
                                elif prev_x > line_pos >= center_x:
                                    crossed = True
                                    direction = "RTL"

                            else:  # horizontal
                                line_pos = roi_height * line["position_pct"] / 100

                                if prev_y < line_pos <= center_y:
                                    crossed = True
                                    direction = "TTB"
                                elif prev_y > line_pos >= center_y:
                                    crossed = True
                                    direction = "BTT"

                            if crossed:
                                crossed_lines[track_id].add(line_key)
                                event_count += 1

                                event = {
                                    "event_type": "LINE_CROSS",
                                    "track_id": track_id,
                                    "object_class": obj_class,
                                    "line_id": line["line_id"],
                                    "direction": direction,
                                    "timestamp_relative": relative_time,
                                }

                                # Add speed data if enabled
                                if speed_enabled and track_id in first_seen_positions:
                                    first_x, first_y = first_seen_positions[track_id]
                                    first_time = first_seen_times[track_id]

                                    distance = (
                                        abs(center_x - first_x)
                                        if line["type"] == "vertical"
                                        else abs(center_y - first_y)
                                    )
                                    time_elapsed = current_time - first_time

                                    if time_elapsed > 0.1:  # Minimum tracking time
                                        event["distance_pixels"] = distance
                                        event["time_elapsed"] = time_elapsed

                                data_queue.put(event)

                    # Check zone entry/exit
                    for zone in zones:
                        # Check if this class is allowed for this zone
                        if obj_class not in zone["allowed_classes"]:
                            continue

                        # Calculate zone boundaries in ROI pixels
                        zone_x1 = roi_width * zone["x1_pct"] / 100
                        zone_x2 = roi_width * zone["x2_pct"] / 100
                        zone_y1 = roi_height * zone["y1_pct"] / 100
                        zone_y2 = roi_height * zone["y2_pct"] / 100

                        # Check if center point is inside zone
                        inside = (
                            zone_x1 <= center_x <= zone_x2
                            and zone_y1 <= center_y <= zone_y2
                        )

                        zone_id = zone["zone_id"]
                        was_inside = zone_id in active_zones[track_id]

                        if inside and not was_inside:
                            # ZONE_ENTER event
                            active_zones[track_id][zone_id] = current_time
                            event_count += 1

                            event = {
                                "event_type": "ZONE_ENTER",
                                "track_id": track_id,
                                "object_class": obj_class,
                                "zone_id": zone_id,
                                "timestamp_relative": relative_time,
                            }

                            data_queue.put(event)

                        elif not inside and was_inside:
                            # ZONE_EXIT event
                            entry_time = active_zones[track_id][zone_id]
                            dwell_time = current_time - entry_time
                            del active_zones[track_id][zone_id]
                            event_count += 1

                            event = {
                                "event_type": "ZONE_EXIT",
                                "track_id": track_id,
                                "object_class": obj_class,
                                "zone_id": zone_id,
                                "timestamp_relative": relative_time,
                                "dwell_time": dwell_time,
                            }

                            data_queue.put(event)

                    # Update position tracking
                    previous_positions[track_id] = (center_x, center_y)

            # Frame saving (if enabled)
            if frame_saving and frame_count % config["frame_saving"]["interval"] == 0:
                annotated_frame = _annotate_frame(
                    frame.copy(),
                    lines,
                    zones,
                    roi_config,
                    (frame_width, frame_height),
                    event_count,
                    current_fps,
                )

                timestamp = datetime.now().strftime("%H%M%S")
                filename = f"{config['frame_saving']['output_dir']}/frame_{frame_count:06d}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)

            # Periodic status update
            if frame_count % 100 == 0:
                avg_fps = sum(fps_list[-100:]) / min(len(fps_list), 100)
                elapsed = time.time() - start_time
                print(
                    f"[{elapsed/60:.1f}min] Frame {frame_count} | "
                    f"FPS: {avg_fps:.1f} | Events: {event_count}"
                )

    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"\nERROR in detection: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cap.release()
        data_queue.put(None)  # Signal end to analyzer

        elapsed = time.time() - start_time
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

        print("\nDetection complete")
        print(f"  Runtime: {elapsed/60:.1f} minutes")
        print(f"  Frames: {frame_count}")
        print(f"  Avg FPS: {avg_fps:.1f}")
        print(f"  Events: {event_count}")


def _parse_lines(config):
    """Parse line configurations and assign IDs"""
    lines = []
    vertical_count = 0
    horizontal_count = 0

    for line_config in config.get("lines", []):
        if line_config["type"] == "vertical":
            vertical_count += 1
            line_id = f"V{vertical_count}"
        else:  # horizontal
            horizontal_count += 1
            line_id = f"H{horizontal_count}"

        # Default allowed_classes to all track_classes if not specified
        allowed_classes = line_config.get(
            "allowed_classes", config["detection"]["track_classes"]
        )

        lines.append(
            {
                "line_id": line_id,
                "type": line_config["type"],
                "position_pct": line_config["position_pct"],
                "description": line_config["description"],
                "allowed_classes": allowed_classes,
            }
        )

    return lines


def _parse_zones(config):
    """Parse zone configurations and assign IDs"""
    zones = []

    for i, zone_config in enumerate(config.get("zones", []), 1):
        zone_id = f"Z{i}"

        # Default allowed_classes to all track_classes if not specified
        allowed_classes = zone_config.get(
            "allowed_classes", config["detection"]["track_classes"]
        )

        zones.append(
            {
                "zone_id": zone_id,
                "x1_pct": zone_config["x1_pct"],
                "y1_pct": zone_config["y1_pct"],
                "x2_pct": zone_config["x2_pct"],
                "y2_pct": zone_config["y2_pct"],
                "description": zone_config["description"],
                "allowed_classes": allowed_classes,
            }
        )

    return zones


def _parse_roi(config):
    """Parse ROI configuration"""
    roi = config.get("roi", {})
    h_roi = roi.get("horizontal", {})
    v_roi = roi.get("vertical", {})

    h_enabled = h_roi.get("enabled", False)
    v_enabled = v_roi.get("enabled", False)

    return {
        "enabled": h_enabled or v_enabled,
        "h_from": h_roi.get("crop_from_left_pct", 0) if h_enabled else 0,
        "h_to": h_roi.get("crop_to_right_pct", 100) if h_enabled else 100,
        "v_from": v_roi.get("crop_from_top_pct", 0) if v_enabled else 0,
        "v_to": v_roi.get("crop_to_bottom_pct", 100) if v_enabled else 100,
    }


def _annotate_frame(frame, lines, zones, roi_config, frame_size, event_count, fps):
    """Add visual annotations to frame for debugging"""
    frame_width, frame_height = frame_size

    # Calculate ROI boundaries
    if roi_config["enabled"]:
        roi_x1 = int(frame_width * roi_config["h_from"] / 100)
        roi_x2 = int(frame_width * roi_config["h_to"] / 100)
        roi_y1 = int(frame_height * roi_config["v_from"] / 100)
        roi_y2 = int(frame_height * roi_config["v_to"] / 100)
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1

        # Draw ROI boundary in blue
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    else:
        roi_x1, roi_y1 = 0, 0
        roi_width, roi_height = frame_width, frame_height

    # Draw counting lines
    for line in lines:
        if line["type"] == "vertical":
            line_x = roi_x1 + int(roi_width * line["position_pct"] / 100)
            cv2.line(frame, (line_x, roi_y1), (line_x, roi_y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                line["line_id"],
                (line_x + 5, roi_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:  # horizontal
            line_y = roi_y1 + int(roi_height * line["position_pct"] / 100)
            cv2.line(frame, (roi_x1, line_y), (roi_x2, line_y), (0, 255, 0), 2)
            cv2.putText(
                frame,
                line["line_id"],
                (roi_x1 + 5, line_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Draw zones
    for zone in zones:
        zone_x1 = roi_x1 + int(roi_width * zone["x1_pct"] / 100)
        zone_x2 = roi_x1 + int(roi_width * zone["x2_pct"] / 100)
        zone_y1 = roi_y1 + int(roi_height * zone["y1_pct"] / 100)
        zone_y2 = roi_y1 + int(roi_height * zone["y2_pct"] / 100)

        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            zone["zone_id"],
            (zone_x1 + 5, zone_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # Overlay stats
    cv2.putText(
        frame,
        f"Events: {event_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return frame


if __name__ == "__main__":
    # Standalone testing
    import yaml
    from multiprocessing import Queue

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    queue = Queue()
    run_detection(queue, config)
