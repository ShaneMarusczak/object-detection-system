"""
Section setup for the config builder.

Handles configuration of individual sections: camera, detection, lines, zones,
reports, and output.
"""

from pathlib import Path

import cv2
import questionary
from questionary import Choice

from .preview import capture_annotated_preview, capture_preview
from .prompts import PROMPT_STYLE, Colors


def setup_camera(
    config: dict,
    state: dict,
) -> bool:
    """
    Setup camera connection.

    Args:
        config: Configuration dict to modify
        state: Builder state dict with cap, frame_width, frame_height, etc.

    Returns:
        True if camera connected successfully
    """
    print(f"\n{Colors.BOLD}--- Camera Setup ---{Colors.RESET}")

    default_url = "http://192.168.86.33:4747/video"
    url = questionary.text(
        "Camera URL:",
        default=default_url,
        style=PROMPT_STYLE,
    ).ask()

    if url is None:
        return False

    print(f"{Colors.GRAY}Testing connection...{Colors.RESET}", end=" ", flush=True)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"{Colors.RED}FAILED{Colors.RESET}")
        print(f"Could not connect to camera at {url}")
        return False

    ret, frame = cap.read()
    if not ret:
        print(f"{Colors.RED}FAILED{Colors.RESET}")
        print("Could not read frame from camera")
        return False

    frame_height, frame_width = frame.shape[:2]
    print(f"{Colors.GREEN}OK{Colors.RESET} ({frame_width}x{frame_height})")

    # Update state
    state["cap"] = cap
    state["camera_url"] = url
    state["frame_width"] = frame_width
    state["frame_height"] = frame_height

    config["camera"] = {"url": url}

    # Capture initial preview
    capture_preview(cap, state["preview_dir"], frame)

    return True


def setup_camera_from_config(config: dict, state: dict) -> bool:
    """
    Setup camera from existing config.

    Args:
        config: Loaded configuration dict
        state: Builder state dict

    Returns:
        True if camera connected successfully
    """
    camera_config = config.get("camera", {})
    url = camera_config.get("url", "")

    if not url:
        print(f"{Colors.RED}No camera URL in config{Colors.RESET}")
        return False

    print(f"{Colors.GRAY}Connecting to camera...{Colors.RESET}", end=" ", flush=True)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"{Colors.RED}FAILED{Colors.RESET}")
        print(f"Could not connect to camera at {url}")
        return False

    ret, frame = cap.read()
    if not ret:
        print(f"{Colors.RED}FAILED{Colors.RESET}")
        print("Could not read frame from camera")
        return False

    frame_height, frame_width = frame.shape[:2]
    state["cap"] = cap
    state["camera_url"] = url
    state["frame_width"] = frame_width
    state["frame_height"] = frame_height

    print(f"{Colors.GREEN}OK{Colors.RESET} ({frame_width}x{frame_height})")

    # Capture initial preview with existing annotations
    capture_annotated_preview(
        cap,
        state["preview_dir"],
        lines=config.get("lines", []),
        zones=config.get("zones", []),
    )
    return True


def setup_detection(config: dict, state: dict) -> None:
    """Setup detection parameters."""
    print(f"\n{Colors.BOLD}--- Detection Settings ---{Colors.RESET}")

    # Model - check for .pt files in current directory and models/
    available_models = _scan_for_models()
    default_model = "yolo11n.pt"

    if available_models:
        model = _select_model_interactively(available_models)
        if not model:
            model = (
                questionary.text(
                    "Model file:",
                    default=default_model,
                    style=PROMPT_STYLE,
                ).ask()
                or default_model
            )
    else:
        model = (
            questionary.text(
                "Model file:",
                default=default_model,
                style=PROMPT_STYLE,
            ).ask()
            or default_model
        )

    # Load model to get its classes
    model_classes = _load_model_classes(model)
    state["model_classes"] = model_classes

    if model_classes:
        print(
            f"  {Colors.CYAN}Model classes:{Colors.RESET} {', '.join(model_classes[:5])}",
            end="",
        )
        if len(model_classes) > 5:
            print(f" (+{len(model_classes) - 5} more)")
        else:
            print()

    # Confidence threshold
    default_conf = "0.3"
    conf_str = (
        questionary.text(
            "Confidence threshold:",
            default=default_conf,
            style=PROMPT_STYLE,
        ).ask()
        or default_conf
    )
    conf = float(conf_str)

    config["detection"] = {"model_file": model, "confidence_threshold": conf}


def _scan_for_models(directories: list[str] | None = None) -> list[Path]:
    """Scan directories for .pt model files."""
    if directories is None:
        directories = [".", "models"]

    all_models = []
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.is_dir():
            all_models.extend(dir_path.glob("*.pt"))

    return sorted(set(all_models))


def _select_model_interactively(models: list[Path]) -> str | None:
    """Let user select a model from available options."""
    if not models:
        return None

    choices = []
    for model_path in models:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        choices.append(
            Choice(
                title=f"{model_path.name} ({size_mb:.1f} MB)",
                value=str(model_path),
            )
        )

    selected = questionary.select(
        "Select model:",
        choices=choices,
        style=PROMPT_STYLE,
    ).ask()

    if selected:
        print(f"  {Colors.GREEN}Selected: {Path(selected).name}{Colors.RESET}")

    return selected


def _load_model_classes(model_path: str) -> list[str]:
    """Load model and extract its class names."""
    try:
        from ultralytics import YOLO

        print(f"  {Colors.GRAY}Loading model...{Colors.RESET}", end=" ", flush=True)
        model = YOLO(model_path)
        classes = list(model.names.values())
        print(f"{Colors.GREEN}OK{Colors.RESET} ({len(classes)} classes)")
        return classes
    except Exception as e:
        print(f"{Colors.RED}Failed{Colors.RESET}")
        print(f"  {Colors.YELLOW}Warning: Could not load model: {e}{Colors.RESET}")
        return []


def setup_lines(config: dict, state: dict) -> None:
    """Setup detection lines with visual preview."""
    print(f"\n{Colors.BOLD}--- Lines Setup ---{Colors.RESET}")
    print(f"{Colors.GRAY}Lines detect objects crossing a boundary{Colors.RESET}")
    print(
        f"{Colors.GRAY}Format: h/v [position%] [name]  (e.g. 'v 50 Driveway'){Colors.RESET}"
    )

    lines = list(config.get("lines", []))
    zones = config.get("zones", [])
    cap = state.get("cap")
    preview_dir = state.get("preview_dir", "data")

    while True:
        line_input = questionary.text(
            "Add line (or Enter when done):",
            style=PROMPT_STYLE,
        ).ask()

        if line_input is None or not line_input.strip():
            break

        parsed = _parse_line_input(line_input.strip())
        if not parsed:
            print(f"  {Colors.RED}Invalid format. Use: h/v [%] [name]{Colors.RESET}")
            continue

        line_type = parsed["type"]
        position = parsed.get("position_pct")
        desc = parsed.get("description")

        # Prompt for missing position
        if position is None:
            default_pos = "50"
            prompt = (
                "Position % from left:"
                if line_type == "vertical"
                else "Position % from top:"
            )
            pos_str = questionary.text(
                prompt,
                default=default_pos,
                style=PROMPT_STYLE,
            ).ask()
            position = int(pos_str) if pos_str else int(default_pos)

        # Prompt for missing description
        if not desc:
            default_desc = f"Line {len(lines) + 1}"
            desc = (
                questionary.text(
                    "Description:",
                    default=default_desc,
                    style=PROMPT_STYLE,
                ).ask()
                or default_desc
            )

        lines.append({"type": line_type, "position_pct": position, "description": desc})

        type_label = "vertical" if line_type == "vertical" else "horizontal"
        print(f'  {Colors.GREEN}✓ {type_label} at {position}% "{desc}"{Colors.RESET}')
        capture_annotated_preview(cap, preview_dir, lines=lines, zones=zones)

        # Quick adjust option
        while True:
            action = questionary.select(
                "Action:",
                choices=[
                    Choice(title="Continue", value="continue"),
                    Choice(title="Adjust position", value="adjust"),
                    Choice(title="Delete this line", value="delete"),
                ],
                style=PROMPT_STYLE,
            ).ask()

            if action == "adjust":
                pos_str = questionary.text(
                    "Position %:",
                    default=str(position),
                    style=PROMPT_STYLE,
                ).ask()
                if pos_str:
                    position = int(pos_str)
                    lines[-1]["position_pct"] = position
                    capture_annotated_preview(
                        cap, preview_dir, lines=lines, zones=zones
                    )
                    print(f"  {Colors.GREEN}✓ Updated to {position}%{Colors.RESET}")
            elif action == "delete":
                deleted = lines.pop()
                print(
                    f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                )
                capture_annotated_preview(cap, preview_dir, lines=lines, zones=zones)
                break
            else:
                break

    if lines:
        config["lines"] = lines
    print(f"{Colors.GRAY}Done - {len(lines)} line(s){Colors.RESET}")


def _parse_line_input(line_input: str) -> dict | None:
    """Parse shorthand line input like 'v 50 Driveway' or 'h 30'."""
    parts = line_input.split(maxsplit=2)
    if not parts:
        return None

    result = {}

    type_char = parts[0].lower()
    if type_char in ("v", "vertical"):
        result["type"] = "vertical"
    elif type_char in ("h", "horizontal"):
        result["type"] = "horizontal"
    else:
        return None

    if len(parts) >= 2:
        try:
            result["position_pct"] = int(parts[1])
        except ValueError:
            result["description"] = " ".join(parts[1:])

    if len(parts) >= 3:
        result["description"] = parts[2]

    return result


def setup_zones(config: dict, state: dict) -> None:
    """Setup detection zones with visual preview."""
    print(f"\n{Colors.BOLD}--- Zones Setup ---{Colors.RESET}")
    print(
        f"{Colors.GRAY}Zones detect objects entering/dwelling in an area{Colors.RESET}"
    )
    print(f"{Colors.GRAY}Format: left% top% right% bottom% [name]{Colors.RESET}")
    print(
        f"{Colors.GRAY}  e.g. '0 0 50 100 Left Half' or '25 25 75 75 Center'{Colors.RESET}"
    )

    zones = list(config.get("zones", []))
    lines = config.get("lines", [])
    cap = state.get("cap")
    preview_dir = state.get("preview_dir", "data")

    while True:
        zone_input = questionary.text(
            "Add zone (or Enter when done):",
            style=PROMPT_STYLE,
        ).ask()

        if zone_input is None or not zone_input.strip():
            break

        parsed = _parse_zone_input(zone_input.strip())
        if not parsed:
            print(
                f"  {Colors.RED}Invalid format. Use: left top right bottom [name]{Colors.RESET}"
            )
            continue

        x1, y1, x2, y2 = (
            parsed["x1_pct"],
            parsed["y1_pct"],
            parsed["x2_pct"],
            parsed["y2_pct"],
        )
        desc = parsed.get("description")

        if x2 <= x1:
            print(
                f"  {Colors.RED}Error: right ({x2}%) must be > left ({x1}%){Colors.RESET}"
            )
            continue
        if y2 <= y1:
            print(
                f"  {Colors.RED}Error: bottom ({y2}%) must be > top ({y1}%){Colors.RESET}"
            )
            continue

        if not desc:
            default_desc = f"Zone {len(zones) + 1}"
            desc = (
                questionary.text(
                    "Description:",
                    default=default_desc,
                    style=PROMPT_STYLE,
                ).ask()
                or default_desc
            )

        zones.append(
            {
                "x1_pct": x1,
                "y1_pct": y1,
                "x2_pct": x2,
                "y2_pct": y2,
                "description": desc,
            }
        )

        print(f'  {Colors.GREEN}✓ {x1},{y1} to {x2},{y2} "{desc}"{Colors.RESET}')
        capture_annotated_preview(cap, preview_dir, lines=lines, zones=zones)

        # Quick adjust option
        while True:
            action = questionary.select(
                "Action:",
                choices=[
                    Choice(title="Continue", value="continue"),
                    Choice(title="Adjust bounds", value="adjust"),
                    Choice(title="Delete this zone", value="delete"),
                ],
                style=PROMPT_STYLE,
            ).ask()

            if action == "adjust":
                print("  Adjust bounds (press Enter to keep current):")
                new_x1 = questionary.text(
                    "Left %:", default=str(x1), style=PROMPT_STYLE
                ).ask()
                new_y1 = questionary.text(
                    "Top %:", default=str(y1), style=PROMPT_STYLE
                ).ask()
                new_x2 = questionary.text(
                    "Right %:", default=str(x2), style=PROMPT_STYLE
                ).ask()
                new_y2 = questionary.text(
                    "Bottom %:", default=str(y2), style=PROMPT_STYLE
                ).ask()

                x1 = int(new_x1) if new_x1 else x1
                y1 = int(new_y1) if new_y1 else y1
                x2 = int(new_x2) if new_x2 else x2
                y2 = int(new_y2) if new_y2 else y2

                if x2 <= x1 or y2 <= y1:
                    print(f"  {Colors.RED}Invalid bounds{Colors.RESET}")
                    continue

                zones[-1] = {
                    "x1_pct": x1,
                    "y1_pct": y1,
                    "x2_pct": x2,
                    "y2_pct": y2,
                    "description": desc,
                }
                capture_annotated_preview(cap, preview_dir, lines=lines, zones=zones)
                print(
                    f"  {Colors.GREEN}✓ Updated to {x1},{y1} to {x2},{y2}{Colors.RESET}"
                )
            elif action == "delete":
                deleted = zones.pop()
                print(
                    f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                )
                capture_annotated_preview(cap, preview_dir, lines=lines, zones=zones)
                break
            else:
                break

    if zones:
        config["zones"] = zones
    print(f"{Colors.GRAY}Done - {len(zones)} zone(s){Colors.RESET}")


def _parse_zone_input(zone_input: str) -> dict | None:
    """Parse shorthand zone input like '0 0 50 100 Left Half'."""
    parts = zone_input.split(maxsplit=4)
    if len(parts) < 4:
        return None

    try:
        result = {
            "x1_pct": int(parts[0]),
            "y1_pct": int(parts[1]),
            "x2_pct": int(parts[2]),
            "y2_pct": int(parts[3]),
        }
        if len(parts) >= 5:
            result["description"] = parts[4]
        return result
    except ValueError:
        return None


def setup_reports(config: dict) -> None:
    """Setup report configurations."""
    events = config.get("events", [])

    report_ids = set()
    for event in events:
        report_id = event.get("actions", {}).get("report")
        if report_id:
            report_ids.add(report_id)

    if not report_ids:
        return

    print(f"\n{Colors.BOLD}--- Reports Setup ---{Colors.RESET}")

    reports = []
    for report_id in report_ids:
        print(f"\n  Report: {report_id}")

        report_events = [
            e["name"] for e in events if e.get("actions", {}).get("report") == report_id
        ]
        print(f"    Events: {', '.join(report_events)}")

        default_title = report_id.replace("_", " ").title()
        title = (
            questionary.text(
                "Title:",
                default=default_title,
                style=PROMPT_STYLE,
            ).ask()
            or default_title
        )

        output_dir = (
            questionary.text(
                "Output directory:",
                default="reports",
                style=PROMPT_STYLE,
            ).ask()
            or "reports"
        )

        has_photos = any(
            e.get("actions", {}).get("frame_capture")
            for e in events
            if e.get("actions", {}).get("report") == report_id
        )

        if not has_photos:
            has_photos = questionary.confirm(
                "Include photos in report?",
                default=True,
                style=PROMPT_STYLE,
            ).ask()

        annotate = False
        if has_photos:
            annotate = questionary.confirm(
                "Annotate photos with lines/zones?",
                default=True,
                style=PROMPT_STYLE,
            ).ask()

        reports.append(
            {
                "id": report_id,
                "title": title,
                "output_dir": output_dir,
                "events": report_events,
                "photos": has_photos,
                "annotate": annotate,
            }
        )

    config["reports"] = reports


def setup_output(config: dict) -> None:
    """Setup output directories and runtime settings."""
    print(f"\n{Colors.BOLD}--- Output Setup ---{Colors.RESET}")

    json_dir = (
        questionary.text(
            "JSON data directory:",
            default="data",
            style=PROMPT_STYLE,
        ).ask()
        or "data"
    )
    config["output"] = {"json_dir": json_dir}

    has_frame_capture = any(
        e.get("actions", {}).get("frame_capture") for e in config.get("events", [])
    )
    if has_frame_capture:
        frames_dir = (
            questionary.text(
                "Frames directory:",
                default="frames",
                style=PROMPT_STYLE,
            ).ask()
            or "frames"
        )
        config["frame_storage"] = {"type": "local", "local_dir": frames_dir}

    config["console_output"] = {"enabled": True, "level": "detailed"}

    print(f"\n{Colors.BOLD}--- Runtime Settings ---{Colors.RESET}")
    duration_str = (
        questionary.text(
            "Default duration in hours:",
            default="0.5",
            style=PROMPT_STYLE,
        ).ask()
        or "0.5"
    )
    duration = float(duration_str)
    duration_seconds = int(duration * 3600)

    config["runtime"] = {"queue_size": 500, "default_duration_hours": duration}

    for event in config.get("events", []):
        frame_capture = event.get("actions", {}).get("frame_capture")
        if frame_capture:
            frame_capture["expected_duration_seconds"] = duration_seconds
