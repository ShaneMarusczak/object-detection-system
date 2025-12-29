"""
Config Builder TUI

Interactive wizard for creating object detection configurations.
Captures preview frames to data/ folder for visual feedback.
"""

import os
import readline  # noqa: F401 - Enables arrow key history for input()
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from ..processor.coco_classes import COCO_CLASSES
from .planner import (
    build_plan,
    generate_sample_events,
    load_config_with_env,
    print_plan,
    print_validation_result,
    simulate_dry_run,
)
from .validator import validate_config_full

# Common object classes for quick selection
COMMON_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "bird",
]


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ConfigBuilder:
    """Interactive config builder with live preview."""

    # Wizard steps for progress indicator
    WIZARD_STEPS = [
        ("camera", "Camera"),
        ("detection", "Detection"),
        ("lines", "Lines"),
        ("zones", "Zones"),
        ("events", "Events"),
        ("output", "Output"),
    ]

    def __init__(self):
        self.config: dict = {}
        self.camera_url: str = ""
        self.cap: cv2.VideoCapture | None = None
        self.preview_dir: str = "data"
        self.http_server: subprocess.Popen | None = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.config_path: str | None = None  # Track source file for edits
        self.current_step: int = 0  # For progress indicator
        self.model_classes: list[str] = []  # Classes from loaded model

    def run(self) -> str | None:
        """Run the config builder wizard. Returns config filename or None."""
        try:
            self._print_header()

            # Step 0: Camera
            self.current_step = 0
            self._print_progress()
            if not self._setup_camera():
                return None
            self._start_preview_server()

            # Step 1: Detection
            self.current_step = 1
            self._print_progress()
            self._setup_detection()

            # Step 2: Lines
            self.current_step = 2
            self._print_progress()
            self._setup_lines()

            # Step 3: Zones
            self.current_step = 3
            self._print_progress()
            self._setup_zones()

            # Step 4: Events
            self.current_step = 4
            self._print_progress()
            self._setup_events()
            self._setup_reports()

            # Step 5: Output
            self.current_step = 5
            self._print_progress()
            self._setup_output()

            # Final review - show edit menu to allow tweaks before saving
            print(f"\n{Colors.CYAN}{Colors.BOLD}=== Final Review ==={Colors.RESET}")
            print(f"{Colors.GRAY}Review your config and make adjustments{Colors.RESET}")
            return self._edit_menu_loop()

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled{Colors.RESET}")
            return None
        finally:
            self._cleanup()

    def run_edit(self, config_path: str) -> str | None:
        """Run the config editor. Returns config filename or None."""
        try:
            # Load existing config
            if not self._load_config(config_path):
                return None

            self._print_edit_header(config_path)

            # Setup camera from loaded config
            if not self._setup_camera_from_config():
                return None

            self._start_preview_server()

            # Show edit menu loop
            return self._edit_menu_loop()

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled{Colors.RESET}")
            return None
        finally:
            self._cleanup()

    def _load_config(self, config_path: str) -> bool:
        """Load an existing config file."""
        try:
            with open(config_path, encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            self.config_path = config_path
            return True
        except FileNotFoundError:
            print(f"{Colors.RED}Config not found: {config_path}{Colors.RESET}")
            return False
        except yaml.YAMLError as e:
            print(f"{Colors.RED}Invalid YAML: {e}{Colors.RESET}")
            return False

    def _setup_camera_from_config(self) -> bool:
        """Setup camera from loaded config."""
        camera_config = self.config.get("camera", {})
        url = camera_config.get("url", "")

        if not url:
            print(f"{Colors.RED}No camera URL in config{Colors.RESET}")
            return False

        print(
            f"{Colors.GRAY}Connecting to camera...{Colors.RESET}", end=" ", flush=True
        )

        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            print(f"Could not connect to camera at {url}")
            return False

        ret, frame = self.cap.read()
        if not ret:
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            print("Could not read frame from camera")
            return False

        self.frame_height, self.frame_width = frame.shape[:2]
        self.camera_url = url
        print(
            f"{Colors.GREEN}OK{Colors.RESET} ({self.frame_width}x{self.frame_height})"
        )

        # Capture initial preview with existing annotations
        self._capture_annotated_preview(
            lines=self.config.get("lines", []), zones=self.config.get("zones", [])
        )
        return True

    def _print_edit_header(self, config_path: str):
        """Print edit mode header."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}=== Config Editor ==={Colors.RESET}")
        print(f"Editing: {Colors.GREEN}{config_path}{Colors.RESET}")
        print()

    def _get_section_summary(self, section: str) -> str:
        """Get a short summary of a config section for the menu."""
        if section == "camera":
            url = self.config.get("camera", {}).get("url", "not set")
            # Truncate long URLs
            if len(url) > 40:
                url = url[:37] + "..."
            return url

        elif section == "detection":
            det = self.config.get("detection", {})
            model = det.get("model_file", "not set")
            conf = det.get("confidence_threshold", 0.3)
            return f"{model}, conf={conf}"

        elif section == "lines":
            lines = self.config.get("lines", [])
            if not lines:
                return "none"
            descs = [ln.get("description", "?")[:12] for ln in lines[:3]]
            summary = ", ".join(descs)
            if len(lines) > 3:
                summary += f" +{len(lines) - 3} more"
            return f"{len(lines)}: {summary}"

        elif section == "zones":
            zones = self.config.get("zones", [])
            if not zones:
                return "none"
            descs = [z.get("description", "?")[:12] for z in zones[:3]]
            summary = ", ".join(descs)
            if len(zones) > 3:
                summary += f" +{len(zones) - 3} more"
            return f"{len(zones)}: {summary}"

        elif section == "events":
            events = self.config.get("events", [])
            if not events:
                return "none"
            return f"{len(events)} defined"

        elif section == "reports":
            reports = self.config.get("reports", [])
            if not reports:
                return "disabled"
            ids = [r.get("id", "?") for r in reports]
            return ", ".join(ids)

        elif section == "output":
            output = self.config.get("output", {})
            json_dir = output.get("json_dir", "data")
            return json_dir

        return "?"

    def _get_config_warnings(self) -> list[str]:
        """Check config for common issues and return warnings."""
        warnings = []
        events = self.config.get("events", [])
        lines = self.config.get("lines", [])
        zones = self.config.get("zones", [])

        # Check for events with no actions
        for event in events:
            if not event.get("actions"):
                warnings.append(f"Event '{event.get('name')}' has no actions")

        # Check for events referencing non-existent lines/zones
        line_names = {ln.get("description") for ln in lines}
        zone_names = {z.get("description") for z in zones}

        for event in events:
            match = event.get("match", {})
            ref_line = match.get("line")
            ref_zone = match.get("zone")

            if ref_line and ref_line not in line_names:
                warnings.append(
                    f"Event '{event.get('name')}' references missing line '{ref_line}'"
                )
            if ref_zone and ref_zone not in zone_names:
                warnings.append(
                    f"Event '{event.get('name')}' references missing zone '{ref_zone}'"
                )

        # Check for no events
        if not events and (lines or zones):
            warnings.append("Lines/zones defined but no events to use them")

        return warnings

    def _edit_menu_loop(self) -> str | None:
        """Show edit menu and handle section selection."""
        sections = [
            ("camera", "Camera"),
            ("detection", "Detection"),
            ("lines", "Lines"),
            ("zones", "Zones"),
            ("events", "Events"),
            ("reports", "Reports"),
            ("output", "Output"),
        ]

        while True:
            # Show warnings if any
            warnings = self._get_config_warnings()
            if warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
                for w in warnings:
                    print(f"  {Colors.YELLOW}! {w}{Colors.RESET}")

            # Show preview URL reminder
            local_ip = self._get_local_ip()
            print(f"\n{Colors.GRAY}Preview: http://{local_ip}:8000/preview.jpg{Colors.RESET}")

            print(f"\n{Colors.BOLD}Which section to edit?{Colors.RESET}")
            for i, (key, label) in enumerate(sections, 1):
                summary = self._get_section_summary(key)
                print(f"  {i}. {label} ({summary})")

            print(f"  {Colors.GRAY}p. Refresh preview{Colors.RESET}")
            print(f"  {Colors.GREEN}r. Save and run{Colors.RESET}")
            print(f"  {Colors.CYAN}s. Save and exit{Colors.RESET}")
            print(f"  {Colors.YELLOW}q. Quit without saving{Colors.RESET}")

            choice = input("Choice: ").strip().lower()

            if choice == "q":
                confirm = input("Are you sure? (y/N): ").strip().lower()
                if confirm == "y":
                    return None
                continue

            if choice == "p":
                self._capture_annotated_preview(
                    lines=self.config.get("lines", []),
                    zones=self.config.get("zones", []),
                )
                print(f"{Colors.GREEN}Preview refreshed{Colors.RESET}")
                continue

            if choice == "s":
                return self._save_edited_config(run_after=False)

            if choice == "r":
                return self._save_edited_config(run_after=True)

            # Section edit
            try:
                section_idx = int(choice) - 1
                if 0 <= section_idx < len(sections):
                    section_key = sections[section_idx][0]
                    self._edit_section(section_key)
                else:
                    print(f"{Colors.RED}Invalid choice{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}Invalid choice{Colors.RESET}")

    def _edit_section(self, section: str):
        """Edit a specific section."""
        title = section.replace("_", " ").title()
        print(f"\n{Colors.BOLD}--- Edit {title} ---{Colors.RESET}")

        if section == "camera":
            self._setup_camera()
        elif section == "detection":
            self._setup_detection()
        elif section == "lines":
            self._edit_lines()
        elif section == "zones":
            self._edit_zones()
        elif section == "events":
            self._edit_events()
        elif section == "reports":
            self._setup_reports()
        elif section == "output":
            self._setup_output()

    def _edit_lines(self):
        """Edit lines with options to keep, modify, or add."""
        lines = self.config.get("lines", [])

        if lines:
            print(f"Current lines ({len(lines)}):")
            for i, line in enumerate(lines, 1):
                print(
                    f"  {i}. {line.get('description')} ({line.get('type')}, {line.get('position_pct')}%)"
                )

            print("\nOptions:")
            print("  1. Keep all")
            print("  2. Delete some")
            print("  3. Add more")
            print("  4. Start fresh")
            choice = input("Choice [1]: ").strip() or "1"

            if choice == "1":
                return
            elif choice == "2":
                to_delete = input("Line numbers to delete (comma-separated): ").strip()
                indices = [
                    int(x.strip()) - 1 for x in to_delete.split(",") if x.strip()
                ]
                self.config["lines"] = [
                    ln for i, ln in enumerate(lines) if i not in indices
                ]
                print(f"{Colors.GREEN}Deleted {len(indices)} line(s){Colors.RESET}")
                # Refresh preview with updated annotations
                self._capture_annotated_preview(
                    lines=self.config.get("lines", []),
                    zones=self.config.get("zones", []),
                )
                return
            elif choice == "3":
                # Fall through to add mode
                pass
            elif choice == "4":
                self.config["lines"] = []
                lines = []
                # Refresh preview with zones only
                self._capture_annotated_preview(
                    lines=[], zones=self.config.get("zones", [])
                )

        # Add new lines (reuse existing method logic)
        self._setup_lines()

    def _edit_zones(self):
        """Edit zones with options to keep, modify, or add."""
        zones = self.config.get("zones", [])

        if zones:
            print(f"Current zones ({len(zones)}):")
            for i, zone in enumerate(zones, 1):
                print(
                    f"  {i}. {zone.get('description')} ({zone.get('x1_pct')}-{zone.get('x2_pct')}%, {zone.get('y1_pct')}-{zone.get('y2_pct')}%)"
                )

            print("\nOptions:")
            print("  1. Keep all")
            print("  2. Delete some")
            print("  3. Add more")
            print("  4. Start fresh")
            choice = input("Choice [1]: ").strip() or "1"

            if choice == "1":
                return
            elif choice == "2":
                to_delete = input("Zone numbers to delete (comma-separated): ").strip()
                indices = [
                    int(x.strip()) - 1 for x in to_delete.split(",") if x.strip()
                ]
                self.config["zones"] = [
                    z for i, z in enumerate(zones) if i not in indices
                ]
                print(f"{Colors.GREEN}Deleted {len(indices)} zone(s){Colors.RESET}")
                # Refresh preview with updated annotations
                self._capture_annotated_preview(
                    lines=self.config.get("lines", []),
                    zones=self.config.get("zones", []),
                )
                return
            elif choice == "3":
                # Fall through to add mode
                pass
            elif choice == "4":
                self.config["zones"] = []
                zones = []
                # Refresh preview with lines only
                self._capture_annotated_preview(
                    lines=self.config.get("lines", []), zones=[]
                )

        # Add new zones
        self._setup_zones()

    def _edit_events(self):
        """Edit events with options to keep, modify, or add."""
        events = self.config.get("events", [])

        if events:
            print(f"Current events ({len(events)}):")
            for i, event in enumerate(events, 1):
                match = event.get("match", {})
                event_type = match.get("event_type", "?")
                name = event.get("name", f"event_{i}")
                print(f"  {i}. {name} ({event_type})")

            print("\nOptions:")
            print("  1. Keep all")
            print("  2. Delete some")
            print("  3. Add more")
            print("  4. Start fresh")
            choice = input("Choice [1]: ").strip() or "1"

            if choice == "1":
                return
            elif choice == "2":
                to_delete = input("Event numbers to delete (comma-separated): ").strip()
                indices = [
                    int(x.strip()) - 1 for x in to_delete.split(",") if x.strip()
                ]
                self.config["events"] = [
                    e for i, e in enumerate(events) if i not in indices
                ]
                print(f"{Colors.GREEN}Deleted {len(indices)} event(s){Colors.RESET}")
                return
            elif choice == "3":
                # Fall through to add mode
                pass
            elif choice == "4":
                self.config["events"] = []

        # Add new events
        self._setup_events()

    def _save_edited_config(self, run_after: bool = False) -> str | None:
        """Save edited config, optionally run after."""
        print(f"\n{Colors.BOLD}--- Save Config ---{Colors.RESET}")

        # Default: original path for edits, timestamped name for new configs
        if self.config_path:
            default_path = self.config_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            default_path = f"configs/config_{timestamp}.yaml"

        filepath = input(f"Save to [{default_path}]: ").strip() or default_path

        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"{Colors.GREEN}Saved:{Colors.RESET} {filepath}")

        if run_after:
            # Update config.yaml to use this config
            self._update_config_pointer(filepath)

            # Run pre-flight stages (validate, plan, dry-run)
            self._run_preflight_stages(filepath)

            self._cleanup()
            print(f"\n{Colors.CYAN}Starting detection...{Colors.RESET}\n")
            os.execvp("python", ["python", "-m", "object_detection", "-c", filepath])

        return filepath

    def _print_header(self):
        """Print welcome header."""
        print(
            f"\n{Colors.CYAN}{Colors.BOLD}=== Object Detection Config Builder ==={Colors.RESET}"
        )
        print(
            f"{Colors.GRAY}Interactive wizard for creating detection configs{Colors.RESET}"
        )
        print()

    def _print_progress(self):
        """Print wizard progress indicator."""
        parts = []
        for i, (key, label) in enumerate(self.WIZARD_STEPS):
            if i < self.current_step:
                parts.append(f"{Colors.GREEN}[✓] {label}{Colors.RESET}")
            elif i == self.current_step:
                parts.append(f"{Colors.CYAN}[•] {label}{Colors.RESET}")
            else:
                parts.append(f"{Colors.GRAY}[ ] {label}{Colors.RESET}")
        print("  ".join(parts))
        print()

    def _get_local_ip(self) -> str:
        """Get local IP address for remote access."""
        import socket

        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def _start_preview_server(self):
        """Start HTTP server in data/ folder for preview images."""
        os.makedirs(self.preview_dir, exist_ok=True)

        # Get local IP for remote access
        local_ip = self._get_local_ip()
        preview_url = f"http://{local_ip}:8000/preview.jpg"

        # Check if something is already running on port 8000
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", 8000))
            sock.close()
            if result == 0:
                print(
                    f"{Colors.GREEN}Preview server already running:{Colors.RESET} {preview_url}"
                )
                return
        except Exception:
            pass

        # Start server (bind to all interfaces for remote access)
        try:
            self.http_server = subprocess.Popen(
                [sys.executable, "-m", "http.server", "8000", "--bind", "0.0.0.0"],
                cwd=self.preview_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"{Colors.GREEN}Preview server started:{Colors.RESET} {preview_url}")
            print(
                f"{Colors.GRAY}(Open in browser to see captured frames){Colors.RESET}"
            )
        except Exception as e:
            print(f"{Colors.YELLOW}Could not start preview server: {e}{Colors.RESET}")
            print(
                f"{Colors.GRAY}Frames will still be saved to {self.preview_dir}/{Colors.RESET}"
            )

    def _setup_camera(self) -> bool:
        """Setup camera connection."""
        print(f"\n{Colors.BOLD}--- Camera Setup ---{Colors.RESET}")

        default_url = "http://192.168.86.33:4747/video"
        url = input(f"Camera URL [{default_url}]: ").strip() or default_url

        print(f"{Colors.GRAY}Testing connection...{Colors.RESET}", end=" ", flush=True)

        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            print(f"Could not connect to camera at {url}")
            return False

        ret, frame = self.cap.read()
        if not ret:
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            print("Could not read frame from camera")
            return False

        self.frame_height, self.frame_width = frame.shape[:2]
        print(
            f"{Colors.GREEN}OK{Colors.RESET} ({self.frame_width}x{self.frame_height})"
        )

        self.camera_url = url
        self.config["camera"] = {"url": url}

        # Capture initial preview (server starts after this)
        self._capture_preview(frame)

        return True

    def _scan_for_models(self, directories: list[str] | None = None) -> list[Path]:
        """Scan directories for .pt model files."""
        if directories is None:
            directories = [".", "models"]

        all_models = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.is_dir():
                all_models.extend(dir_path.glob("*.pt"))

        return sorted(set(all_models))

    def _select_model_interactively(self) -> str | None:
        """
        Scan for .pt files and let user select one.

        Returns the selected model path or None if no models found.
        """
        models = self._scan_for_models()

        if not models:
            return None

        print(f"\n  {Colors.CYAN}Available models:{Colors.RESET}")
        print("  " + "-" * 45)
        for i, model_path in enumerate(models, 1):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"    {i}. {model_path.name} ({size_mb:.1f} MB)")
        print("  " + "-" * 45)

        while True:
            try:
                choice = input(f"  Select model [1-{len(models)}]: ").strip()
                if not choice:
                    continue
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"  {Colors.GREEN}Selected: {selected.name}{Colors.RESET}")
                    return str(selected)
                else:
                    print(
                        f"  {Colors.RED}Enter a number between 1 and {len(models)}{Colors.RESET}"
                    )
            except ValueError:
                print(f"  {Colors.RED}Please enter a number{Colors.RESET}")

    def _load_model_classes(self, model_path: str) -> list[str]:
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

    def _setup_detection(self):
        """Setup detection parameters."""
        print(f"\n{Colors.BOLD}--- Detection Settings ---{Colors.RESET}")

        # Model - check for .pt files in current directory and models/
        available_models = self._scan_for_models()
        default_model = "yolo11n.pt"

        if available_models:
            # Show available models for selection
            model = self._select_model_interactively()
            if not model:
                model = (
                    input(f"Model file [{default_model}]: ").strip() or default_model
                )
        else:
            # No models found, ask for path
            model = input(f"Model file [{default_model}]: ").strip() or default_model

        # Load model to get its classes
        self.model_classes = self._load_model_classes(model)
        if self.model_classes:
            print(
                f"  {Colors.CYAN}Model classes:{Colors.RESET} {', '.join(self.model_classes[:5])}",
                end="",
            )
            if len(self.model_classes) > 5:
                print(f" (+{len(self.model_classes) - 5} more)")
            else:
                print()

        # Confidence threshold
        default_conf = "0.3"
        conf_str = (
            input(f"Confidence threshold [{default_conf}]: ").strip() or default_conf
        )
        conf = float(conf_str)

        self.config["detection"] = {"model_file": model, "confidence_threshold": conf}

    def _parse_line_input(self, line_input: str) -> dict | None:
        """Parse shorthand line input like 'v 50 Driveway' or 'h 30'."""
        parts = line_input.split(maxsplit=2)
        if not parts:
            return None

        result = {}

        # First part: type (v/h)
        type_char = parts[0].lower()
        if type_char in ("v", "vertical"):
            result["type"] = "vertical"
        elif type_char in ("h", "horizontal"):
            result["type"] = "horizontal"
        else:
            return None  # Invalid type

        # Second part: position (optional)
        if len(parts) >= 2:
            try:
                result["position_pct"] = int(parts[1])
            except ValueError:
                # Second part might be description, not position
                result["description"] = " ".join(parts[1:])

        # Third part: description (optional)
        if len(parts) >= 3:
            result["description"] = parts[2]

        return result

    def _setup_lines(self):
        """Setup detection lines with visual preview."""
        print(f"\n{Colors.BOLD}--- Lines Setup ---{Colors.RESET}")
        print(f"{Colors.GRAY}Lines detect objects crossing a boundary{Colors.RESET}")
        print(
            f"{Colors.GRAY}Format: h/v [position%] [name]  (e.g. 'v 50 Driveway'){Colors.RESET}"
        )

        # Start with existing lines (for edit mode)
        lines = list(self.config.get("lines", []))
        zones = self.config.get("zones", [])

        while True:
            line_input = input("\nAdd line (or Enter when done): ").strip()
            if not line_input:
                break

            # Parse shorthand input
            parsed = self._parse_line_input(line_input)
            if not parsed:
                print(
                    f"  {Colors.RED}Invalid format. Use: h/v [%] [name]{Colors.RESET}"
                )
                continue

            line_type = parsed["type"]
            position = parsed.get("position_pct")
            desc = parsed.get("description")

            # Prompt for missing position
            if position is None:
                default_pos = "50"
                if line_type == "vertical":
                    pos_str = input(f"  Position % from left [{default_pos}]: ").strip()
                else:
                    pos_str = input(f"  Position % from top [{default_pos}]: ").strip()
                position = int(pos_str) if pos_str else int(default_pos)

            # Prompt for missing description
            if not desc:
                default_desc = f"Line {len(lines) + 1}"
                desc = (
                    input(f"  Description [{default_desc}]: ").strip() or default_desc
                )

            lines.append(
                {"type": line_type, "position_pct": position, "description": desc}
            )

            # Show confirmation and capture preview
            type_label = "vertical" if line_type == "vertical" else "horizontal"
            print(
                f'  {Colors.GREEN}✓ {type_label} at {position}% "{desc}"{Colors.RESET}'
            )
            self._capture_annotated_preview(lines=lines, zones=zones)

            # Quick adjust option
            while True:
                action = input("  [a]djust [d]elete [Enter] continue: ").strip().lower()
                if action == "a":
                    if line_type == "vertical":
                        pos_str = input(f"  Position % [{position}]: ").strip()
                    else:
                        pos_str = input(f"  Position % [{position}]: ").strip()
                    if pos_str:
                        position = int(pos_str)
                        lines[-1]["position_pct"] = position
                        self._capture_annotated_preview(lines=lines, zones=zones)
                        print(f"  {Colors.GREEN}✓ Updated to {position}%{Colors.RESET}")
                elif action == "d":
                    deleted = lines.pop()
                    print(
                        f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                    )
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    break
                else:
                    break

        if lines:
            self.config["lines"] = lines
        print(f"{Colors.GRAY}Done - {len(lines)} line(s){Colors.RESET}")

    def _parse_zone_input(self, zone_input: str) -> dict | None:
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

    def _setup_zones(self):
        """Setup detection zones with visual preview."""
        print(f"\n{Colors.BOLD}--- Zones Setup ---{Colors.RESET}")
        print(
            f"{Colors.GRAY}Zones detect objects entering/dwelling in an area{Colors.RESET}"
        )
        print(f"{Colors.GRAY}Format: left% top% right% bottom% [name]{Colors.RESET}")
        print(
            f"{Colors.GRAY}  e.g. '0 0 50 100 Left Half' or '25 25 75 75 Center'{Colors.RESET}"
        )

        # Start with existing zones and lines (for edit mode)
        zones = list(self.config.get("zones", []))
        lines = self.config.get("lines", [])

        while True:
            zone_input = input("\nAdd zone (or Enter when done): ").strip()
            if not zone_input:
                break

            # Parse shorthand input
            parsed = self._parse_zone_input(zone_input)
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

            # Validate bounds
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

            # Prompt for missing description
            if not desc:
                default_desc = f"Zone {len(zones) + 1}"
                desc = (
                    input(f"  Description [{default_desc}]: ").strip() or default_desc
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

            # Show confirmation and capture preview
            print(f'  {Colors.GREEN}✓ {x1},{y1} to {x2},{y2} "{desc}"{Colors.RESET}')
            self._capture_annotated_preview(lines=lines, zones=zones)

            # Quick adjust option
            while True:
                action = input("  [a]djust [d]elete [Enter] continue: ").strip().lower()
                if action == "a":
                    print("  Adjust bounds (press Enter to keep current):")
                    new_x1 = input(f"    Left % [{x1}]: ").strip()
                    new_y1 = input(f"    Top % [{y1}]: ").strip()
                    new_x2 = input(f"    Right % [{x2}]: ").strip()
                    new_y2 = input(f"    Bottom % [{y2}]: ").strip()

                    x1 = int(new_x1) if new_x1 else x1
                    y1 = int(new_y1) if new_y1 else y1
                    x2 = int(new_x2) if new_x2 else x2
                    y2 = int(new_y2) if new_y2 else y2

                    # Validate adjusted bounds
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
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    print(
                        f"  {Colors.GREEN}✓ Updated to {x1},{y1} to {x2},{y2}{Colors.RESET}"
                    )
                elif action == "d":
                    deleted = zones.pop()
                    print(
                        f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                    )
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    break
                else:
                    break

        if zones:
            self.config["zones"] = zones
        print(f"{Colors.GRAY}Done - {len(zones)} zone(s){Colors.RESET}")

    def _setup_events(self):
        """Setup event definitions."""
        print(f"\n{Colors.BOLD}--- Events Setup ---{Colors.RESET}")
        print(
            f"{Colors.GRAY}Events define what to detect and what actions to take{Colors.RESET}"
        )

        events = []
        lines = self.config.get("lines", [])
        zones = self.config.get("zones", [])

        while True:
            add = input("\nAdd an event? (Y/n): ").strip().lower()
            if add == "n":
                break

            event = {}

            # Event name
            name = input("  Event name: ").strip() or f"event_{len(events) + 1}"
            event["name"] = name

            # Match criteria
            print("  Match criteria:")
            match = {}

            # Build dynamic event type menu based on available lines/zones
            event_options = []
            if lines:
                event_options.append(("LINE_CROSS", "object crosses a line"))
            if zones:
                event_options.append(("ZONE_ENTER", "object enters a zone"))
                event_options.append(("ZONE_EXIT", "object exits a zone"))
                event_options.append(("NIGHTTIME_CAR", "headlight blob detection in zone"))
            event_options.append(("DETECTED", "any detection, no geometry required"))

            print("    Event type:")
            for i, (etype, desc) in enumerate(event_options, 1):
                print(f"      {i}. {etype} ({desc})")
            type_choice = input("    Choice [1]: ").strip() or "1"

            is_nighttime_event = False
            is_detected_event = False

            try:
                selected_type = event_options[int(type_choice) - 1][0]
            except (ValueError, IndexError):
                selected_type = event_options[0][0]

            match["event_type"] = selected_type

            if selected_type == "LINE_CROSS":
                print("    Which line?")
                for i, line in enumerate(lines, 1):
                    print(f"      {i}. {line['description']}")
                try:
                    line_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    if 0 <= line_choice < len(lines):
                        match["line"] = lines[line_choice]["description"]
                    else:
                        match["line"] = lines[0]["description"]
                except (ValueError, IndexError):
                    match["line"] = lines[0]["description"]
            elif selected_type == "ZONE_ENTER":
                print("    Which zone?")
                for i, zone in enumerate(zones, 1):
                    print(f"      {i}. {zone['description']}")
                try:
                    zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    if 0 <= zone_choice < len(zones):
                        match["zone"] = zones[zone_choice]["description"]
                    else:
                        match["zone"] = zones[0]["description"]
                except (ValueError, IndexError):
                    match["zone"] = zones[0]["description"]
            elif selected_type == "ZONE_EXIT":
                print("    Which zone?")
                for i, zone in enumerate(zones, 1):
                    print(f"      {i}. {zone['description']}")
                try:
                    zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    if 0 <= zone_choice < len(zones):
                        match["zone"] = zones[zone_choice]["description"]
                    else:
                        match["zone"] = zones[0]["description"]
                except (ValueError, IndexError):
                    match["zone"] = zones[0]["description"]
            elif selected_type == "DETECTED":
                is_detected_event = True
                print(
                    f"    {Colors.GRAY}DETECTED fires for every detection - no tracking needed{Colors.RESET}"
                )
            elif selected_type == "NIGHTTIME_CAR":
                is_nighttime_event = True
                print("    Which zone to monitor for headlights?")
                for i, zone in enumerate(zones, 1):
                    print(f"      {i}. {zone['description']}")
                try:
                    zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    if 0 <= zone_choice < len(zones):
                        match["zone"] = zones[zone_choice]["description"]
                    else:
                        match["zone"] = zones[0]["description"]
                except (ValueError, IndexError):
                    match["zone"] = zones[0]["description"]

                # Nighttime detection parameters
                print("    Nighttime detection settings (press Enter for defaults):")
                brightness = input("      Brightness threshold [30]: ").strip() or "30"
                min_blob = input("      Min blob size [100]: ").strip() or "100"
                max_blob = input("      Max blob size [10000]: ").strip() or "10000"
                score = input("      Score threshold [85]: ").strip() or "85"
                taillight_str = (
                    input("      Require taillight match? (Y/n): ").strip().lower()
                )
                taillight = taillight_str != "n"

                match["nighttime_detection"] = {
                    "brightness_threshold": int(brightness),
                    "min_blob_size": int(min_blob),
                    "max_blob_size": int(max_blob),
                    "score_threshold": int(score),
                    "taillight_color_match": taillight,
                }

            # Object classes with validation (skip for NIGHTTIME_CAR, optional for DETECTED)
            if not is_nighttime_event and not is_detected_event:
                # Use model classes if loaded, otherwise fall back to COCO
                if self.model_classes:
                    valid_classes = set(self.model_classes)
                    display_classes = self.model_classes
                    # Default to first class if single-class model
                    default_classes = (
                        self.model_classes[0]
                        if len(self.model_classes) == 1
                        else "car, truck, bus"
                    )
                else:
                    valid_classes = set(COCO_CLASSES.values())
                    display_classes = COMMON_CLASSES
                    default_classes = "car, truck, bus"

                while True:
                    print("    Object classes (comma-separated):")
                    print(
                        f"    {Colors.CYAN}Available:{Colors.RESET} {', '.join(display_classes)}"
                    )
                    classes_str = (
                        input(f"    Classes [{default_classes}]: ").strip()
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

            # DETECTED events can optionally filter by class
            if is_detected_event:
                # Single-class model: auto-select the only class
                if self.model_classes and len(self.model_classes) == 1:
                    single_class = self.model_classes[0]
                    match["object_class"] = single_class
                    print(
                        f"    {Colors.CYAN}Single-class model: using '{single_class}'{Colors.RESET}"
                    )
                else:
                    filter_class = (
                        input("    Filter by object class? (y/N): ").strip().lower()
                    )
                    if filter_class == "y":
                        if self.model_classes:
                            valid_classes = set(self.model_classes)
                            display_classes = self.model_classes
                            default_class = self.model_classes[0]
                        else:
                            valid_classes = set(COCO_CLASSES.values())
                            display_classes = COMMON_CLASSES
                            default_class = "car"

                        print(
                            f"    {Colors.CYAN}Available:{Colors.RESET} {', '.join(display_classes)}"
                        )
                        class_str = (
                            input(f"    Class [{default_class}]: ").strip() or default_class
                        )
                        if class_str in valid_classes:
                            match["object_class"] = class_str
                        else:
                            print(
                                f"    {Colors.YELLOW}Unknown class, skipping filter{Colors.RESET}"
                            )

            event["match"] = match

            # Actions - outcome based selection
            print("  Actions:")
            print("    1. Run command/script")
            print("    2. Report with photos")
            print("    3. Report stats only")
            print("    4. JSON log only")
            action_input = input("  Choose (e.g. 1,2) [1]: ").strip() or "1"

            # Parse choices
            choices = {c.strip() for c in action_input.replace(" ", ",").split(",")}
            actions = {}

            want_command = "1" in choices
            want_report_photos = "2" in choices
            want_report_only = "3" in choices
            want_json_only = "4" in choices

            # Build enabled list for display
            enabled = []

            # Command action
            if want_command:
                enabled.append("command")
            # Report with photos (implies json)
            if want_report_photos:
                enabled.extend(["report", "frame_capture", "json_log"])
            # Report only (implies json)
            if want_report_only:
                if "report" not in enabled:
                    enabled.append("report")
                if "json_log" not in enabled:
                    enabled.append("json_log")
            # JSON only
            if want_json_only:
                if "json_log" not in enabled:
                    enabled.append("json_log")

            if enabled:
                print(f"    {Colors.GRAY}→ Enables: {', '.join(enabled)}{Colors.RESET}")

            # Configure command action
            if want_command:
                exec_path = input("    Command/script path: ").strip()
                if exec_path:
                    timeout = input("    Timeout seconds [30]: ").strip() or "30"
                    actions["command"] = {
                        "exec": exec_path,
                        "timeout_seconds": int(timeout),
                    }

                    # Ask about shutdown
                    shutdown_str = (
                        input("    Stop detector after this event? (y/N): ")
                        .strip()
                        .lower()
                    )
                    if shutdown_str == "y":
                        actions["shutdown"] = True
                        print(
                            f"    {Colors.RED}SHUTDOWN enabled - detector will stop after this event{Colors.RESET}"
                        )

            # Configure report
            if want_report_photos or want_report_only:
                report_id = (
                    input("    Report ID [traffic_report]: ").strip()
                    or "traffic_report"
                )
                actions["report"] = report_id

            if want_report_photos:
                max_photos_str = input("    Max photos [100]: ").strip() or "100"
                actions["frame_capture"] = {"max_photos": int(max_photos_str)}

            # JSON is implicit for reports, explicit for json-only
            if want_report_photos or want_report_only or want_json_only:
                actions["json_log"] = True

            event["actions"] = actions
            events.append(event)

            print(f"  {Colors.GREEN}Event '{name}' added{Colors.RESET}")

            # Inline validation warning
            if not actions:
                print(
                    f"  {Colors.YELLOW}Warning: No actions defined - event won't do anything{Colors.RESET}"
                )

        if events:
            self.config["events"] = events

    def _setup_reports(self):
        """Setup report configurations (generates HTML)."""
        events = self.config.get("events", [])

        # Collect unique report IDs from events
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

            # Get events for this report
            report_events = [
                e["name"]
                for e in events
                if e.get("actions", {}).get("report") == report_id
            ]
            print(f"    Events: {', '.join(report_events)}")

            title = input(
                f"    Title [{report_id.replace('_', ' ').title()}]: "
            ).strip()
            if not title:
                title = report_id.replace("_", " ").title()

            output_dir = input("    Output directory [reports]: ").strip() or "reports"

            # Check if any event has frame_capture
            has_photos = any(
                e.get("actions", {}).get("frame_capture")
                for e in events
                if e.get("actions", {}).get("report") == report_id
            )

            # Ask about photos if not already enabled via frame_capture
            if not has_photos:
                photos_str = (
                    input("    Include photos in report? (Y/n): ").strip().lower()
                )
                has_photos = photos_str != "n"

            annotate = False
            if has_photos:
                annotate_str = (
                    input("    Annotate photos with lines/zones? (Y/n): ")
                    .strip()
                    .lower()
                )
                annotate = annotate_str != "n"

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

        self.config["reports"] = reports

    def _setup_output(self):
        """Setup output directories and runtime settings."""
        print(f"\n{Colors.BOLD}--- Output Setup ---{Colors.RESET}")

        json_dir = input("JSON data directory [data]: ").strip() or "data"
        self.config["output"] = {"json_dir": json_dir}

        # Frame storage if needed
        has_frame_capture = any(
            e.get("actions", {}).get("frame_capture")
            for e in self.config.get("events", [])
        )
        if has_frame_capture:
            frames_dir = input("Frames directory [frames]: ").strip() or "frames"
            self.config["frame_storage"] = {"type": "local", "local_dir": frames_dir}

        # Console output
        self.config["console_output"] = {"enabled": True, "level": "detailed"}

        # Runtime settings
        print(f"\n{Colors.BOLD}--- Runtime Settings ---{Colors.RESET}")
        duration_str = input("Default duration in hours [0.5]: ").strip() or "0.5"
        duration = float(duration_str)
        duration_seconds = int(duration * 3600)

        self.config["runtime"] = {"queue_size": 500, "default_duration_hours": duration}

        # Apply duration to all frame_capture configs
        for event in self.config.get("events", []):
            frame_capture = event.get("actions", {}).get("frame_capture")
            if frame_capture:
                frame_capture["expected_duration_seconds"] = duration_seconds

    def _update_config_pointer(self, config_path: str):
        """Update config.yaml to point to the specified config."""
        pointer_content = f"""# Config pointer - specifies which config to use
use: {config_path}
"""
        with open("config.yaml", "w", encoding="utf-8") as f:
            f.write(pointer_content)

        print(f"{Colors.GRAY}config.yaml -> {config_path}{Colors.RESET}")

    def _run_preflight_stages(self, config_path: str):
        """Run validate, plan, and dry-run stages before detection."""
        # Load config from file
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = load_config_with_env(config)

        # Stage 1: Validate
        print(f"\n{Colors.GREEN}=== Validating ==={Colors.RESET}")
        result = validate_config_full(config)
        print_validation_result(result)
        if not result.valid:
            print(
                f"\n{Colors.RED}Validation failed. Fix errors before running.{Colors.RESET}"
            )
            sys.exit(1)
        input("Press Enter to continue...")

        # Stage 2: Plan
        print(f"\n{Colors.GREEN}=== Planning ==={Colors.RESET}")
        plan = build_plan(config)
        print_plan(plan)
        input("Press Enter to continue...")

        # Stage 3: Dry Run
        print(f"\n{Colors.GREEN}=== Dry Run ==={Colors.RESET}")
        sample_events = generate_sample_events(config)
        print(f"Generated {len(sample_events)} sample events from config")
        simulate_dry_run(config, sample_events)

    def _capture_preview(self, frame=None):
        """Capture a raw preview frame."""
        if frame is None:
            if self.cap is None:
                return
            ret, frame = self.cap.read()
            if not ret:
                return

        preview_path = os.path.join(self.preview_dir, "preview.jpg")
        cv2.imwrite(preview_path, frame)

    def _capture_annotated_preview(
        self, lines: list[dict] | None = None, zones: list[dict] | None = None
    ):
        """Capture preview with annotations."""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        height, width = frame.shape[:2]

        # Draw lines (yellow)
        if lines:
            for line in lines:
                line_type = line.get("type", "vertical")
                position_pct = line.get("position_pct", 50)
                description = line.get("description", "")

                if line_type == "vertical":
                    x = int(width * position_pct / 100)
                    cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        description,
                        (x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    # Show percentage
                    cv2.putText(
                        frame,
                        f"{position_pct}%",
                        (x + 5, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                else:
                    y = int(height * position_pct / 100)
                    cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        description,
                        (10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{position_pct}%",
                        (10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

        # Draw zones (cyan)
        if zones:
            for zone in zones:
                x1 = int(width * zone.get("x1_pct", 0) / 100)
                y1 = int(height * zone.get("y1_pct", 0) / 100)
                x2 = int(width * zone.get("x2_pct", 100) / 100)
                y2 = int(height * zone.get("y2_pct", 100) / 100)
                description = zone.get("description", "")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    description,
                    (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

        # Save
        preview_path = os.path.join(self.preview_dir, "preview.jpg")
        cv2.imwrite(preview_path, frame)

    def _cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.http_server is not None:
            self.http_server.terminate()
            try:
                self.http_server.wait(timeout=2)
            except Exception:
                self.http_server.kill()
            self.http_server = None


def run_builder() -> str | None:
    """Run the config builder and return the config path."""
    builder = ConfigBuilder()
    return builder.run()


def run_editor(config_path: str) -> str | None:
    """Run the config editor and return the config path."""
    builder = ConfigBuilder()
    return builder.run_edit(config_path)


if __name__ == "__main__":
    run_builder()
