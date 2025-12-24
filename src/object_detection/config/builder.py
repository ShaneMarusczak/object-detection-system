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

import cv2
import yaml

from ..processor.coco_classes import COCO_CLASSES
from .planner import (
    load_config_with_env,
    build_plan,
    print_plan,
    print_validation_result,
    simulate_dry_run,
    generate_sample_events,
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
            self._setup_pdf_reports()
            self._setup_digests()
            self._setup_email()

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
            lines=self.config.get("lines", []),
            zones=self.config.get("zones", [])
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
                summary += f" +{len(lines)-3} more"
            return f"{len(lines)}: {summary}"

        elif section == "zones":
            zones = self.config.get("zones", [])
            if not zones:
                return "none"
            descs = [z.get("description", "?")[:12] for z in zones[:3]]
            summary = ", ".join(descs)
            if len(zones) > 3:
                summary += f" +{len(zones)-3} more"
            return f"{len(zones)}: {summary}"

        elif section == "events":
            events = self.config.get("events", [])
            if not events:
                return "none"
            return f"{len(events)} defined"

        elif section == "pdf_reports":
            reports = self.config.get("pdf_reports", [])
            if not reports:
                return "disabled"
            ids = [r.get("id", "?") for r in reports]
            return ", ".join(ids)

        elif section == "digests":
            digests = self.config.get("digests", [])
            if not digests:
                return "disabled"
            ids = [d.get("id", "?") for d in digests]
            return ", ".join(ids)

        elif section == "email":
            email = self.config.get("email", {})
            if not email:
                return "not configured"
            to_addrs = email.get("to_addresses", [])
            if to_addrs:
                return (
                    to_addrs[0]
                    if len(to_addrs) == 1
                    else f"{to_addrs[0]} +{len(to_addrs)-1}"
                )
            return "configured"

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

        # Check for email actions without email config
        has_email_action = any(
            e.get("actions", {}).get("email_immediate")
            or e.get("actions", {}).get("email_digest")
            for e in events
        )
        if has_email_action and not self.config.get("email"):
            warnings.append("Email actions configured but email settings missing")

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
            ("pdf_reports", "PDF Reports"),
            ("digests", "Digests"),
            ("email", "Email"),
            ("output", "Output"),
        ]

        while True:
            # Show warnings if any
            warnings = self._get_config_warnings()
            if warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
                for w in warnings:
                    print(f"  {Colors.YELLOW}! {w}{Colors.RESET}")

            print(f"\n{Colors.BOLD}Which section to edit?{Colors.RESET}")
            for i, (key, label) in enumerate(sections, 1):
                summary = self._get_section_summary(key)
                print(f"  {i}. {label} ({summary})")

            print(f"  {Colors.GREEN}r. Save and run{Colors.RESET}")
            print(f"  {Colors.CYAN}s. Save and exit{Colors.RESET}")
            print(f"  {Colors.YELLOW}q. Quit without saving{Colors.RESET}")

            choice = input("Choice: ").strip().lower()

            if choice == "q":
                confirm = input("Are you sure? (y/N): ").strip().lower()
                if confirm == "y":
                    return None
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
        elif section == "pdf_reports":
            self._setup_pdf_reports()
        elif section == "digests":
            self._setup_digests()
        elif section == "email":
            self._setup_email()
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
                    zones=self.config.get("zones", [])
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
                    lines=[],
                    zones=self.config.get("zones", [])
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
                    zones=self.config.get("zones", [])
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
                    lines=self.config.get("lines", []),
                    zones=[]
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
            os.execvp(
                "python", ["python", "-m", "object_detection", "-c", filepath]
            )

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

    def _setup_detection(self):
        """Setup detection parameters."""
        print(f"\n{Colors.BOLD}--- Detection Settings ---{Colors.RESET}")

        # Model
        default_model = "yolo11n.pt"
        model = input(f"Model file [{default_model}]: ").strip() or default_model

        # Confidence threshold
        default_conf = "0.3"
        conf_str = (
            input(f"Confidence threshold [{default_conf}]: ").strip() or default_conf
        )
        conf = float(conf_str)

        self.config["detection"] = {"model_file": model, "confidence_threshold": conf}

    def _parse_line_input(self, line_input: str, line_count: int) -> dict | None:
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
        print(f"{Colors.GRAY}Format: h/v [position%] [name]  (e.g. 'v 50 Driveway'){Colors.RESET}")

        # Start with existing lines (for edit mode)
        lines = list(self.config.get("lines", []))
        zones = self.config.get("zones", [])

        while True:
            line_input = input("\nAdd line (or Enter when done): ").strip()
            if not line_input:
                break

            # Parse shorthand input
            parsed = self._parse_line_input(line_input, len(lines))
            if not parsed:
                print(f"  {Colors.RED}Invalid format. Use: h/v [%] [name]{Colors.RESET}")
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
                desc = input(f"  Description [{default_desc}]: ").strip() or default_desc

            lines.append(
                {"type": line_type, "position_pct": position, "description": desc}
            )

            # Show confirmation and capture preview
            type_label = "vertical" if line_type == "vertical" else "horizontal"
            print(f"  {Colors.GREEN}✓ {type_label} at {position}% \"{desc}\"{Colors.RESET}")
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
                    print(f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}")
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
        print(f"{Colors.GRAY}Zones detect objects entering/dwelling in an area{Colors.RESET}")
        print(f"{Colors.GRAY}Format: left% top% right% bottom% [name]{Colors.RESET}")
        print(f"{Colors.GRAY}  e.g. '0 0 50 100 Left Half' or '25 25 75 75 Center'{Colors.RESET}")

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
                print(f"  {Colors.RED}Invalid format. Use: left top right bottom [name]{Colors.RESET}")
                continue

            x1, y1, x2, y2 = parsed["x1_pct"], parsed["y1_pct"], parsed["x2_pct"], parsed["y2_pct"]
            desc = parsed.get("description")

            # Validate bounds
            if x2 <= x1:
                print(f"  {Colors.RED}Error: right ({x2}%) must be > left ({x1}%){Colors.RESET}")
                continue
            if y2 <= y1:
                print(f"  {Colors.RED}Error: bottom ({y2}%) must be > top ({y1}%){Colors.RESET}")
                continue

            # Prompt for missing description
            if not desc:
                default_desc = f"Zone {len(zones) + 1}"
                desc = input(f"  Description [{default_desc}]: ").strip() or default_desc

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
            print(f"  {Colors.GREEN}✓ {x1},{y1} to {x2},{y2} \"{desc}\"{Colors.RESET}")
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
                    print(f"  {Colors.GREEN}✓ Updated to {x1},{y1} to {x2},{y2}{Colors.RESET}")
                elif action == "d":
                    deleted = zones.pop()
                    print(f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}")
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

        if not lines and not zones:
            print(
                f"{Colors.YELLOW}No lines or zones defined - skipping events{Colors.RESET}"
            )
            return

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

            # Event type
            print("    Event type:")
            print("      1. LINE_CROSS (object crosses a line)")
            if zones:
                print("      2. ZONE_ENTER (object enters a zone)")
                print("      3. ZONE_DWELL (object stays in zone)")
                print("      4. NIGHTTIME_CAR (headlight blob detection in zone)")
            type_choice = input("    Choice [1]: ").strip() or "1"

            is_nighttime_event = False

            if type_choice == "1":
                match["event_type"] = "LINE_CROSS"
                if lines:
                    print("    Which line?")
                    for i, line in enumerate(lines, 1):
                        print(f"      {i}. {line['description']}")
                    line_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    match["line"] = lines[line_choice]["description"]
            elif type_choice == "2":
                match["event_type"] = "ZONE_ENTER"
                if zones:
                    print("    Which zone?")
                    for i, zone in enumerate(zones, 1):
                        print(f"      {i}. {zone['description']}")
                    zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    match["zone"] = zones[zone_choice]["description"]
            elif type_choice == "3":
                match["event_type"] = "ZONE_DWELL"
                if zones:
                    print("    Which zone?")
                    for i, zone in enumerate(zones, 1):
                        print(f"      {i}. {zone['description']}")
                    zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                    match["zone"] = zones[zone_choice]["description"]
            elif type_choice == "4" and zones:
                match["event_type"] = "NIGHTTIME_CAR"
                is_nighttime_event = True
                print("    Which zone to monitor for headlights?")
                for i, zone in enumerate(zones, 1):
                    print(f"      {i}. {zone['description']}")
                zone_choice = int(input("    Choice [1]: ").strip() or "1") - 1
                match["zone"] = zones[zone_choice]["description"]

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

            # Object classes with validation (skip for NIGHTTIME_CAR)
            if not is_nighttime_event:
                valid_classes = set(COCO_CLASSES.values())
                while True:
                    print("    Object classes (comma-separated):")
                    print(f"    Common: {', '.join(COMMON_CLASSES)}")
                    classes_str = (
                        input("    Classes [car, truck, bus]: ").strip()
                        or "car, truck, bus"
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

            event["match"] = match

            # Actions - outcome based selection
            print("  Actions:")
            print("    1. PDF with photos")
            print("    2. PDF stats only")
            print("    3. Email alert")
            print("    4. Email digest")
            print("    5. JSON only")
            action_input = input("  Choose (e.g. 1,3) [1]: ").strip() or "1"

            # Parse choices
            choices = {c.strip() for c in action_input.replace(" ", ",").split(",")}
            actions = {}

            want_pdf_photos = "1" in choices
            want_pdf_only = "2" in choices
            want_email = "3" in choices
            want_digest = "4" in choices
            want_json_only = "5" in choices

            # Build enabled list for display
            enabled = []

            # PDF with photos (implies json)
            if want_pdf_photos:
                enabled.extend(["pdf_report", "frame_capture", "json_log"])
            # PDF only (implies json)
            if want_pdf_only:
                if "pdf_report" not in enabled:
                    enabled.append("pdf_report")
                if "json_log" not in enabled:
                    enabled.append("json_log")
            # Email alert (implies json)
            if want_email:
                enabled.append("email_immediate")
                if "json_log" not in enabled:
                    enabled.append("json_log")
            # Digest (implies json)
            if want_digest:
                enabled.append("email_digest")
                if "json_log" not in enabled:
                    enabled.append("json_log")
            # JSON only
            if want_json_only:
                if "json_log" not in enabled:
                    enabled.append("json_log")

            if enabled:
                print(f"    {Colors.GRAY}→ Enables: {', '.join(enabled)}{Colors.RESET}")

            # Configure each selected action
            if want_pdf_photos or want_pdf_only:
                report_id = (
                    input("    PDF report ID [traffic_report]: ").strip()
                    or "traffic_report"
                )
                actions["pdf_report"] = report_id

            if want_pdf_photos:
                max_photos_str = input("    Max photos [100]: ").strip() or "100"
                actions["frame_capture"] = {"max_photos": int(max_photos_str)}

            if want_email:
                actions["email_immediate"] = True

            if want_digest:
                digest_id = (
                    input("    Digest ID [daily_digest]: ").strip() or "daily_digest"
                )
                actions["email_digest"] = digest_id

            # JSON is implicit for all except json-only where it's explicit
            if enabled:
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

    def _setup_pdf_reports(self):
        """Setup PDF report configurations."""
        events = self.config.get("events", [])

        # Collect unique report IDs from events
        report_ids = set()
        for event in events:
            report_id = event.get("actions", {}).get("pdf_report")
            if report_id:
                report_ids.add(report_id)

        if not report_ids:
            return

        print(f"\n{Colors.BOLD}--- PDF Reports Setup ---{Colors.RESET}")

        pdf_reports = []
        for report_id in report_ids:
            print(f"\n  Report: {report_id}")

            # Get events for this report
            report_events = [
                e["name"]
                for e in events
                if e.get("actions", {}).get("pdf_report") == report_id
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
                if e.get("actions", {}).get("pdf_report") == report_id
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

            pdf_reports.append(
                {
                    "id": report_id,
                    "title": title,
                    "output_dir": output_dir,
                    "events": report_events,
                    "photos": has_photos,
                    "annotate": annotate,
                }
            )

        self.config["pdf_reports"] = pdf_reports

    def _setup_digests(self):
        """Setup email digest configurations."""
        events = self.config.get("events", [])

        # Collect unique digest IDs from events
        digest_ids = set()
        for event in events:
            digest_id = event.get("actions", {}).get("email_digest")
            if digest_id:
                digest_ids.add(digest_id)

        if not digest_ids:
            return

        print(f"\n{Colors.BOLD}--- Email Digests Setup ---{Colors.RESET}")

        digests = []
        for digest_id in digest_ids:
            print(f"\n  Digest: {digest_id}")

            # Get events for this digest
            digest_events = [
                e["name"]
                for e in events
                if e.get("actions", {}).get("email_digest") == digest_id
            ]
            print(f"    Events: {', '.join(digest_events)}")

            # Schedule
            print("    Schedule type:")
            print("      1. interval (every N minutes)")
            print("      2. daily (at specific time)")
            sched_choice = input("    Choice [1]: ").strip() or "1"

            schedule = {}
            if sched_choice == "1":
                interval = input("    Interval in minutes [60]: ").strip() or "60"
                schedule = {"interval_minutes": int(interval)}
            else:
                time_str = input("    Time (HH:MM) [08:00]: ").strip() or "08:00"
                schedule = {"daily_at": time_str}

            # Include photos?
            photos = input("    Include photos in digest? (y/N): ").strip().lower()

            digests.append(
                {
                    "id": digest_id,
                    "events": digest_events,
                    "schedule": schedule,
                    "photos": photos == "y",
                }
            )

        self.config["digests"] = digests

    def _setup_email(self):
        """Setup email configuration (SMTP settings)."""
        events = self.config.get("events", [])

        # Check if any email actions are configured
        has_email = any(
            e.get("actions", {}).get("email_immediate")
            or e.get("actions", {}).get("email_digest")
            for e in events
        )

        if not has_email:
            return

        print(f"\n{Colors.BOLD}--- Email Configuration ---{Colors.RESET}")
        print(
            f"{Colors.GRAY}Configure SMTP settings for email notifications{Colors.RESET}"
        )

        smtp_host = input("  SMTP host [smtp.gmail.com]: ").strip() or "smtp.gmail.com"
        smtp_port = input("  SMTP port [587]: ").strip() or "587"
        smtp_user = input("  SMTP username (email): ").strip()
        smtp_pass = input("  SMTP password (app password): ").strip()
        from_addr = input(f"  From address [{smtp_user}]: ").strip() or smtp_user
        to_addr = input("  To address(es) (comma-separated): ").strip()

        self.config["email"] = {
            "smtp_host": smtp_host,
            "smtp_port": int(smtp_port),
            "smtp_user": smtp_user,
            "smtp_pass": smtp_pass,
            "from_address": from_addr,
            "to_addresses": [addr.strip() for addr in to_addr.split(",")],
        }

        print(f"  {Colors.GREEN}Email configured{Colors.RESET}")

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
            print(f"\n{Colors.RED}Validation failed. Fix errors before running.{Colors.RESET}")
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
