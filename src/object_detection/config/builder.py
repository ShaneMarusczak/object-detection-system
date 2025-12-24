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

    def __init__(self):
        self.config: dict = {}
        self.camera_url: str = ""
        self.cap: cv2.VideoCapture | None = None
        self.preview_dir: str = "data"
        self.http_server: subprocess.Popen | None = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.config_path: str | None = None  # Track source file for edits

    def run(self) -> str | None:
        """Run the config builder wizard. Returns config filename or None."""
        try:
            self._print_header()

            # Setup camera first, then start preview server
            if not self._setup_camera():
                return None

            self._start_preview_server()

            self._setup_detection()
            self._setup_lines()
            self._setup_zones()
            self._setup_events()
            self._setup_pdf_reports()
            self._setup_digests()
            self._setup_email()
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

        # Set as default
        set_default = input("Set as default? (Y/n): ").strip().lower()
        if set_default != "n":
            self._set_as_default(filepath)

        if run_after:
            self._cleanup()
            print(f"\n{Colors.CYAN}Starting detection...{Colors.RESET}\n")
            # Run with the saved config directly (not the default)
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

    def _setup_lines(self):
        """Setup detection lines with visual preview."""
        print(f"\n{Colors.BOLD}--- Lines Setup ---{Colors.RESET}")
        print(f"{Colors.GRAY}Lines detect objects crossing a boundary{Colors.RESET}")

        # Start with existing lines (for edit mode)
        lines = list(self.config.get("lines", []))
        zones = self.config.get("zones", [])

        while True:
            add = input("\nAdd a line? (Y/n): ").strip().lower()
            if add == "n":
                break

            # Line type
            print("  Line type:")
            print("    1. vertical (left-right crossing)")
            print("    2. horizontal (top-bottom crossing)")
            type_choice = input("  Choice [1]: ").strip() or "1"
            line_type = "vertical" if type_choice == "1" else "horizontal"

            # Position
            if line_type == "vertical":
                pos_str = input("  Position % from left [50]: ").strip() or "50"
            else:
                pos_str = input("  Position % from top [50]: ").strip() or "50"
            position = int(pos_str)

            # Description
            desc = input("  Description: ").strip() or f"Line {len(lines) + 1}"

            lines.append(
                {"type": line_type, "position_pct": position, "description": desc}
            )

            # Capture annotated preview
            self._capture_annotated_preview(lines=lines, zones=zones)
            print(f"  {Colors.GREEN}Preview updated{Colors.RESET} - refresh browser")

            # Adjust option
            while True:
                action = (
                    input("  [c] Capture  [a] Adjust  [d] Delete  [n] Next: ")
                    .strip()
                    .lower()
                )
                if action == "c":
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    print(f"  {Colors.GREEN}Preview updated{Colors.RESET}")
                elif action == "a":
                    if line_type == "vertical":
                        pos_str = input(
                            f"  New position % from left [{position}]: "
                        ).strip() or str(position)
                    else:
                        pos_str = input(
                            f"  New position % from top [{position}]: "
                        ).strip() or str(position)
                    position = int(pos_str)
                    lines[-1]["position_pct"] = position
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    print(f"  {Colors.GREEN}Preview updated{Colors.RESET}")
                elif action == "d":
                    deleted = lines.pop()
                    print(
                        f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                    )
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    break  # Go back to "Add a line?" prompt
                elif action == "n" or action == "":
                    break

        if lines:
            self.config["lines"] = lines

    def _setup_zones(self):
        """Setup detection zones with visual preview."""
        print(f"\n{Colors.BOLD}--- Zones Setup ---{Colors.RESET}")
        print(
            f"{Colors.GRAY}Zones detect objects entering/dwelling in an area{Colors.RESET}"
        )

        # Start with existing zones and lines (for edit mode)
        zones = list(self.config.get("zones", []))
        lines = self.config.get("lines", [])

        while True:
            add = input("\nAdd a zone? (y/N): ").strip().lower()
            if add != "y":
                break

            # Zone bounds with validation
            while True:
                print("  Zone bounds (percentages):")
                x1 = int(input("    Left edge % [0]: ").strip() or "0")
                y1 = int(input("    Top edge % [0]: ").strip() or "0")
                x2 = int(input("    Right edge % [100]: ").strip() or "100")
                y2 = int(input("    Bottom edge % [100]: ").strip() or "100")

                # Validate bounds
                if x2 <= x1:
                    print(
                        f"  {Colors.RED}Error: Right edge ({x2}%) must be greater than left edge ({x1}%){Colors.RESET}"
                    )
                    continue
                if y2 <= y1:
                    print(
                        f"  {Colors.RED}Error: Bottom edge ({y2}%) must be greater than top edge ({y1}%){Colors.RESET}"
                    )
                    continue
                break

            # Description
            desc = input("  Description: ").strip() or f"Zone {len(zones) + 1}"

            zones.append(
                {
                    "x1_pct": x1,
                    "y1_pct": y1,
                    "x2_pct": x2,
                    "y2_pct": y2,
                    "description": desc,
                }
            )

            # Capture annotated preview
            self._capture_annotated_preview(lines=lines, zones=zones)
            print(f"  {Colors.GREEN}Preview updated{Colors.RESET} - refresh browser")

            # Adjust option
            while True:
                action = (
                    input("  [c] Capture  [a] Adjust  [d] Delete  [n] Next: ")
                    .strip()
                    .lower()
                )
                if action == "c":
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    print(f"  {Colors.GREEN}Preview updated{Colors.RESET}")
                elif action == "a":
                    while True:
                        x1 = int(
                            input(f"    Left edge % [{zones[-1]['x1_pct']}]: ").strip()
                            or str(zones[-1]["x1_pct"])
                        )
                        y1 = int(
                            input(f"    Top edge % [{zones[-1]['y1_pct']}]: ").strip()
                            or str(zones[-1]["y1_pct"])
                        )
                        x2 = int(
                            input(f"    Right edge % [{zones[-1]['x2_pct']}]: ").strip()
                            or str(zones[-1]["x2_pct"])
                        )
                        y2 = int(
                            input(
                                f"    Bottom edge % [{zones[-1]['y2_pct']}]: "
                            ).strip()
                            or str(zones[-1]["y2_pct"])
                        )
                        if x2 <= x1:
                            print(
                                f"  {Colors.RED}Error: Right ({x2}%) must be > left ({x1}%){Colors.RESET}"
                            )
                            continue
                        if y2 <= y1:
                            print(
                                f"  {Colors.RED}Error: Bottom ({y2}%) must be > top ({y1}%){Colors.RESET}"
                            )
                            continue
                        break
                    zones[-1] = {
                        "x1_pct": x1,
                        "y1_pct": y1,
                        "x2_pct": x2,
                        "y2_pct": y2,
                        "description": desc,
                    }
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    print(f"  {Colors.GREEN}Preview updated{Colors.RESET}")
                elif action == "d":
                    deleted = zones.pop()
                    print(
                        f"  {Colors.YELLOW}Deleted: {deleted.get('description')}{Colors.RESET}"
                    )
                    self._capture_annotated_preview(lines=lines, zones=zones)
                    break  # Go back to "Add a zone?" prompt
                elif action == "n" or action == "":
                    break

        if zones:
            self.config["zones"] = zones

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

            # Actions
            print("  Actions:")
            actions = {}

            # PDF report
            pdf = input("    Add to PDF report? (Y/n): ").strip().lower()
            if pdf != "n":
                report_id = (
                    input("    Report ID [traffic_report]: ").strip()
                    or "traffic_report"
                )
                actions["pdf_report"] = report_id

                # Frame capture for photos
                capture = (
                    input("    Capture photos for report? (Y/n): ").strip().lower()
                )
                if capture != "n":
                    max_photos_str = input("    Max photos [100]: ").strip() or "100"
                    # Duration comes from runtime settings, applied later
                    actions["frame_capture"] = {"max_photos": int(max_photos_str)}

            # Immediate email
            email_imm = input("    Send immediate email? (y/N): ").strip().lower()
            if email_imm == "y":
                actions["email_immediate"] = True

            # Email digest
            email_dig = input("    Add to email digest? (y/N): ").strip().lower()
            if email_dig == "y":
                digest_id = (
                    input("    Digest ID [daily_digest]: ").strip() or "daily_digest"
                )
                actions["email_digest"] = digest_id

            event["actions"] = actions
            events.append(event)

            print(f"  {Colors.GREEN}Event '{name}' added{Colors.RESET}")

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

    def _set_as_default(self, config_path: str):
        """Set config as default by updating config.yaml's use: pointer."""
        pointer_content = f"""# Active configuration pointer
# Change this to switch configs, or run: python -m object_detection --build-config
use: {config_path}
"""
        with open("config.yaml", "w", encoding="utf-8") as f:
            f.write(pointer_content)

        print(
            f"{Colors.GREEN}Set as default:{Colors.RESET} config.yaml now points to {config_path}"
        )
        print(f"{Colors.GRAY}Run with: ./run.sh -y{Colors.RESET}")

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
