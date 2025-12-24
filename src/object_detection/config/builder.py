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

            # Save and optionally run
            return self._save_config()

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled{Colors.RESET}")
            return None
        finally:
            self._cleanup()

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

        lines = []

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
            self._capture_annotated_preview(lines=lines)
            print(f"  {Colors.GREEN}Preview updated{Colors.RESET} - refresh browser")

            # Adjust option
            while True:
                action = (
                    input("  [c] Capture again  [a] Adjust position  [n] Next: ")
                    .strip()
                    .lower()
                )
                if action == "c":
                    self._capture_annotated_preview(lines=lines)
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
                    self._capture_annotated_preview(lines=lines)
                    print(f"  {Colors.GREEN}Preview updated{Colors.RESET}")
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

        zones = []
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
                    input("  [c] Capture again  [a] Adjust bounds  [n] Next: ")
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

    def _save_config(self) -> str | None:
        """Save config to YAML file with options for what to do next."""
        print(f"\n{Colors.BOLD}--- Finish ---{Colors.RESET}")

        # Generate default name from timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        default_name = f"config_{timestamp}"

        name = input(f"Config name [{default_name}]: ").strip() or default_name

        # Ensure .yaml extension
        if not name.endswith(".yaml") and not name.endswith(".yml"):
            filename = f"{name}.yaml"
        else:
            filename = name

        filepath = os.path.join("configs", filename)

        # Show options
        print(f"\n{Colors.BOLD}What next?{Colors.RESET}")
        print("  1. Save and run (auto)   - quick start, no pauses")
        print("  2. Save and run (verify) - pause between steps to check")
        print("  3. Save and exit")
        print("  4. Exit without saving")
        choice = input("Choice [3]: ").strip() or "3"

        # Handle exit without saving
        if choice == "4":
            print(f"{Colors.YELLOW}Exiting without saving{Colors.RESET}")
            return None

        # Save the config (options 1, 2, 3)
        os.makedirs("configs", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"{Colors.GREEN}Saved:{Colors.RESET} {filepath}")

        # Handle based on choice
        if choice in ("1", "2"):
            # Running - auto-set as default (no need to ask)
            self._set_as_default(filepath)
            self._cleanup()

            if choice == "1":
                print(
                    f"\n{Colors.CYAN}Starting detection (auto mode)...{Colors.RESET}\n"
                )
                os.execvp("./run.sh", ["./run.sh", "-sy"])  # skip menu + auto
            else:
                print(
                    f"\n{Colors.CYAN}Starting detection (verify mode)...{Colors.RESET}\n"
                )
                os.execvp("./run.sh", ["./run.sh", "-s"])  # skip menu only

        else:  # choice == '3' - save and exit
            # Only ask about default if just saving
            set_default = input("\nSet as default? (Y/n): ").strip().lower()
            if set_default != "n":
                self._set_as_default(filepath)

            print(f"\n{Colors.GRAY}Run later with: ./run.sh{Colors.RESET}")

        return filepath

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


if __name__ == "__main__":
    run_builder()
