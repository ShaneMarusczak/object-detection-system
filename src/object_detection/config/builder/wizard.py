"""
Config Builder Wizard - Main orchestration class.

Ties together all builder modules to provide the interactive configuration
wizard for creating and editing detection configurations.
"""

import subprocess

import questionary
from questionary import Choice

from .events import setup_events
from .io import load_config, save_config
from .preview import (
    capture_annotated_preview,
    get_local_ip,
    start_preview_server,
    stop_preview_server,
)
from .prompts import PROMPT_STYLE, Colors
from .sections import (
    setup_camera,
    setup_camera_from_config,
    setup_detection,
    setup_lines,
    setup_output,
    setup_reports,
    setup_zones,
)


class ConfigBuilder:
    """Interactive config builder with live preview."""

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
        self.state: dict = {
            "cap": None,
            "camera_url": "",
            "frame_width": 0,
            "frame_height": 0,
            "preview_dir": "data",
            "model_classes": [],
        }
        self.http_server: subprocess.Popen | None = None
        self.config_path: str | None = None
        self.current_step: int = 0

    def run(self) -> str | None:
        """Run the config builder wizard. Returns config filename or None."""
        try:
            self._print_header()

            # Step 0: Camera
            self.current_step = 0
            self._print_progress()
            if not setup_camera(self.config, self.state):
                return None
            self.http_server = start_preview_server(self.state["preview_dir"])

            # Step 1: Detection
            self.current_step = 1
            self._print_progress()
            setup_detection(self.config, self.state)

            # Step 2: Lines
            self.current_step = 2
            self._print_progress()
            setup_lines(self.config, self.state)

            # Step 3: Zones
            self.current_step = 3
            self._print_progress()
            setup_zones(self.config, self.state)

            # Step 4: Events
            self.current_step = 4
            self._print_progress()
            setup_events(self.config, self.state.get("model_classes", []))
            setup_reports(self.config)

            # Step 5: Output
            self.current_step = 5
            self._print_progress()
            setup_output(self.config)

            # Final review
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
            loaded = load_config(config_path)
            if loaded is None:
                return None
            self.config = loaded
            self.config_path = config_path

            self._print_edit_header(config_path)

            if not setup_camera_from_config(self.config, self.state):
                return None

            self.http_server = start_preview_server(self.state["preview_dir"])

            return self._edit_menu_loop()

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

    def _print_edit_header(self, config_path: str):
        """Print edit mode header."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}=== Config Editor ==={Colors.RESET}")
        print(f"Editing: {Colors.GREEN}{config_path}{Colors.RESET}")
        print()

    def _print_progress(self):
        """Print wizard progress indicator."""
        parts = []
        for i, (_, label) in enumerate(self.WIZARD_STEPS):
            if i < self.current_step:
                parts.append(f"{Colors.GREEN}[✓] {label}{Colors.RESET}")
            elif i == self.current_step:
                parts.append(f"{Colors.CYAN}[•] {label}{Colors.RESET}")
            else:
                parts.append(f"{Colors.GRAY}[ ] {label}{Colors.RESET}")
        print("  ".join(parts))
        print()

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
            warnings = self._get_config_warnings()
            if warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
                for w in warnings:
                    print(f"  {Colors.YELLOW}! {w}{Colors.RESET}")

            local_ip = get_local_ip()
            print(
                f"\n{Colors.GRAY}Preview: http://{local_ip}:8000/preview.jpg{Colors.RESET}"
            )

            choices = []
            for key, label in sections:
                summary = self._get_section_summary(key)
                choices.append(Choice(title=f"{label} ({summary})", value=key))

            choices.append(Choice(title="Refresh preview", value="refresh"))
            choices.append(Choice(title="Save and run", value="run"))
            choices.append(Choice(title="Save and exit", value="save"))
            choices.append(Choice(title="Quit without saving", value="quit"))

            choice = questionary.select(
                "Which section to edit?",
                choices=choices,
                style=PROMPT_STYLE,
            ).ask()

            if choice is None:
                return None

            if choice == "quit":
                if questionary.confirm(
                    "Are you sure you want to quit without saving?",
                    default=False,
                    style=PROMPT_STYLE,
                ).ask():
                    return None
                continue

            if choice == "refresh":
                cap = self.state.get("cap")
                preview_dir = self.state.get("preview_dir", "data")
                capture_annotated_preview(
                    cap,
                    preview_dir,
                    lines=self.config.get("lines", []),
                    zones=self.config.get("zones", []),
                )
                print(f"{Colors.GREEN}Preview refreshed{Colors.RESET}")
                continue

            if choice == "save":
                return save_config(
                    self.config,
                    self.config_path,
                    run_after=False,
                    cleanup_callback=self._cleanup,
                )

            if choice == "run":
                return save_config(
                    self.config,
                    self.config_path,
                    run_after=True,
                    cleanup_callback=self._cleanup,
                )

            self._edit_section(choice)

    def _edit_section(self, section: str):
        """Edit a specific section."""
        title = section.replace("_", " ").title()
        print(f"\n{Colors.BOLD}--- Edit {title} ---{Colors.RESET}")

        if section == "camera":
            setup_camera(self.config, self.state)
        elif section == "detection":
            setup_detection(self.config, self.state)
        elif section == "lines":
            self._edit_lines()
        elif section == "zones":
            self._edit_zones()
        elif section == "events":
            self._edit_events()
        elif section == "reports":
            setup_reports(self.config)
        elif section == "output":
            setup_output(self.config)

    def _edit_lines(self):
        """Edit lines with options to keep, modify, or add."""
        lines = self.config.get("lines", [])
        cap = self.state.get("cap")
        preview_dir = self.state.get("preview_dir", "data")

        if lines:
            print(f"Current lines ({len(lines)}):")
            for i, line in enumerate(lines, 1):
                print(
                    f"  {i}. {line.get('description')} ({line.get('type')}, {line.get('position_pct')}%)"
                )

            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    Choice(title="Keep all", value="keep"),
                    Choice(title="Delete some", value="delete"),
                    Choice(title="Add more", value="add"),
                    Choice(title="Start fresh", value="fresh"),
                ],
                style=PROMPT_STYLE,
            ).ask()

            if choice == "keep" or choice is None:
                return
            elif choice == "delete":
                line_choices = [
                    Choice(
                        title=f"{ln.get('description')} ({ln.get('type')}, {ln.get('position_pct')}%)",
                        value=i,
                    )
                    for i, ln in enumerate(lines)
                ]
                to_delete = questionary.checkbox(
                    "Select lines to delete:",
                    choices=line_choices,
                    style=PROMPT_STYLE,
                ).ask()

                if to_delete:
                    self.config["lines"] = [
                        ln for i, ln in enumerate(lines) if i not in to_delete
                    ]
                    print(
                        f"{Colors.GREEN}Deleted {len(to_delete)} line(s){Colors.RESET}"
                    )
                    capture_annotated_preview(
                        cap,
                        preview_dir,
                        lines=self.config.get("lines", []),
                        zones=self.config.get("zones", []),
                    )
                return
            elif choice == "add":
                pass
            elif choice == "fresh":
                self.config["lines"] = []
                capture_annotated_preview(
                    cap, preview_dir, lines=[], zones=self.config.get("zones", [])
                )

        setup_lines(self.config, self.state)

    def _edit_zones(self):
        """Edit zones with options to keep, modify, or add."""
        zones = self.config.get("zones", [])
        cap = self.state.get("cap")
        preview_dir = self.state.get("preview_dir", "data")

        if zones:
            print(f"Current zones ({len(zones)}):")
            for i, zone in enumerate(zones, 1):
                print(
                    f"  {i}. {zone.get('description')} ({zone.get('x1_pct')}-{zone.get('x2_pct')}%, {zone.get('y1_pct')}-{zone.get('y2_pct')}%)"
                )

            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    Choice(title="Keep all", value="keep"),
                    Choice(title="Delete some", value="delete"),
                    Choice(title="Add more", value="add"),
                    Choice(title="Start fresh", value="fresh"),
                ],
                style=PROMPT_STYLE,
            ).ask()

            if choice == "keep" or choice is None:
                return
            elif choice == "delete":
                zone_choices = [
                    Choice(
                        title=f"{z.get('description')} ({z.get('x1_pct')}-{z.get('x2_pct')}%, {z.get('y1_pct')}-{z.get('y2_pct')}%)",
                        value=i,
                    )
                    for i, z in enumerate(zones)
                ]
                to_delete = questionary.checkbox(
                    "Select zones to delete:",
                    choices=zone_choices,
                    style=PROMPT_STYLE,
                ).ask()

                if to_delete:
                    self.config["zones"] = [
                        z for i, z in enumerate(zones) if i not in to_delete
                    ]
                    print(
                        f"{Colors.GREEN}Deleted {len(to_delete)} zone(s){Colors.RESET}"
                    )
                    capture_annotated_preview(
                        cap,
                        preview_dir,
                        lines=self.config.get("lines", []),
                        zones=self.config.get("zones", []),
                    )
                return
            elif choice == "add":
                pass
            elif choice == "fresh":
                self.config["zones"] = []
                capture_annotated_preview(
                    cap, preview_dir, lines=self.config.get("lines", []), zones=[]
                )

        setup_zones(self.config, self.state)

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

            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    Choice(title="Keep all", value="keep"),
                    Choice(title="Delete some", value="delete"),
                    Choice(title="Add more", value="add"),
                    Choice(title="Start fresh", value="fresh"),
                ],
                style=PROMPT_STYLE,
            ).ask()

            if choice == "keep" or choice is None:
                return
            elif choice == "delete":
                event_choices = [
                    Choice(
                        title=f"{e.get('name')} ({e.get('match', {}).get('event_type', '?')})",
                        value=i,
                    )
                    for i, e in enumerate(events)
                ]
                to_delete = questionary.checkbox(
                    "Select events to delete:",
                    choices=event_choices,
                    style=PROMPT_STYLE,
                ).ask()

                if to_delete:
                    self.config["events"] = [
                        e for i, e in enumerate(events) if i not in to_delete
                    ]
                    print(
                        f"{Colors.GREEN}Deleted {len(to_delete)} event(s){Colors.RESET}"
                    )
                return
            elif choice == "add":
                pass
            elif choice == "fresh":
                self.config["events"] = []

        setup_events(self.config, self.state.get("model_classes", []))

    def _get_section_summary(self, section: str) -> str:
        """Get a short summary of a config section for the menu."""
        if section == "camera":
            url = self.config.get("camera", {}).get("url", "not set")
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
            return output.get("json_dir", "data")

        return "?"

    def _get_config_warnings(self) -> list[str]:
        """Check config for common issues and return warnings."""
        warnings = []
        events = self.config.get("events", [])
        lines = self.config.get("lines", [])
        zones = self.config.get("zones", [])

        for event in events:
            if not event.get("actions"):
                warnings.append(f"Event '{event.get('name')}' has no actions")

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

        if not events and (lines or zones):
            warnings.append("Lines/zones defined but no events to use them")

        return warnings

    def _cleanup(self):
        """Clean up resources."""
        cap = self.state.get("cap")
        if cap is not None:
            cap.release()
            self.state["cap"] = None

        stop_preview_server(self.http_server)
        self.http_server = None


def run_builder() -> str | None:
    """Run the config builder wizard."""
    builder = ConfigBuilder()
    return builder.run()


def run_editor(config_path: str) -> str | None:
    """Run the config editor."""
    builder = ConfigBuilder()
    return builder.run_edit(config_path)
