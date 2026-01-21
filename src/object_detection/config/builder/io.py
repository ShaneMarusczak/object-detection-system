"""
Config I/O for the config builder.

Handles loading, saving, and preflight validation of configurations.
"""

import os
import sys
from datetime import datetime

import questionary
import yaml

from ..planner import (
    build_plan,
    generate_sample_events,
    load_config_with_env,
    print_plan,
    print_validation_result,
    simulate_dry_run,
)
from ..validator import validate_config_full
from .prompts import PROMPT_STYLE, Colors


def load_config(config_path: str) -> dict | None:
    """
    Load a configuration file.

    Args:
        config_path: Path to the config file

    Returns:
        Configuration dict, or None on error
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"{Colors.RED}Config not found: {config_path}{Colors.RESET}")
        return None
    except yaml.YAMLError as e:
        print(f"{Colors.RED}Invalid YAML: {e}{Colors.RESET}")
        return None


def save_config(
    config: dict,
    config_path: str | None,
    run_after: bool = False,
    cleanup_callback=None,
) -> str | None:
    """
    Save configuration to file.

    Args:
        config: Configuration dict to save
        config_path: Original config path (for default suggestion)
        run_after: Whether to run detection after saving
        cleanup_callback: Optional callback to cleanup resources before running

    Returns:
        Saved filepath, or None if cancelled
    """
    print(f"\n{Colors.BOLD}--- Save Config ---{Colors.RESET}")

    if config_path:
        default_path = config_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        default_path = f"configs/config_{timestamp}.yaml"

    filepath = questionary.text(
        "Save to:",
        default=default_path,
        style=PROMPT_STYLE,
    ).ask()

    if filepath is None:
        return None

    # Ensure directory exists
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"{Colors.GREEN}Saved:{Colors.RESET} {filepath}")

    if run_after:
        update_config_pointer(filepath)
        run_preflight_stages(filepath)

        if cleanup_callback:
            cleanup_callback()

        print(f"\n{Colors.CYAN}Starting detection...{Colors.RESET}\n")
        os.execvp("python", ["python", "-m", "object_detection", "-c", filepath])

    return filepath


def update_config_pointer(config_path: str) -> None:
    """Update config.yaml to point to the specified config."""
    pointer_content = f"""# Config pointer - specifies which config to use
use: {config_path}
"""
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(pointer_content)

    print(f"{Colors.GRAY}config.yaml -> {config_path}{Colors.RESET}")


def run_preflight_stages(config_path: str) -> None:
    """Run validate, plan, and dry-run stages before detection."""
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

    questionary.press_any_key_to_continue(
        "Press any key to continue...",
        style=PROMPT_STYLE,
    ).ask()

    # Stage 2: Plan
    print(f"\n{Colors.GREEN}=== Planning ==={Colors.RESET}")
    plan = build_plan(config)
    print_plan(plan)

    questionary.press_any_key_to_continue(
        "Press any key to continue...",
        style=PROMPT_STYLE,
    ).ask()

    # Stage 3: Dry Run
    print(f"\n{Colors.GREEN}=== Dry Run ==={Colors.RESET}")
    sample_events = generate_sample_events(config)
    print(f"Generated {len(sample_events)} sample events from config")
    simulate_dry_run(config, sample_events)
