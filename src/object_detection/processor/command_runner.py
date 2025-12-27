"""
Command Runner - Execute shell commands as event actions.

Runs commands with event data passed as environment variables.
Supports timeouts to prevent blocking the detection loop.
"""

import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout for commands (seconds)
DEFAULT_TIMEOUT = 30


def run_command(
    command_config: dict,
    event: dict,
    temp_frame_dir: str | None = None,
) -> tuple[bool, str | None]:
    """
    Execute a command with event data as environment variables.

    Args:
        command_config: Command configuration with 'exec' and optional 'timeout_seconds'
        event: Enriched event dict with all event data
        temp_frame_dir: Directory where temp frames are stored

    Returns:
        (success, error_message) tuple
    """
    exec_path = command_config.get("exec")
    if not exec_path:
        return False, "No 'exec' path specified in command config"

    timeout = command_config.get("timeout_seconds", DEFAULT_TIMEOUT)

    # Build environment with event data
    env = _build_event_env(event, temp_frame_dir)

    logger.info(f"Running command: {exec_path}")
    logger.debug(f"Command environment: {env}")

    try:
        result = subprocess.run(
            exec_path,
            shell=True,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode == 0:
            logger.info("Command completed successfully")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout.strip()}")
            return True, None
        else:
            error_msg = f"Command failed with code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr.strip()}"
            logger.error(error_msg)
            return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout}s"
        logger.error(error_msg)
        return False, error_msg

    except Exception as e:
        error_msg = f"Command execution error: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def _build_event_env(event: dict, temp_frame_dir: str | None) -> dict[str, str]:
    """
    Build environment variables from event data.

    Passes all relevant event fields as environment variables.
    """
    # Start with current environment
    env = os.environ.copy()

    # Core event fields
    _set_env(env, "EVENT_TYPE", event.get("event_type"))
    _set_env(env, "EVENT_NAME", event.get("event_definition"))
    _set_env(env, "TIMESTAMP", event.get("timestamp"))
    _set_env(env, "TIMESTAMP_RELATIVE", event.get("timestamp_relative"))

    # Object identification
    _set_env(env, "OBJECT_CLASS", event.get("object_class_name"))
    _set_env(env, "OBJECT_CLASS_ID", event.get("object_class"))
    _set_env(env, "CONFIDENCE", event.get("confidence"))
    _set_env(env, "TRACK_ID", event.get("track_id"))

    # Bounding box (as comma-separated: x1,y1,x2,y2)
    bbox = event.get("bbox")
    if bbox:
        _set_env(env, "BBOX", ",".join(str(v) for v in bbox))

    # Location context
    _set_env(env, "ZONE_ID", event.get("zone_id"))
    _set_env(env, "ZONE_NAME", event.get("zone_description"))
    _set_env(env, "LINE_ID", event.get("line_id"))
    _set_env(env, "LINE_NAME", event.get("line_description"))
    _set_env(env, "DIRECTION", event.get("direction"))

    # Frame reference
    frame_id = event.get("frame_id")
    if frame_id and temp_frame_dir:
        frame_path = os.path.join(temp_frame_dir, f"{frame_id}.jpg")
        if os.path.exists(frame_path):
            _set_env(env, "FRAME_PATH", frame_path)
    _set_env(env, "FRAME_ID", frame_id)

    # Nighttime car specific
    _set_env(env, "SCORE", event.get("score"))
    _set_env(env, "HAD_TAILLIGHT", event.get("had_taillight"))

    # Zone dwell specific
    _set_env(env, "DWELL_TIME", event.get("dwell_time"))

    return env


def _set_env(env: dict, key: str, value: Any) -> None:
    """Set environment variable if value is not None."""
    if value is not None:
        env[key] = str(value)
