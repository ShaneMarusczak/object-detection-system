"""
Geometry Configuration Parsing - Shared parsers for zones and lines.

This module provides a single source of truth for parsing zone and line
configurations from the raw config dict. Used by emitters and dispatcher.
"""

from ..models import LineConfig, ZoneConfig


def parse_zones(
    config: dict,
    default_classes: list[int] | None = None,
) -> list[ZoneConfig]:
    """
    Parse zone configurations from raw config dict.

    Args:
        config: Full configuration dictionary with 'zones' key
        default_classes: Default allowed classes if not specified per-zone.
                        If None, reads from config['detection']['track_classes']

    Returns:
        List of ZoneConfig objects with generated zone_ids (Z1, Z2, ...)
    """
    if default_classes is None:
        default_classes = config.get("detection", {}).get("track_classes", [])

    zones = []
    for i, zone_config in enumerate(config.get("zones", []), 1):
        allowed_classes = zone_config.get("allowed_classes", default_classes)

        zones.append(
            ZoneConfig(
                zone_id=f"Z{i}",
                x1_pct=zone_config["x1_pct"],
                y1_pct=zone_config["y1_pct"],
                x2_pct=zone_config["x2_pct"],
                y2_pct=zone_config["y2_pct"],
                description=zone_config["description"],
                allowed_classes=allowed_classes,
            )
        )

    return zones


def parse_lines(
    config: dict,
    default_classes: list[int] | None = None,
) -> list[LineConfig]:
    """
    Parse line configurations from raw config dict.

    Args:
        config: Full configuration dictionary with 'lines' key
        default_classes: Default allowed classes if not specified per-line.
                        If None, reads from config['detection']['track_classes']

    Returns:
        List of LineConfig objects with generated line_ids (V1, H1, ...)
    """
    if default_classes is None:
        default_classes = config.get("detection", {}).get("track_classes", [])

    lines = []
    vertical_count = 0
    horizontal_count = 0

    for line_config in config.get("lines", []):
        if line_config["type"] == "vertical":
            vertical_count += 1
            line_id = f"V{vertical_count}"
        else:
            horizontal_count += 1
            line_id = f"H{horizontal_count}"

        allowed_classes = line_config.get("allowed_classes", default_classes)

        lines.append(
            LineConfig(
                line_id=line_id,
                type=line_config["type"],
                position_pct=line_config["position_pct"],
                description=line_config["description"],
                allowed_classes=allowed_classes,
            )
        )

    return lines


def build_zone_lookup(config: dict) -> dict[str, dict]:
    """
    Build lookup from zone_id to zone info.

    Used by dispatcher to enrich events with zone descriptions.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary mapping zone_id (Z1, Z2, ...) to zone info dict
        containing 'description' and 'config' keys
    """
    zones = parse_zones(config)
    lookup = {}

    for i, zone in enumerate(zones):
        # Get the raw config for this zone
        raw_config = config.get("zones", [])[i] if i < len(config.get("zones", [])) else {}
        lookup[zone.zone_id] = {
            "description": zone.description,
            "config": raw_config,
        }

    return lookup


def build_line_lookup(config: dict) -> dict[str, dict]:
    """
    Build lookup from line_id to line info.

    Used by dispatcher to enrich events with line descriptions.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary mapping line_id (V1, H1, ...) to line info dict
        containing 'description' and 'config' keys
    """
    lines = parse_lines(config)
    lookup = {}

    for i, line in enumerate(lines):
        # Get the raw config for this line
        raw_config = config.get("lines", [])[i] if i < len(config.get("lines", [])) else {}
        lookup[line.line_id] = {
            "description": line.description,
            "config": raw_config,
        }

    return lookup


__all__ = [
    "build_line_lookup",
    "build_zone_lookup",
    "parse_lines",
    "parse_zones",
]
