"""
ZoneEnterEmitter - Detects objects entering monitoring zones.

Requires tracking state to detect entry transitions.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@dataclass
class ZoneConfig:
    """Configuration for a monitoring zone."""

    zone_id: str
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    description: str
    allowed_classes: list[int]


@register("ZONE_ENTER")
class ZoneEnterEmitter:
    """Emits ZONE_ENTER events when tracked objects enter zones."""

    event_type = "ZONE_ENTER"
    needs_yolo = True
    needs_tracking = True

    def __init__(self, config: dict, frame_dims: tuple):
        """
        Initialize with zone configurations.

        Args:
            config: Full config dict
            frame_dims: (width, height) for calculating zone boundaries
        """
        self.frame_dims = frame_dims
        self.zones = self._parse_zones(config)

    def _parse_zones(self, config: dict) -> list[ZoneConfig]:
        """Parse zone configurations from config."""
        zones = []
        default_classes = config.get("detection", {}).get("track_classes", [])

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

    def process(
        self,
        frame,
        yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Check for zone entries and emit events.

        Args:
            frame: Raw frame (unused)
            yolo_results: YOLO results (unused, we use tracking_state)
            timestamp: Relative timestamp
            tracking_state: Shared tracking state with object positions
            frame_id: Optional saved frame reference

        Returns:
            List of ZONE_ENTER events
        """
        if tracking_state is None:
            return []

        events = []
        roi_width, roi_height = self.frame_dims

        for tracked_obj in tracking_state.get_active_objects():
            curr_x, curr_y = tracked_obj.current_pos

            for zone in self.zones:
                # Check class permission
                if tracked_obj.object_class not in zone.allowed_classes:
                    continue

                # Calculate zone boundaries
                zone_x1 = roi_width * zone.x1_pct / 100
                zone_x2 = roi_width * zone.x2_pct / 100
                zone_y1 = roi_height * zone.y1_pct / 100
                zone_y2 = roi_height * zone.y2_pct / 100

                # Check if inside zone
                inside = zone_x1 <= curr_x <= zone_x2 and zone_y1 <= curr_y <= zone_y2
                was_inside = zone.zone_id in tracked_obj.active_zones

                if inside and not was_inside:
                    # ZONE_ENTER - record entry time in tracking state
                    tracked_obj.active_zones[zone.zone_id] = tracking_state.current_time

                    events.append({
                        "event_type": "ZONE_ENTER",
                        "track_id": tracked_obj.track_id,
                        "object_class": tracked_obj.object_class,
                        "bbox": tracked_obj.bbox,
                        "frame_id": frame_id,
                        "zone_id": zone.zone_id,
                        "timestamp_relative": timestamp,
                    })

        return events
