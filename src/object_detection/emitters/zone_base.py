"""
Base Zone Emitter - Shared logic for zone enter/exit detection.

Handles zone boundary calculations and object position checking.
Subclasses implement only the specific event condition.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..config.geometry import parse_zones
from ..models import ZoneConfig

if TYPE_CHECKING:
    from .tracking_state import TrackingState


class BaseZoneEmitter(ABC):
    """Base class for zone-based emitters (enter/exit)."""

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
        self.zones = parse_zones(config)

    def _is_inside_zone(
        self, x: float, y: float, zone: ZoneConfig, roi_width: int, roi_height: int
    ) -> bool:
        """Check if point is inside zone boundaries."""
        zone_x1 = roi_width * zone.x1_pct / 100
        zone_x2 = roi_width * zone.x2_pct / 100
        zone_y1 = roi_height * zone.y1_pct / 100
        zone_y2 = roi_height * zone.y2_pct / 100
        return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2

    @property
    @abstractmethod
    def event_type(self) -> str:
        """The event type this emitter produces."""
        ...

    @abstractmethod
    def _check_transition(
        self,
        inside: bool,
        was_inside: bool,
        tracked_obj,
        zone: ZoneConfig,
        tracking_state: "TrackingState",
        frame_id: str | None,
    ) -> dict | None:
        """
        Check for zone transition and return event if triggered.

        Args:
            inside: Whether object is currently inside zone
            was_inside: Whether object was previously inside zone
            tracked_obj: The tracked object
            zone: The zone being checked
            tracking_state: Shared tracking state
            frame_id: Optional frame reference

        Returns:
            Event dict if transition occurred, None otherwise
        """
        ...

    def process(
        self,
        _frame,
        _yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Check for zone transitions and emit events.

        Args:
            _frame: Raw frame (unused - interface requirement)
            _yolo_results: YOLO results (unused - we use tracking_state)
            timestamp: Relative timestamp
            tracking_state: Shared tracking state with object positions
            frame_id: Optional saved frame reference

        Returns:
            List of zone events
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

                inside = self._is_inside_zone(
                    curr_x, curr_y, zone, roi_width, roi_height
                )
                was_inside = zone.zone_id in tracked_obj.active_zones

                event = self._check_transition(
                    inside, was_inside, tracked_obj, zone, tracking_state, frame_id
                )
                if event:
                    events.append(event)

        return events
