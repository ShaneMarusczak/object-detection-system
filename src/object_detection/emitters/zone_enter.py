"""
ZoneEnterEmitter - Detects objects entering monitoring zones.

Requires tracking state to detect entry transitions.
"""

from typing import TYPE_CHECKING

from ..models import ZoneConfig
from .registry import register
from .zone_base import BaseZoneEmitter

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@register("ZONE_ENTER")
class ZoneEnterEmitter(BaseZoneEmitter):
    """Emits ZONE_ENTER events when tracked objects enter zones."""

    @property
    def event_type(self) -> str:
        return "ZONE_ENTER"

    def _check_transition(
        self,
        inside: bool,
        was_inside: bool,
        tracked_obj,
        zone: ZoneConfig,
        tracking_state: "TrackingState",
        frame_id: str | None,
    ) -> dict | None:
        """Check for zone entry and return event if triggered."""
        if inside and not was_inside:
            # Record entry time in tracking state
            tracked_obj.active_zones[zone.zone_id] = tracking_state.current_time

            return {
                "event_type": "ZONE_ENTER",
                "track_id": tracked_obj.track_id,
                "object_class": tracked_obj.object_class,
                "bbox": tracked_obj.bbox,
                "frame_id": frame_id,
                "zone_id": zone.zone_id,
            }
        return None
