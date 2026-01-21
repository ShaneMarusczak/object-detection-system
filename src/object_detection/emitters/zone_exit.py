"""
ZoneExitEmitter - Detects objects exiting monitoring zones.

Requires tracking state to detect exit transitions and calculate dwell time.
"""

from typing import TYPE_CHECKING

from ..models import ZoneConfig
from .registry import register
from .zone_base import BaseZoneEmitter

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@register("ZONE_EXIT")
class ZoneExitEmitter(BaseZoneEmitter):
    """Emits ZONE_EXIT events when tracked objects exit zones."""

    @property
    def event_type(self) -> str:
        return "ZONE_EXIT"

    def _check_transition(
        self,
        inside: bool,
        was_inside: bool,
        tracked_obj,
        zone: ZoneConfig,
        tracking_state: "TrackingState",
        frame_id: str | None,
    ) -> dict | None:
        """Check for zone exit and return event with dwell time."""
        if not inside and was_inside:
            # Calculate dwell time and clean up
            entry_time = tracked_obj.active_zones[zone.zone_id]
            dwell_time = tracking_state.current_time - entry_time
            del tracked_obj.active_zones[zone.zone_id]

            return {
                "event_type": "ZONE_EXIT",
                "track_id": tracked_obj.track_id,
                "object_class": tracked_obj.object_class,
                "bbox": tracked_obj.bbox,
                "frame_id": frame_id,
                "zone_id": zone.zone_id,
                "dwell_time": dwell_time,
            }
        return None
