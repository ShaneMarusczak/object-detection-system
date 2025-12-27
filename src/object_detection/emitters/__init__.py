"""
Event Emitters - Pratt-style dispatch for detection event generation.

Each emitter handles a single event type. The registry maps event types
to emitter classes. Active emitters are determined from config at startup.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .registry import EMITTER_REGISTRY, build_active_emitters
from .tracking_state import TrackingState

if TYPE_CHECKING:
    from .tracking_state import TrackingState

__all__ = ["Emitter", "EMITTER_REGISTRY", "build_active_emitters", "TrackingState"]


@runtime_checkable
class Emitter(Protocol):
    """Protocol for event emitters."""

    event_type: str  # The single event type this emitter produces
    needs_yolo: bool  # Does this emitter need YOLO inference results?
    needs_tracking: bool  # Does this emitter need tracking state?

    def process(
        self,
        frame,
        yolo_results,
        timestamp: float,
        tracking_state: TrackingState | None = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Process frame and YOLO results, emit events.

        Args:
            frame: Raw BGR frame (for emitters that need pixel analysis)
            yolo_results: YOLO detection results (boxes, classes, track_ids)
            timestamp: Relative timestamp since detection started
            tracking_state: Shared tracking state (for tracking-dependent emitters)
            frame_id: Optional frame ID for linking to saved frames

        Returns:
            List of event dicts to queue
        """
        ...
