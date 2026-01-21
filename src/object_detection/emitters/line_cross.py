"""
LineCrossEmitter - Detects objects crossing counting lines.

Requires tracking state to detect movement across line boundaries.
"""

from typing import TYPE_CHECKING

from ..config.geometry import parse_lines
from ..models import LineConfig
from .registry import register

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@register("LINE_CROSS")
class LineCrossEmitter:
    """Emits LINE_CROSS events when tracked objects cross counting lines."""

    event_type = "LINE_CROSS"
    needs_yolo = True
    needs_tracking = True

    def __init__(self, config: dict, frame_dims: tuple):
        """
        Initialize with line configurations.

        Args:
            config: Full config dict
            frame_dims: (width, height) for calculating line positions
        """
        self.frame_dims = frame_dims
        self.lines = parse_lines(config)

    def process(
        self,
        _frame,
        _yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Check for line crossings and emit events.

        Args:
            _frame: Raw frame (unused - interface requirement)
            _yolo_results: YOLO results (unused - we use tracking_state)
            timestamp: Relative timestamp
            tracking_state: Shared tracking state with object positions
            frame_id: Optional saved frame reference

        Returns:
            List of LINE_CROSS events
        """
        if tracking_state is None:
            return []

        events = []
        roi_width, roi_height = self.frame_dims

        for tracked_obj in tracking_state.get_active_objects():
            # Need previous position to detect crossing
            if tracked_obj.is_new():
                continue

            prev_x, prev_y = tracked_obj.previous_pos
            curr_x, curr_y = tracked_obj.current_pos

            for line in self.lines:
                # Check class permission
                if tracked_obj.object_class not in line.allowed_classes:
                    continue

                # Check if already crossed
                if line.line_id in tracked_obj.crossed_lines:
                    continue

                # Check for crossing
                crossed, direction = self._detect_crossing(
                    prev_x, prev_y, curr_x, curr_y, line, roi_width, roi_height
                )

                if crossed:
                    tracked_obj.crossed_lines.add(line.line_id)

                    event = {
                        "event_type": "LINE_CROSS",
                        "track_id": tracked_obj.track_id,
                        "object_class": tracked_obj.object_class,
                        "bbox": tracked_obj.bbox,
                        "frame_id": frame_id,
                        "line_id": line.line_id,
                        "direction": direction,
                    }

                    events.append(event)

        return events

    def _detect_crossing(
        self,
        prev_x: float,
        prev_y: float,
        curr_x: float,
        curr_y: float,
        line: LineConfig,
        roi_width: int,
        roi_height: int,
    ) -> tuple[bool, str | None]:
        """Detect if movement crossed a line."""
        if line.type == "vertical":
            line_pos = roi_width * line.position_pct / 100
            if prev_x < line_pos <= curr_x:
                return True, "LTR"
            elif prev_x > line_pos >= curr_x:
                return True, "RTL"
        else:  # horizontal
            line_pos = roi_height * line.position_pct / 100
            if prev_y < line_pos <= curr_y:
                return True, "TTB"
            elif prev_y > line_pos >= curr_y:
                return True, "BTT"

        return False, None
