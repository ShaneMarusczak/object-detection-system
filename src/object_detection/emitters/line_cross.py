"""
LineCrossEmitter - Detects objects crossing counting lines.

Requires tracking state to detect movement across line boundaries.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@dataclass
class LineConfig:
    """Configuration for a counting line."""

    line_id: str
    type: str  # 'vertical' or 'horizontal'
    position_pct: float
    description: str
    allowed_classes: list[int]


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
        self.lines = self._parse_lines(config)
        self.speed_enabled = config.get("speed_calculation", {}).get("enabled", False)

    def _parse_lines(self, config: dict) -> list[LineConfig]:
        """Parse line configurations from config."""
        lines = []
        vertical_count = 0
        horizontal_count = 0
        default_classes = config.get("detection", {}).get("track_classes", [])

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

    def process(
        self,
        frame,
        yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Check for line crossings and emit events.

        Args:
            frame: Raw frame (unused)
            yolo_results: YOLO results (unused, we use tracking_state)
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
                        "timestamp_relative": timestamp,
                    }

                    # Add speed data if enabled
                    if self.speed_enabled:
                        self._add_speed_data(event, tracked_obj, line, tracking_state.current_time)

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

    def _add_speed_data(
        self, event: dict, tracked_obj, line: LineConfig, current_time: float
    ) -> None:
        """Add speed calculation data to event."""
        if not tracked_obj.first_pos or not tracked_obj.first_seen_time:
            return

        first_x, first_y = tracked_obj.first_pos
        curr_x, curr_y = tracked_obj.current_pos

        if line.type == "vertical":
            distance = abs(curr_x - first_x)
        else:
            distance = abs(curr_y - first_y)

        time_elapsed = current_time - tracked_obj.first_seen_time

        if time_elapsed > 0.1:  # MIN_TRACKING_TIME
            event["distance_pixels"] = distance
            event["time_elapsed"] = time_elapsed
