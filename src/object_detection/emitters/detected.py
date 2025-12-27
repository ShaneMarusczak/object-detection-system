"""
DetectedEmitter - Raw detection to event.

No geometry, no tracking state. Every YOLO detection becomes an event.
Simplest emitter - pure pass-through.
"""

from typing import TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .tracking_state import TrackingState


@register("DETECTED")
class DetectedEmitter:
    """Emits DETECTED events for every YOLO detection."""

    event_type = "DETECTED"
    needs_yolo = True
    needs_tracking = False

    def __init__(self, _config: dict, frame_dims: tuple):
        """Initialize emitter (no config needed for this simple emitter)."""
        self.frame_dims = frame_dims

    def process(
        self,
        frame,  # noqa: ARG002 - unused, interface requirement
        yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,  # noqa: ARG002
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Emit DETECTED event for each detection in frame.

        Args:
            _frame: Raw frame (unused - interface requirement)
            yolo_results: YOLO results with boxes
            timestamp: Relative timestamp
            _tracking_state: Tracking state (unused - no tracking needed)
            frame_id: Optional saved frame reference

        Returns:
            List of DETECTED events
        """
        events = []

        boxes = yolo_results[0].boxes
        if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
            return events

        # Handle case where tracking is disabled (no track IDs)
        has_track_ids = boxes.id is not None

        classes = boxes.cls.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist() if has_track_ids else [None] * len(classes)
        )

        for obj_class, box, conf, track_id in zip(classes, xyxy, confs, track_ids):
            x1, y1, x2, y2 = box
            bbox = (int(x1), int(y1), int(x2), int(y2))

            event = {
                "event_type": "DETECTED",
                "object_class": obj_class,
                "confidence": conf,
                "bbox": bbox,
                "frame_id": frame_id,
            }

            # Include track_id if available (tracking enabled)
            if track_id is not None:
                event["track_id"] = track_id

            events.append(event)

        return events
