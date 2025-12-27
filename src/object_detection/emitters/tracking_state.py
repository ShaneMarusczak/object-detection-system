"""
TrackingState - Shared state for tracking-dependent emitters.

Maintains tracked objects, their positions, and zone membership.
Updated once per frame, then queried by emitters.
"""

from ..models import TrackedObject


class TrackingState:
    """
    Shared tracking state for emitters.

    Updated once per frame with YOLO results. Emitters read from this
    to determine state changes (line crossings, zone enter/exit).
    """

    def __init__(self):
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.current_time: float = 0.0

    def update(self, yolo_results, current_time: float) -> None:
        """
        Update tracking state from YOLO results.

        Args:
            yolo_results: YOLO detection results
            current_time: Current timestamp
        """
        self.current_time = current_time

        boxes = yolo_results[0].boxes
        if boxes is None or boxes.id is None or len(boxes.id) == 0:
            return

        track_ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()

        for track_id, box, obj_class in zip(track_ids, xyxy, classes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = TrackedObject(
                    track_id=track_id,
                    object_class=obj_class,
                    current_pos=(center_x, center_y),
                    bbox=bbox,
                )
            else:
                self.tracked_objects[track_id].update_position(center_x, center_y, bbox)

    def get_active_objects(self) -> list[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self.tracked_objects.values())
