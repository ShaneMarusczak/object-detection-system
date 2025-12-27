"""
Tracking data models - objects, lines, zones, and ROI configuration.
"""

from dataclasses import dataclass, field


@dataclass
class TrackedObject:
    """
    Represents a tracked object with its state across frames.

    Attributes:
        track_id: Unique identifier from ByteTrack
        object_class: COCO class ID
        current_pos: Current (x, y) center position
        bbox: Current bounding box (x1, y1, x2, y2) in pixels
        previous_pos: Previous (x, y) center position for movement detection
        crossed_lines: Set of line IDs this object has crossed
        active_zones: Dict mapping zone_id to entry timestamp
    """

    track_id: int
    object_class: int
    current_pos: tuple[float, float]
    bbox: tuple[int, int, int, int] | None = None
    previous_pos: tuple[float, float] | None = None
    crossed_lines: set[str] = field(default_factory=set)
    active_zones: dict[str, float] = field(default_factory=dict)

    def update_position(
        self, x: float, y: float, bbox: tuple[int, int, int, int] | None = None
    ) -> None:
        """Update position and bbox, moving current to previous."""
        self.previous_pos = self.current_pos
        self.current_pos = (x, y)
        if bbox:
            self.bbox = bbox

    def is_new(self) -> bool:
        """Check if this is the first detection of this object."""
        return self.previous_pos is None


@dataclass
class LineConfig:
    """Configuration for a counting line."""

    line_id: str
    type: str  # 'vertical' or 'horizontal'
    position_pct: float
    description: str
    allowed_classes: list[int]


@dataclass
class ZoneConfig:
    """Configuration for a detection zone."""

    zone_id: str
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    description: str
    allowed_classes: list[int]


@dataclass
class ROIConfig:
    """Region of Interest configuration."""

    enabled: bool
    h_from: float = 0
    h_to: float = 100
    v_from: float = 0
    v_to: float = 100
