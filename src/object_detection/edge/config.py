"""
Edge Detector Configuration

Minimal configuration for Jetson edge detector.
Only what's needed for detection - no event routing, emails, or digests.
"""

from dataclasses import dataclass, field
import yaml


@dataclass
class ROIConfig:
    """Region of interest configuration."""

    enabled: bool = False
    h_from: float = 0
    h_to: float = 100
    v_from: float = 0
    v_to: float = 100


@dataclass
class LineConfig:
    """Counting line configuration."""

    line_id: str
    type: str  # 'vertical' or 'horizontal'
    position_pct: float
    allowed_classes: list[int] = field(default_factory=list)


@dataclass
class ZoneConfig:
    """Zone configuration."""

    zone_id: str
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    allowed_classes: list[int] = field(default_factory=list)


@dataclass
class NighttimeDetectionParams:
    """Detection parameters for nighttime car detection."""

    brightness_threshold: int = 30
    min_blob_size: int = 100
    max_blob_size: int = 10000
    score_threshold: int = 85
    taillight_color_match: bool = True


@dataclass
class NighttimeCarEventConfig:
    """Nighttime car detection event configuration (from events with event_type=NIGHTTIME_CAR)."""

    name: str  # Event name
    zone_id: str  # Zone identifier (e.g., "Z1")
    zone_description: str  # Zone description for matching
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    detection_params: NighttimeDetectionParams = field(
        default_factory=NighttimeDetectionParams
    )


@dataclass
class EdgeConfig:
    """
    Minimal configuration for edge detector.

    Only includes what's needed on Jetson:
    - Model and detection settings
    - Camera connection
    - Lines and zones geometry
    - ROI cropping
    - Output destination (Redis)
    """

    # Detection
    model_file: str
    confidence_threshold: float
    track_classes: list[int]

    # Camera
    camera_url: str

    # Geometry
    lines: list[LineConfig] = field(default_factory=list)
    zones: list[ZoneConfig] = field(default_factory=list)
    nighttime_car_events: list[NighttimeCarEventConfig] = field(default_factory=list)
    roi: ROIConfig = field(default_factory=ROIConfig)

    # Output
    redis_url: str = "redis://localhost:6379"
    redis_stream: str = "detections"
    device_id: str = "jetson-01"

    # Runtime
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

    @classmethod
    def from_yaml(cls, path: str) -> "EdgeConfig":
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeConfig":
        """Create config from dictionary."""
        detection = data.get("detection", {})
        camera = data.get("camera", {})
        output = data.get("output", {})

        # Parse ROI
        roi_data = data.get("roi", {})
        h_roi = roi_data.get("horizontal", {})
        v_roi = roi_data.get("vertical", {})
        roi = ROIConfig(
            enabled=h_roi.get("enabled", False) or v_roi.get("enabled", False),
            h_from=h_roi.get("crop_from_left_pct", 0),
            h_to=h_roi.get("crop_to_right_pct", 100),
            v_from=v_roi.get("crop_from_top_pct", 0),
            v_to=v_roi.get("crop_to_bottom_pct", 100),
        )

        # Parse lines
        lines = []
        v_count, h_count = 0, 0
        for line_data in data.get("lines", []):
            if line_data["type"] == "vertical":
                v_count += 1
                line_id = f"V{v_count}"
            else:
                h_count += 1
                line_id = f"H{h_count}"

            lines.append(
                LineConfig(
                    line_id=line_id,
                    type=line_data["type"],
                    position_pct=line_data["position_pct"],
                    allowed_classes=line_data.get(
                        "allowed_classes", detection.get("track_classes", [])
                    ),
                )
            )

        # Parse zones
        zones = []
        for i, zone_data in enumerate(data.get("zones", []), 1):
            zones.append(
                ZoneConfig(
                    zone_id=f"Z{i}",
                    x1_pct=zone_data["x1_pct"],
                    y1_pct=zone_data["y1_pct"],
                    x2_pct=zone_data["x2_pct"],
                    y2_pct=zone_data["y2_pct"],
                    allowed_classes=zone_data.get(
                        "allowed_classes", detection.get("track_classes", [])
                    ),
                )
            )

        # Build zone lookup for NIGHTTIME_CAR events
        zone_lookup = {}
        for i, zone_data in enumerate(data.get("zones", []), 1):
            zone_id = f"Z{i}"
            zone_desc = zone_data.get("description", zone_id)
            zone_lookup[zone_desc] = {
                "zone_id": zone_id,
                "x1_pct": zone_data["x1_pct"],
                "y1_pct": zone_data["y1_pct"],
                "x2_pct": zone_data["x2_pct"],
                "y2_pct": zone_data["y2_pct"],
            }

        # Parse NIGHTTIME_CAR events
        nighttime_car_events = []
        for event in data.get("events", []):
            match = event.get("match", {})
            if match.get("event_type") != "NIGHTTIME_CAR":
                continue

            zone_desc = match.get("zone")
            if not zone_desc or zone_desc not in zone_lookup:
                continue

            zone_geo = zone_lookup[zone_desc]
            nighttime_config = match.get("nighttime_detection", {})

            nighttime_car_events.append(
                NighttimeCarEventConfig(
                    name=event.get("name", ""),
                    zone_id=zone_geo["zone_id"],
                    zone_description=zone_desc,
                    x1_pct=zone_geo["x1_pct"],
                    y1_pct=zone_geo["y1_pct"],
                    x2_pct=zone_geo["x2_pct"],
                    y2_pct=zone_geo["y2_pct"],
                    detection_params=NighttimeDetectionParams(
                        brightness_threshold=nighttime_config.get(
                            "brightness_threshold", 30
                        ),
                        min_blob_size=nighttime_config.get("min_blob_size", 100),
                        max_blob_size=nighttime_config.get("max_blob_size", 10000),
                        score_threshold=nighttime_config.get("score_threshold", 85),
                        taillight_color_match=nighttime_config.get(
                            "taillight_color_match", True
                        ),
                    ),
                )
            )

        return cls(
            model_file=detection["model_file"],
            confidence_threshold=detection.get("confidence_threshold", 0.25),
            track_classes=detection.get("track_classes", []),
            camera_url=camera["url"],
            lines=lines,
            zones=zones,
            nighttime_car_events=nighttime_car_events,
            roi=roi,
            redis_url=output.get("redis_url", "redis://localhost:6379"),
            redis_stream=output.get("stream", "detections"),
            device_id=data.get("device_id", "jetson-01"),
        )
