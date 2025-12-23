"""
Pydantic schemas for configuration validation.

Provides type-safe, declarative validation with clear error messages.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class DetectionConfig(BaseModel):
    """Detection settings."""

    model_file: str = Field(..., description="YOLO model file path (.pt)")
    confidence_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence threshold"
    )
    # track_classes is derived from events by planner, not user-configurable

    @field_validator("model_file")
    @classmethod
    def validate_model_file(cls, v: str) -> str:
        if not v.endswith(".pt"):
            raise ValueError("Model file must be .pt format")
        return v


class ROIHorizontalConfig(BaseModel):
    """Horizontal ROI settings."""

    enabled: bool = False
    crop_from_left_pct: float = Field(default=0, ge=0, le=100)
    crop_to_right_pct: float = Field(default=100, ge=0, le=100)

    @model_validator(mode="after")
    def validate_range(self):
        if self.enabled and self.crop_from_left_pct >= self.crop_to_right_pct:
            raise ValueError("crop_to_right_pct must be > crop_from_left_pct")
        return self


class ROIVerticalConfig(BaseModel):
    """Vertical ROI settings."""

    enabled: bool = False
    crop_from_top_pct: float = Field(default=0, ge=0, le=100)
    crop_to_bottom_pct: float = Field(default=100, ge=0, le=100)

    @model_validator(mode="after")
    def validate_range(self):
        if self.enabled and self.crop_from_top_pct >= self.crop_to_bottom_pct:
            raise ValueError("crop_to_bottom_pct must be > crop_from_top_pct")
        return self


class ROIConfig(BaseModel):
    """Region of interest settings."""

    horizontal: ROIHorizontalConfig = Field(default_factory=ROIHorizontalConfig)
    vertical: ROIVerticalConfig = Field(default_factory=ROIVerticalConfig)


class LineConfig(BaseModel):
    """Counting line configuration."""

    type: Literal["vertical", "horizontal"]
    position_pct: float = Field(..., ge=0, le=100, description="Position as percentage")
    description: str = Field(..., min_length=1, description="Human-readable line name")
    allowed_classes: list[int] | None = Field(
        default=None, description="COCO class IDs allowed to cross"
    )


class ZoneConfig(BaseModel):
    """Zone configuration."""

    x1_pct: float = Field(..., ge=0, le=100)
    y1_pct: float = Field(..., ge=0, le=100)
    x2_pct: float = Field(..., ge=0, le=100)
    y2_pct: float = Field(..., ge=0, le=100)
    description: str = Field(..., min_length=1)
    allowed_classes: list[int] | None = None

    @model_validator(mode="after")
    def validate_coordinates(self):
        if self.x2_pct <= self.x1_pct:
            raise ValueError("x2_pct must be > x1_pct")
        if self.y2_pct <= self.y1_pct:
            raise ValueError("y2_pct must be > y1_pct")
        return self


class EventMatch(BaseModel):
    """Event matching criteria."""

    event_type: (
        None | (Literal["LINE_CROSS", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"])
    ) = None
    line: str | None = None
    zone: str | None = None
    object_class: str | list[str] | None = None
    direction: Literal["LTR", "RTL", "TTB", "BTT"] | None = None


class EmailImmediateAction(BaseModel):
    """Immediate email action configuration."""

    enabled: bool = True
    cooldown_minutes: int = Field(default=30, ge=0)
    message: str | None = None
    subject: str | None = None
    include_frame: bool = False


class FrameCaptureAction(BaseModel):
    """Frame capture action configuration."""

    enabled: bool = True
    cooldown_seconds: int = Field(default=300, ge=0)
    annotate: bool = False


class EventActions(BaseModel):
    """Event actions configuration."""

    json_log: bool | None = None
    email_immediate: bool | EmailImmediateAction | None = None
    email_digest: str | None = None
    pdf_report: str | None = None
    frame_capture: bool | FrameCaptureAction | None = None


class EventConfig(StrictModel):
    """Event definition."""

    name: str = Field(..., min_length=1)
    match: EventMatch
    actions: EventActions


class FrameConfig(BaseModel):
    """Frame configuration for digests."""

    cooldown_seconds: int = Field(default=300, ge=0)


class DigestConfig(StrictModel):
    """Digest configuration."""

    id: str = Field(..., min_length=1)
    period_minutes: int = Field(
        ..., gt=0, description="Period in minutes between digest emails"
    )
    period_label: str = Field(default="")
    subject: str = Field(default="", description="Email subject line")
    events: list[str] = Field(
        default_factory=list, description="Event definition names to include"
    )
    photos: bool = False
    frame_config: FrameConfig | None = None


class PDFReportConfig(StrictModel):
    """PDF report configuration. Generated on shutdown covering the entire run."""

    id: str = Field(..., min_length=1)
    output_dir: str = Field(default="reports", description="Directory for PDF output")
    title: str = Field(default="Object Detection Report", description="Report title")
    events: list[str] = Field(
        default_factory=list, description="Event definition names to include"
    )
    photos: bool = False
    annotate: bool = False  # Draw lines/zones/bboxes on photos
    frame_config: FrameConfig | None = None


class EmailConfig(BaseModel):
    """Email notification settings."""

    enabled: bool = False
    smtp_server: str | None = None
    smtp_port: int | None = Field(default=587, ge=1, le=65535)
    use_tls: bool = True
    username: str | None = None
    password: str | None = None
    from_address: str | None = None
    to_addresses: list[str] | None = None


class NotificationsConfig(BaseModel):
    """Notifications configuration."""

    enabled: bool = False
    email: EmailConfig = Field(default_factory=EmailConfig)


class OutputConfig(BaseModel):
    """Output configuration."""

    json_dir: str = Field(default="data")


class CameraConfig(BaseModel):
    """Camera configuration."""

    url: str = Field(..., min_length=1)


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    default_duration_hours: float = Field(default=1.0, gt=0)
    queue_size: int = Field(default=1000, gt=0)
    analyzer_startup_delay: int = Field(default=1, ge=0)
    detector_shutdown_timeout: int = Field(default=5, ge=0)
    analyzer_shutdown_timeout: int = Field(default=10, ge=0)


class ConsoleOutputConfig(BaseModel):
    """Console output configuration."""

    enabled: bool = True
    level: Literal["detailed", "summary", "silent"] = "detailed"


class FrameStorageConfig(BaseModel):
    """Frame storage configuration."""

    type: Literal["local"] = "local"
    local_dir: str = "frames"
    retention_days: int = Field(default=7, ge=1, description="Days to retain frames")


class SpeedCalculationConfig(BaseModel):
    """Speed calculation configuration."""

    enabled: bool = False


class Config(StrictModel):
    """Complete configuration schema."""

    detection: DetectionConfig
    roi: ROIConfig = Field(default_factory=ROIConfig)
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    events: list[EventConfig] = Field(default_factory=list)
    digests: list[DigestConfig] = Field(default_factory=list)
    pdf_reports: list[PDFReportConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    camera: CameraConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    console_output: ConsoleOutputConfig = Field(default_factory=ConsoleOutputConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    frame_storage: FrameStorageConfig = Field(default_factory=FrameStorageConfig)
    speed_calculation: SpeedCalculationConfig = Field(
        default_factory=SpeedCalculationConfig
    )
    temp_frames_enabled: bool = False
    temp_frame_dir: str = "/tmp/frames"
    temp_frame_interval: int = 5
    temp_frame_max_age_seconds: int = 30

    @model_validator(mode="after")
    def validate_references(self):
        """Validate that events reference existing lines/zones."""
        line_names = {line.description for line in self.lines}
        zone_names = {zone.description for zone in self.zones}
        digest_ids = {digest.id for digest in self.digests}
        pdf_report_ids = {report.id for report in self.pdf_reports}

        for event in self.events:
            if event.match.line and event.match.line not in line_names:
                raise ValueError(
                    f"Event '{event.name}' references non-existent line: '{event.match.line}'"
                )
            if event.match.zone and event.match.zone not in zone_names:
                raise ValueError(
                    f"Event '{event.name}' references non-existent zone: '{event.match.zone}'"
                )
            if (
                event.actions.email_digest
                and event.actions.email_digest not in digest_ids
            ):
                raise ValueError(
                    f"Event '{event.name}' references non-existent digest: '{event.actions.email_digest}'"
                )
            if (
                event.actions.pdf_report
                and event.actions.pdf_report not in pdf_report_ids
            ):
                raise ValueError(
                    f"Event '{event.name}' references non-existent pdf_report: '{event.actions.pdf_report}'"
                )

        return self


def validate_config_pydantic(config: dict) -> Config:
    """
    Validate configuration using Pydantic.

    Args:
        config: Raw configuration dictionary

    Returns:
        Validated Config object

    Raises:
        pydantic.ValidationError: If validation fails
    """
    return Config(**config)
