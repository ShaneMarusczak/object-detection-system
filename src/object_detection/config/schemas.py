"""
Pydantic schemas for configuration validation.

Provides type-safe, declarative validation with clear error messages.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.constants import DEFAULT_TEMP_FRAME_DIR, DEFAULT_TEMP_FRAME_MAX_AGE


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


class NighttimeDetectionConfig(BaseModel):
    """
    Nighttime car detection parameters.

    Used when event_type is NIGHTTIME_CAR to configure blob detection.
    """

    brightness_threshold: int = Field(
        default=30, ge=0, le=255, description="Max brightness for nighttime mode"
    )
    min_blob_size: int = Field(
        default=100, ge=10, description="Minimum blob size in pixels"
    )
    max_blob_size: int = Field(
        default=10000, ge=100, description="Maximum blob size in pixels"
    )
    score_threshold: int = Field(
        default=85, ge=0, le=200, description="Minimum score to trigger detection"
    )
    taillight_color_match: bool = Field(
        default=True, description="Require red/orange color match"
    )


class EventMatch(BaseModel):
    """Event matching criteria."""

    event_type: (
        None
        | (
            Literal[
                "LINE_CROSS",
                "ZONE_ENTER",
                "ZONE_EXIT",
                "NIGHTTIME_CAR",
                "DETECTED",
            ]
        )
    ) = None
    line: str | None = None
    zone: str | None = None
    object_class: str | list[str] | None = None
    direction: Literal["LTR", "RTL", "TTB", "BTT"] | None = None
    # Nighttime detection parameters (only used when event_type is NIGHTTIME_CAR)
    nighttime_detection: NighttimeDetectionConfig | None = None


class CommandAction(BaseModel):
    """Command execution action configuration."""

    exec: str = Field(..., min_length=1, description="Command or script to execute")
    timeout_seconds: int = Field(
        default=30, ge=1, le=300, description="Command timeout in seconds"
    )


class FrameCaptureAction(BaseModel):
    """Frame capture action configuration."""

    enabled: bool = True
    cooldown_seconds: int = Field(default=300, ge=0)
    annotate: bool = False


class VLMAnalyzeAction(BaseModel):
    """VLM analysis action configuration."""

    analyzer: str = Field(..., min_length=1, description="References analyzers[].id")
    prompt: str = Field(..., min_length=1, description="Prompt to send with image")
    notify: list[str] = Field(
        default_factory=list, description="References notifiers[].id"
    )


class NotifyActionItem(BaseModel):
    """
    Direct notification action item.

    Sends notifications directly without VLM analysis.
    """

    notifier: str = Field(..., min_length=1, description="References notifiers[].id")
    message: str = Field(
        ..., min_length=1, description="Message template with {variable} placeholders"
    )
    include_image: bool = Field(
        default=False, description="Attach captured frame to notification"
    )


class EventActions(BaseModel):
    """Event actions configuration."""

    json_log: bool | None = None
    command: CommandAction | None = None
    report: str | None = None
    frame_capture: bool | FrameCaptureAction | None = None
    vlm_analyze: VLMAnalyzeAction | None = None
    notify: list[NotifyActionItem] | None = None
    shutdown: bool = Field(
        default=False, description="Stop detector after this event triggers"
    )


class EventConfig(StrictModel):
    """Event definition."""

    name: str = Field(..., min_length=1)
    match: EventMatch
    actions: EventActions


class FrameConfig(BaseModel):
    """Frame configuration for reports."""

    cooldown_seconds: int = Field(default=300, ge=0)


class ReportConfig(StrictModel):
    """Report configuration. Generated as HTML on shutdown covering the entire run."""

    id: str = Field(..., min_length=1)
    output_dir: str = Field(
        default="reports", description="Directory for report output"
    )
    title: str = Field(default="Object Detection Report", description="Report title")
    events: list[str] = Field(
        default_factory=list, description="Event definition names to include"
    )
    photos: bool = False
    annotate: bool = False  # Draw lines/zones/bboxes on photos
    frame_config: FrameConfig | None = None


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


class ConsoleOutputConfig(BaseModel):
    """Console output configuration."""

    enabled: bool = True
    level: Literal["detailed", "summary", "silent"] = "detailed"


class FrameStorageConfig(BaseModel):
    """Frame storage configuration."""

    type: Literal["local"] = "local"
    local_dir: str = "frames"
    retention_days: int = Field(default=7, ge=1, description="Days to retain frames")


class NotifierConfig(StrictModel):
    """
    Top-level notifier definition.

    Notifiers are reusable notification endpoints referenced by vlm_analyze actions.
    """

    id: str = Field(..., min_length=1, description="Unique notifier identifier")
    type: Literal["ntfy", "webhook"] = Field(..., description="Notifier type")
    # ntfy-specific
    topic: str | None = Field(default=None, description="ntfy topic name")
    # webhook-specific
    url: str | None = Field(default=None, description="Webhook URL")
    # Common
    priority: Literal["min", "low", "default", "high", "urgent"] = Field(
        default="default", description="Notification priority"
    )
    title_template: str | None = Field(
        default=None, description="Title template with {variable} placeholders"
    )

    @model_validator(mode="after")
    def validate_type_fields(self):
        """Validate that required fields are present for each type."""
        if self.type == "ntfy" and not self.topic:
            raise ValueError("ntfy notifier requires 'topic' field")
        if self.type == "webhook" and not self.url:
            raise ValueError("webhook notifier requires 'url' field")
        return self


class AnalyzerConfig(StrictModel):
    """
    Top-level analyzer definition.

    Analyzers are VLM endpoints (e.g., Orin 2) that process images.
    """

    id: str = Field(..., min_length=1, description="Unique analyzer identifier")
    url: str = Field(..., min_length=1, description="Analyzer endpoint URL")
    timeout_seconds: int = Field(
        default=60, ge=1, le=300, description="Request timeout"
    )


class Config(StrictModel):
    """Complete configuration schema."""

    detection: DetectionConfig
    roi: ROIConfig = Field(default_factory=ROIConfig)
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    events: list[EventConfig] = Field(default_factory=list)
    reports: list[ReportConfig] = Field(default_factory=list)
    notifiers: list[NotifierConfig] = Field(default_factory=list)
    analyzers: list[AnalyzerConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    camera: CameraConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    console_output: ConsoleOutputConfig = Field(default_factory=ConsoleOutputConfig)
    frame_storage: FrameStorageConfig = Field(default_factory=FrameStorageConfig)
    temp_frames_enabled: bool = False
    temp_frame_dir: str = DEFAULT_TEMP_FRAME_DIR
    temp_frame_max_age_seconds: int = DEFAULT_TEMP_FRAME_MAX_AGE

    @model_validator(mode="after")
    def validate_references(self):
        """Validate that events reference existing lines/zones/analyzers/notifiers."""
        line_names = {line.description for line in self.lines}
        zone_names = {zone.description for zone in self.zones}
        report_ids = {report.id for report in self.reports}
        analyzer_ids = {analyzer.id for analyzer in self.analyzers}
        notifier_ids = {notifier.id for notifier in self.notifiers}

        for event in self.events:
            if event.match.line and event.match.line not in line_names:
                raise ValueError(
                    f"Event '{event.name}' references non-existent line: '{event.match.line}'"
                )
            if event.match.zone and event.match.zone not in zone_names:
                raise ValueError(
                    f"Event '{event.name}' references non-existent zone: '{event.match.zone}'"
                )
            # NIGHTTIME_CAR events must reference a zone
            if event.match.event_type == "NIGHTTIME_CAR" and not event.match.zone:
                raise ValueError(
                    f"Event '{event.name}' with event_type NIGHTTIME_CAR must specify a zone"
                )
            if event.actions.report and event.actions.report not in report_ids:
                raise ValueError(
                    f"Event '{event.name}' references non-existent report: '{event.actions.report}'"
                )
            # Validate vlm_analyze references
            if event.actions.vlm_analyze:
                vlm = event.actions.vlm_analyze
                if vlm.analyzer not in analyzer_ids:
                    raise ValueError(
                        f"Event '{event.name}' references non-existent analyzer: '{vlm.analyzer}'"
                    )
                for notify_id in vlm.notify:
                    if notify_id not in notifier_ids:
                        raise ValueError(
                            f"Event '{event.name}' references non-existent notifier: '{notify_id}'"
                        )
            # Validate direct notify references
            if event.actions.notify:
                for notify_item in event.actions.notify:
                    if notify_item.notifier not in notifier_ids:
                        raise ValueError(
                            f"Event '{event.name}' references non-existent notifier: '{notify_item.notifier}'"
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
