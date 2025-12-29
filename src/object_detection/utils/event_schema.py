"""
Event Schema - Contract between edge detector and processor.

Defines the event format used for communication between detection and processing.
In local mode, events flow via multiprocessing.Queue.
In distributed mode, events flow via Redis Streams.

All producers (detector, edge detector, nighttime zones) and consumers
(dispatcher, json_writer, command_runner, etc.) must adhere to this schema.

Event Types:
    LINE_CROSS: Object crossed a counting line
    ZONE_ENTER: Object entered a zone
    ZONE_EXIT: Object exited a zone
    NIGHTTIME_CAR: Vehicle detected at night via blob scoring
    DETECTED: Raw object detection (no geometry/tracking required)
"""

from typing import Literal, TypedDict

# Event type constants
EVENT_TYPE_LINE_CROSS = "LINE_CROSS"
EVENT_TYPE_ZONE_ENTER = "ZONE_ENTER"
EVENT_TYPE_ZONE_EXIT = "ZONE_EXIT"
EVENT_TYPE_NIGHTTIME_CAR = "NIGHTTIME_CAR"
EVENT_TYPE_DETECTED = "DETECTED"

EventType = Literal[
    "LINE_CROSS", "ZONE_ENTER", "ZONE_EXIT", "NIGHTTIME_CAR", "DETECTED"
]

# Direction constants for line crossings
DIRECTION_LTR = "LTR"  # Left to right
DIRECTION_RTL = "RTL"  # Right to left
DIRECTION_TTB = "TTB"  # Top to bottom
DIRECTION_BTT = "BTT"  # Bottom to top

Direction = Literal["LTR", "RTL", "TTB", "BTT"]


class BaseEvent(TypedDict, total=False):
    """
    Common fields present in all events.

    Required fields:
        event_type: Type of event (LINE_CROSS, ZONE_ENTER, etc.)
        track_id: Unique identifier for the tracked object
        object_class: COCO class ID (0-79) or synthetic ID (1000 for nighttime_car)

    Optional fields:
        bbox: Bounding box as (x1, y1, x2, y2) tuple
        frame_id: UUID of saved frame in temp storage
    """

    event_type: EventType
    track_id: int | str  # int for YOLO, "nc_N" for nighttime car
    object_class: int
    bbox: tuple[int, int, int, int]
    frame_id: str | None


class LineCrossEvent(BaseEvent):
    """
    LINE_CROSS event - object crossed a counting line.

    Additional fields:
        line_id: Line identifier (V1, V2, H1, etc.)
        direction: Crossing direction (LTR, RTL, TTB, BTT)
    """

    line_id: str
    direction: Direction


class ZoneEnterEvent(BaseEvent):
    """
    ZONE_ENTER event - object entered a monitoring zone.

    Additional fields:
        zone_id: Zone identifier (Z1, Z2, etc.)
    """

    zone_id: str


class ZoneExitEvent(BaseEvent):
    """
    ZONE_EXIT event - object exited a monitoring zone.

    Additional fields:
        zone_id: Zone identifier (Z1, Z2, etc.)
        dwell_time: Seconds spent inside zone
    """

    zone_id: str
    dwell_time: float


class NighttimeCarEvent(BaseEvent):
    """
    NIGHTTIME_CAR event - vehicle detected via blob scoring.

    Uses synthetic class ID 1000 (not a real COCO class).
    Detection is based on headlight/taillight blob analysis,
    not YOLO inference.

    Additional fields:
        zone_id: Zone where detection occurred
        score: Detection confidence score (0-100+)
        had_taillight: True if taillight matched headlight
    """

    zone_id: str
    score: float
    had_taillight: bool


class DetectedEvent(BaseEvent):
    """
    DETECTED event - raw object detection.

    Fires for every YOLO detection. No zone/line geometry required.
    Simplest event type - pure detection to event.

    Additional fields:
        confidence: Detection confidence (0-1)

    Optional fields:
        track_id: Only present if tracking is enabled
    """

    confidence: float


# Synthetic class ID for nighttime car (outside COCO range 0-79)
NIGHTTIME_CAR_CLASS_ID = 1000

# Union type for all events
Event = (
    LineCrossEvent | ZoneEnterEvent | ZoneExitEvent | NighttimeCarEvent | DetectedEvent
)


def is_valid_event(event: dict) -> bool:
    """
    Validate that an event has required fields.

    Args:
        event: Event dictionary to validate

    Returns:
        True if event has required base fields
    """
    # DETECTED events don't require track_id (tracking may be disabled)
    if event.get("event_type") == EVENT_TYPE_DETECTED:
        required = {"event_type", "object_class", "confidence"}
    else:
        required = {"event_type", "track_id", "object_class"}
    return required.issubset(event.keys())


def get_event_summary(event: dict) -> str:
    """
    Get a human-readable summary of an event.

    Args:
        event: Event dictionary

    Returns:
        Summary string for logging
    """
    event_type = event.get("event_type", "UNKNOWN")
    track_id = event.get("track_id", "?")

    if event_type == EVENT_TYPE_LINE_CROSS:
        line_id = event.get("line_id", "?")
        direction = event.get("direction", "?")
        return f"LINE_CROSS track={track_id} line={line_id} dir={direction}"

    elif event_type == EVENT_TYPE_ZONE_ENTER:
        zone_id = event.get("zone_id", "?")
        return f"ZONE_ENTER track={track_id} zone={zone_id}"

    elif event_type == EVENT_TYPE_ZONE_EXIT:
        zone_id = event.get("zone_id", "?")
        dwell = event.get("dwell_time", 0)
        return f"ZONE_EXIT track={track_id} zone={zone_id} dwell={dwell:.1f}s"

    elif event_type == EVENT_TYPE_NIGHTTIME_CAR:
        zone_id = event.get("zone_id", "?")
        score = event.get("score", 0)
        return f"NIGHTTIME_CAR track={track_id} zone={zone_id} score={score:.0f}"

    elif event_type == EVENT_TYPE_DETECTED:
        conf = event.get("confidence", 0)
        return f"DETECTED track={track_id} conf={conf:.2f}"

    else:
        return f"{event_type} track={track_id}"
