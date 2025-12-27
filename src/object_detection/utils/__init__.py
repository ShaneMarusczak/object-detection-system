"""
Utility modules for constants and abstractions.

Note: COCO class mappings are in processor/coco_classes.py
(enrichment is a processor responsibility, not edge)
"""

from .constants import (
    DEFAULT_QUEUE_SIZE,
    DEFAULT_TEMP_FRAME_DIR,
    DEFAULT_TEMP_FRAME_MAX_AGE,
    ENV_CAMERA_URL,
    SUMMARY_EVENT_INTERVAL,
)
from .event_schema import (
    EVENT_TYPE_LINE_CROSS,
    EVENT_TYPE_NIGHTTIME_CAR,
    EVENT_TYPE_ZONE_ENTER,
    EVENT_TYPE_ZONE_EXIT,
    NIGHTTIME_CAR_CLASS_ID,
    get_event_summary,
    is_valid_event,
)
from .queue_protocol import CallbackQueueAdapter, EventQueue

__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "DEFAULT_TEMP_FRAME_DIR",
    "DEFAULT_TEMP_FRAME_MAX_AGE",
    "ENV_CAMERA_URL",
    # Event schema
    "EVENT_TYPE_LINE_CROSS",
    "EVENT_TYPE_NIGHTTIME_CAR",
    "EVENT_TYPE_ZONE_ENTER",
    "EVENT_TYPE_ZONE_EXIT",
    "NIGHTTIME_CAR_CLASS_ID",
    "SUMMARY_EVENT_INTERVAL",
    "CallbackQueueAdapter",
    # Queue abstraction for distributed deployment
    "EventQueue",
    "get_event_summary",
    "is_valid_event",
]
