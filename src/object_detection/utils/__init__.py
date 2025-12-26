"""
Utility modules for constants and abstractions.

Note: COCO class mappings are in processor/coco_classes.py
(enrichment is a processor responsibility, not edge)
"""

from .constants import (
    ENV_CAMERA_URL,
    DEFAULT_QUEUE_SIZE,
    SUMMARY_EVENT_INTERVAL,
    DEFAULT_TEMP_FRAME_DIR,
    DEFAULT_TEMP_FRAME_MAX_AGE,
)
from .queue_protocol import EventQueue, CallbackQueueAdapter
from .event_schema import (
    EVENT_TYPE_LINE_CROSS,
    EVENT_TYPE_ZONE_ENTER,
    EVENT_TYPE_ZONE_EXIT,
    EVENT_TYPE_NIGHTTIME_CAR,
    NIGHTTIME_CAR_CLASS_ID,
    is_valid_event,
    get_event_summary,
)

__all__ = [
    "ENV_CAMERA_URL",
    "DEFAULT_QUEUE_SIZE",
    "SUMMARY_EVENT_INTERVAL",
    "DEFAULT_TEMP_FRAME_DIR",
    "DEFAULT_TEMP_FRAME_MAX_AGE",
    # Queue abstraction for distributed deployment
    "EventQueue",
    "CallbackQueueAdapter",
    # Event schema
    "EVENT_TYPE_LINE_CROSS",
    "EVENT_TYPE_ZONE_ENTER",
    "EVENT_TYPE_ZONE_EXIT",
    "EVENT_TYPE_NIGHTTIME_CAR",
    "NIGHTTIME_CAR_CLASS_ID",
    "is_valid_event",
    "get_event_summary",
]
