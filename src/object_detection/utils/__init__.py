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

__all__ = [
    "ENV_CAMERA_URL",
    "DEFAULT_QUEUE_SIZE",
    "SUMMARY_EVENT_INTERVAL",
    "DEFAULT_TEMP_FRAME_DIR",
    "DEFAULT_TEMP_FRAME_MAX_AGE",
    # Queue abstraction for distributed deployment
    "EventQueue",
    "CallbackQueueAdapter",
]
