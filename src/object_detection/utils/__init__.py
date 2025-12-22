"""
Utility modules for constants.

Note: COCO class mappings are in processor/coco_classes.py
(enrichment is a processor responsibility, not edge)
"""

from .constants import ENV_CAMERA_URL, DEFAULT_QUEUE_SIZE, SUMMARY_EVENT_INTERVAL

__all__ = [
    "ENV_CAMERA_URL",
    "DEFAULT_QUEUE_SIZE",
    "SUMMARY_EVENT_INTERVAL",
]
