"""
Utility modules for constants and class mappings.
"""

from .constants import ENV_CAMERA_URL, DEFAULT_QUEUE_SIZE, SUMMARY_EVENT_INTERVAL
from .coco_classes import COCO_CLASSES, COCO_NAME_TO_ID, get_class_id, get_class_name

__all__ = [
    "ENV_CAMERA_URL",
    "DEFAULT_QUEUE_SIZE",
    "SUMMARY_EVENT_INTERVAL",
    "COCO_CLASSES",
    "COCO_NAME_TO_ID",
    "get_class_id",
    "get_class_name",
]
