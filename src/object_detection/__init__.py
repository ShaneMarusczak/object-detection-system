"""
Object Detection System

A production-quality object detection system for tracking movement across
boundaries and through zones. Built on YOLO and ByteTrack.
"""

__version__ = "2.0.0"
__author__ = "Shane"

from .detector import run_detection
from .dispatcher import dispatch_events, derive_track_classes, EventDefinition
from .config import validate_config, ConfigValidationError
from .models import TrackedObject

__all__ = [
    "run_detection",
    "dispatch_events",
    "derive_track_classes",
    "EventDefinition",
    "validate_config",
    "ConfigValidationError",
    "TrackedObject",
]
