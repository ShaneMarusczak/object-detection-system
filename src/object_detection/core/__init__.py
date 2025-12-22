"""
Core detection and dispatching components.
"""

from .detector import run_detection
from .dispatcher import dispatch_events, derive_track_classes, EventDefinition
from .models import TrackedObject

__all__ = [
    "run_detection",
    "dispatch_events",
    "derive_track_classes",
    "EventDefinition",
    "TrackedObject",
]
