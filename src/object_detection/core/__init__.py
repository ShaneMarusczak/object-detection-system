"""
Core detection components.

NOTE: dispatcher has moved to processor/ module.
This module contains the detector and foundational classes.
"""

from .detector import run_detection
from .models import TrackedObject, LineConfig, ZoneConfig, ROIConfig
from .event_definition import EventDefinition

__all__ = [
    "run_detection",
    "TrackedObject",
    "LineConfig",
    "ZoneConfig",
    "ROIConfig",
    "EventDefinition",
]
