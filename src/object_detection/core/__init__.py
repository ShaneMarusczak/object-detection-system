"""
Core detection components.

NOTE: dispatcher has moved to processor/ module.
This module contains the detector and foundational classes.
Models have moved to the models/ package.
"""

from .detector import run_detection
from ..models import TrackedObject, LineConfig, ZoneConfig, ROIConfig, EventDefinition

__all__ = [
    "run_detection",
    "TrackedObject",
    "LineConfig",
    "ZoneConfig",
    "ROIConfig",
    "EventDefinition",
]
