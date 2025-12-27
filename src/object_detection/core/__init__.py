"""
Core detection components.

NOTE: dispatcher has moved to processor/ module.
This module contains the detector and foundational classes.
Models have moved to the models/ package.
"""

from ..models import EventDefinition, LineConfig, ROIConfig, TrackedObject, ZoneConfig
from .detector import run_detection

__all__ = [
    "EventDefinition",
    "LineConfig",
    "ROIConfig",
    "TrackedObject",
    "ZoneConfig",
    "run_detection",
]
