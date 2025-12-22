"""
Core detection components.

NOTE: dispatcher has moved to processor/ module.
This module now only contains the detector (for local/legacy mode).
"""

from .detector import run_detection
from .models import TrackedObject, LineConfig, ZoneConfig, ROIConfig

__all__ = [
    "run_detection",
    "TrackedObject",
    "LineConfig",
    "ZoneConfig",
    "ROIConfig",
]
