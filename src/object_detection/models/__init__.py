"""
Consolidated data models for object detection.

This package contains all core data structures used across the application.
Edge deployment (edge/) maintains its own models for self-contained operation.
"""

from .tracking import TrackedObject, LineConfig, ZoneConfig, ROIConfig
from .events import EventDefinition
from .detector import Detector

__all__ = [
    # Tracking models
    "TrackedObject",
    "LineConfig",
    "ZoneConfig",
    "ROIConfig",
    # Event models
    "EventDefinition",
    # Protocols
    "Detector",
]
