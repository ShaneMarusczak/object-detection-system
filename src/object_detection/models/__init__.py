"""
Consolidated data models for object detection.

This package contains all core data structures used across the application.
Edge deployment (edge/) maintains its own models for self-contained operation.
"""

from .detector import Detector
from .events import EventDefinition
from .tracking import LineConfig, ROIConfig, TrackedObject, ZoneConfig

__all__ = [
    # Protocols
    "Detector",
    # Event models
    "EventDefinition",
    "LineConfig",
    "ROIConfig",
    # Tracking models
    "TrackedObject",
    "ZoneConfig",
]
