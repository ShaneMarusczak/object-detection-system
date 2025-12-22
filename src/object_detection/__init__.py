"""
Object Detection System

A production-quality object detection system for tracking movement across
boundaries and through zones. Built on YOLO and ByteTrack.

Supports Terraform-like workflow:
  --validate  Check configuration validity
  --plan      Show event routing plan
  --dry-run   Simulate with sample events
"""

__version__ = "2.1.0"
__author__ = "Shane"

from .detector import run_detection
from .dispatcher import dispatch_events, derive_track_classes, EventDefinition
from .config import ConfigValidationError
from .models import TrackedObject
from .planner import validate_config_full, build_plan, ConfigPlan, ValidationResult

__all__ = [
    "run_detection",
    "dispatch_events",
    "derive_track_classes",
    "EventDefinition",
    "validate_config_full",
    "build_plan",
    "ConfigPlan",
    "ValidationResult",
    "ConfigValidationError",
    "TrackedObject",
]
