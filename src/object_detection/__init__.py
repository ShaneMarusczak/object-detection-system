"""
Object Detection System

A production-quality object detection system for tracking movement across
boundaries and through zones. Built on YOLO and ByteTrack.

Supports Terraform-like workflow:
  --validate  Check configuration validity
  --plan      Show event routing plan
  --dry-run   Simulate with sample events

Package structure:
  core/       - Detection and event dispatching
  config/     - Configuration loading and validation
  consumers/  - Event consumers (JSON, email, frames)
  utils/      - Constants and COCO class mappings
"""

__version__ = "2.1.0"
__author__ = "Shane"

# Core components
from .core import (
    run_detection,
    dispatch_events,
    derive_track_classes,
    EventDefinition,
    TrackedObject,
)

# Configuration
from .config import (
    ConfigValidationError,
    validate_config_full,
    build_plan,
    ConfigPlan,
    ValidationResult,
)

__all__ = [
    # Core
    "run_detection",
    "dispatch_events",
    "derive_track_classes",
    "EventDefinition",
    "TrackedObject",
    # Config
    "ConfigValidationError",
    "validate_config_full",
    "build_plan",
    "ConfigPlan",
    "ValidationResult",
]
