"""
Object Detection System

A production-quality object detection system for tracking movement across
boundaries and through zones. Built on YOLO and ByteTrack.

Supports Terraform-like workflow:
  --validate  Check configuration validity
  --plan      Show event routing plan
  --dry-run   Simulate with sample events

Package structure:
  edge/       - Minimal detector for Jetson (detection only)
  processor/  - Event processing for k3s (dispatcher, consumers)
  core/       - Legacy combined mode (runs both on same machine)
  config/     - Configuration loading and validation
  utils/      - Constants
"""

__version__ = "2.2.0"
__author__ = "Shane"

# Core detection
from .core import (
    run_detection,
    TrackedObject,
)

# Processor (dispatcher, consumers)
from .processor import (
    dispatch_events,
    derive_track_classes,
    EventDefinition,
    COCO_CLASSES,
    get_class_name,
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
    "TrackedObject",
    # Processor
    "dispatch_events",
    "derive_track_classes",
    "EventDefinition",
    "COCO_CLASSES",
    "get_class_name",
    # Config
    "ConfigValidationError",
    "validate_config_full",
    "build_plan",
    "ConfigPlan",
    "ValidationResult",
]
