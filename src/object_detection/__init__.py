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
# Configuration
from .config import (
    ConfigPlan,
    ConfigValidationError,
    ValidationResult,
    build_plan,
    validate_config_full,
)
from .core import (
    TrackedObject,
    run_detection,
)

# Processor (dispatcher, consumers)
from .processor import (
    COCO_CLASSES,
    EventDefinition,
    dispatch_events,
    get_class_name,
)

__all__ = [
    "COCO_CLASSES",
    "ConfigPlan",
    # Config
    "ConfigValidationError",
    "EventDefinition",
    "TrackedObject",
    "ValidationResult",
    "build_plan",
    # Processor
    "dispatch_events",
    "get_class_name",
    # Core
    "run_detection",
    "validate_config_full",
]
