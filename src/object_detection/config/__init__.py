"""
Configuration loading, validation, and planning.

Provides Terraform-like workflow:
- validate_config_full: Comprehensive validation with errors/warnings
- build_plan: Show what events will be routed where
- simulate_dry_run: Test with sample events
- load_config_with_env: Apply environment variable overrides

Pydantic schemas available for type-safe validation:
- Config: Complete configuration schema
- validate_config_pydantic: Validate and parse config to Pydantic model
"""

from .builder import run_builder, run_editor
from .planner import (
    ConfigPlan,
    # Exception
    ConfigValidationError,
    EventPlan,
    # Planning
    build_plan,
    generate_sample_events,
    # Config loading
    load_config_with_env,
    load_sample_events,
    print_plan,
    # Display
    print_validation_result,
    # Dry-run
    simulate_dry_run,
)
from .resolver import (
    derive_track_classes,
    prepare_runtime_config,
)
from .schemas import (
    Config,
    # Sub-schemas for type hints
    DetectionConfig,
    EventConfig,
    LineConfig,
    ReportConfig,
    ZoneConfig,
    validate_config_pydantic,
)
from .validator import (
    ValidationResult,
    validate_config_full,
)

__all__ = [
    # Pydantic validation
    "Config",
    "ConfigPlan",
    # Exception
    "ConfigValidationError",
    "DetectionConfig",
    "EventConfig",
    "EventPlan",
    "LineConfig",
    "ReportConfig",
    "ValidationResult",
    "ZoneConfig",
    # Planning
    "build_plan",
    "derive_track_classes",
    "generate_sample_events",
    # Config loading & preparation
    "load_config_with_env",
    "load_sample_events",
    "prepare_runtime_config",
    "print_plan",
    # Display
    "print_validation_result",
    # Builder
    "run_builder",
    "run_editor",
    # Dry-run
    "simulate_dry_run",
    # Validation
    "validate_config_full",
    "validate_config_pydantic",
]
