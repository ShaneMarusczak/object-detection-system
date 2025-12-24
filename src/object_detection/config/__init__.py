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

from .planner import (
    # Exception
    ConfigValidationError,
    # Config loading
    load_config_with_env,
    # Planning
    build_plan,
    ConfigPlan,
    EventPlan,
    # Display
    print_validation_result,
    print_plan,
    # Dry-run
    simulate_dry_run,
    generate_sample_events,
    load_sample_events,
)

from .validator import (
    validate_config_full,
    ValidationResult,
)

from .resolver import (
    prepare_runtime_config,
    derive_track_classes,
)

from .schemas import (
    Config,
    validate_config_pydantic,
    # Sub-schemas for type hints
    DetectionConfig,
    EventConfig,
    LineConfig,
    ZoneConfig,
    DigestConfig,
    PDFReportConfig,
)

from .builder import run_builder, run_editor

__all__ = [
    # Exception
    "ConfigValidationError",
    # Config loading & preparation
    "load_config_with_env",
    "prepare_runtime_config",
    "derive_track_classes",
    # Validation
    "validate_config_full",
    "ValidationResult",
    # Pydantic validation
    "Config",
    "validate_config_pydantic",
    "DetectionConfig",
    "EventConfig",
    "LineConfig",
    "ZoneConfig",
    "DigestConfig",
    "PDFReportConfig",
    # Planning
    "build_plan",
    "ConfigPlan",
    "EventPlan",
    # Display
    "print_validation_result",
    "print_plan",
    # Dry-run
    "simulate_dry_run",
    "generate_sample_events",
    "load_sample_events",
    # Builder
    "run_builder",
    "run_editor",
]
