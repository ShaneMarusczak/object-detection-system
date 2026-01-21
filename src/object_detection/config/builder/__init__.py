"""
Config Builder Package - Interactive configuration wizard.

Provides modular components for building and editing detection configs:
- wizard: Main ConfigBuilder class and entry points
- sections: Individual section setup functions
- events: Event configuration logic
- io: Config loading, saving, and preflight
- preview: Camera preview and HTTP server
- prompts: Styling and constants
"""

from .wizard import ConfigBuilder, run_builder, run_editor

__all__ = [
    "ConfigBuilder",
    "run_builder",
    "run_editor",
]
