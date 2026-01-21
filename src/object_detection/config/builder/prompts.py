"""
Prompt styling and constants for the config builder TUI.

Provides shared styling for questionary prompts and ANSI color codes.
"""

from questionary import Style

# Custom style for questionary prompts
PROMPT_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("instruction", "fg:gray"),
    ]
)

# Common object classes for quick selection
COMMON_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "bird",
]


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
