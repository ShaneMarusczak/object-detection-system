"""
Entry point for running the object detection system as a module.

Usage:
    python -m object_detection [hours]
"""

from .cli import main

if __name__ == "__main__":
    main()
