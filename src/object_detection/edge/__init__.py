"""
Edge Detection Module (Jetson)

Minimal detection-only component for edge deployment.
Runs YOLO + ByteTrack and publishes raw events.

Does NOT include:
- Event enrichment (class names, descriptions)
- Email/notifications
- Digests
- Frame capture for events
- Speed calculation
"""

from .detector import EdgeDetector
from .config import EdgeConfig

__all__ = ['EdgeDetector', 'EdgeConfig']
