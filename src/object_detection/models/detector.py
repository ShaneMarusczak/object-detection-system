"""
Detector Protocol - Common interface for all detection algorithms.

Any detection method (YOLO, nighttime blob detection, thermal, etc.)
can implement this protocol to integrate with the detection loop.
"""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class Detector(Protocol):
    """
    Protocol for detection algorithms.

    Implementations process frames and emit events via callback.
    This abstraction allows mixing different detection methods
    in the same detection loop.

    Example:
        detectors: list[Detector] = [yolo_detector, nighttime_detector]
        for detector in detectors:
            event_count += detector.process_frame(frame, on_event, relative_time)
    """

    def process_frame(
        self,
        frame: np.ndarray,
        on_event: Callable[[dict[str, Any]], None],
        relative_time: float,
    ) -> int:
        """
        Process a single frame and emit detected events.

        Args:
            frame: BGR frame from camera (numpy array)
            on_event: Callback to invoke for each detected event
            relative_time: Seconds since detection started

        Returns:
            Number of events emitted
        """
        ...
