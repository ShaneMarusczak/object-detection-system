"""
Camera initialization and management.
"""

import logging
import time

import cv2

from ..utils.constants import MAX_CAMERA_RECONNECT_ATTEMPTS, CAMERA_RECONNECT_DELAY

logger = logging.getLogger(__name__)


def initialize_camera(camera_url: str) -> cv2.VideoCapture:
    """
    Initialize camera with retry logic.

    Args:
        camera_url: Camera URL or device path

    Returns:
        OpenCV VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened after retries
    """
    for attempt in range(MAX_CAMERA_RECONNECT_ATTEMPTS + 1):
        logger.info(f"Connecting to camera: {camera_url} (attempt {attempt + 1})")
        cap = cv2.VideoCapture(camera_url)

        if cap.isOpened():
            logger.info("Camera connected successfully")
            return cap

        if attempt < MAX_CAMERA_RECONNECT_ATTEMPTS:
            logger.warning(
                f"Failed to connect, retrying in {CAMERA_RECONNECT_DELAY}s..."
            )
            time.sleep(CAMERA_RECONNECT_DELAY)
        else:
            logger.error(
                f"Failed to connect to camera after {MAX_CAMERA_RECONNECT_ATTEMPTS + 1} attempts"
            )
            raise RuntimeError(f"Cannot connect to camera: {camera_url}")

    raise RuntimeError(f"Cannot connect to camera: {camera_url}")
