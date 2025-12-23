"""
Nighttime Detection Module

Detects headlights when it's dark by looking for bright blobs.
Uses streetlight presence in upper frame as darkness indicator.
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Virtual class ID for headlight detections (not in COCO)
HEADLIGHT_CLASS_ID = 1000
HEADLIGHT_CLASS_NAME = "headlight"


@dataclass
class BlobDetection:
    """A detected bright blob (potential headlight)."""

    center: tuple[float, float]  # (x, y) center position
    size: float  # blob size/radius
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box


class LightingMonitor:
    """
    Monitors lighting conditions to determine day/night mode.

    Uses streetlight presence in upper frame as darkness indicator:
    - Streetlight ON → it's dark → night mode
    - High overall brightness → it's day → day mode

    Checks on startup and every 15 minutes thereafter.
    """

    # Streetlight detection region (upper portion of frame)
    STREETLIGHT_Y_PCT = (0, 25)  # Top 25%
    STREETLIGHT_X_PCT = (20, 90)  # Wide range to catch edge streetlights

    # Thresholds
    BRIGHTNESS_THRESHOLD = 200  # L channel threshold for "bright"
    MIN_BRIGHT_PIXELS = (
        100  # Minimum bright pixels to detect streetlight (small lights ok)
    )
    DAYLIGHT_BRIGHTNESS = 80  # Avg frame brightness for "it's daytime"

    # Timing
    CHECK_INTERVAL_SECONDS = 900  # 15 minutes

    def __init__(self, enabled: bool = True):
        """
        Initialize lighting monitor.

        Args:
            enabled: If False, always returns day mode (YOLO only)
        """
        self.enabled = enabled
        self.is_dark = False
        self.last_check_time: float | None = None
        self.frames_checked = 0

        if enabled:
            logger.info("Nighttime detection enabled")
            logger.info(
                f"  Streetlight ROI: Y {self.STREETLIGHT_Y_PCT}, X {self.STREETLIGHT_X_PCT}"
            )
            logger.info(f"  Check interval: {self.CHECK_INTERVAL_SECONDS}s")

    def should_check(self) -> bool:
        """Determine if we should check lighting conditions."""
        if not self.enabled:
            return False

        # Always check on first few frames
        if self.frames_checked < 5:
            return True

        # Check every CHECK_INTERVAL_SECONDS
        if self.last_check_time is None:
            return True

        return (time.time() - self.last_check_time) >= self.CHECK_INTERVAL_SECONDS

    def check_lighting(self, frame: np.ndarray) -> bool:
        """
        Check lighting conditions and update mode.

        Args:
            frame: BGR frame from camera

        Returns:
            True if dark (night mode), False if bright (day mode)
        """
        if not self.enabled:
            return False

        self.frames_checked += 1
        self.last_check_time = time.time()

        h, w = frame.shape[:2]

        if self.is_dark:
            # Currently night mode - check if it's getting bright (dawn)
            avg_brightness = self._get_avg_brightness(frame)
            if avg_brightness > self.DAYLIGHT_BRIGHTNESS:
                logger.info(
                    f"Daylight detected (brightness: {avg_brightness:.0f}) - switching to day mode"
                )
                self.is_dark = False
        else:
            # Currently day mode - check for streetlight
            has_streetlight = self._detect_streetlight(frame, h, w)
            if has_streetlight:
                logger.info("Streetlight detected - switching to night mode")
                self.is_dark = True

        return self.is_dark

    def _detect_streetlight(self, frame: np.ndarray, h: int, w: int) -> bool:
        """Check for bright blob in streetlight region."""
        # Extract ROI
        y1 = int(h * self.STREETLIGHT_Y_PCT[0] / 100)
        y2 = int(h * self.STREETLIGHT_Y_PCT[1] / 100)
        x1 = int(w * self.STREETLIGHT_X_PCT[0] / 100)
        x2 = int(w * self.STREETLIGHT_X_PCT[1] / 100)

        roi = frame[y1:y2, x1:x2]

        # Convert to LAB and threshold L channel
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, binary = cv2.threshold(
            l_channel, self.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # Count bright pixels
        bright_pixels = cv2.countNonZero(binary)

        detected = bright_pixels >= self.MIN_BRIGHT_PIXELS
        logger.debug(
            f"Streetlight check: ROI y={y1}-{y2} x={x1}-{x2}, "
            f"bright_pixels={bright_pixels}, threshold={self.MIN_BRIGHT_PIXELS}, detected={detected}"
        )

        return detected

    def _get_avg_brightness(self, frame: np.ndarray) -> float:
        """Get average brightness of frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())


class HeadlightDetector:
    """
    Detects bright blobs (headlights) in the lower portion of frame.

    Used during night mode when YOLO struggles to detect vehicles.
    Outputs detections in same format as YOLO for seamless integration.
    """

    # Detection region (lower 70% of frame - excludes streetlight area)
    DETECTION_Y_START_PCT = 30  # Start at 30% from top

    # Blob detector parameters
    BRIGHTNESS_THRESHOLD = 200
    MIN_BLOB_AREA = 50
    MAX_BLOB_AREA = 5000
    MIN_CIRCULARITY = 0.3

    def __init__(self):
        """Initialize headlight detector with blob detection parameters."""
        self.blob_detector = self._create_blob_detector()
        logger.debug("HeadlightDetector initialized")

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
        """Create OpenCV blob detector with tuned parameters."""
        params = cv2.SimpleBlobDetector_Params()

        # Filter by color (bright blobs)
        params.filterByColor = True
        params.blobColor = 255

        # Filter by area
        params.filterByArea = True
        params.minArea = self.MIN_BLOB_AREA
        params.maxArea = self.MAX_BLOB_AREA

        # Filter by circularity (headlights are roughly circular)
        params.filterByCircularity = True
        params.minCircularity = self.MIN_CIRCULARITY

        # Don't filter by convexity or inertia
        params.filterByConvexity = False
        params.filterByInertia = False

        return cv2.SimpleBlobDetector_create(params)

    def detect(self, frame: np.ndarray) -> list[BlobDetection]:
        """
        Detect bright blobs (headlights) in frame.

        Args:
            frame: BGR frame from camera

        Returns:
            List of BlobDetection objects
        """
        h, w = frame.shape[:2]

        # Only look at lower portion of frame (exclude streetlight region)
        y_start = int(h * self.DETECTION_Y_START_PCT / 100)
        detection_region = frame[y_start:, :]

        # Convert to LAB and threshold L channel
        lab = cv2.cvtColor(detection_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, binary = cv2.threshold(
            l_channel, self.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # Detect blobs
        keypoints = self.blob_detector.detect(binary)

        # Convert to BlobDetection objects with adjusted coordinates
        detections = []
        for kp in keypoints:
            # Adjust Y coordinate back to full frame
            cx = kp.pt[0]
            cy = kp.pt[1] + y_start
            radius = kp.size / 2

            # Create bounding box around blob
            x1 = int(cx - radius)
            y1 = int(cy - radius)
            x2 = int(cx + radius)
            y2 = int(cy + radius)

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            detections.append(
                BlobDetection(center=(cx, cy), size=kp.size, bbox=(x1, y1, x2, y2))
            )

        return detections

    def detections_to_tracks(
        self,
        detections: list[BlobDetection],
        existing_tracks: dict,
        current_time: float,
        max_distance: float = 50.0,
    ) -> list[tuple[int, BlobDetection]]:
        """
        Associate blob detections with track IDs.

        Simple nearest-neighbor association with existing headlight tracks.
        Creates new tracks for unmatched detections.

        Args:
            detections: List of blob detections
            existing_tracks: Dict of track_id -> TrackedObject for headlight tracks
            current_time: Current timestamp
            max_distance: Max pixels to associate with existing track

        Returns:
            List of (track_id, detection) tuples
        """
        # Track IDs for headlights start at 10000 to avoid collision with YOLO
        if not hasattr(self, "_next_track_id"):
            self._next_track_id = 10000

        results = []
        used_tracks = set()

        for detection in detections:
            cx, cy = detection.center
            best_track_id = None
            best_distance = max_distance

            # Find nearest existing track
            for track_id, tracked_obj in existing_tracks.items():
                if track_id in used_tracks:
                    continue
                if tracked_obj.object_class != HEADLIGHT_CLASS_ID:
                    continue

                tx, ty = tracked_obj.current_pos
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Associate with existing track
                used_tracks.add(best_track_id)
                results.append((best_track_id, detection))
            else:
                # Create new track
                results.append((self._next_track_id, detection))
                self._next_track_id += 1

        return results
