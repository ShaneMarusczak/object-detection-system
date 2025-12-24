"""
Headlight Detection Module

Detects headlights by looking for bright blobs when YOLO detection fails.
Used as fallback detection for low-light conditions.
"""

import logging
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


class HeadlightDetector:
    """
    Detects bright blobs (headlights) in the lower portion of frame.

    Used as fallback when YOLO fails to detect vehicles.
    Runs every frame to maintain tracking, but results are only used
    when YOLO has no detections.
    """

    # Detection region (lower 80% of frame - excludes sky/streetlights)
    DETECTION_Y_START_PCT = 20  # Start at 20% from top

    # Blob detector parameters - tuned for wide range of headlight types
    BRIGHTNESS_THRESHOLD = 180  # Lower to catch dimmer LED headlights
    MIN_BLOB_AREA = 20  # Smaller to catch distant headlights
    MAX_BLOB_AREA = 15000  # Larger to catch close headlights
    MIN_CIRCULARITY = 0.1  # Lower to allow LED light bars and non-circular lights

    # Temporal filtering - require N consecutive frames before confirming
    MIN_FRAMES_TO_CONFIRM = 5

    def __init__(self):
        """Initialize headlight detector with blob detection parameters."""
        self.blob_detector = self._create_blob_detector()
        self._track_frame_counts: dict[int, int] = {}  # track_id -> consecutive frames seen
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
    ) -> list[tuple[int, BlobDetection, bool]]:
        """
        Associate blob detections with track IDs.

        Simple nearest-neighbor association with existing headlight tracks.
        Creates new tracks for unmatched detections.
        Tracks must be seen for MIN_FRAMES_TO_CONFIRM consecutive frames
        before being confirmed (to filter out reflections/noise).

        Args:
            detections: List of blob detections
            existing_tracks: Dict of track_id -> TrackedObject for headlight tracks
            current_time: Current timestamp
            max_distance: Max pixels to associate with existing track

        Returns:
            List of (track_id, detection, is_confirmed) tuples.
            is_confirmed is True only after track seen for MIN_FRAMES_TO_CONFIRM frames.
        """
        # Track IDs for headlights start at 10000 to avoid collision with YOLO
        if not hasattr(self, "_next_track_id"):
            self._next_track_id = 10000

        results = []
        seen_this_frame = set()

        for detection in detections:
            cx, cy = detection.center
            best_track_id = None
            best_distance = max_distance

            # Find nearest existing track
            for track_id, tracked_obj in existing_tracks.items():
                if track_id in seen_this_frame:
                    continue
                if tracked_obj.object_class != HEADLIGHT_CLASS_ID:
                    continue

                tx, ty = tracked_obj.current_pos
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Associate with existing track - increment frame count
                seen_this_frame.add(best_track_id)
                self._track_frame_counts[best_track_id] = (
                    self._track_frame_counts.get(best_track_id, 0) + 1
                )
                is_confirmed = (
                    self._track_frame_counts[best_track_id] >= self.MIN_FRAMES_TO_CONFIRM
                )
                results.append((best_track_id, detection, is_confirmed))
            else:
                # Create new track - starts at frame 1
                new_id = self._next_track_id
                self._next_track_id += 1
                self._track_frame_counts[new_id] = 1
                seen_this_frame.add(new_id)
                results.append((new_id, detection, False))  # New tracks not confirmed

        # Clean up stale tracks not seen this frame (reset their counts)
        stale_tracks = [
            tid for tid in self._track_frame_counts if tid not in seen_this_frame
        ]
        for tid in stale_tracks:
            del self._track_frame_counts[tid]

        return results
