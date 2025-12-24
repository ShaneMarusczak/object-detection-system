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

    # Headlight pairing - merge two close blobs as one vehicle
    PAIR_MAX_X_DISTANCE = 150  # Max horizontal distance between paired headlights
    PAIR_MAX_Y_DISTANCE = 30  # Max vertical difference (should be same height)
    PAIR_MAX_SIZE_RATIO = 2.5  # Max size ratio between paired headlights

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

        # Morphological cleanup - removes noise while preserving headlight shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

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

        # Pair headlights - merge two close blobs into one vehicle detection
        detections = self._pair_headlights(detections)

        return detections

    def _pair_headlights(
        self, detections: list[BlobDetection]
    ) -> list[BlobDetection]:
        """
        Merge pairs of headlights into single vehicle detections.

        Two blobs are paired if they're close horizontally, at similar height,
        and similar size. The result is a single detection centered between them.

        Args:
            detections: List of individual blob detections

        Returns:
            List with paired headlights merged into single detections
        """
        if len(detections) < 2:
            return detections

        paired = set()  # Indices of already-paired detections
        result = []

        for i, det1 in enumerate(detections):
            if i in paired:
                continue

            best_pair = None
            best_distance = self.PAIR_MAX_X_DISTANCE

            for j, det2 in enumerate(detections):
                if j <= i or j in paired:
                    continue

                # Check vertical alignment (same height)
                y_diff = abs(det1.center[1] - det2.center[1])
                if y_diff > self.PAIR_MAX_Y_DISTANCE:
                    continue

                # Check horizontal distance
                x_diff = abs(det1.center[0] - det2.center[0])
                if x_diff > self.PAIR_MAX_X_DISTANCE:
                    continue

                # Check size similarity
                size_ratio = max(det1.size, det2.size) / max(min(det1.size, det2.size), 1)
                if size_ratio > self.PAIR_MAX_SIZE_RATIO:
                    continue

                # Found a valid pair - pick closest
                if x_diff < best_distance:
                    best_distance = x_diff
                    best_pair = j

            if best_pair is not None:
                # Merge the pair into a single detection
                det2 = detections[best_pair]
                paired.add(i)
                paired.add(best_pair)

                # Center is midpoint between the two headlights
                cx = (det1.center[0] + det2.center[0]) / 2
                cy = (det1.center[1] + det2.center[1]) / 2

                # Bounding box encompasses both
                x1 = min(det1.bbox[0], det2.bbox[0])
                y1 = min(det1.bbox[1], det2.bbox[1])
                x2 = max(det1.bbox[2], det2.bbox[2])
                y2 = max(det1.bbox[3], det2.bbox[3])

                # Size is the combined width
                combined_size = x2 - x1

                result.append(
                    BlobDetection(
                        center=(cx, cy),
                        size=combined_size,
                        bbox=(x1, y1, x2, y2),
                    )
                )
            else:
                # No pair found - keep as single headlight
                result.append(det1)

        return result

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
