"""
Nighttime Car Zone Detection

Detects vehicles at night using a scoring-based system that combines:
- Zone brightness changes (headlights approaching)
- White blob detection (headlights in zone)
- Red blob detection (taillights following)
- Temporal features (duration, rising patterns)

Unlike line/zone crossings, nighttime car zones use mathematical
scoring to confirm detections rather than simple thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from multiprocessing import Queue

logger = logging.getLogger(__name__)


@dataclass
class NighttimeCarZoneConfig:
    """Configuration for a nighttime car detection zone."""

    name: str
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    # Output wiring
    pdf_report: str | None = None
    email_immediate: bool = False
    email_digest: str | None = None
    # Debug mode - logs scoring details every N frames
    debug: bool = False
    # Debug save frames - saves zone images to debug detection issues
    debug_save_frames: str | None = None  # Directory to save debug frames


@dataclass
class TrackedBlob:
    """A tracked blob (headlight or taillight) with scoring state."""

    blob_id: int
    center: tuple[float, float]
    size: float
    bbox: tuple[int, int, int, int]
    first_seen_frame: int
    frames_seen: int = 1
    is_disqualified: bool = False  # Set True after event emitted
    is_taillight: bool = False
    last_score: int = -1  # For debug: only log when score changes


@dataclass
class ZoneBrightnessState:
    """Tracks brightness history for a zone."""

    history: list[float] = field(default_factory=list)
    max_history: int = 30  # ~1 second at 30fps

    def add(self, brightness: float) -> None:
        """Add a brightness sample."""
        self.history.append(brightness)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    @property
    def current(self) -> float:
        """Current brightness (0-1 normalized)."""
        return self.history[-1] / 255.0 if self.history else 0.0

    @property
    def baseline(self) -> float:
        """Baseline brightness (minimum over history)."""
        if len(self.history) < 5:
            return self.current
        return min(self.history) / 255.0

    @property
    def delta(self) -> float:
        """Brightness delta from baseline (0-1)."""
        return max(0.0, self.current - self.baseline)

    @property
    def is_rising(self) -> bool:
        """True if brightness has been rising over recent frames."""
        if len(self.history) < 5:
            return False
        recent = self.history[-5:]
        # Must have actually increased from start to end of window
        total_rise = recent[-1] - recent[0]
        if total_rise < 3:  # Require at least 3 units (~1%) actual increase
            return False
        # Check trend is mostly upward (allowing small noise)
        rising_count = sum(
            1 for i in range(1, len(recent)) if recent[i] >= recent[i - 1] - 2
        )
        return rising_count >= 3


class TaillightDetector:
    """
    Detects red blobs (taillights) in frame regions.

    Uses HSV color space to find red pixels, then blob detection
    to identify taillight candidates.
    """

    # Red color thresholds in HSV
    # Red wraps around in HSV, so we need two ranges
    RED_LOW_H1, RED_HIGH_H1 = 0, 10
    RED_LOW_H2, RED_HIGH_H2 = 160, 180
    RED_LOW_S, RED_HIGH_S = 100, 255
    RED_LOW_V, RED_HIGH_V = 100, 255

    MIN_BLOB_AREA = 15
    MAX_BLOB_AREA = 5000

    def detect_in_region(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> list[tuple[float, float, float]]:
        """
        Detect red blobs (taillights) in a frame region.

        Args:
            frame: BGR frame
            x1, y1, x2, y2: Region bounds in pixels

        Returns:
            List of (center_x, center_y, size) tuples in frame coordinates
        """
        # Extract region
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Create masks for both red ranges
        mask1 = cv2.inRange(
            hsv,
            (self.RED_LOW_H1, self.RED_LOW_S, self.RED_LOW_V),
            (self.RED_HIGH_H1, self.RED_HIGH_S, self.RED_HIGH_V),
        )
        mask2 = cv2.inRange(
            hsv,
            (self.RED_LOW_H2, self.RED_LOW_S, self.RED_LOW_V),
            (self.RED_HIGH_H2, self.RED_HIGH_S, self.RED_HIGH_V),
        )
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_BLOB_AREA <= area <= self.MAX_BLOB_AREA:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] + x1  # Convert to frame coords
                    cy = M["m01"] / M["m00"] + y1
                    size = np.sqrt(area)
                    results.append((cx, cy, size))

        return results


class NighttimeCarZone:
    """
    Detects vehicles at night using brightness + blob scoring.

    Scoring features (all normalized 0-1):
    - brightness_delta: How much brighter than baseline (weight: 15)
    - brightness_rising: 1.0 if brightness rising, else 0.0 (weight: 10)
    - blob_present: 1.0 if white blob in zone, else 0.0 (weight: 20)
    - blob_duration: frames_seen / 30, capped at 1.0 (weight: 15)
    - blob_size: normalized blob size (weight: 5)
    - primed_bonus: 1.0 if zone was primed (brightness rose before blob) (weight: 25)
    - taillight_match: 1.0 if red blob follows at same Y (weight: 30)

    Event emitted when score >= 100. Blob is then disqualified.
    """

    # Scoring weights
    WEIGHT_BRIGHTNESS_DELTA = 15
    WEIGHT_BRIGHTNESS_RISING = 10
    WEIGHT_BLOB_PRESENT = 20
    WEIGHT_BLOB_DURATION = 15
    WEIGHT_BLOB_SIZE = 5
    WEIGHT_PRIMED_BONUS = 25
    WEIGHT_TAILLIGHT_MATCH = 30

    SCORE_THRESHOLD = 100

    # Blob detection parameters
    BRIGHTNESS_THRESHOLD = 180
    MIN_BLOB_AREA = 20
    MAX_BLOB_AREA = 15000
    MIN_CIRCULARITY = 0.1

    # Taillight matching
    TAILLIGHT_Y_TOLERANCE = 30  # pixels
    TAILLIGHT_X_MIN_OFFSET = 20  # taillight should be behind headlight
    TAILLIGHT_DELAY_FRAMES = 5  # frames to wait for taillight

    def __init__(
        self, config: NighttimeCarZoneConfig, frame_width: int, frame_height: int
    ):
        """
        Initialize nighttime car zone.

        Args:
            config: Zone configuration
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.config = config
        self.name = config.name

        # Calculate pixel bounds
        self.x1 = int(frame_width * config.x1_pct / 100)
        self.y1 = int(frame_height * config.y1_pct / 100)
        self.x2 = int(frame_width * config.x2_pct / 100)
        self.y2 = int(frame_height * config.y2_pct / 100)

        # State
        self.brightness_state = ZoneBrightnessState()
        self.tracked_blobs: dict[int, TrackedBlob] = {}
        self._next_blob_id = 0
        self._primed = False  # Zone was primed by rising brightness
        self._primed_frame = 0
        self._frame_count = 0

        # Debug mode
        self._debug = config.debug
        self._debug_interval = 150  # Log status every N frames (5s at 30fps) when idle
        self._last_debug_state = ""  # Track state changes to reduce noise
        self._debug_save_frames = config.debug_save_frames
        self._debug_frame_interval = 30  # Save debug frames every N frames
        if self._debug_save_frames:
            import os
            os.makedirs(self._debug_save_frames, exist_ok=True)

        # Taillight detector
        self.taillight_detector = TaillightDetector()

        # Blob detector
        self._blob_detector = self._create_blob_detector()

        log_msg = f"NighttimeCarZone '{self.name}' initialized: ({self.x1},{self.y1})-({self.x2},{self.y2})"
        if self._debug:
            logger.info(f"{log_msg} [DEBUG MODE ENABLED]")
        if self._debug_save_frames:
            logger.info(f"  Debug frames will be saved to: {self._debug_save_frames}")
        if not self._debug and not self._debug_save_frames:
            logger.debug(log_msg)

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
        """Create OpenCV blob detector for headlight detection."""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = self.MIN_BLOB_AREA
        params.maxArea = self.MAX_BLOB_AREA
        params.filterByCircularity = True
        params.minCircularity = self.MIN_CIRCULARITY
        params.filterByConvexity = False
        params.filterByInertia = False
        return cv2.SimpleBlobDetector_create(params)

    def process_frame(
        self,
        frame: np.ndarray,
        data_queue: "Queue",
        relative_time: float,
        temp_frame_dir: str | None = None,
        temp_frame_max_age: int = 30,
    ) -> int:
        """
        Process a frame and emit events if score threshold reached.

        Args:
            frame: BGR frame from camera
            data_queue: Queue for sending events
            relative_time: Time since detection started
            temp_frame_dir: Directory for temporary frame storage
            temp_frame_max_age: Max age of temp frames in seconds

        Returns:
            Number of events emitted
        """
        self._frame_count += 1
        event_count = 0

        # Extract zone region
        region = frame[self.y1 : self.y2, self.x1 : self.x2]
        if region.size == 0:
            return 0

        # Update brightness state
        brightness = self._calculate_zone_brightness(region)
        self.brightness_state.add(brightness)

        # Check for priming (brightness rising without blob)
        if self.brightness_state.is_rising and not self.tracked_blobs:
            if not self._primed:
                self._primed = True
                self._primed_frame = self._frame_count
                logger.debug(f"Zone '{self.name}' primed at frame {self._frame_count}")

        # Detect white blobs (headlights)
        blobs = self._detect_blobs(region)

        # Update blob tracking
        self._update_blob_tracking(blobs)

        # Detect taillights in zone
        taillights = self.taillight_detector.detect_in_region(
            frame, self.x1, self.y1, self.x2, self.y2
        )

        # Debug logging - only when state changes or blobs are present
        if self._debug:
            active_blobs = sum(1 for b in self.tracked_blobs.values() if not b.is_disqualified)
            # Create state signature to detect changes
            current_state = f"{len(blobs)}_{active_blobs}_{len(taillights)}_{self._primed}"

            # Log immediately if blobs detected, or periodically if idle
            should_log = (
                len(blobs) > 0 or
                active_blobs > 0 or
                current_state != self._last_debug_state or
                self._frame_count % self._debug_interval == 0
            )

            if should_log:
                self._last_debug_state = current_state
                if len(blobs) == 0 and active_blobs == 0:
                    # Compact idle status
                    logger.info(
                        f"[DEBUG] '{self.name}' frame {self._frame_count}: "
                        f"idle (brightness={self.brightness_state.current:.2f}, primed={self._primed})"
                    )
                else:
                    # Full status when blobs are present
                    logger.info(
                        f"[DEBUG] '{self.name}' frame {self._frame_count}: "
                        f"brightness={self.brightness_state.current:.2f} "
                        f"(delta={self.brightness_state.delta:.2f}, rising={self.brightness_state.is_rising}) | "
                        f"blobs={len(blobs)}, tracked={active_blobs}, taillights={len(taillights)}, primed={self._primed}"
                    )

        # Score each active blob
        for blob_id, blob in list(self.tracked_blobs.items()):
            if blob.is_disqualified:
                continue

            score = self._calculate_score(blob, taillights, log_debug=self._debug)

            if score >= self.SCORE_THRESHOLD:
                # Emit event
                event_count += 1
                blob.is_disqualified = True

                # Save frame if configured
                frame_id = None
                if temp_frame_dir:
                    frame_id = self._save_temp_frame(
                        frame, temp_frame_dir, temp_frame_max_age
                    )

                event = {
                    "event_type": "NIGHTTIME_CAR",
                    "zone_name": self.name,
                    "score": score,
                    "blob_center": blob.center,
                    "bbox": blob.bbox,  # For frame annotation
                    "frame_id": frame_id,
                    "timestamp_relative": relative_time,
                    "was_primed": self._primed,
                    "had_taillight": any(
                        self._taillight_matches(blob, t) for t in taillights
                    ),
                }

                data_queue.put(event)
                logger.info(
                    f"NIGHTTIME_CAR in '{self.name}' (score={score:.0f}, "
                    f"primed={self._primed}, taillight={event['had_taillight']})"
                )

                # Reset primed state after event
                self._primed = False

        # Cleanup old blobs
        self._cleanup_stale_blobs()

        return event_count

    def _calculate_zone_brightness(self, region: np.ndarray) -> float:
        """Calculate average brightness of zone region."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _detect_blobs(self, region: np.ndarray) -> list[tuple[float, float, float]]:
        """
        Detect white blobs (headlights) in zone region.

        Returns:
            List of (center_x, center_y, size) in zone-relative coordinates
        """
        # Convert to LAB and threshold L channel
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, binary = cv2.threshold(
            l_channel, self.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Detect blobs
        keypoints = self._blob_detector.detect(binary)

        # Save debug frames periodically
        if self._debug_save_frames and self._frame_count % self._debug_frame_interval == 0:
            self._save_debug_frame(region, l_channel, binary, keypoints)

        results = []
        for kp in keypoints:
            # Convert to frame coordinates
            cx = kp.pt[0] + self.x1
            cy = kp.pt[1] + self.y1
            results.append((cx, cy, kp.size))

        return results

    def _save_debug_frame(
        self,
        region: np.ndarray,
        l_channel: np.ndarray,
        binary: np.ndarray,
        keypoints: list,
    ) -> None:
        """Save debug images showing detection pipeline."""
        import os

        base_path = os.path.join(self._debug_save_frames, f"frame_{self._frame_count:06d}")

        # Save original region
        cv2.imwrite(f"{base_path}_1_region.jpg", region)

        # Save L channel (brightness)
        cv2.imwrite(f"{base_path}_2_brightness.jpg", l_channel)

        # Save binary threshold result
        cv2.imwrite(f"{base_path}_3_binary.jpg", binary)

        # Save with keypoints drawn
        if len(keypoints) > 0:
            with_keypoints = cv2.drawKeypoints(
                binary, keypoints, None, (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imwrite(f"{base_path}_4_keypoints.jpg", with_keypoints)

        # Log max brightness in region for debugging threshold
        max_brightness = int(l_channel.max())
        avg_brightness = int(l_channel.mean())
        logger.info(
            f"[DEBUG FRAME] '{self.name}' saved to {base_path}_*.jpg | "
            f"brightness: max={max_brightness}, avg={avg_brightness}, threshold={self.BRIGHTNESS_THRESHOLD}"
        )

    def _update_blob_tracking(self, blobs: list[tuple[float, float, float]]) -> None:
        """Associate detected blobs with tracked blobs."""
        max_distance = 50.0
        seen_this_frame = set()

        for cx, cy, size in blobs:
            best_blob_id = None
            best_distance = max_distance

            # Find nearest existing blob
            for blob_id, tracked in self.tracked_blobs.items():
                if blob_id in seen_this_frame:
                    continue
                tx, ty = tracked.center
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_blob_id = blob_id

            if best_blob_id is not None:
                # Update existing blob
                blob = self.tracked_blobs[best_blob_id]
                blob.center = (cx, cy)
                blob.size = size
                blob.frames_seen += 1
                seen_this_frame.add(best_blob_id)
            else:
                # Create new blob
                new_id = self._next_blob_id
                self._next_blob_id += 1
                self.tracked_blobs[new_id] = TrackedBlob(
                    blob_id=new_id,
                    center=(cx, cy),
                    size=size,
                    bbox=(
                        int(cx - size / 2),
                        int(cy - size / 2),
                        int(cx + size / 2),
                        int(cy + size / 2),
                    ),
                    first_seen_frame=self._frame_count,
                )
                seen_this_frame.add(new_id)

    def _calculate_score(
        self, blob: TrackedBlob, taillights: list[tuple[float, float, float]],
        log_debug: bool = False
    ) -> float:
        """Calculate detection score for a blob."""
        # Brightness delta (0-1, capped)
        brightness_delta = min(1.0, self.brightness_state.delta * 5)

        # Brightness rising (binary)
        brightness_rising = 1.0 if self.brightness_state.is_rising else 0.0

        # Blob present (binary)
        blob_present = 1.0

        # Blob duration (0-1, normalized by 30 frames)
        blob_duration = min(1.0, blob.frames_seen / 30.0)

        # Blob size (0-1, normalized)
        blob_size = min(1.0, blob.size / 100.0)

        # Primed bonus (zone was primed before blob appeared)
        primed_bonus = 0.0
        if self._primed and blob.first_seen_frame > self._primed_frame:
            primed_bonus = 1.0

        # Taillight match
        taillight_match = 0.0
        if blob.frames_seen >= self.TAILLIGHT_DELAY_FRAMES:
            for taillight in taillights:
                if self._taillight_matches(blob, taillight):
                    taillight_match = 1.0
                    break

        # Calculate weighted score with individual contributions
        score_brightness_delta = self.WEIGHT_BRIGHTNESS_DELTA * brightness_delta
        score_brightness_rising = self.WEIGHT_BRIGHTNESS_RISING * brightness_rising
        score_blob_present = self.WEIGHT_BLOB_PRESENT * blob_present
        score_blob_duration = self.WEIGHT_BLOB_DURATION * blob_duration
        score_blob_size = self.WEIGHT_BLOB_SIZE * blob_size
        score_primed_bonus = self.WEIGHT_PRIMED_BONUS * primed_bonus
        score_taillight_match = self.WEIGHT_TAILLIGHT_MATCH * taillight_match

        score = (
            score_brightness_delta
            + score_brightness_rising
            + score_blob_present
            + score_blob_duration
            + score_blob_size
            + score_primed_bonus
            + score_taillight_match
        )

        # Only log if score changed from last frame (reduces noise)
        score_int = int(score)
        if log_debug and score_int != blob.last_score:
            blob.last_score = score_int
            logger.info(
                f"[DEBUG] '{self.name}' blob#{blob.blob_id} score={score:.0f}/{self.SCORE_THRESHOLD} | "
                f"bright_delta={score_brightness_delta:.0f} bright_rise={score_brightness_rising:.0f} "
                f"present={score_blob_present:.0f} duration={score_blob_duration:.0f} "
                f"size={score_blob_size:.0f} primed={score_primed_bonus:.0f} taillight={score_taillight_match:.0f} | "
                f"blob: frames={blob.frames_seen}, size={blob.size:.1f}, center={blob.center}"
            )

        return score

    def _taillight_matches(
        self, blob: TrackedBlob, taillight: tuple[float, float, float]
    ) -> bool:
        """Check if a taillight matches a headlight blob."""
        tx, ty, _ = taillight
        bx, by = blob.center

        # Taillight should be at similar Y (same height)
        y_diff = abs(ty - by)
        if y_diff > self.TAILLIGHT_Y_TOLERANCE:
            return False

        # Taillight should be behind headlight (higher X = further from camera)
        # This assumes cars are moving toward the camera
        x_diff = tx - bx
        if x_diff < self.TAILLIGHT_X_MIN_OFFSET:
            return False

        return True

    def _cleanup_stale_blobs(self) -> None:
        """Remove blobs not seen for several frames."""
        stale_threshold = 10  # frames
        stale_ids = [
            blob_id
            for blob_id, blob in self.tracked_blobs.items()
            if self._frame_count - blob.first_seen_frame - blob.frames_seen
            > stale_threshold
            or blob.is_disqualified
        ]
        for blob_id in stale_ids:
            del self.tracked_blobs[blob_id]

    def _save_temp_frame(
        self, frame: np.ndarray, temp_dir: str, max_age_seconds: int
    ) -> str | None:
        """Save temporary frame for event capture."""
        import glob
        import os
        import time
        import uuid

        try:
            frame_id = str(uuid.uuid4())
            filepath = os.path.join(temp_dir, f"{frame_id}.jpg")
            cv2.imwrite(filepath, frame)

            # Cleanup old frames
            current_time = time.time()
            for temp_path in glob.glob(os.path.join(temp_dir, "*.jpg")):
                try:
                    if current_time - os.path.getmtime(temp_path) > max_age_seconds:
                        os.remove(temp_path)
                except Exception:
                    pass

            return frame_id
        except Exception as e:
            logger.debug(f"Error saving temp frame: {e}")
            return None


def create_nighttime_car_zones(
    config: dict, frame_width: int, frame_height: int
) -> list[NighttimeCarZone]:
    """
    Create NighttimeCarZone instances from config.

    Args:
        config: Full configuration dictionary
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        List of NighttimeCarZone instances
    """
    zones = []
    for zone_config in config.get("nighttime_car_zones", []):
        zone = NighttimeCarZone(
            NighttimeCarZoneConfig(
                name=zone_config["name"],
                x1_pct=zone_config["x1_pct"],
                y1_pct=zone_config["y1_pct"],
                x2_pct=zone_config["x2_pct"],
                y2_pct=zone_config["y2_pct"],
                pdf_report=zone_config.get("pdf_report"),
                email_immediate=zone_config.get("email_immediate", False),
                email_digest=zone_config.get("email_digest"),
                debug=zone_config.get("debug", False),
                debug_save_frames=zone_config.get("debug_save_frames"),
            ),
            frame_width,
            frame_height,
        )
        zones.append(zone)

    if zones:
        logger.info(f"Created {len(zones)} nighttime car zone(s)")

    return zones
