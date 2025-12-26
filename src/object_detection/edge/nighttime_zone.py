"""
Nighttime Car Zone Detection (Edge-local copy)

THIS IS A SELF-CONTAINED COPY FOR EDGE DEPLOYMENT.
See edge/ISOLATION.txt for details on why this exists.

Detects vehicles at night using a scoring-based system that combines:
- Zone brightness changes (headlights approaching)
- White blob detection (headlights in zone)
- Red blob detection (taillights following)
- Temporal features (duration, rising patterns)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Synthetic COCO class ID for nighttime car detection
# Regular COCO classes are 0-79, so 1000 is safe
NIGHTTIME_CAR_CLASS_ID = 1000


@dataclass
class NighttimeDetectionParams:
    """Detection parameters for nighttime car detection."""

    brightness_threshold: int = 30
    min_blob_size: int = 100
    max_blob_size: int = 10000
    score_threshold: int = 85
    taillight_color_match: bool = True


@dataclass
class NighttimeCarZoneConfig:
    """Configuration for a nighttime car detection zone."""

    name: str
    zone_id: str
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    detection_params: NighttimeDetectionParams = field(
        default_factory=NighttimeDetectionParams
    )
    event_name: str = ""
    debug: bool = False
    debug_save_frames: str | None = None


@dataclass
class TrackedBlob:
    """A tracked blob (headlight or taillight) with scoring state."""

    blob_id: int
    center: tuple[float, float]
    size: float
    bbox: tuple[int, int, int, int]
    first_seen_frame: int
    frames_seen: int = 1
    is_disqualified: bool = False
    is_taillight: bool = False
    last_score: int = -1


@dataclass
class ZoneBrightnessState:
    """Tracks brightness history for a zone."""

    history: list[float] = field(default_factory=list)
    max_history: int = 30

    def add(self, brightness: float) -> None:
        self.history.append(brightness)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    @property
    def current(self) -> float:
        return self.history[-1] / 255.0 if self.history else 0.0

    @property
    def baseline(self) -> float:
        if len(self.history) < 5:
            return self.current
        return min(self.history) / 255.0

    @property
    def delta(self) -> float:
        return max(0.0, self.current - self.baseline)

    @property
    def is_rising(self) -> bool:
        if len(self.history) < 5:
            return False
        recent = self.history[-5:]
        total_rise = recent[-1] - recent[0]
        if total_rise < 3:
            return False
        rising_count = sum(
            1 for i in range(1, len(recent)) if recent[i] >= recent[i - 1] - 2
        )
        return rising_count >= 3


class TaillightDetector:
    """Detects red blobs (taillights) in frame regions."""

    RED_LOW_H1, RED_HIGH_H1 = 0, 10
    RED_LOW_H2, RED_HIGH_H2 = 160, 180
    RED_LOW_S, RED_HIGH_S = 100, 255
    RED_LOW_V, RED_HIGH_V = 100, 255

    MIN_BLOB_AREA = 15
    MAX_BLOB_AREA = 5000

    def detect_in_region(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> list[tuple[float, float, float]]:
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return []

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_BLOB_AREA <= area <= self.MAX_BLOB_AREA:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] + x1
                    cy = M["m01"] / M["m00"] + y1
                    size = np.sqrt(area)
                    results.append((cx, cy, size))

        return results


class NighttimeCarZone:
    """
    Detects vehicles at night using brightness + blob scoring.

    Scoring features (all normalized 0-1):
    - brightness_delta: How much brighter than baseline (weight: 15)
    - brightness_rising: 1.0 if brightness rising (weight: 10)
    - blob_present: 1.0 if white blob in zone (weight: 20)
    - blob_duration: frames_seen / 30, capped at 1.0 (weight: 15)
    - blob_size: normalized blob size (weight: 5)
    - primed_bonus: 1.0 if zone was primed (weight: 15)
    - taillight_match: 1.0 if red blob follows (weight: 50)

    Event emitted when score >= 85.
    """

    WEIGHT_BRIGHTNESS_DELTA = 15
    WEIGHT_BRIGHTNESS_RISING = 10
    WEIGHT_BLOB_PRESENT = 20
    WEIGHT_BLOB_DURATION = 15
    WEIGHT_BLOB_SIZE = 5
    WEIGHT_PRIMED_BONUS = 15
    WEIGHT_TAILLIGHT_MATCH = 50

    SCORE_THRESHOLD = 85
    BRIGHTNESS_THRESHOLD = 180
    MIN_BLOB_AREA = 20
    MAX_BLOB_AREA = 15000
    MIN_CIRCULARITY = 0.1

    TAILLIGHT_Y_TOLERANCE = 30
    TAILLIGHT_X_MIN_OFFSET = 20
    TAILLIGHT_DELAY_FRAMES = 5

    def __init__(
        self, config: NighttimeCarZoneConfig, frame_width: int, frame_height: int
    ):
        self.config = config
        self.name = config.name
        self.zone_id = config.zone_id
        self.event_name = config.event_name

        self.x1 = int(frame_width * config.x1_pct / 100)
        self.y1 = int(frame_height * config.y1_pct / 100)
        self.x2 = int(frame_width * config.x2_pct / 100)
        self.y2 = int(frame_height * config.y2_pct / 100)

        params = config.detection_params
        self.SCORE_THRESHOLD = params.score_threshold
        self.MIN_BLOB_AREA = params.min_blob_size
        self.MAX_BLOB_AREA = params.max_blob_size

        self.brightness_state = ZoneBrightnessState()
        self.tracked_blobs: dict[int, TrackedBlob] = {}
        self._next_blob_id = 0
        self._primed = False
        self._primed_frame = 0
        self._primed_until_frame = 0
        self._frame_count = 0

        self._debug = config.debug
        self._debug_interval = 150
        self._last_debug_state = ""
        self._debug_save_frames = config.debug_save_frames
        self._debug_frame_interval = 30
        if self._debug_save_frames:
            import os

            os.makedirs(self._debug_save_frames, exist_ok=True)

        self.taillight_detector = TaillightDetector()
        self._blob_detector = self._create_blob_detector()

        log_msg = f"NighttimeCarZone '{self.name}' initialized: ({self.x1},{self.y1})-({self.x2},{self.y2})"
        if self._debug:
            logger.info(f"{log_msg} [DEBUG MODE ENABLED]")
        else:
            logger.debug(log_msg)

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
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
        on_event: Callable[[dict[str, Any]], None],
        relative_time: float,
    ) -> int:
        """
        Process a frame and emit events if score threshold reached.

        Args:
            frame: BGR frame from camera
            on_event: Callback to invoke when event is detected
            relative_time: Time since detection started

        Returns:
            Number of events emitted
        """
        self._frame_count += 1
        event_count = 0

        region = frame[self.y1 : self.y2, self.x1 : self.x2]
        if region.size == 0:
            return 0

        brightness = self._calculate_zone_brightness(region)
        self.brightness_state.add(brightness)

        brightness_elevated = self.brightness_state.delta >= 0.02
        if (
            brightness_elevated or self.brightness_state.is_rising
        ) and not self.tracked_blobs:
            if not self._primed:
                self._primed = True
                self._primed_frame = self._frame_count
                self._primed_until_frame = self._frame_count + 45
                logger.debug(f"Zone '{self.name}' primed at frame {self._frame_count}")
            elif self._frame_count < self._primed_until_frame:
                self._primed_until_frame = self._frame_count + 45
        elif (
            self._primed
            and self._frame_count >= self._primed_until_frame
            and not self.tracked_blobs
        ):
            self._primed = False
            logger.debug(
                f"Zone '{self.name}' primed state reset at frame {self._frame_count}"
            )

        blobs = self._detect_blobs(region)
        self._update_blob_tracking(blobs)

        taillights = self.taillight_detector.detect_in_region(
            frame, self.x1, self.y1, self.x2, self.y2
        )

        if self._debug:
            active_blobs = sum(
                1 for b in self.tracked_blobs.values() if not b.is_disqualified
            )
            current_state = (
                f"{len(blobs)}_{active_blobs}_{len(taillights)}_{self._primed}"
            )

            should_log = (
                len(blobs) > 0
                or active_blobs > 0
                or current_state != self._last_debug_state
                or self._frame_count % self._debug_interval == 0
            )

            if should_log:
                self._last_debug_state = current_state
                if len(blobs) == 0 and active_blobs == 0:
                    logger.info(
                        f"[DEBUG] '{self.name}' frame {self._frame_count}: "
                        f"idle (brightness={self.brightness_state.current:.2f}, primed={self._primed})"
                    )
                else:
                    logger.info(
                        f"[DEBUG] '{self.name}' frame {self._frame_count}: "
                        f"brightness={self.brightness_state.current:.2f} "
                        f"(delta={self.brightness_state.delta:.2f}, rising={self.brightness_state.is_rising}) | "
                        f"blobs={len(blobs)}, tracked={active_blobs}, taillights={len(taillights)}, primed={self._primed}"
                    )

        for blob_id, blob in list(self.tracked_blobs.items()):
            if blob.is_disqualified:
                continue

            score = self._calculate_score(blob, taillights, log_debug=self._debug)

            if score >= self.SCORE_THRESHOLD:
                event_count += 1
                blob.is_disqualified = True

                track_id = f"nc_{blob.blob_id}"

                event = {
                    "event_type": "NIGHTTIME_CAR",
                    "zone_id": self.zone_id,
                    "object_class": NIGHTTIME_CAR_CLASS_ID,
                    "track_id": track_id,
                    "bbox": blob.bbox,
                    "timestamp_relative": relative_time,
                    "score": score,
                    "was_primed": self._primed,
                    "had_taillight": any(
                        self._taillight_matches(blob, t) for t in taillights
                    ),
                }

                on_event(event)
                logger.info(
                    f"NIGHTTIME_CAR in '{self.name}' (score={score:.0f}, "
                    f"primed={self._primed}, taillight={event['had_taillight']})"
                )

                self._primed = False

        self._cleanup_stale_blobs()
        return event_count

    def _calculate_zone_brightness(self, region: np.ndarray) -> float:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _detect_blobs(self, region: np.ndarray) -> list[tuple[float, float, float]]:
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, binary = cv2.threshold(
            l_channel, self.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        keypoints = self._blob_detector.detect(binary)

        results = []
        for kp in keypoints:
            cx = kp.pt[0] + self.x1
            cy = kp.pt[1] + self.y1
            results.append((cx, cy, kp.size))

        return results

    def _update_blob_tracking(self, blobs: list[tuple[float, float, float]]) -> None:
        max_distance = 100.0
        seen_this_frame = set()

        for cx, cy, size in blobs:
            best_blob_id = None
            best_distance = max_distance

            for blob_id, tracked in self.tracked_blobs.items():
                if blob_id in seen_this_frame:
                    continue
                tx, ty = tracked.center
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_blob_id = blob_id

            if best_blob_id is not None:
                blob = self.tracked_blobs[best_blob_id]
                blob.center = (cx, cy)
                blob.size = size
                blob.bbox = (
                    int(cx - size / 2),
                    int(cy - size / 2),
                    int(cx + size / 2),
                    int(cy + size / 2),
                )
                blob.frames_seen += 1
                seen_this_frame.add(best_blob_id)
            else:
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
        self,
        blob: TrackedBlob,
        taillights: list[tuple[float, float, float]],
        log_debug: bool = False,
    ) -> float:
        brightness_delta = min(1.0, self.brightness_state.delta * 5)
        brightness_rising = 1.0 if self.brightness_state.is_rising else 0.0
        blob_present = 1.0
        blob_duration = 1.0 if blob.frames_seen >= 5 else 0.0
        blob_size = min(1.0, blob.size / 100.0)

        primed_bonus = 0.0
        if self._primed and blob.first_seen_frame >= self._primed_frame:
            primed_bonus = 1.0

        taillight_match = 0.0
        if blob.frames_seen >= self.TAILLIGHT_DELAY_FRAMES:
            for taillight in taillights:
                if self._taillight_matches(blob, taillight):
                    taillight_match = 1.0
                    break

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

        score_int = int(score)
        if log_debug and score_int != blob.last_score:
            blob.last_score = score_int
            logger.info(
                f"[DEBUG] '{self.name}' blob#{blob.blob_id} score={score:.0f}/{self.SCORE_THRESHOLD} | "
                f"bright_delta={score_brightness_delta:.0f} bright_rise={score_brightness_rising:.0f} "
                f"present={score_blob_present:.0f} duration={score_blob_duration:.0f} "
                f"size={score_blob_size:.0f} primed={score_primed_bonus:.0f} taillight={score_taillight_match:.0f}"
            )

        return score

    def _taillight_matches(
        self, blob: TrackedBlob, taillight: tuple[float, float, float]
    ) -> bool:
        tx, ty, _ = taillight
        bx, by = blob.center

        y_diff = abs(ty - by)
        if y_diff > self.TAILLIGHT_Y_TOLERANCE:
            return False

        x_diff = tx - bx
        if x_diff < self.TAILLIGHT_X_MIN_OFFSET:
            return False

        return True

    def _cleanup_stale_blobs(self) -> None:
        stale_threshold = 10
        stale_ids = [
            blob_id
            for blob_id, blob in self.tracked_blobs.items()
            if self._frame_count - blob.first_seen_frame - blob.frames_seen
            > stale_threshold
            or blob.is_disqualified
        ]
        for blob_id in stale_ids:
            del self.tracked_blobs[blob_id]
