"""
NighttimeCarEmitter - Detects vehicles at night via blob scoring.

Does NOT use YOLO - instead analyzes brightness patterns and blob detection
for headlights/taillights. Completely independent detection path.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .registry import register

if TYPE_CHECKING:
    from .tracking_state import TrackingState

logger = logging.getLogger(__name__)

# Synthetic class ID for nighttime car (outside COCO range)
NIGHTTIME_CAR_CLASS_ID = 1000


@dataclass
class BrightnessState:
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


@dataclass
class TrackedBlob:
    """A tracked blob with scoring state."""

    blob_id: int
    center: tuple[float, float]
    size: float
    bbox: tuple[int, int, int, int]
    first_seen_frame: int
    frames_seen: int = 1
    is_disqualified: bool = False
    is_taillight: bool = False


@dataclass
class NighttimeZoneState:
    """State for a single nighttime detection zone."""

    name: str
    zone_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    score_threshold: int
    brightness_state: BrightnessState = field(default_factory=BrightnessState)
    tracked_blobs: dict[int, TrackedBlob] = field(default_factory=dict)
    next_blob_id: int = 0
    primed: bool = False
    primed_frame: int = 0
    primed_until_frame: int = 0
    frame_count: int = 0


@register("NIGHTTIME_CAR")
class NighttimeCarEmitter:
    """Emits NIGHTTIME_CAR events via brightness/blob scoring."""

    event_type = "NIGHTTIME_CAR"
    needs_yolo = False
    needs_tracking = False

    # Scoring weights
    WEIGHT_BRIGHTNESS_DELTA = 15
    WEIGHT_BRIGHTNESS_RISING = 10
    WEIGHT_BLOB_PRESENT = 20
    WEIGHT_BLOB_DURATION = 15
    WEIGHT_BLOB_SIZE = 5
    WEIGHT_PRIMED_BONUS = 15
    WEIGHT_TAILLIGHT_MATCH = 50

    # Detection parameters
    BRIGHTNESS_THRESHOLD = 180
    MIN_BLOB_AREA = 20
    MAX_BLOB_AREA = 15000
    MIN_CIRCULARITY = 0.1

    # Taillight matching
    TAILLIGHT_Y_TOLERANCE = 30
    TAILLIGHT_X_MIN_OFFSET = 20

    def __init__(self, config: dict, frame_dims: tuple):
        """
        Initialize nighttime car emitter.

        Scans event definitions for NIGHTTIME_CAR events and creates
        zone states for each referenced zone.
        """
        self.frame_dims = frame_dims
        self.zones: list[NighttimeZoneState] = []
        self._blob_detector = self._create_blob_detector()

        # Find NIGHTTIME_CAR events and their zones
        self._init_zones_from_config(config, frame_dims)

    def _init_zones_from_config(self, config: dict, frame_dims: tuple) -> None:
        """Initialize zone states from config."""
        width, height = frame_dims

        # Build zone lookup from config
        zone_lookup = {}
        for i, zone_config in enumerate(config.get("zones", []), 1):
            zone_id = f"Z{i}"
            zone_lookup[zone_config["description"]] = {
                "zone_id": zone_id,
                **zone_config,
            }

        # Find NIGHTTIME_CAR events
        for event_def in config.get("events", []):
            match = event_def.get("match", {})
            if match.get("event_type") != "NIGHTTIME_CAR":
                continue

            zone_name = match.get("zone")
            if not zone_name or zone_name not in zone_lookup:
                continue

            zone_config = zone_lookup[zone_name]
            nighttime_params = match.get("nighttime_detection", {})

            self.zones.append(
                NighttimeZoneState(
                    name=zone_name,
                    zone_id=zone_config["zone_id"],
                    x1=int(width * zone_config["x1_pct"] / 100),
                    y1=int(height * zone_config["y1_pct"] / 100),
                    x2=int(width * zone_config["x2_pct"] / 100),
                    y2=int(height * zone_config["y2_pct"] / 100),
                    score_threshold=nighttime_params.get("score_threshold", 85),
                )
            )
            logger.info(f"NighttimeCarEmitter: zone '{zone_name}' initialized")

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

    def process(
        self,
        frame,
        yolo_results,
        timestamp: float,
        tracking_state: "TrackingState | None" = None,
        frame_id: str | None = None,
    ) -> list[dict]:
        """
        Process frame for nighttime car detection.

        Args:
            frame: Raw BGR frame for blob analysis
            yolo_results: Unused (we don't use YOLO)
            timestamp: Relative timestamp
            tracking_state: Unused (we don't use tracking)
            frame_id: Optional saved frame reference

        Returns:
            List of NIGHTTIME_CAR events
        """
        events = []

        for zone in self.zones:
            zone_events = self._process_zone(frame, zone, timestamp, frame_id)
            events.extend(zone_events)

        return events

    def _process_zone(
        self,
        frame,
        zone: NighttimeZoneState,
        timestamp: float,
        frame_id: str | None,
    ) -> list[dict]:
        """Process a single zone for nighttime detection."""
        events = []
        zone.frame_count += 1

        # Extract zone region
        region = frame[zone.y1 : zone.y2, zone.x1 : zone.x2]
        if region.size == 0:
            return events

        # Update brightness state
        brightness = self._calculate_brightness(region)
        zone.brightness_state.add(brightness)

        # Check for priming
        self._update_priming(zone)

        # Detect white blobs (headlights)
        blobs = self._detect_blobs(region, zone)

        # Update blob tracking
        self._update_blob_tracking(zone, blobs)

        # Detect taillights
        taillights = self._detect_taillights(frame, zone)

        # Score each active blob
        for blob_id, blob in list(zone.tracked_blobs.items()):
            if blob.is_disqualified:
                continue

            score = self._calculate_score(zone, blob, taillights)

            if score >= zone.score_threshold:
                blob.is_disqualified = True
                zone.primed = False

                has_taillight = any(
                    self._taillight_matches(blob, t) for t in taillights
                )

                events.append(
                    {
                        "event_type": "NIGHTTIME_CAR",
                        "zone_id": zone.zone_id,
                        "object_class": NIGHTTIME_CAR_CLASS_ID,
                        "track_id": f"nc_{blob.blob_id}",
                        "bbox": blob.bbox,
                        "frame_id": frame_id,
                        "timestamp_relative": timestamp,
                        "score": score,
                        "was_primed": zone.primed,
                        "had_taillight": has_taillight,
                    }
                )

                logger.info(
                    f"NIGHTTIME_CAR in '{zone.name}' (score={score:.0f}, "
                    f"primed={zone.primed}, taillight={has_taillight})"
                )

        # Cleanup stale blobs
        self._cleanup_stale_blobs(zone)

        return events

    def _calculate_brightness(self, region) -> float:
        """Calculate average brightness of region."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _update_priming(self, zone: NighttimeZoneState) -> None:
        """Update zone priming state based on brightness."""
        brightness_elevated = zone.brightness_state.delta >= 0.02

        if (
            brightness_elevated or zone.brightness_state.is_rising
        ) and not zone.tracked_blobs:
            if not zone.primed:
                zone.primed = True
                zone.primed_frame = zone.frame_count
                zone.primed_until_frame = zone.frame_count + 45
            elif zone.frame_count < zone.primed_until_frame:
                zone.primed_until_frame = zone.frame_count + 45
        elif (
            zone.primed
            and zone.frame_count >= zone.primed_until_frame
            and not zone.tracked_blobs
        ):
            zone.primed = False

    def _detect_blobs(
        self, region, zone: NighttimeZoneState
    ) -> list[tuple[float, float, float]]:
        """Detect white blobs (headlights) in region."""
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
            cx = kp.pt[0] + zone.x1
            cy = kp.pt[1] + zone.y1
            results.append((cx, cy, kp.size))

        return results

    def _detect_taillights(
        self, frame, zone: NighttimeZoneState
    ) -> list[tuple[float, float, float]]:
        """Detect red blobs (taillights) in zone."""
        region = frame[zone.y1 : zone.y2, zone.x1 : zone.x2]
        if region.size == 0:
            return []

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 <= area <= 5000:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] + zone.x1
                    cy = M["m01"] / M["m00"] + zone.y1
                    size = np.sqrt(area)
                    results.append((cx, cy, size))

        return results

    def _update_blob_tracking(self, zone: NighttimeZoneState, blobs: list) -> None:
        """Update blob tracking state."""
        matched_blob_ids = set()

        for cx, cy, size in blobs:
            # Try to match to existing blob
            best_match = None
            best_dist = 50  # Max matching distance

            for blob_id, blob in zone.tracked_blobs.items():
                if blob.is_disqualified:
                    continue
                dist = np.sqrt((cx - blob.center[0]) ** 2 + (cy - blob.center[1]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_match = blob_id

            if best_match is not None:
                # Update existing blob
                blob = zone.tracked_blobs[best_match]
                blob.center = (cx, cy)
                blob.size = size
                blob.frames_seen += 1
                matched_blob_ids.add(best_match)
            else:
                # Create new blob
                blob_id = zone.next_blob_id
                zone.next_blob_id += 1
                half_size = int(size / 2)
                bbox = (
                    int(cx - half_size),
                    int(cy - half_size),
                    int(cx + half_size),
                    int(cy + half_size),
                )
                zone.tracked_blobs[blob_id] = TrackedBlob(
                    blob_id=blob_id,
                    center=(cx, cy),
                    size=size,
                    bbox=bbox,
                    first_seen_frame=zone.frame_count,
                )
                matched_blob_ids.add(blob_id)

    def _calculate_score(
        self,
        zone: NighttimeZoneState,
        blob: TrackedBlob,
        taillights: list,
    ) -> float:
        """Calculate detection score for a blob."""
        score = 0.0

        # Brightness features
        score += zone.brightness_state.delta * 5 * self.WEIGHT_BRIGHTNESS_DELTA
        if zone.brightness_state.is_rising:
            score += self.WEIGHT_BRIGHTNESS_RISING

        # Blob features
        score += self.WEIGHT_BLOB_PRESENT
        duration_norm = min(blob.frames_seen / 30, 1.0)
        score += duration_norm * self.WEIGHT_BLOB_DURATION
        size_norm = min(blob.size / 50, 1.0)
        score += size_norm * self.WEIGHT_BLOB_SIZE

        # Primed bonus
        if zone.primed:
            score += self.WEIGHT_PRIMED_BONUS

        # Taillight match (decisive)
        for taillight in taillights:
            if self._taillight_matches(blob, taillight):
                score += self.WEIGHT_TAILLIGHT_MATCH
                break

        return score

    def _taillight_matches(self, blob: TrackedBlob, taillight: tuple) -> bool:
        """Check if taillight position matches headlight."""
        tx, ty, _ = taillight
        bx, by = blob.center

        y_match = abs(ty - by) <= self.TAILLIGHT_Y_TOLERANCE
        x_behind = (tx - bx) >= self.TAILLIGHT_X_MIN_OFFSET

        return y_match and x_behind

    def _cleanup_stale_blobs(self, zone: NighttimeZoneState) -> None:
        """Remove blobs that haven't been seen recently."""
        stale_ids = []
        for blob_id, blob in zone.tracked_blobs.items():
            frames_missing = zone.frame_count - blob.first_seen_frame - blob.frames_seen
            if frames_missing > 10 or blob.is_disqualified:
                stale_ids.append(blob_id)

        for blob_id in stale_ids:
            del zone.tracked_blobs[blob_id]
