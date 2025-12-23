"""
Tests for data models
"""

import unittest
from src.object_detection.core.models import (
    TrackedObject,
    LineConfig,
    ZoneConfig,
    ROIConfig,
)


class TestTrackedObject(unittest.TestCase):
    """Test TrackedObject dataclass functionality."""

    def test_creation(self):
        """Test creating a tracked object."""
        obj = TrackedObject(track_id=1, object_class=15, current_pos=(100.0, 200.0))

        self.assertEqual(obj.track_id, 1)
        self.assertEqual(obj.object_class, 15)
        self.assertEqual(obj.current_pos, (100.0, 200.0))
        self.assertIsNone(obj.previous_pos)
        self.assertEqual(len(obj.crossed_lines), 0)
        self.assertEqual(len(obj.active_zones), 0)

    def test_update_position(self):
        """Test position update moves current to previous."""
        obj = TrackedObject(track_id=1, object_class=15, current_pos=(100.0, 200.0))

        obj.update_position(150.0, 250.0)

        self.assertEqual(obj.previous_pos, (100.0, 200.0))
        self.assertEqual(obj.current_pos, (150.0, 250.0))

    def test_is_new(self):
        """Test is_new returns True when no previous position."""
        obj = TrackedObject(track_id=1, object_class=15, current_pos=(100.0, 200.0))

        self.assertTrue(obj.is_new())

        obj.update_position(150.0, 250.0)
        self.assertFalse(obj.is_new())

    def test_crossed_lines_tracking(self):
        """Test tracking crossed lines."""
        obj = TrackedObject(track_id=1, object_class=15, current_pos=(100.0, 200.0))

        obj.crossed_lines.add("V1")
        obj.crossed_lines.add("H1")

        self.assertIn("V1", obj.crossed_lines)
        self.assertIn("H1", obj.crossed_lines)
        self.assertEqual(len(obj.crossed_lines), 2)

    def test_active_zones_tracking(self):
        """Test tracking active zones with timestamps."""
        obj = TrackedObject(track_id=1, object_class=15, current_pos=(100.0, 200.0))

        obj.active_zones["Z1"] = 123.456
        obj.active_zones["Z2"] = 789.012

        self.assertEqual(obj.active_zones["Z1"], 123.456)
        self.assertEqual(obj.active_zones["Z2"], 789.012)
        self.assertEqual(len(obj.active_zones), 2)


class TestLineConfig(unittest.TestCase):
    """Test LineConfig dataclass."""

    def test_vertical_line(self):
        """Test vertical line configuration."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center line",
            allowed_classes=[15, 16],
        )

        self.assertEqual(line.line_id, "V1")
        self.assertEqual(line.type, "vertical")
        self.assertEqual(line.position_pct, 50.0)
        self.assertEqual(line.allowed_classes, [15, 16])

    def test_horizontal_line(self):
        """Test horizontal line configuration."""
        line = LineConfig(
            line_id="H1",
            type="horizontal",
            position_pct=30.0,
            description="upper boundary",
            allowed_classes=[0],
        )

        self.assertEqual(line.type, "horizontal")
        self.assertEqual(line.position_pct, 30.0)


class TestZoneConfig(unittest.TestCase):
    """Test ZoneConfig dataclass."""

    def test_zone_creation(self):
        """Test zone configuration."""
        zone = ZoneConfig(
            zone_id="Z1",
            x1_pct=10.0,
            y1_pct=20.0,
            x2_pct=30.0,
            y2_pct=40.0,
            description="food bowl",
            allowed_classes=[15],
        )

        self.assertEqual(zone.zone_id, "Z1")
        self.assertEqual(zone.x1_pct, 10.0)
        self.assertEqual(zone.y2_pct, 40.0)
        self.assertEqual(zone.allowed_classes, [15])


class TestROIConfig(unittest.TestCase):
    """Test ROIConfig dataclass."""

    def test_roi_disabled(self):
        """Test ROI configuration when disabled."""
        roi = ROIConfig(enabled=False)

        self.assertFalse(roi.enabled)
        self.assertEqual(roi.h_from, 0)
        self.assertEqual(roi.h_to, 100)

    def test_roi_enabled(self):
        """Test ROI configuration when enabled."""
        roi = ROIConfig(enabled=True, h_from=10.0, h_to=90.0, v_from=20.0, v_to=80.0)

        self.assertTrue(roi.enabled)
        self.assertEqual(roi.h_from, 10.0)
        self.assertEqual(roi.h_to, 90.0)
        self.assertEqual(roi.v_from, 20.0)
        self.assertEqual(roi.v_to, 80.0)


if __name__ == "__main__":
    unittest.main()
