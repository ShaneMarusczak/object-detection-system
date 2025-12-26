"""
Tests for detector logic (line crossing, zone detection)
"""

import unittest
from src.object_detection.core.detector import _detect_line_crossing
from src.object_detection.models import LineConfig


class TestLineCrossing(unittest.TestCase):
    """Test line crossing detection logic."""

    def test_vertical_line_left_to_right(self):
        """Test crossing vertical line from left to right."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves from x=40 to x=60, crossing line at x=50
        crossed, direction = _detect_line_crossing(
            prev_x=40.0,
            prev_y=100.0,
            curr_x=60.0,
            curr_y=100.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertTrue(crossed)
        self.assertEqual(direction, "LTR")

    def test_vertical_line_right_to_left(self):
        """Test crossing vertical line from right to left."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves from x=60 to x=40, crossing line at x=50
        crossed, direction = _detect_line_crossing(
            prev_x=60.0,
            prev_y=100.0,
            curr_x=40.0,
            curr_y=100.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertTrue(crossed)
        self.assertEqual(direction, "RTL")

    def test_vertical_line_no_crossing(self):
        """Test no crossing when object stays on same side."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves from x=30 to x=40, not crossing line at x=50
        crossed, direction = _detect_line_crossing(
            prev_x=30.0,
            prev_y=100.0,
            curr_x=40.0,
            curr_y=100.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertFalse(crossed)
        self.assertIsNone(direction)

    def test_horizontal_line_top_to_bottom(self):
        """Test crossing horizontal line from top to bottom."""
        line = LineConfig(
            line_id="H1",
            type="horizontal",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves from y=40 to y=60, crossing line at y=50
        crossed, direction = _detect_line_crossing(
            prev_x=100.0,
            prev_y=40.0,
            curr_x=100.0,
            curr_y=60.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertTrue(crossed)
        self.assertEqual(direction, "TTB")

    def test_horizontal_line_bottom_to_top(self):
        """Test crossing horizontal line from bottom to top."""
        line = LineConfig(
            line_id="H1",
            type="horizontal",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves from y=60 to y=40, crossing line at y=50
        crossed, direction = _detect_line_crossing(
            prev_x=100.0,
            prev_y=60.0,
            curr_x=100.0,
            curr_y=40.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertTrue(crossed)
        self.assertEqual(direction, "BTT")

    def test_diagonal_crossing(self):
        """Test diagonal movement that crosses line."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object moves diagonally, crossing vertical line
        crossed, direction = _detect_line_crossing(
            prev_x=30.0,
            prev_y=30.0,
            curr_x=70.0,
            curr_y=70.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        self.assertTrue(crossed)
        self.assertEqual(direction, "LTR")

    def test_edge_case_exactly_on_line(self):
        """Test movement starting exactly on the line."""
        line = LineConfig(
            line_id="V1",
            type="vertical",
            position_pct=50.0,
            description="center",
            allowed_classes=[15],
        )

        # Object starts on line, moves right
        crossed, direction = _detect_line_crossing(
            prev_x=50.0,
            prev_y=100.0,
            curr_x=60.0,
            curr_y=100.0,
            line=line,
            roi_width=100,
            roi_height=100,
        )

        # Should detect as crossing (from left side of line to right)
        self.assertTrue(crossed)
        self.assertEqual(direction, "LTR")


class TestZoneDetection(unittest.TestCase):
    """Test zone entry/exit detection logic."""

    def test_point_inside_zone(self):
        """Test if point is inside zone boundaries."""
        # Zone from (10, 20) to (30, 40) in percentage
        zone_x1 = 10.0  # 10% of 100 = 10
        zone_x2 = 30.0  # 30% of 100 = 30
        zone_y1 = 20.0  # 20% of 100 = 20
        zone_y2 = 40.0  # 40% of 100 = 40

        # Point at (20, 30) should be inside
        point_x = 20.0
        point_y = 30.0

        inside = zone_x1 <= point_x <= zone_x2 and zone_y1 <= point_y <= zone_y2

        self.assertTrue(inside)

    def test_point_outside_zone(self):
        """Test if point is outside zone boundaries."""
        zone_x1 = 10.0
        zone_x2 = 30.0
        zone_y1 = 20.0
        zone_y2 = 40.0

        # Point at (50, 50) should be outside
        point_x = 50.0
        point_y = 50.0

        inside = zone_x1 <= point_x <= zone_x2 and zone_y1 <= point_y <= zone_y2

        self.assertFalse(inside)

    def test_point_on_zone_boundary(self):
        """Test point exactly on zone boundary."""
        zone_x1 = 10.0
        zone_x2 = 30.0
        zone_y1 = 20.0
        zone_y2 = 40.0

        # Point exactly on left boundary
        point_x = 10.0
        point_y = 30.0

        inside = zone_x1 <= point_x <= zone_x2 and zone_y1 <= point_y <= zone_y2

        self.assertTrue(inside)  # Boundaries are inclusive


if __name__ == "__main__":
    unittest.main()
