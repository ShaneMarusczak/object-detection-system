"""
Tests for configuration validation and event-driven architecture.
"""

import unittest
import tempfile
import os

from src.object_detection.config import (
    validate_config_full,
    load_config_with_env,
    prepare_runtime_config,
    derive_track_classes,
    build_plan,
)
from src.object_detection.utils.constants import ENV_CAMERA_URL, DEFAULT_QUEUE_SIZE


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation logic."""

    def setUp(self):
        """Create a temporary model file for testing."""
        self.temp_model = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self.temp_model.close()
        self.model_path = self.temp_model.name

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def get_valid_config(self):
        """Return a minimal valid configuration."""
        return {
            "detection": {
                "model_file": self.model_path,
                "track_classes": [],
                "confidence_threshold": 0.25,
            },
            "output": {"json_dir": "data"},
            "camera": {"url": "http://test:4747/video"},
            "runtime": {"default_duration_hours": 1.0},
            "lines": [
                {
                    "type": "vertical",
                    "position_pct": 50,
                    "description": "test line",
                    "allowed_classes": [2],
                }
            ],
            "events": [
                {
                    "name": "test-event",
                    "match": {
                        "event_type": "LINE_CROSS",
                        "line": "test line",
                        "object_class": "car",
                    },
                    "actions": {"json_log": True},
                }
            ],
        }

    def test_valid_config_passes(self):
        """Test that a valid config passes validation."""
        config = self.get_valid_config()
        result = validate_config_full(config)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

    def test_missing_detection_section(self):
        """Test that missing detection section fails validation."""
        config = self.get_valid_config()
        del config["detection"]
        result = validate_config_full(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("detection" in e for e in result.errors))

    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold fails validation."""
        config = self.get_valid_config()
        config["detection"]["confidence_threshold"] = 1.5
        result = validate_config_full(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("confidence" in e.lower() for e in result.errors))

    def test_invalid_zone_coordinates(self):
        """Test that zone x2 <= x1 fails validation."""
        config = self.get_valid_config()
        config["zones"] = [
            {
                "x1_pct": 50,
                "y1_pct": 20,
                "x2_pct": 30,  # Invalid: x2 < x1
                "y2_pct": 40,
                "description": "test zone",
            }
        ]
        result = validate_config_full(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("x2_pct" in e for e in result.errors))

    def test_invalid_event_zone_reference(self):
        """Test that event referencing non-existent zone fails."""
        config = self.get_valid_config()
        config["events"][0]["match"]["zone"] = "non-existent zone"
        del config["events"][0]["match"]["line"]
        result = validate_config_full(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("non-existent zone" in e for e in result.errors))


class TestEventDrivenWiring(unittest.TestCase):
    """Test event-driven configuration features."""

    def get_event_config(self):
        """Return config with events defined."""
        return {
            "detection": {
                "model_file": "yolo11n.pt",
                "track_classes": [],
                "confidence_threshold": 0.25,
            },
            "output": {"json_dir": "data"},
            "camera": {"url": "http://test/video"},
            "runtime": {"default_duration_hours": 1.0},
            "lines": [
                {"type": "vertical", "position_pct": 50, "description": "driveway"}
            ],
            "zones": [
                {
                    "x1_pct": 10,
                    "y1_pct": 10,
                    "x2_pct": 30,
                    "y2_pct": 30,
                    "description": "food bowl",
                }
            ],
            "events": [
                {
                    "name": "car-crossing",
                    "match": {
                        "event_type": "LINE_CROSS",
                        "line": "driveway",
                        "object_class": ["car", "truck"],
                    },
                    "actions": {"json_log": True},
                },
                {
                    "name": "cat-food",
                    "match": {
                        "event_type": "ZONE_ENTER",
                        "zone": "food bowl",
                        "object_class": "cat",
                    },
                    "actions": {"json_log": True},
                },
            ],
        }

    def test_derive_track_classes_from_events(self):
        """Test that track_classes are derived from event definitions."""
        config = self.get_event_config()
        classes = derive_track_classes(config)

        # Should have car (2), truck (7), cat (15)
        self.assertIn(2, classes)  # car
        self.assertIn(7, classes)  # truck
        self.assertIn(15, classes)  # cat
        self.assertEqual(len(classes), 3)

    def test_prepare_runtime_config_injects_classes(self):
        """Test that prepare_runtime_config injects derived classes."""
        config = self.get_event_config()
        self.assertEqual(config["detection"]["track_classes"], [])

        prepared = prepare_runtime_config(config)

        # Should now have the derived classes
        self.assertEqual(len(prepared["detection"]["track_classes"]), 3)
        self.assertIn(2, prepared["detection"]["track_classes"])

    def test_build_plan_shows_events(self):
        """Test that build_plan creates plan for all events."""
        config = self.get_event_config()
        plan = build_plan(config)

        self.assertEqual(len(plan.events), 2)
        event_names = [e.name for e in plan.events]
        self.assertIn("car-crossing", event_names)
        self.assertIn("cat-food", event_names)

    def test_build_plan_shows_geometry(self):
        """Test that build_plan includes geometry."""
        config = self.get_event_config()
        plan = build_plan(config)

        self.assertIn("driveway", plan.geometry["lines"])
        self.assertIn("food bowl", plan.geometry["zones"])


class TestImpliedActions(unittest.TestCase):
    """Test AWS-style implied action composition."""

    def get_digest_config(self):
        """Return config with digest that requires implied actions."""
        return {
            "detection": {
                "model_file": "yolo11n.pt",
                "track_classes": [],
                "confidence_threshold": 0.25,
            },
            "output": {"json_dir": "data"},
            "camera": {"url": "http://test/video"},
            "runtime": {"default_duration_hours": 1.0},
            "lines": [
                {"type": "vertical", "position_pct": 50, "description": "driveway"}
            ],
            "events": [
                {
                    "name": "car-crossing",
                    "match": {
                        "event_type": "LINE_CROSS",
                        "line": "driveway",
                        "object_class": "car",
                    },
                    "actions": {"email_digest": "daily_traffic"},
                }
            ],
            "digests": [
                {
                    "id": "daily_traffic",
                    "period_minutes": 1440,
                    "period_label": "Daily Traffic",
                    "photos": True,
                }
            ],
            "notifications": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 587,
                    "username": "test",
                    "password": "test",
                    "from_address": "test@test.com",
                    "to_addresses": ["test@test.com"],
                },
            },
        }

    def test_email_digest_implies_json_log(self):
        """Test that email_digest action implies json_log."""
        config = self.get_digest_config()
        plan = build_plan(config)

        event = plan.events[0]
        # email_digest should imply json_log
        self.assertIn("json_log (required by email_digest)", event.implied_actions)

    def test_photo_digest_implies_frame_capture(self):
        """Test that digest with photos implies frame_capture."""
        config = self.get_digest_config()
        plan = build_plan(config)

        event = plan.events[0]
        # photos: true should imply frame_capture
        implied_str = " ".join(event.implied_actions)
        self.assertIn("frame_capture", implied_str)


class TestNighttimeCarZones(unittest.TestCase):
    """Test nighttime car zone configuration."""

    def get_nighttime_config(self):
        """Return config with nighttime car zones."""
        return {
            "detection": {
                "model_file": "yolo11n.pt",
                "track_classes": [],
                "confidence_threshold": 0.25,
            },
            "output": {"json_dir": "data"},
            "camera": {"url": "http://test/video"},
            "runtime": {"default_duration_hours": 1.0},
            "nighttime_car_zones": [
                {
                    "name": "driveway entrance",
                    "x1_pct": 0,
                    "y1_pct": 50,
                    "x2_pct": 100,
                    "y2_pct": 100,
                    "pdf_report": "traffic_report",
                    "email_immediate": True,
                    "email_digest": "daily_digest",
                }
            ],
            "pdf_reports": [
                {"id": "traffic_report", "output_dir": "reports", "events": []}
            ],
            "digests": [{"id": "daily_digest", "period_minutes": 1440, "events": []}],
            "notifications": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 587,
                    "username": "test",
                    "password": "test",
                    "from_address": "test@test.com",
                    "to_addresses": ["test@test.com"],
                },
            },
        }

    def test_nighttime_car_zones_in_geometry(self):
        """Test that nighttime car zones appear in build_plan geometry."""
        config = self.get_nighttime_config()
        plan = build_plan(config)

        self.assertIn("nighttime_car_zones", plan.geometry)
        self.assertIn("driveway entrance", plan.geometry["nighttime_car_zones"])

    def test_nighttime_car_zones_consumers(self):
        """Test that nighttime car zone actions resolve to consumers."""
        config = self.get_nighttime_config()
        prepared = prepare_runtime_config(config)

        consumers = prepared.get("_resolved_consumers", [])
        self.assertIn("json_writer", consumers)
        self.assertIn("email_notifier", consumers)
        self.assertIn("email_digest", consumers)
        self.assertIn("pdf_report", consumers)

    def test_nighttime_car_zones_in_plan_consumers(self):
        """Test that build_plan includes nighttime car zone consumers."""
        config = self.get_nighttime_config()
        plan = build_plan(config)

        self.assertIn("json_writer", plan.consumers)
        self.assertIn("email_notifier", plan.consumers)
        self.assertIn("email_digest", plan.consumers)
        self.assertIn("pdf_report", plan.consumers)


class TestLoadConfigWithEnv(unittest.TestCase):
    """Test environment variable handling."""

    def test_env_override_camera_url(self):
        """Test that CAMERA_URL env var overrides config."""
        config = {"camera": {"url": "http://original/video"}, "runtime": {}}

        # Set env var
        os.environ[ENV_CAMERA_URL] = "http://override/video"
        try:
            result = load_config_with_env(config)
            self.assertEqual(result["camera"]["url"], "http://override/video")
        finally:
            del os.environ[ENV_CAMERA_URL]

    def test_default_queue_size(self):
        """Test that default queue size is set."""
        config = {"runtime": {}}
        result = load_config_with_env(config)
        self.assertEqual(result["runtime"]["queue_size"], DEFAULT_QUEUE_SIZE)


if __name__ == "__main__":
    unittest.main()
