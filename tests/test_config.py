"""
Tests for configuration validation
"""

import unittest
import tempfile
import os
from pathlib import Path

from src.object_detection.config import (
    validate_config,
    ConfigValidationError,
    load_config_with_env
)
from src.object_detection.constants import ENV_CAMERA_URL, DEFAULT_QUEUE_SIZE


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation logic."""

    def setUp(self):
        """Create a temporary model file for testing."""
        self.temp_model = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model.close()
        self.model_path = self.temp_model.name

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def get_valid_config(self):
        """Return a minimal valid configuration."""
        return {
            'detection': {
                'model_file': self.model_path,
                'track_classes': [15, 16],
                'confidence_threshold': 0.25
            },
            'output': {
                'json_dir': 'data'
            },
            'camera': {
                'url': 'http://test:4747/video'
            },
            'runtime': {
                'default_duration_hours': 1.0
            }
        }

    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = self.get_valid_config()
        self.assertTrue(validate_config(config))

    def test_missing_detection_section(self):
        """Test that missing detection section fails."""
        config = self.get_valid_config()
        del config['detection']

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("Missing 'detection' section", str(cm.exception))

    def test_missing_model_file(self):
        """Test that missing model file fails."""
        config = self.get_valid_config()
        config['detection']['model_file'] = 'nonexistent.pt'

        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_invalid_model_extension(self):
        """Test that non-.pt model file fails."""
        config = self.get_valid_config()
        config['detection']['model_file'] = 'model.txt'

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("must be .pt format", str(cm.exception))

    def test_empty_track_classes(self):
        """Test that empty track_classes fails."""
        config = self.get_valid_config()
        config['detection']['track_classes'] = []

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("must contain at least one class", str(cm.exception))

    def test_invalid_track_classes(self):
        """Test that invalid class IDs fail."""
        config = self.get_valid_config()
        config['detection']['track_classes'] = [999]

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("COCO class IDs (0-79)", str(cm.exception))

    def test_invalid_confidence_threshold(self):
        """Test that out-of-range confidence fails."""
        config = self.get_valid_config()
        config['detection']['confidence_threshold'] = 1.5

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("between 0.0 and 1.0", str(cm.exception))

    def test_invalid_roi_ordering(self):
        """Test that invalid ROI boundaries fail."""
        config = self.get_valid_config()
        config['roi'] = {
            'horizontal': {
                'enabled': True,
                'crop_from_left_pct': 90,
                'crop_to_right_pct': 10  # Invalid: less than from
            }
        }

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("must be >", str(cm.exception))

    def test_line_with_invalid_classes(self):
        """Test that line with classes not in track_classes fails."""
        config = self.get_valid_config()
        config['lines'] = [{
            'type': 'vertical',
            'position_pct': 50,
            'description': 'test line',
            'allowed_classes': [0, 1]  # Not in track_classes
        }]

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("not in detection.track_classes", str(cm.exception))

    def test_zone_invalid_coordinates(self):
        """Test that zone with invalid coordinates fails."""
        config = self.get_valid_config()
        config['zones'] = [{
            'x1_pct': 50,
            'y1_pct': 50,
            'x2_pct': 30,  # Invalid: less than x1
            'y2_pct': 80,
            'description': 'test zone'
        }]

        with self.assertRaises(ConfigValidationError) as cm:
            validate_config(config)

        self.assertIn("x2_pct must be > x1_pct", str(cm.exception))


class TestConfigEnvironmentOverride(unittest.TestCase):
    """Test environment variable overrides."""

    def test_camera_url_from_env(self):
        """Test that camera URL can be loaded from environment."""
        config = {
            'camera': {
                'url': 'http://original:4747/video'
            },
            'runtime': {}
        }

        # Set environment variable
        os.environ[ENV_CAMERA_URL] = 'http://env:4747/video'

        try:
            updated_config = load_config_with_env(config)
            self.assertEqual(updated_config['camera']['url'], 'http://env:4747/video')
        finally:
            # Clean up
            if ENV_CAMERA_URL in os.environ:
                del os.environ[ENV_CAMERA_URL]

    def test_default_queue_size(self):
        """Test that default queue size is set if not specified."""
        config = {
            'runtime': {}
        }

        updated_config = load_config_with_env(config)
        self.assertEqual(updated_config['runtime']['queue_size'], DEFAULT_QUEUE_SIZE)

    def test_preserve_existing_queue_size(self):
        """Test that existing queue size is preserved."""
        config = {
            'runtime': {
                'queue_size': 5000
            }
        }

        updated_config = load_config_with_env(config)
        self.assertEqual(updated_config['runtime']['queue_size'], 5000)


if __name__ == '__main__':
    unittest.main()
