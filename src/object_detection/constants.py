"""
Constants used throughout the object detection system
"""

# Performance and monitoring
FPS_REPORT_INTERVAL = 100  # Report FPS every N frames
FPS_WINDOW_SIZE = 100  # Number of frames to average for FPS calculation
SUMMARY_EVENT_INTERVAL = 50  # Print summary every N events

# Detection thresholds
MIN_TRACKING_TIME = 0.1  # Minimum seconds to track before calculating speed

# Queue configuration
DEFAULT_QUEUE_SIZE = 1000  # Default max queue size if not in config

# Timeouts (seconds)
DEFAULT_DETECTOR_SHUTDOWN_TIMEOUT = 5
DEFAULT_ANALYZER_SHUTDOWN_TIMEOUT = 10
DEFAULT_ANALYZER_STARTUP_DELAY = 1

# Camera reconnection
MAX_CAMERA_RECONNECT_ATTEMPTS = 2
CAMERA_RECONNECT_DELAY = 2.0  # Seconds between reconnection attempts

# Environment variables
ENV_CAMERA_URL = "CAMERA_URL"

# OpenCV settings
QT_PLATFORM = 'offscreen'  # For headless operation
