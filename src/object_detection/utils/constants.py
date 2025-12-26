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

# Frame storage
# Note: In distributed mode, this should be a shared path (NFS/S3/etc)
DEFAULT_TEMP_FRAME_DIR = "/tmp/frames"
DEFAULT_TEMP_FRAME_MAX_AGE = 30  # Seconds before cleanup

# Camera reconnection
MAX_CAMERA_RECONNECT_ATTEMPTS = 2
CAMERA_RECONNECT_DELAY = 2.0  # Seconds between reconnection attempts

# Environment variables
ENV_CAMERA_URL = "CAMERA_URL"
