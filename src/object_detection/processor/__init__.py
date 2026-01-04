"""
Event Processor Module (k3s Cluster)

Handles everything after detection:
- Event enrichment (COCO class names, descriptions)
- Event routing/dispatching
- Consumers (JSON log, command execution, frame capture)
- State persistence

In local mode: receives events via multiprocessing.Queue
In distributed mode: receives events via Redis Streams
"""

from .coco_classes import COCO_CLASSES, COCO_NAME_TO_ID, get_class_name
from .dispatcher import dispatch_events
from ..models import EventDefinition

# Consumers
from .json_writer import json_writer_consumer
from .frame_capture import frame_capture_consumer
from .vlm_analyzer import vlm_analyzer_consumer

# Command execution
from .command_runner import run_command

# Notifiers
from .notifiers import Notifier, create_notifier, create_notifiers

# Services
from .frame_service import FrameService

__all__ = [
    # Dispatcher
    "dispatch_events",
    "EventDefinition",
    # Enrichment
    "COCO_CLASSES",
    "COCO_NAME_TO_ID",
    "get_class_name",
    # Consumers
    "json_writer_consumer",
    "frame_capture_consumer",
    "vlm_analyzer_consumer",
    # Command execution
    "run_command",
    # Notifiers
    "Notifier",
    "create_notifier",
    "create_notifiers",
    # Services
    "FrameService",
]
