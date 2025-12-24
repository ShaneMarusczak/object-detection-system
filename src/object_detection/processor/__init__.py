"""
Event Processor Module (k3s Cluster)

Handles everything after detection:
- Event enrichment (COCO class names, descriptions)
- Event routing/dispatching
- Consumers (JSON log, email, digest, frame capture)
- State persistence

In local mode: receives events via multiprocessing.Queue
In distributed mode: receives events via Redis Streams
"""

from .coco_classes import COCO_CLASSES, COCO_NAME_TO_ID, get_class_name
from .digest_state import DigestStateManager, DigestPeriodState
from .dispatcher import dispatch_events
from ..models import EventDefinition

# Consumers
from .json_writer import json_writer_consumer
from .frame_capture import frame_capture_consumer

# Email handlers (fire-and-forget immediate, shutdown-time digest)
from .email_immediate import ImmediateEmailHandler
from .email_digest import generate_email_digest

# Services
from .email_service import EmailService
from .frame_service import FrameService

__all__ = [
    # Dispatcher
    "dispatch_events",
    "EventDefinition",
    # Enrichment
    "COCO_CLASSES",
    "COCO_NAME_TO_ID",
    "get_class_name",
    # State
    "DigestStateManager",
    "DigestPeriodState",
    # Consumers
    "json_writer_consumer",
    "frame_capture_consumer",
    # Email handlers
    "ImmediateEmailHandler",
    "generate_email_digest",
    # Services
    "EmailService",
    "FrameService",
]
