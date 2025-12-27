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

from ..models import EventDefinition
from .coco_classes import COCO_CLASSES, COCO_NAME_TO_ID, get_class_name
from .digest_scheduler import DigestScheduler
from .digest_state import DigestPeriodState, DigestStateManager
from .dispatcher import dispatch_events
from .email_digest import generate_email_digest

# Email handlers (fire-and-forget immediate, scheduled/shutdown digest)
from .email_immediate import ImmediateEmailHandler

# Services
from .email_service import EmailService
from .frame_capture import frame_capture_consumer
from .frame_service import FrameService

# Consumers
from .json_writer import json_writer_consumer

__all__ = [
    # Enrichment
    "COCO_CLASSES",
    "COCO_NAME_TO_ID",
    "DigestPeriodState",
    "DigestScheduler",
    # State
    "DigestStateManager",
    # Services
    "EmailService",
    "EventDefinition",
    "FrameService",
    # Email handlers
    "ImmediateEmailHandler",
    # Dispatcher
    "dispatch_events",
    "frame_capture_consumer",
    "generate_email_digest",
    "get_class_name",
    # Consumers
    "json_writer_consumer",
]
