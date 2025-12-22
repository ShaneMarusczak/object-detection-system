"""
Event Processor Module (k3s Cluster)

Handles everything after detection:
- Event enrichment (COCO class names, descriptions)
- Event routing/dispatching
- Consumers (JSON log, email, digest, frame capture)
- State persistence

Runs on k3s cluster, receives events from Jetson via Redis.
"""

from .enricher import EventEnricher
from .coco_classes import COCO_CLASSES, COCO_NAME_TO_ID, get_class_name

__all__ = [
    'EventEnricher',
    'COCO_CLASSES',
    'COCO_NAME_TO_ID',
    'get_class_name',
]
