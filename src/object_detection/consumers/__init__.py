"""
Event consumers for the object detection system.
Each consumer processes enriched events independently.
"""

from .json_writer import json_writer_consumer
from .email_notifier import email_notifier_consumer
from .email_digest import email_digest_consumer

__all__ = ['json_writer_consumer', 'email_notifier_consumer', 'email_digest_consumer']
