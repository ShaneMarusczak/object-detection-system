"""
Event Enricher

Takes raw events from edge detector and adds:
- COCO class names (from class ID)
- Line/zone descriptions (from config)
- ISO timestamps
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .coco_classes import get_class_name


class EventEnricher:
    """
    Enriches raw events from edge detector.

    The edge detector sends minimal events with just IDs.
    This enricher adds human-readable context.
    """

    def __init__(self, config: dict):
        """
        Initialize enricher with processor config.

        Args:
            config: Processor configuration with lines, zones, events definitions
        """
        # Build lookup tables for descriptions
        self.line_descriptions: Dict[str, str] = {}
        self.zone_descriptions: Dict[str, str] = {}

        # Parse line descriptions
        v_count, h_count = 0, 0
        for line in config.get('lines', []):
            if line['type'] == 'vertical':
                v_count += 1
                line_id = f"V{v_count}"
            else:
                h_count += 1
                line_id = f"H{h_count}"
            self.line_descriptions[line_id] = line['description']

        # Parse zone descriptions
        for i, zone in enumerate(config.get('zones', []), 1):
            zone_id = f"Z{i}"
            self.zone_descriptions[zone_id] = zone['description']

    def enrich(self, event: dict) -> dict:
        """
        Enrich a raw event with human-readable context.

        Args:
            event: Raw event from edge detector with:
                - event_type: LINE_CROSS, ZONE_ENTER, ZONE_EXIT
                - device_id: Source device
                - track_id: Object tracking ID
                - object_class: COCO class ID (integer)
                - line_id or zone_id: Geometry reference
                - timestamp_relative: Seconds since detection start
                - direction (for LINE_CROSS)
                - dwell_time (for ZONE_EXIT)

        Returns:
            Enriched event with:
                - object_class_name: Human-readable class name
                - line_description or zone_description
                - timestamp: ISO 8601 timestamp
        """
        enriched = event.copy()

        # Add class name
        class_id = event.get('object_class')
        if class_id is not None:
            enriched['object_class_name'] = get_class_name(class_id)

        # Add line description
        line_id = event.get('line_id')
        if line_id and line_id in self.line_descriptions:
            enriched['line_description'] = self.line_descriptions[line_id]

        # Add zone description
        zone_id = event.get('zone_id')
        if zone_id and zone_id in self.zone_descriptions:
            enriched['zone_description'] = self.zone_descriptions[zone_id]

        # Add ISO timestamp
        if 'timestamp' not in enriched:
            enriched['timestamp'] = datetime.now(timezone.utc).isoformat()

        return enriched
