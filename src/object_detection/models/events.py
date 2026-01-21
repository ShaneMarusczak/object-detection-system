"""
Event Definition - Core class for declarative event matching.

Provides a runtime representation of event configurations that can efficiently
match raw events from the detector to their defined actions.
"""

from typing import Any


class EventDefinition:
    """
    Declarative event definition - specifies what to match and what actions to take.

    This is the runtime representation of an event config. It's created from
    config by the dispatcher and used to route events to consumers.

    Attributes:
        name: Unique identifier for this event definition
        match: Original match criteria from config
        actions: Resolved actions (already processed by prepare_runtime_config)
        event_type: Event type filter (LINE_CROSS, ZONE_ENTER, etc.)
        zone: Zone description filter
        line: Line description filter
        direction: Direction filter (for LINE_CROSS)
        object_classes: Set of object classes to match (None = match any)
    """

    def __init__(
        self,
        name: str,
        match: dict[str, Any],
        actions: dict[str, Any],
        cooldown_seconds: int = 0,
    ):
        """
        Create event definition from pre-resolved config.

        Actions should already be resolved by prepare_runtime_config() -
        no inference happens here at runtime.

        Args:
            name: Event definition name
            match: Match criteria dict
            actions: Resolved actions dict
            cooldown_seconds: Seconds to wait before event can fire again
        """
        self.name = name
        self.match = match
        self.actions = actions  # Already resolved - no inference needed
        self.cooldown_seconds = cooldown_seconds

        # Parse match criteria
        self.event_type = match.get("event_type")
        self.zone = match.get("zone")
        self.line = match.get("line")
        self.direction = match.get("direction")

        # Handle single class or list of classes
        obj_class = match.get("object_class")
        if isinstance(obj_class, list):
            self.object_classes = set(obj_class)
        elif obj_class:
            self.object_classes = {obj_class}
        else:
            self.object_classes = None  # Match any class

    def matches(self, event: dict[str, Any]) -> bool:
        """
        Check if raw event matches this definition.

        All criteria must match (AND logic). Unspecified criteria match anything.

        Args:
            event: Enriched event dict from detector

        Returns:
            True if event matches all specified criteria
        """
        # Check event type
        if self.event_type and event.get("event_type") != self.event_type:
            return False

        # Check zone
        if self.zone and event.get("zone_description") != self.zone:
            return False

        # Check line
        if self.line and event.get("line_description") != self.line:
            return False

        # Check direction (for LINE_CROSS events)
        if self.direction and event.get("direction") != self.direction:
            return False

        # Check object class
        if (
            self.object_classes
            and event.get("object_class_name") not in self.object_classes
        ):
            return False

        return True

    def get_object_classes(self) -> set[str]:
        """Get all object classes this event definition matches."""
        return self.object_classes if self.object_classes else set()

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        criteria = []
        if self.event_type:
            criteria.append(f"type={self.event_type}")
        if self.zone:
            criteria.append(f"zone={self.zone}")
        if self.line:
            criteria.append(f"line={self.line}")
        if self.object_classes:
            criteria.append(f"classes={self.object_classes}")
        return f"EventDefinition({self.name}: {', '.join(criteria)})"
