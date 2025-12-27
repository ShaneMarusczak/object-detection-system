"""
Emitter Registry - Maps event types to emitter classes.

Registry is populated by emitter modules. Active emitters are built
from config event definitions at startup.
"""

import logging

logger = logging.getLogger(__name__)

# Registry: event_type -> emitter class
EMITTER_REGISTRY: dict[str, type] = {}


def register(event_type: str):
    """Decorator to register an emitter class for an event type."""

    def decorator(cls):
        EMITTER_REGISTRY[event_type] = cls
        return cls

    return decorator


def build_active_emitters(config: dict, frame_dims: tuple) -> list:
    """
    Build emitters needed for the event definitions in config.

    Scans event definitions, finds unique event_types, instantiates
    the corresponding emitters.

    Args:
        config: Full config dict
        frame_dims: (width, height) of frame for geometry calculations

    Returns:
        List of configured emitter instances
    """
    # Import emitters to populate registry (decorators register on import)
    from . import detected, line_cross, nighttime_car, zone_enter, zone_exit  # noqa: F401

    # Find event types used in config
    event_types_needed = set()
    for event_def in config.get("events", []):
        event_type = event_def.get("match", {}).get("event_type")
        if event_type:
            event_types_needed.add(event_type)

    logger.info(f"Event types in config: {event_types_needed}")

    # Build emitters
    active_emitters = []
    for event_type in event_types_needed:
        if event_type not in EMITTER_REGISTRY:
            logger.warning(f"No emitter registered for event type: {event_type}")
            continue

        emitter_class = EMITTER_REGISTRY[event_type]
        emitter = emitter_class(config, frame_dims)
        active_emitters.append(emitter)
        logger.info(f"Activated emitter: {emitter_class.__name__} -> {event_type}")

    return active_emitters
