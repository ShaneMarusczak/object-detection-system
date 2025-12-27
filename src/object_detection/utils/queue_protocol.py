"""
Event Queue Protocol - Abstract interface for event transport.

Defines the interface that both multiprocessing.Queue and future
Redis Streams implementations must satisfy. This enables swapping
the transport layer without changing consumer code.

Usage:
    # Current (local mode):
    from multiprocessing import Queue
    queue: EventQueue = Queue()

    # Future (distributed mode):
    queue: EventQueue = RedisStreamQueue("redis://...", "detections")
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventQueue(Protocol):
    """
    Protocol for event queue implementations.

    Both multiprocessing.Queue and future Redis implementations
    must satisfy this interface.
    """

    def put(self, event: dict[str, Any]) -> None:
        """
        Send an event to the queue.

        Args:
            event: Event dictionary to send
        """
        ...

    def get(
        self,
        _block: bool = True,
        _timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Receive an event from the queue.

        Args:
            _block: Whether to block waiting for event
            _timeout: Maximum time to wait (None = forever)

        Returns:
            Event dictionary

        Raises:
            Empty: If non-blocking and no event available
        """
        ...


class CallbackQueueAdapter:
    """
    Adapter that wraps a callback function as an EventQueue.

    Useful for edge deployment where events are published via callback
    (e.g., to Redis) rather than put on a local queue.

    Example:
        def publish_to_redis(event):
            redis.xadd("detections", event)

        queue = CallbackQueueAdapter(publish_to_redis)
        queue.put({"event_type": "LINE_CROSS", ...})  # Calls publish_to_redis
    """

    def __init__(self, callback):
        """
        Create adapter from callback function.

        Args:
            callback: Function that accepts event dict
        """
        self._callback = callback

    def put(self, event: dict[str, Any]) -> None:
        """Forward event to callback."""
        self._callback(event)

    def get(
        self,
        _block: bool = True,
        _timeout: float | None = None,
    ) -> dict[str, Any]:
        """Not supported - this is a write-only adapter."""
        raise NotImplementedError(
            "CallbackQueueAdapter is write-only. Use for publishing, not consuming."
        )
