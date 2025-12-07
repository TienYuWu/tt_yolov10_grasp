"""Simple event bus for publish-subscribe pattern."""

from typing import Callable, Dict, List, Any
from dataclasses import dataclass


@dataclass
class Event:
    """Base event class."""
    pass


@dataclass
class BatchProgressEvent(Event):
    """Batch processing progress event."""
    current: int
    total: int
    image_name: str


@dataclass
class BatchCompletedEvent(Event):
    """Batch processing completed event."""
    stats: Dict[str, int]  # {success, failed, total}


@dataclass
class BatchErrorEvent(Event):
    """Batch processing error event."""
    image_index: int
    error: str


@dataclass
class TrainingProgressEvent(Event):
    """Training progress event."""
    epoch: int
    total_epochs: int
    metrics: Dict[str, Any]


@dataclass
class TrainingCompletedEvent(Event):
    """Training completed event."""
    results: Dict[str, Any]


class EventBus:
    """Simple event bus implementation."""

    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[type, List[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle the event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Callable) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Callback function to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event instance to publish
        """
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")
