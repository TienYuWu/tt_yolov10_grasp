"""Core infrastructure for Smart Label application."""

from .event_bus import EventBus
from .service_container import ServiceContainer

__all__ = ['EventBus', 'ServiceContainer']
