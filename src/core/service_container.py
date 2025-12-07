"""Simple dependency injection container."""

from typing import Any, Dict, Optional


class ServiceContainer:
    """Simple DI container for managing services."""

    def __init__(self):
        """Initialize service container."""
        self._services: Dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        """Register a service.

        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service

    def get(self, name: str) -> Optional[Any]:
        """Get a service by name.

        Args:
            name: Service name

        Returns:
            Service instance or None if not found
        """
        return self._services.get(name)

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service name

        Returns:
            True if service is registered
        """
        return name in self._services

    def remove(self, name: str) -> None:
        """Remove a service.

        Args:
            name: Service name
        """
        if name in self._services:
            del self._services[name]
