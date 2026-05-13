from __future__ import annotations


class ServiceError(Exception):
    """Base class for expected service-layer errors."""


class DuplicateResourceError(ServiceError):
    """Raised when a unique business identifier already exists."""

    def __init__(self, resource: str, identifier: str) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} already exists: {identifier}")


class ValidationError(ServiceError):
    """Raised when a business rule rejects a request."""
