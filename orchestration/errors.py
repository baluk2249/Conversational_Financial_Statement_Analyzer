"""Explicit errors raised at the application orchestration boundary."""


class OrchestrationInputError(ValueError):
    """Raised when a valid request cannot be routed with the supplied inputs."""


class CitationValidationError(RuntimeError):
    """Raised if the final response fails citation validation."""
