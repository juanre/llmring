"""
Transport factory for creating and managing transport instances.

Provides a factory pattern for dynamic transport selection and
a registry for custom transport types.
"""

from typing import Any

from llmring.mcp.client.transports.base import Transport

# Global transport registry
_transport_registry: dict[str, type[Transport]] = {}


def register_transport(transport_type: str, transport_class: type[Transport]) -> None:
    """
    Register a custom transport type.

    Args:
        transport_type: String identifier for the transport type
        transport_class: Transport class that implements the Transport interface

    Raises:
        ValueError: If transport_class doesn't implement Transport interface
    """
    if not issubclass(transport_class, Transport):
        raise ValueError(
            f"Transport class {transport_class} must inherit from Transport"
        )

    _transport_registry[transport_type] = transport_class


def create_transport(transport_type: str, config: dict[str, Any]) -> Transport:
    """
    Create a transport instance based on type and configuration.

    Args:
        transport_type: Transport type ("http", "stdio", "sse", or custom registered type)
        config: Configuration dictionary specific to the transport type

    Returns:
        Transport instance configured according to the provided config

    Raises:
        ValueError: If transport_type is not supported or config is invalid
    """
    if transport_type not in _transport_registry:
        raise ValueError(f"Unsupported transport type: {transport_type}")

    transport_class = _transport_registry[transport_type]

    try:
        return transport_class(**config)
    except TypeError as e:
        raise ValueError(f"Invalid configuration for {transport_type} transport: {e}")


def get_supported_transports() -> list[str]:
    """
    Get list of supported transport types.

    Returns:
        List of transport type strings
    """
    return list(_transport_registry.keys())


def _register_builtin_transports() -> None:
    """Register built-in transport types."""
    # Import here to avoid circular imports
    try:
        from .http import HTTPTransport

        register_transport("http", HTTPTransport)
    except ImportError:
        pass  # HTTP transport not available

    try:
        from .stdio import STDIOTransport

        register_transport("stdio", STDIOTransport)
    except ImportError:
        pass  # STDIO transport not available

    try:
        from .sse import SSETransport

        register_transport("sse", SSETransport)
    except ImportError:
        pass  # SSE transport not available

    try:
        from .streamable_http import StreamableHTTPTransport

        register_transport("streamable_http", StreamableHTTPTransport)
    except ImportError:
        pass  # Streamable HTTP transport not available


# Register built-in transports on module import
_register_builtin_transports()
