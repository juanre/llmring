"""
Transport layer for MCP client.

This module provides a pluggable transport system that allows the MCP client
to work with different communication protocols (HTTP, STDIO, SSE) through
a unified interface.
"""

from .base import ConnectionState, Transport
from .factory import create_transport, get_supported_transports, register_transport

# Import transports to ensure they're registered
try:
    from llmring.mcp.client.transports.http import HTTPTransport
except ImportError:
    HTTPTransport = None

try:
    from llmring.mcp.client.transports.stdio import STDIOTransport
except ImportError:
    STDIOTransport = None

try:
    from llmring.mcp.client.transports.sse import SSETransport
except ImportError:
    SSETransport = None

try:
    from llmring.mcp.client.transports.streamable_http import StreamableHTTPTransport
except ImportError:
    StreamableHTTPTransport = None

__all__ = [
    "ConnectionState",
    "Transport",
    "create_transport",
    "get_supported_transports",
    "register_transport",
]

if HTTPTransport:
    __all__.append("HTTPTransport")
if STDIOTransport:
    __all__.append("STDIOTransport")
if SSETransport:
    __all__.append("SSETransport")
if StreamableHTTPTransport:
    __all__.append("StreamableHTTPTransport")
