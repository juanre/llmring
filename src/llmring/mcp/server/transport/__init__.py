"""
Transport implementations for MCP Server Engine.
"""

from llmring.mcp.server.transport.base import Transport
from llmring.mcp.server.transport.stdio import StdioTransport, StdioServerTransport
from llmring.mcp.server.transport.websocket import (
    WebSocketTransport,
    WebSocketServerTransport,
)

# Core transports always available
__all__ = [
    "Transport",
    "StdioTransport",
    "StdioServerTransport",
    "WebSocketTransport",
    "WebSocketServerTransport",
]

# Optional HTTP transports - require FastAPI
try:
    from .streamable_http import StreamableHTTPTransport, ResponseMode
    from .http import HTTPTransport, SessionManager  # Legacy, deprecated

    __all__.extend(
        [
            "StreamableHTTPTransport",
            "ResponseMode",
            "HTTPTransport",  # Legacy, deprecated
            "SessionManager",  # Legacy, deprecated
        ]
    )
except ImportError:
    # HTTP transports not available without FastAPI
    pass
