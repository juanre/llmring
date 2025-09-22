"""
Transport implementations for MCP Server Engine.
"""

from llmring.mcp.server.transport.base import Transport
from llmring.mcp.server.transport.stdio import StdioServerTransport, StdioTransport
from llmring.mcp.server.transport.websocket import WebSocketServerTransport, WebSocketTransport

# Core transports always available
__all__ = [
    "Transport",
    "StdioTransport",
    "StdioServerTransport",
    "WebSocketTransport",
    "WebSocketServerTransport",
]

# Optional HTTP transports - require FastAPI (currently not exposed)
# These imports are kept for potential future use but not exported
# try:
#     from .http import HTTPTransport, SessionManager  # Legacy, deprecated
#     from .streamable_http import ResponseMode, StreamableHTTPTransport
#
#     __all__.extend(
#         [
#             "StreamableHTTPTransport",
#             "ResponseMode",
#             "HTTPTransport",  # Legacy, deprecated
#             "SessionManager",  # Legacy, deprecated
#         ]
#     )
# except ImportError:
#     # HTTP transports not available without FastAPI
#     pass
