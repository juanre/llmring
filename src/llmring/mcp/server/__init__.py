"""
MCP Server - Core Model Context Protocol implementation.

A modular, extensible implementation of the Model Context Protocol (MCP) that
provides a clean separation between the protocol implementation and authentication/storage layers.
"""

from llmring.mcp.server.interfaces import (
    AuthProvider,
    MCPMiddleware,
    Prompt,
    Resource,
    StorageProvider,
    Tool,
)
from llmring.mcp.server.mcp_server import MCPServer
from llmring.mcp.server.protocol import JSONRPCError, JSONRPCRequest, JSONRPCResponse, JSONRPCRouter
from llmring.mcp.server.registries import FunctionRegistry, PromptRegistry, ResourceRegistry
from llmring.mcp.server.transport import (
    StdioServerTransport,
    StdioTransport,
    Transport,
    WebSocketServerTransport,
    WebSocketTransport,
)

__version__ = "0.1.0"

__all__ = [
    # Main server
    "MCPServer",
    # Interfaces
    "AuthProvider",
    "StorageProvider",
    "MCPMiddleware",
    "Tool",
    "Prompt",
    "Resource",
    # Protocol
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "JSONRPCRouter",
    # Registries
    "FunctionRegistry",
    "ResourceRegistry",
    "PromptRegistry",
    # Transport
    "Transport",
    "StdioTransport",
    "StdioServerTransport",
    "WebSocketTransport",
    "WebSocketServerTransport",
]
