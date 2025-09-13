"""
Abstract interfaces for MCP Server Engine.
"""

from llmring.mcp.server.interfaces.auth import AuthProvider
from llmring.mcp.server.interfaces.storage import (
    StorageProvider,
    Tool,
    Prompt,
    Resource,
)
from llmring.mcp.server.interfaces.middleware import MCPMiddleware

__all__ = [
    "AuthProvider",
    "StorageProvider",
    "Tool",
    "Prompt",
    "Resource",
    "MCPMiddleware",
]
