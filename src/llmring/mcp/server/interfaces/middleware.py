"""
Middleware interface for MCP Server Engine.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any


class MCPMiddleware(ABC):
    """Abstract middleware interface for MCP request processing."""

    @abstractmethod
    async def process_request(self, request: Any, call_next: Callable) -> Any:
        """
        Process request before handler.

        Args:
            request: The incoming request
            call_next: The next middleware or handler in the chain

        Returns:
            Response from the handler chain
        """
        pass
