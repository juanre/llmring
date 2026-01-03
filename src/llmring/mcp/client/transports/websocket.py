"""WebSocket transport for MCP client.

The client currently supports stdio and HTTP-based transports. This module
provides a placeholder WebSocket transport to keep the public factory methods
stable while the implementation is completed.
"""

from __future__ import annotations

from typing import Any

from llmring.mcp.client.transports.base import Transport


class WebSocketTransport(Transport):
    """WebSocket transport implementation (not implemented)."""

    def __init__(self, url: str, timeout: float = 30.0, **kwargs: object):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.kwargs = kwargs

    async def start(self) -> None:
        raise NotImplementedError("WebSocket transport not yet implemented")

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("WebSocket transport not yet implemented")

    async def close(self) -> None:
        raise NotImplementedError("WebSocket transport not yet implemented")
