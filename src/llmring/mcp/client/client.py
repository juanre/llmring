"""Client-side connection pool abstractions for MCP transports.

This module exists primarily to provide a stable import location for optional
connection-pool support in transports (HTTP/SSE/Streamable HTTP). The current
codebase does not ship a concrete pool implementation, but transports accept a
pool-like object to reuse `httpx.AsyncClient` instances when embedded in larger
applications.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class ConnectionPool(Protocol):
    """Protocol for an object that can vend shared `httpx.AsyncClient` instances."""

    def get_async_client(
        self,
        *,
        base_url: str,
        timeout: float | None,
        headers: dict[str, str] | None = None,
        **kwargs: object,
    ) -> httpx.AsyncClient: ...
