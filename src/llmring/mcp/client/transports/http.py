"""
HTTP transport implementation for MCP client.

Provides HTTP/HTTPS transport for JSON-RPC communication with MCP servers,
including connection pooling, retry logic, and authentication support.
"""

import logging
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llmring.mcp.client.transports.base import ConnectionState, Transport

logger = logging.getLogger(__name__)


class HTTPTransport(Transport):
    """
    HTTP transport implementation for MCP client.

    Uses httpx for HTTP communication with support for connection pooling,
    retry logic, and various authentication methods.
    """

    def __init__(
        self,
        base_url: str,
        enable_retry: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
        pool: Optional["ConnectionPool"] = None,
        **client_kwargs,
    ):
        """
        Initialize HTTP transport.

        Args:
            base_url: Base URL of the MCP server
            enable_retry: Whether to enable automatic retry on failures
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            pool: Optional connection pool to use
            **client_kwargs: Additional httpx client arguments
        """
        super().__init__()
        self.base_url = base_url
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.timeout = timeout

        # Build headers
        self.headers = self._build_headers()
        self._protocol_version: str | None = None

        # Connection pool management
        self.pool = pool
        self.client: httpx.AsyncClient | None = None
        self.client_kwargs = client_kwargs

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
        }

        return headers

    def set_protocol_version(self, version: str) -> None:
        """Persist negotiated MCP protocol version for outgoing requests."""
        self._protocol_version = version
        # Update existing headers and ensure future clients inherit it
        self.headers["MCP-Protocol-Version"] = version

    async def start(self) -> None:
        """Initialize the HTTP transport connection."""
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Create HTTP client
            if self.pool:
                self.client = self.pool.get_async_client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    follow_redirects=True,
                    **self.client_kwargs,
                )
            else:
                self.client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    follow_redirects=True,
                    **self.client_kwargs,
                )

            # Test connection with a simple request (we'll validate this works)
            # For now, just mark as connected since HTTP is connectionless
            self._set_state(ConnectionState.CONNECTED)

        except Exception as e:
            self._set_state(ConnectionState.ERROR)
            self._handle_error(e)
            raise

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send JSON-RPC message via HTTP POST and return response.

        Args:
            message: JSON-RPC message dictionary

        Returns:
            JSON-RPC response dictionary

        Raises:
            Exception: If transport is not connected or request fails
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")

        if not self.client:
            raise RuntimeError("HTTP client not initialized")

        # Use retry logic if enabled
        if self.enable_retry:
            return await self._send_with_retry(message)
        else:
            return await self._send_request(message)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def _send_with_retry(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send request with automatic retry logic."""
        return await self._send_request(message)

    async def _send_request(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send HTTP request with JSON-RPC message.

        Args:
            message: JSON-RPC message dictionary

        Returns:
            JSON-RPC response result

        Raises:
            httpx.HTTPError: If there's an HTTP error
            ValueError: If there's an error in the JSON-RPC response
        """
        # Log request details
        logger.debug(f"Making HTTP JSON-RPC request to {self.base_url}: {message.get('method')}")

        try:
            # Ensure protocol header is present if negotiated
            headers = dict(self.headers)
            if self._protocol_version:
                headers["MCP-Protocol-Version"] = self._protocol_version
            response = await self.client.post("/", headers=headers, json=message)

            # Raise HTTP errors
            response.raise_for_status()

            # Parse the response
            result = response.json()
        except Exception as e:
            logger.error(f"Error making HTTP request: {e!s}")
            self._handle_error(e)
            raise

        # Check for JSON-RPC errors
        if "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                message = error.get("message", str(error))
                code = error.get("code", -32000)
                raise ValueError(f"JSON-RPC error {code}: {message}")
            else:
                raise ValueError(f"JSON-RPC error: {error}")

        # Return the result
        return result.get("result", {})

    async def send_response(self, message: dict[str, Any]) -> None:
        """
        HTTP transport cannot receive server-initiated requests; responses are not applicable.
        """
        raise NotImplementedError("HTTP transport does not support server-initiated responses")

    async def close(self) -> None:
        """Close the HTTP transport."""
        self._set_state(ConnectionState.DISCONNECTED)

        # Note: We don't close the client here if it's from a pool
        # The pool manages the lifecycle of shared clients
        if self.client and not self.pool:
            await self.client.aclose()

        self.client = None


# Import ConnectionPool from client module to avoid circular imports
# We'll need to refactor this in the next step
try:
    from ..client import ConnectionPool
except ImportError:
    # Fallback for when ConnectionPool is moved
    ConnectionPool = None
