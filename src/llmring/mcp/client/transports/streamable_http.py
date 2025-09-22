"""
Streamable HTTP transport implementation for MCP client.

Implements the MCP streamable HTTP transport specification which uses a single
endpoint that handles both HTTP POST (for client-to-server JSON-RPC) and
HTTP GET (for server-to-client SSE streaming).

This is the new base transport recommended by the MCP specification.
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx

from llmring.mcp.client.transports.base import ConnectionState, Transport

logger = logging.getLogger(__name__)


class StreamableHTTPTransport(Transport):
    """
    Streamable HTTP transport implementation for MCP client.

    Uses a single HTTP endpoint that supports both:
    - HTTP POST for client-to-server JSON-RPC requests
    - HTTP GET with Accept: text/event-stream for server-to-client SSE streaming

    This is the new base transport as specified in MCP 2025-03-26.
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/",
        timeout: float = 30.0,
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 60.0,
        max_reconnect_attempts: int | None = None,
        allowed_origins: list[str] | None = None,
        enable_cors_validation: bool = True,
        rate_limit_requests_per_minute: int | None = None,
        session_id: str | None = None,
        pool: Optional["ConnectionPool"] = None,
        **client_kwargs,
    ):
        """
        Initialize streamable HTTP transport.

        Args:
            base_url: Base URL of the MCP server
            endpoint: Single endpoint path for both POST and GET (default: "/")
            timeout: Request timeout in seconds
            reconnect_interval: Initial reconnection interval in seconds
            max_reconnect_interval: Maximum reconnection interval in seconds
            max_reconnect_attempts: Maximum number of reconnection attempts (None = unlimited)
            allowed_origins: List of allowed origins for CORS validation
            enable_cors_validation: Whether to enable CORS origin validation
            rate_limit_requests_per_minute: Client-side rate limiting (requests per minute)
            session_id: Optional session ID for resumable connections
            pool: Optional connection pool to use
            **client_kwargs: Additional httpx client arguments
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.allowed_origins = allowed_origins or []
        self.enable_cors_validation = enable_cors_validation
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute
        self.session_id = session_id or str(uuid.uuid4())
        self.pool = pool
        self.client_kwargs = client_kwargs

        # Build headers
        self.headers = self._build_headers()
        self._protocol_version: str | None = None

        # Connection management
        self.http_client: httpx.AsyncClient | None = None
        self.sse_client: httpx.AsyncClient | None = None
        self._sse_task: asyncio.Agent | None = None
        self._should_reconnect = False
        self._reconnect_count = 0
        self._last_event_id: str | None = None

        # Request/response management
        self._pending_requests: dict[str, asyncio.Future] = {}

        # Rate limiting
        self._request_timestamps: list[float] = []

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "User-Agent": "MCP-Client/1.0",
        }

        return headers

    def set_protocol_version(self, version: str) -> None:
        """Persist negotiated MCP protocol version for outgoing requests."""
        self._protocol_version = version
        self.headers["MCP-Protocol-Version"] = version

    def _validate_url_security(self, url: str) -> None:
        """Validate URL for security requirements."""
        parsed = urlparse(url)

        # Ensure HTTPS for non-localhost connections
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            if parsed.scheme != "https":
                raise ValueError(f"Non-localhost connections must use HTTPS: {url}")

        # Additional security checks could be added here
        if not parsed.hostname:
            raise ValueError(f"Invalid hostname in URL: {url}")

    def _validate_origin(self, origin: str) -> bool:
        """Validate origin against allowed origins for CORS."""
        if not self.enable_cors_validation:
            return True

        if not self.allowed_origins:
            # If no origins specified, allow localhost by default
            parsed = urlparse(origin)
            return parsed.hostname in ("localhost", "127.0.0.1", "::1")

        return origin in self.allowed_origins

    async def _check_rate_limit(self) -> None:
        """Check and enforce client-side rate limiting."""
        if not self.rate_limit_requests_per_minute:
            return

        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Remove old timestamps
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > cutoff]

        # Check if we're over the limit
        if len(self._request_timestamps) >= self.rate_limit_requests_per_minute:
            raise RuntimeError(
                f"Rate limit exceeded: {self.rate_limit_requests_per_minute} requests per minute"
            )

        # Record this request
        self._request_timestamps.append(now)

    def _validate_response_headers(self, response: httpx.Response) -> None:
        """Validate response headers for security."""
        # Check for required security headers in production
        if self.base_url.startswith("https://"):
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "Strict-Transport-Security": None,  # Should be present
            }

            for header, expected_values in security_headers.items():
                value = response.headers.get(header)
                if expected_values is None:
                    # Just check presence
                    if not value:
                        logger.warning(f"Missing security header: {header}")
                elif isinstance(expected_values, list):
                    # Check for specific values
                    if value not in expected_values:
                        logger.warning(f"Unexpected value for {header}: {value}")

    async def start(self) -> None:
        """Initialize the streamable HTTP transport."""
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Validate URL security
            full_url = urljoin(self.base_url, self.endpoint)
            self._validate_url_security(full_url)

            # Create HTTP clients
            await self._create_clients()

            self._set_state(ConnectionState.CONNECTED)

        except Exception as e:
            self._set_state(ConnectionState.ERROR)
            self._handle_error(e)
            raise

    async def _create_clients(self) -> None:
        """Create HTTP clients for requests and SSE."""
        if self.pool:
            # Use shared client from pool for regular requests
            self.http_client = self.pool.get_async_client(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
                **self.client_kwargs,
            )
            # Create separate client for SSE (no timeout)
            self.sse_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=None,  # SSE connections should not timeout
                follow_redirects=True,
                **self.client_kwargs,
            )
        else:
            # Create dedicated clients
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
                **self.client_kwargs,
            )
            self.sse_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=None,  # SSE connections should not timeout
                follow_redirects=True,
                **self.client_kwargs,
            )

    async def _start_sse_connection(self) -> None:
        """Start the SSE connection and background reading agent."""
        self._should_reconnect = True
        self._sse_task = asyncio.create_task(self._sse_reader())

    async def _sse_reader(self) -> None:
        """Background agent to read SSE events from server."""
        while self._should_reconnect and self.state != ConnectionState.DISCONNECTED:
            try:
                await self._connect_sse()
                # Reset reconnect count on successful connection
                self._reconnect_count = 0

            except Exception as e:
                logger.error(f"SSE connection error: {e}")

                if self._should_reconnect and self.state != ConnectionState.DISCONNECTED:
                    await self._handle_reconnection()
                else:
                    break

    async def _connect_sse(self) -> None:
        """Establish SSE connection and read events from the single endpoint."""
        if not self.sse_client:
            raise RuntimeError("SSE client not initialized")

        # Prepare SSE request headers - single endpoint with Accept header
        sse_headers = self.headers.copy()
        sse_headers["Accept"] = "text/event-stream"
        sse_headers["Cache-Control"] = "no-cache"

        # Add session ID for resumable connections (server expects 'Mcp-Session-Id')
        if self.session_id:
            sse_headers["Mcp-Session-Id"] = self.session_id

        # Add Last-Event-ID for resumable connections
        if self._last_event_id:
            sse_headers["Last-Event-ID"] = self._last_event_id

        logger.debug(f"Connecting to streamable HTTP endpoint: {self.endpoint}")

        if self._protocol_version:
            sse_headers["MCP-Protocol-Version"] = self._protocol_version
        async with self.sse_client.stream("GET", self.endpoint, headers=sse_headers) as response:
            response.raise_for_status()

            # Validate response headers for security
            self._validate_response_headers(response)

            # Validate content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("text/event-stream"):
                raise ValueError(f"Invalid SSE content type: {content_type}")

            logger.debug("Streamable HTTP SSE connection established")

            # Read SSE events
            async for line in response.aiter_lines():
                if not self._should_reconnect:
                    break

                await self._process_sse_line(line)

    async def _process_sse_line(self, line: str) -> None:
        """Process a single SSE line."""
        line = line.strip()

        if not line or line.startswith(":"):
            # Empty line or comment, ignore
            return

        if line.startswith("id:"):
            # Event ID for resumable connections
            self._last_event_id = line[3:].strip()

        elif line.startswith("data:"):
            # Event data
            data = line[5:].strip()
            if data:
                await self._process_sse_data(data)

    async def _process_sse_data(self, data: str) -> None:
        """Process SSE event data."""
        try:
            message = json.loads(data)

            # Check if this is a response to a pending request
            if "id" in message and message["id"] in self._pending_requests:
                request_id = message["id"]
                future = self._pending_requests.pop(request_id)

                if "error" in message:
                    # JSON-RPC error response
                    error = message["error"]
                    if isinstance(error, dict):
                        error_msg = error.get("message", str(error))
                        code = error.get("code", -32000)
                        future.set_exception(ValueError(f"JSON-RPC error {code}: {error_msg}"))
                    else:
                        future.set_exception(ValueError(f"JSON-RPC error: {error}"))
                else:
                    # Successful response
                    result = message.get("result", {})
                    future.set_result(result)

            elif message.get("method"):
                # Server notification - call notification handler if set
                if self.onmessage:
                    try:
                        await self.onmessage(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in SSE data: {data}, error: {e}")
        except Exception as e:
            logger.error(f"Error processing SSE data: {e}")

    async def _handle_reconnection(self) -> None:
        """Handle SSE reconnection with exponential backoff."""
        if self.max_reconnect_attempts and self._reconnect_count >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self._should_reconnect = False
            self._set_state(ConnectionState.ERROR)
            return

        self._reconnect_count += 1

        # Calculate backoff delay with exponential backoff
        delay = min(
            self.reconnect_interval * (2 ** (self._reconnect_count - 1)),
            self.max_reconnect_interval,
        )

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send JSON-RPC message via HTTP POST to the single endpoint.

        Args:
            message: JSON-RPC message dictionary

        Returns:
            JSON-RPC response result

        Raises:
            RuntimeError: If transport is not connected
            ValueError: If there's a JSON-RPC error
            TimeoutError: If response times out
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")

        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        request_id = str(message.get("id"))
        if not request_id:
            raise ValueError("Message must have an 'id' field for streamable HTTP transport")

        # Create future for response (will be fulfilled via SSE)
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Check rate limiting
            await self._check_rate_limit()

            # Send request via HTTP POST to the same endpoint
            post_headers = self.headers.copy()
            post_headers["Content-Type"] = "application/json"
            # Ensure server can select JSON response mode without 406
            post_headers.setdefault("Accept", "application/json")

            logger.debug(f"Sending streamable HTTP request: {message.get('method', 'unknown')}")

            if self._protocol_version:
                post_headers["MCP-Protocol-Version"] = self._protocol_version
            response = await self.http_client.post(
                self.endpoint, headers=post_headers, json=message
            )

            # Check for immediate HTTP errors
            response.raise_for_status()

            # For streamable HTTP, we might get:
            # 1. Immediate JSON response (non-streaming)
            # 2. 202 Accepted (response will come via SSE)
            # 3. SSE stream response

            content_type = response.headers.get("content-type", "")

            if content_type.startswith("application/json"):
                # Immediate JSON response
                try:
                    result = response.json()
                    # Capture session id from response headers if provided
                    session_header = response.headers.get("Mcp-Session-Id") or response.headers.get(
                        "mcp-session-id"
                    )
                    if session_header:
                        self.session_id = session_header
                    self._pending_requests.pop(request_id, None)

                    if "error" in result:
                        error = result["error"]
                        if isinstance(error, dict):
                            message = error.get("message", str(error))
                            code = error.get("code", -32000)
                            raise ValueError(f"JSON-RPC error {code}: {message}")
                        else:
                            raise ValueError(f"JSON-RPC error: {error}")

                    return result.get("result", {})
                except json.JSONDecodeError:
                    # Fall through to wait for SSE response
                    pass

            # Wait for response via SSE
            result = await asyncio.wait_for(response_future, timeout=self.timeout)
            return result

        except asyncio.TimeoutError:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {request_id} timed out after {self.timeout}s")
        except Exception as e:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            logger.error(f"Error sending streamable HTTP request: {e}")
            raise

    async def close(self) -> None:
        """Close the streamable HTTP transport."""
        self._set_state(ConnectionState.DISCONNECTED)
        self._should_reconnect = False

        # Cancel SSE reading agent
        if self._sse_task:
            self._sse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sse_task
            self._sse_task = None

        # Reject any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("Transport closed"))
        self._pending_requests.clear()

        # Close HTTP clients
        if self.sse_client and not self.pool:
            await self.sse_client.aclose()
        if self.http_client and not self.pool:
            await self.http_client.aclose()

        self.http_client = None
        self.sse_client = None

    async def send_response(self, message: dict[str, Any]) -> None:
        """Send JSON-RPC response for server-initiated request to the single endpoint."""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        post_headers = self.headers.copy()
        post_headers["Content-Type"] = "application/json"
        if self._protocol_version:
            post_headers["MCP-Protocol-Version"] = self._protocol_version
        await self.http_client.post(self.endpoint, headers=post_headers, json=message)

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the current connection."""
        return {
            "transport_type": "streamable_http",
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "state": self.state.value,
            "session_id": self.session_id,
            "reconnect_count": self._reconnect_count,
            "last_event_id": self._last_event_id,
            "pending_requests": len(self._pending_requests),
        }


# Import ConnectionPool from client module to avoid circular imports
try:
    from ..client import ConnectionPool
except ImportError:
    # Fallback for when ConnectionPool is moved
    ConnectionPool = None
