"""
Server-Sent Events (SSE) transport implementation for MCP client.

Provides SSE transport for server-to-client streaming with HTTP POST
for client-to-server requests, enabling real-time bidirectional
communication with MCP servers.
"""

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from llmring.mcp.client.transports.base import ConnectionState, Transport

logger = logging.getLogger(__name__)


class SSETransport(Transport):
    """
    Server-Sent Events (SSE) transport implementation for MCP client.

    Uses SSE for server-to-client streaming and HTTP POST for client-to-server
    requests, providing full bidirectional communication capability.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 60.0,
        max_reconnect_attempts: int | None = None,
        allowed_origins: list[str] | None = None,
        enable_cors_validation: bool = True,
        rate_limit_requests_per_minute: int | None = None,
        pool: Optional["ConnectionPool"] = None,
        **client_kwargs,
    ):
        """
        Initialize SSE transport.

        Args:
            base_url: Base URL of the MCP server
            timeout: Request timeout in seconds
            reconnect_interval: Initial reconnection interval in seconds
            max_reconnect_interval: Maximum reconnection interval in seconds
            max_reconnect_attempts: Maximum number of reconnection attempts (None = unlimited)
            allowed_origins: List of allowed origins for CORS validation
            enable_cors_validation: Whether to enable CORS origin validation
            rate_limit_requests_per_minute: Client-side rate limiting (requests per minute)
            pool: Optional connection pool to use
            **client_kwargs: Additional httpx client arguments
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.allowed_origins = allowed_origins or []
        self.enable_cors_validation = enable_cors_validation
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute

        # Build headers
        self.headers = self._build_headers()
        self._protocol_version: str | None = None

        # Connection management
        self.pool = pool
        self.http_client: httpx.AsyncClient | None = None
        self.sse_client: httpx.AsyncClient | None = None
        self.client_kwargs = client_kwargs

        # SSE connection management
        self._sse_task: asyncio.Agent | None = None
        self._reconnect_count = 0
        self._last_event_id: str | None = None
        self._should_reconnect = True

        # Request/response handling
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()

        # Rate limiting
        self._request_timestamps: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }

        return headers

    def set_protocol_version(self, version: str) -> None:
        """Persist negotiated MCP protocol version for outgoing requests."""
        self._protocol_version = version
        self.headers["MCP-Protocol-Version"] = version

    def _validate_url_security(self, url: str) -> None:
        """
        Validate URL for security compliance.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL is not secure for production use
        """
        parsed = urlparse(url)

        # Warn about non-HTTPS URLs in production
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            logger.warning(
                f"Using HTTP for SSE connection to {parsed.hostname}. "
                "Consider using HTTPS for production deployments."
            )

        # Check for suspicious hostnames
        if parsed.hostname and any(
            suspicious in parsed.hostname.lower()
            for suspicious in ["0.0.0.0", "metadata", "internal"]
        ):
            logger.warning(f"Potentially unsafe hostname in URL: {parsed.hostname}")

    async def _check_rate_limit(self) -> None:
        """
        Check and enforce client-side rate limiting.

        Raises:
            RuntimeError: If rate limit is exceeded
        """
        if not self.rate_limit_requests_per_minute:
            return

        async with self._rate_limit_lock:
            now = time.time()

            # Remove timestamps older than 1 minute
            cutoff = now - 60.0
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > cutoff
            ]

            # Check if we're at the limit
            if len(self._request_timestamps) >= self.rate_limit_requests_per_minute:
                oldest_allowed = self._request_timestamps[0] + 60.0
                wait_time = oldest_allowed - now
                if wait_time > 0:
                    raise RuntimeError(
                        f"Rate limit exceeded: {self.rate_limit_requests_per_minute} requests/minute. "
                        f"Wait {wait_time:.1f} seconds."
                    )

            # Add current timestamp
            self._request_timestamps.append(now)

    async def start(self) -> None:
        """Start the SSE transport and establish connections."""
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Validate URL security
            self._validate_url_security(self.base_url)

            await self._create_clients()
            await self._start_sse_connection()
            self._set_state(ConnectionState.CONNECTED)
            logger.info(f"SSE transport connected to {self.base_url}")
        except Exception as e:
            self._set_state(ConnectionState.ERROR)
            self._handle_error(e)
            raise

    async def _create_clients(self) -> None:
        """Create HTTP clients for requests and SSE."""
        if self.pool:
            # Use pooled clients
            self.http_client = self.pool.get_async_client(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
                **self.client_kwargs,
            )
            self.sse_client = self.pool.get_async_client(
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

                if (
                    self._should_reconnect
                    and self.state != ConnectionState.DISCONNECTED
                ):
                    await self._handle_reconnection()
                else:
                    break

    async def _connect_sse(self) -> None:
        """Establish SSE connection and read events."""
        if not self.sse_client:
            raise RuntimeError("SSE client not initialized")

        # Prepare SSE request headers
        sse_headers = self.headers.copy()
        if self._last_event_id:
            sse_headers["Last-Event-ID"] = self._last_event_id

        # Connect to SSE endpoint
        sse_url = "/events"
        logger.debug(f"Connecting to SSE endpoint: {sse_url}")

        # Apply protocol version header if negotiated
        if self._protocol_version:
            sse_headers["MCP-Protocol-Version"] = self._protocol_version
        async with self.sse_client.stream(
            "GET", sse_url, headers=sse_headers
        ) as response:
            response.raise_for_status()

            # Validate response headers for security
            self._validate_response_headers(response)

            # Validate content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("text/event-stream"):
                raise ValueError(f"Invalid SSE content type: {content_type}")

            logger.debug("SSE connection established")

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
            # Event ID
            self._last_event_id = line[3:].strip()

        elif line.startswith("event:"):
            # Event type (currently not used, but could be extended)
            pass

        elif line.startswith("data:"):
            # Event data
            data = line[5:].strip()
            if data:
                try:
                    message = json.loads(data)
                    await self._handle_sse_message(message)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in SSE data: {data} - {e}")
                except Exception as e:
                    logger.error(f"Error processing SSE message: {e}")

    async def _handle_sse_message(self, message: dict[str, Any]) -> None:
        """Handle incoming SSE message from server."""
        if "id" in message:
            # This is a response to a request
            request_id = str(message["id"])
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    # JSON-RPC error response
                    error = message["error"]
                    if isinstance(error, dict):
                        error_msg = error.get("message", str(error))
                        code = error.get("code", -32000)
                        future.set_exception(
                            ValueError(f"JSON-RPC error {code}: {error_msg}")
                        )
                    else:
                        future.set_exception(ValueError(f"JSON-RPC error: {error}"))
                else:
                    # Successful response
                    future.set_result(message.get("result", {}))
            else:
                logger.warning(
                    f"Received response for unknown request ID: {request_id}"
                )
        else:
            # This is a notification (no id field)
            self._handle_message(message)

    async def _handle_reconnection(self) -> None:
        """Handle SSE reconnection with exponential backoff."""
        if (
            self.max_reconnect_attempts
            and self._reconnect_count >= self.max_reconnect_attempts
        ):
            logger.error(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            self._set_state(ConnectionState.ERROR)
            return

        self._reconnect_count += 1

        # Calculate backoff interval with exponential increase
        interval = min(
            self.reconnect_interval * (2 ** (self._reconnect_count - 1)),
            self.max_reconnect_interval,
        )

        logger.info(f"Reconnecting in {interval}s (attempt {self._reconnect_count})")
        await asyncio.sleep(interval)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send JSON-RPC message via HTTP POST and wait for response via SSE.

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
            raise ValueError("Message must have an 'id' field for SSE transport")

        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Check rate limiting
            await self._check_rate_limit()

            # Send request via HTTP POST
            rpc_url = "/rpc"
            post_headers = {
                "Content-Type": "application/json",
            }

            logger.debug(
                f"Sending SSE HTTP request: {message.get('method', 'unknown')}"
            )

            if self._protocol_version:
                post_headers["MCP-Protocol-Version"] = self._protocol_version
            response = await self.http_client.post(
                rpc_url, headers=post_headers, json=message
            )

            # Check for immediate HTTP errors
            response.raise_for_status()

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
            logger.error(f"Error sending SSE request: {e}")
            raise

    async def close(self) -> None:
        """Close the SSE transport and all connections."""
        self._set_state(ConnectionState.DISCONNECTED)
        self._should_reconnect = False

        # Cancel SSE reader agent
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sse_task

        # Close clients if not using pool
        if not self.pool:
            if self.http_client:
                await self.http_client.aclose()
            if self.sse_client:
                await self.sse_client.aclose()

        self.http_client = None
        self.sse_client = None

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        logger.debug("SSE transport closed")

    async def send_response(self, message: dict[str, Any]) -> None:
        """
        Send JSON-RPC response for a server-initiated request back to the server via POST.
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        post_headers = {"Content-Type": "application/json"}
        if self._protocol_version:
            post_headers["MCP-Protocol-Version"] = self._protocol_version
        await self.http_client.post("/rpc", headers=post_headers, json=message)

    def validate_origin(self, origin: str) -> bool:
        """
        Validate origin header for CORS compliance.

        Args:
            origin: Origin header value to validate

        Returns:
            True if origin is allowed, False otherwise
        """
        if not self.enable_cors_validation:
            return True

        if not self.allowed_origins:
            # If CORS validation is enabled but no origins specified,
            # only allow localhost origins for security
            parsed = urlparse(origin)
            return parsed.hostname in ("localhost", "127.0.0.1", "::1")

        return origin in self.allowed_origins

    def _validate_response_headers(self, response: httpx.Response) -> None:
        """
        Validate response headers for security compliance.

        Args:
            response: HTTP response to validate

        Raises:
            ValueError: If response headers indicate security issues
        """
        # Check for CORS headers
        if self.enable_cors_validation:
            access_control_origin = response.headers.get("access-control-allow-origin")
            if access_control_origin == "*":
                logger.warning(
                    "Server allows all origins (*) which may be a security risk"
                )

        # Check for security headers
        security_headers = {
            "x-frame-options": "Clickjacking protection",
            "x-content-type-options": "MIME type sniffing protection",
            "strict-transport-security": "HTTPS enforcement",
        }

        missing_headers = []
        for header, description in security_headers.items():
            if header not in response.headers:
                missing_headers.append(f"{header} ({description})")

        if missing_headers:
            logger.info(
                f"Server missing recommended security headers: {', '.join(missing_headers)}"
            )

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get current connection information for debugging.

        Returns:
            Dictionary with connection details
        """
        return {
            "state": self.state.value,
            "reconnect_count": self._reconnect_count,
            "last_event_id": self._last_event_id,
            "pending_requests": len(self._pending_requests),
            "should_reconnect": self._should_reconnect,
        }


# Import ConnectionPool from client module to avoid circular imports
try:
    from ..client import ConnectionPool
except ImportError:
    # Fallback for when ConnectionPool is moved
    ConnectionPool = None
