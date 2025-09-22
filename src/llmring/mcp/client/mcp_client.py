"""
MCP (Model Context Protocol) client implementation.

This module provides a clean, efficient client for the Model Context Protocol,
supporting multiple transport protocols including HTTP/SSE, WebSocket, and stdio.

Key features:
- Factory-based API for transport selection
- Sync and async API support
- Multiple transport protocols (HTTP, WebSocket, stdio)
- Resource subscription management
- Tool execution with capability caching
- Connection state management
- Robust error handling
"""

import asyncio
import inspect
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from llmring.exceptions import (
    MCPError,
    MCPProtocolError,
    MCPTransportError,
    NetworkError,
    TimeoutError,
)
from llmring.mcp.client.transports import Transport
from llmring.mcp.client.transports.http import HTTPTransport
from llmring.mcp.client.transports.stdio import STDIOTransport

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration for bidirectional communication."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class CapabilityCache:
    """Cache for server capabilities to reduce initialization calls."""

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the capability cache.

        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[dict[str, Any], datetime]] = OrderedDict()
        self._max_size = 100

    def get(self, server_key: str) -> dict[str, Any] | None:
        """Get cached capabilities for a server."""
        if server_key not in self._cache:
            return None

        capabilities, cached_at = self._cache[server_key]
        if datetime.now() - cached_at > timedelta(seconds=self.ttl_seconds):
            del self._cache[server_key]
            return None

        # Move to end for LRU
        self._cache.move_to_end(server_key)
        return capabilities

    def set(self, server_key: str, capabilities: dict[str, Any]):
        """Cache capabilities for a server."""
        self._cache[server_key] = (capabilities, datetime.now())

        # Evict oldest if at capacity
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached capabilities."""
        self._cache.clear()


class MCPClient:
    """MCP (Model Context Protocol) client with clean factory design.

    Use factory methods to create clients for specific transports:
    - MCPClient.stdio() for stdio transport
    - MCPClient.http() for HTTP transport
    - MCPClient.websocket() for WebSocket transport

    Example:
        >>> # stdio transport
        >>> client = MCPClient.stdio(
        ...     command=["python", "-m", "my_server"]
        ... )
        >>>
        >>> # HTTP transport
        >>> client = MCPClient.http("http://localhost:8080")
        >>>
        >>> # Initialize and use
        >>> with client:
        ...     client.initialize()
        ...     tools = client.list_tools()
    """

    def __init__(self, transport: Transport, protocol_version: str = "2025-06-18"):
        """
        Private constructor - use factory methods instead.

        Args:
            transport: Configured transport instance
            protocol_version: MCP protocol version
        """
        self.logger = logging.getLogger(__name__)
        self.transport = transport
        self.protocol_version = protocol_version
        self.capability_cache = CapabilityCache(ttl_seconds=3600)
        self.session_id = None
        self.server_capabilities: dict[str, Any] | None = None
        self.negotiated_protocol_version: str | None = None

        # Dedicated background loop for sync calls to avoid losing agents between calls
        self._loop_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Resource subscription management
        self._resource_subscriptions = {}
        self._resource_uri_subscriptions = {}

        # Bidirectional communication support
        self._notification_handlers: dict[str, list[Callable]] = defaultdict(list)
        self._method_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._connection_state = ConnectionState.DISCONNECTED
        self._connection_state_handlers: list[Callable[[ConnectionState], None]] = []
        self._connected_handlers: list[Callable[[], None]] = []

    @classmethod
    def stdio(
        cls,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        allow_unsafe_commands: bool = False,
        **kwargs,
    ) -> "MCPClient":
        """
        Create client with stdio transport.

        Args:
            command: Command to execute (e.g., ["python", "-m", "mcp_server"])
            cwd: Working directory for the process
            env: Environment variables
            timeout: Request timeout
            allow_unsafe_commands: Allow potentially unsafe commands
            **kwargs: Additional STDIOTransport parameters

        Returns:
            MCPClient configured for stdio transport
        """
        transport = STDIOTransport(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            allow_unsafe_commands=allow_unsafe_commands,
            **kwargs,
        )
        return cls(transport=transport)

    @classmethod
    def http(
        cls,
        url: str,
        timeout: float = 30.0,
        enable_retry: bool = True,
        max_retries: int = 3,
        **kwargs,
    ) -> "MCPClient":
        """
        Create client with HTTP transport.

        Args:
            url: HTTP URL (e.g., "http://localhost:8080")
            timeout: Request timeout
            enable_retry: Enable retry logic
            max_retries: Maximum retry attempts
            **kwargs: Additional HTTPTransport parameters

        Returns:
            MCPClient configured for HTTP transport
        """
        transport = HTTPTransport(
            base_url=url,
            timeout=timeout,
            enable_retry=enable_retry,
            max_retries=max_retries,
            **kwargs,
        )
        return cls(transport=transport)

    @classmethod
    def websocket(cls, url: str, timeout: float = 30.0, **kwargs) -> "MCPClient":
        """
        Create client with WebSocket transport.

        Args:
            url: WebSocket URL (e.g., "ws://localhost:8765")
            timeout: Request timeout
            **kwargs: Additional WebSocketTransport parameters

        Returns:
            MCPClient configured for WebSocket transport
        """
        # Import here to avoid circular imports
        try:
            from .transports.websocket import WebSocketTransport

            transport = WebSocketTransport(url=url, timeout=timeout, **kwargs)
            return cls(transport=transport)
        except ImportError:
            raise ImportError("WebSocket transport not available")

    def _ensure_loop(self) -> None:
        if self._loop and self._loop_thread and self._loop_thread.is_alive():
            return

        def loop_runner():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(None)

        self._loop_thread = threading.Thread(target=loop_runner, daemon=True)
        self._loop_thread.start()

        # Wait briefly for loop to start
        for _ in range(100):
            if self._loop is not None:
                break
            time.sleep(0.01)

    def _run_async(self, coro):
        if not inspect.iscoroutine(coro):
            coro = coro()

        # Always use a persistent background event loop to avoid losing agents
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120.0)

    def _make_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a JSON-RPC request via transport."""
        request_id = str(uuid.uuid4())

        request_data = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        self.logger.debug(f"Making JSON-RPC request: {method}")

        try:
            # Ensure transport is connected - only start if not already connected
            from .transports.base import ConnectionState as TransportConnectionState

            if (
                hasattr(self.transport, "state")
                and self.transport.state != TransportConnectionState.CONNECTED
            ):
                self.logger.debug(
                    f"Transport not connected, starting... (current state: {self.transport.state})"
                )
                self._run_async(self.transport.start())
                self.logger.debug(f"Transport state after start: {self.transport.state}")
            elif not hasattr(self.transport, "state"):
                # Fallback for transports without state tracking
                self.logger.debug("Transport has no state tracking, checking if already started")
                if not getattr(self.transport, "_started", False):
                    self._run_async(self.transport.start())
                    self.transport._started = True

            # Wire onmessage handler if the transport supports it
            try:
                self.transport.set_onmessage(self._on_transport_message)
            except (MCPTransportError, AttributeError) as e:
                logger.debug(f"Transport doesn't support onmessage handler: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error setting onmessage handler: {e}")

            # Send request
            result = self._run_async(self.transport.send(request_data))
            return result
        except (MCPTransportError, NetworkError, TimeoutError) as e:
            self.logger.error(f"MCP transport request failed: {e}")
            raise MCPError(f"Transport request failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in transport request: {e}")
            raise MCPError(f"Unexpected transport error: {e}")

    def initialize(self) -> dict[str, Any]:
        """Initialize MCP session."""
        params = {
            "clientInfo": {"name": "MCP Python Client", "version": "1.0.0"},
            "protocolVersion": self.protocol_version,
            "capabilities": self._build_client_capabilities(),
        }

        result = self._make_request("initialize", params)

        # Store session ID if provided
        if "sessionInfo" in result and "id" in result["sessionInfo"]:
            self.session_id = result["sessionInfo"]["id"]

        # Store capabilities and negotiated protocol version if provided
        if isinstance(result, dict):
            self.server_capabilities = result.get("capabilities") or result.get(
                "serverCapabilities"
            )
            self.negotiated_protocol_version = (
                result.get("protocolVersion") or self.protocol_version
            )
            try:
                self.transport.set_protocol_version(self.negotiated_protocol_version)
            except (MCPProtocolError, AttributeError) as e:
                logger.debug(f"Transport doesn't support protocol version setting: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error setting protocol version: {e}")

        # Send initialized notification as required by MCP protocol
        self._send_notification("initialized", {})

        # Give server a moment to process the notification
        import time

        time.sleep(0.5)  # Increased delay for persistent event loop

        self._set_connection_state(ConnectionState.CONNECTED)
        return result

    def _build_client_capabilities(self) -> dict[str, Any]:
        """Advertise client capabilities per MCP spec."""
        return {
            "tools": {"list": True, "call": True},
            "prompts": {"list": True, "get": True},
            "resources": {"list": True, "read": True, "subscribe": True},
            "roots": {"list": True},
            "logging": {"message": True},
            "sampling": {"createMessage": False},
            "notifications": True,
        }

    def get_server_capabilities(self) -> dict[str, Any] | None:
        return self.server_capabilities

    def get_negotiated_protocol_version(self) -> str | None:
        return self.negotiated_protocol_version

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            # No id for notifications
        }

        self.logger.info(f"Sending notification: {method} with payload: {notification}")

        async def _send_notification_async():
            try:
                self.logger.info("Trying transport.send_notification")
                await self.transport.send_notification(notification)
                self.logger.info("Successfully sent via transport.send_notification")
            except AttributeError as e:
                self.logger.info(f"send_notification not available: {e}, trying direct send")
                # Transport might not support notifications, try direct send
                if hasattr(self.transport, "process") and hasattr(self.transport.process, "stdin"):
                    # Direct send for stdio transport
                    message_json = json.dumps(notification) + "\n"
                    self.logger.info(f"Sending directly: {message_json.strip()}")
                    self.transport.process.stdin.write(message_json.encode("utf-8"))
                    await self.transport.process.stdin.drain()
                    self.logger.info("Successfully sent via direct write")
                else:
                    self.logger.error("No process stdin available for direct send")
            except Exception as e:
                self.logger.error(f"Exception sending notification: {e}")

        self._run_async(_send_notification_async())

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        result = self._make_request("tools/list", {})
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return result if isinstance(result, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool with arguments."""
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")

        params = {"name": name, "arguments": arguments}
        return self._make_request("tools/call", params)

    def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts."""
        result = self._make_request("prompts/list", {})
        if isinstance(result, dict) and "prompts" in result:
            return result["prompts"]
        return result if isinstance(result, list) else []

    def list_roots(self) -> list[dict[str, Any]]:
        """List available roots (roots/list)."""
        result = self._make_request("roots/list", {})
        if isinstance(result, dict) and "roots" in result:
            return result["roots"]
        return result if isinstance(result, list) else []

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get a prompt with arguments."""
        if not name or not isinstance(name, str):
            raise ValueError("Prompt name must be a non-empty string")

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self._make_request("prompts/get", params)

    def list_resources(self) -> list[dict[str, Any]]:
        """List available resources."""
        result = self._make_request("resources/list", {})
        if isinstance(result, dict) and "resources" in result:
            return result["resources"]
        return result if isinstance(result, list) else []

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        if not uri or not isinstance(uri, str):
            raise ValueError("URI must be a non-empty string")

        params = {"uri": uri}
        return self._make_request("resources/read", params)

    def subscribe_to_resource(self, uri: str, callback: Callable[[dict[str, Any]], None]) -> str:
        """
        Subscribe to resource change notifications.

        Args:
            uri: The URI of the resource to subscribe to
            callback: Function to call when resource changes

        Returns:
            Subscription ID for later unsubscription

        Raises:
            ValueError: If URI format is invalid or transport doesn't support subscriptions
        """
        if not uri or not isinstance(uri, str):
            raise ValueError("URI must be a non-empty string")

        # Check if transport supports bidirectional communication
        transport_type = getattr(self.transport, "__class__", None)
        if transport_type and hasattr(transport_type, "__name__"):
            transport_name = transport_type.__name__
            if transport_name == "HTTPTransport":
                raise ValueError(
                    "HTTP transport does not support resource subscriptions. Use SSE or WebSocket transport."
                )

        # Check if already subscribed to this URI
        if uri in self._resource_uri_subscriptions:
            existing_subscription_id = self._resource_uri_subscriptions[uri]
            # Update callback for existing subscription
            self._resource_subscriptions[existing_subscription_id] = callback
            return existing_subscription_id

        params = {"uri": uri}

        try:
            result = self._make_request("resources/subscribe", params)

            # Extract subscription ID from result
            subscription_id = result.get("subscriptionId") or str(uuid.uuid4())

            # Store subscription mapping
            self._resource_subscriptions[subscription_id] = callback
            self._resource_uri_subscriptions[uri] = subscription_id

            return subscription_id
        except Exception as e:
            logger.error(f"Failed to subscribe to resource {uri}: {e}")
            raise

    def unsubscribe_from_resource(self, uri: str) -> bool:
        """
        Unsubscribe from resource change notifications.

        Args:
            uri: The URI of the resource to unsubscribe from

        Returns:
            True if successfully unsubscribed, False if not subscribed

        Raises:
            ValueError: If URI format is invalid
        """
        if not uri or not isinstance(uri, str):
            raise ValueError("URI must be a non-empty string")

        # Check if subscribed to this URI
        if uri not in self._resource_uri_subscriptions:
            return False

        subscription_id = self._resource_uri_subscriptions[uri]
        params = {"subscriptionId": subscription_id}

        try:
            self._make_request("resources/unsubscribe", params)

            # Remove from tracking
            del self._resource_subscriptions[subscription_id]
            del self._resource_uri_subscriptions[uri]

            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from resource {uri}: {e}")
            raise

    def _handle_resource_notification(self, notification: dict[str, Any]):
        """
        Handle incoming resource change notifications.

        Args:
            notification: The notification message from the server
        """
        method = notification.get("method")
        if method == "notifications/resources/updated":
            params = notification.get("params", {})
            uri = params.get("uri")

            if uri and uri in self._resource_uri_subscriptions:
                subscription_id = self._resource_uri_subscriptions[uri]
                callback = self._resource_subscriptions.get(subscription_id)

                if callback:
                    try:
                        callback(params)
                    except Exception as e:
                        logger.error(f"Error in resource notification callback for {uri}: {e}")
        elif method == "notifications/resources/list_changed":
            # Handle resource list changes - call all callbacks
            params = notification.get("params", {})
            for callback in self._resource_subscriptions.values():
                try:
                    callback(params)
                except Exception as e:
                    logger.error(f"Error in resource list change callback: {e}")

    def register_notification_handler(self, method: str, handler: Callable):
        """Register handler for specific notification method."""
        self._notification_handlers[method].append(handler)

    def register_method_handler(
        self, method: str, handler: Callable[[dict[str, Any]], dict[str, Any]]
    ):
        """Register handler for server-initiated request methods."""
        self._method_handlers[method] = handler

    def unregister_notification_handler(self, method: str, handler: Callable):
        """Unregister handler for specific notification method."""
        if method in self._notification_handlers:
            try:
                self._notification_handlers[method].remove(handler)
                if not self._notification_handlers[method]:
                    del self._notification_handlers[method]
            except ValueError:
                pass  # Handler not found

    def _handle_notification(self, notification: dict[str, Any]):
        """Process incoming server notification."""
        method = notification.get("method")
        params = notification.get("params", {})

        if not method:
            logger.warning(f"Received notification without method: {notification}")
            return

        handlers = self._notification_handlers.get(method, [])
        if not handlers:
            logger.debug(f"No handlers registered for notification method: {method}")
            return

        for handler in handlers:
            try:
                handler(params)
            except Exception as e:
                logger.error(f"Notification handler error for method {method}: {e}")

    def on_connection_state_changed(self, callback: Callable[[ConnectionState], None]):
        """Register callback for connection state changes."""
        self._connection_state_handlers.append(callback)

    def on_connected(self, callback: Callable[[], None]):
        """Register callback for successful connection."""
        self._connected_handlers.append(callback)

    def _set_connection_state(self, state: ConnectionState):
        """Set connection state and notify handlers."""
        if self._connection_state != state:
            self._connection_state = state

            # Notify state change handlers
            for handler in self._connection_state_handlers:
                try:
                    handler(state)
                except Exception as e:
                    logger.error(f"Connection state handler error: {e}")

            # Notify connection handlers when connected
            if state == ConnectionState.CONNECTED:
                for handler in self._connected_handlers:
                    try:
                        handler()
                    except Exception as e:
                        logger.error(f"Connected handler error: {e}")

    def _on_transport_message(self, message: dict[str, Any]) -> None:
        """Dispatch incoming messages from transport (notifications or server-initiated requests)."""
        try:
            if "id" not in message:
                # Notification
                self._handle_notification(message)
                return

            # Potential server-initiated request
            method = message.get("method")
            if method:
                handler = self._method_handlers.get(method)
                if not handler:
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                else:
                    try:
                        params = message.get("params") or {}
                        result = handler(params)
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "result": result,
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "error": {
                                "code": -32603,
                                "message": "Internal error",
                                "data": str(e),
                            },
                        }
                # Send response via transport
                self._run_async(self.transport.send_response(response))
        except Exception as e:
            self.logger.error(f"Error in onmessage dispatcher: {e}")

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state

    def close(self) -> None:
        """Close client and cleanup resources."""
        # Clean up resource subscriptions
        for uri in list(self._resource_uri_subscriptions.keys()):
            try:
                self.unsubscribe_from_resource(uri)
            except Exception as e:
                self.logger.warning(f"Error unsubscribing from resource {uri} during close: {e}")

        try:
            self._run_async(self.transport.close())
        except Exception as e:
            self.logger.warning(f"Error closing transport: {e}")

        # Update connection state
        self._set_connection_state(ConnectionState.DISCONNECTED)

        # Stop background loop
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except (RuntimeError, AttributeError) as e:
                logger.debug(f"Error stopping event loop: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error stopping event loop: {e}")
        if self._loop_thread and self._loop_thread.is_alive():
            try:
                self._loop_thread.join(timeout=2.0)
            except (RuntimeError, OSError) as e:
                logger.debug(f"Error joining thread: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error joining thread: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncMCPClient:
    """
    Async version of the MCP client.

    Same factory methods but with async interface.
    """

    def __init__(self, transport: Transport, protocol_version: str = "2025-06-18"):
        """Private constructor - use factory methods instead."""
        self.transport = transport
        self.protocol_version = protocol_version
        self.capability_cache = CapabilityCache(ttl_seconds=3600)
        self.session_id = None
        self.server_capabilities: dict[str, Any] | None = None
        self.negotiated_protocol_version: str | None = None

        # Resource subscription management
        self._resource_subscriptions = {}
        self._resource_uri_subscriptions = {}

        # Bidirectional communication support
        self._notification_handlers: dict[str, list[Callable]] = defaultdict(list)
        self._method_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._connection_state = ConnectionState.DISCONNECTED

    @classmethod
    def stdio(cls, command: list[str], **kwargs) -> "AsyncMCPClient":
        """Create async client with stdio transport."""
        transport = STDIOTransport(command=command, **kwargs)
        return cls(transport=transport)

    @classmethod
    def http(cls, url: str, **kwargs) -> "AsyncMCPClient":
        """Create async client with HTTP transport."""
        transport = HTTPTransport(base_url=url, **kwargs)
        return cls(transport=transport)

    @classmethod
    def websocket(cls, url: str, **kwargs) -> "AsyncMCPClient":
        """Create async client with WebSocket transport."""
        try:
            from .transports.websocket import WebSocketTransport

            transport = WebSocketTransport(url=url, **kwargs)
            return cls(transport=transport)
        except ImportError:
            raise ImportError("WebSocket transport not available")

    async def initialize(self) -> dict[str, Any]:
        """Initialize MCP session asynchronously."""
        params = {
            "clientInfo": {"name": "MCP Python Client", "version": "1.0.0"},
            "protocolVersion": self.protocol_version,
            "capabilities": self._build_client_capabilities(),
        }

        # Ensure transport is connected
        if self.transport.state != ConnectionState.CONNECTED:
            await self.transport.start()
        # Wire onmessage handler if supported
        try:
            self.transport.set_onmessage(self._on_transport_message)
        except (MCPTransportError, AttributeError) as e:
            logger.debug(f"Transport doesn't support onmessage handler: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error setting onmessage handler: {e}")

        # Make request
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": params,
        }

        result = await self.transport.send(request_data)

        # Store capabilities and negotiated protocol version if provided
        if isinstance(result, dict):
            self.server_capabilities = result.get("capabilities") or result.get(
                "serverCapabilities"
            )
            self.negotiated_protocol_version = (
                result.get("protocolVersion") or self.protocol_version
            )
            try:
                self.transport.set_protocol_version(self.negotiated_protocol_version)
            except (MCPProtocolError, AttributeError) as e:
                logger.debug(f"Transport doesn't support protocol version setting: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error setting protocol version: {e}")

        # Send initialized notification as required by MCP protocol
        await self._send_notification("initialized", {})

        # Give server a moment to process the notification
        await asyncio.sleep(0.1)

        self._connection_state = ConnectionState.CONNECTED
        return result

    def _build_client_capabilities(self) -> dict[str, Any]:
        return {
            "tools": {"list": True, "call": True},
            "prompts": {"list": True, "get": True},
            "resources": {"list": True, "read": True, "subscribe": True},
            "roots": {"list": True},
            "logging": {"message": True},
            "sampling": {"createMessage": False},
            "notifications": True,
        }

    def register_notification_handler(self, method: str, handler: Callable):
        self._notification_handlers[method].append(handler)

    def register_method_handler(
        self, method: str, handler: Callable[[dict[str, Any]], dict[str, Any]]
    ):
        self._method_handlers[method] = handler

    def _on_transport_message(self, message: dict[str, Any]) -> None:
        """Schedule async dispatch for incoming transport messages."""
        try:
            asyncio.create_task(self._dispatch_incoming_message(message))
        except (RuntimeError, AttributeError) as e:
            # No event loop available - this is expected in some scenarios
            logger.debug(f"No event loop available for message dispatch: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in message dispatch: {e}")

    async def _dispatch_incoming_message(self, message: dict[str, Any]) -> None:
        if "id" not in message:
            # Notification
            method = message.get("method")
            params = message.get("params", {})
            for handler in self._notification_handlers.get(method, []):
                try:
                    handler(params)
                except (MCPError, RuntimeError) as e:
                    logger.debug(f"Error in notification handler: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error in notification handler: {e}")
            return

        method = message.get("method")
        if method:
            handler = self._method_handlers.get(method)
            if not handler:
                response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": "Method not found"},
                }
            else:
                try:
                    params = message.get("params") or {}
                    result = handler(params)
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": result,
                    }
                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32603,
                            "message": "Internal error",
                            "data": str(e),
                        },
                    }
            try:
                await self.transport.send_response(response)
            except (MCPTransportError, NetworkError) as e:
                logger.debug(f"Error sending response via transport: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error sending response: {e}")

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            # No id for notifications
        }

        try:
            await self.transport.send_notification(notification)
        except AttributeError:
            # Transport might not support notifications, try direct send
            if hasattr(self.transport, "process") and hasattr(self.transport.process, "stdin"):
                # Direct send for stdio transport
                message_json = json.dumps(notification) + "\n"
                self.transport.process.stdin.write(message_json.encode("utf-8"))
                await self.transport.process.stdin.drain()

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools asynchronously."""
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }

        result = await self.transport.send(request_data)
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return result if isinstance(result, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool asynchronously."""
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }

        return await self.transport.send(request_data)

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts asynchronously."""
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "prompts/list",
            "params": {},
        }

        result = await self.transport.send(request_data)
        return result if isinstance(result, list) else []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get a prompt asynchronously."""
        if not name or not isinstance(name, str):
            raise ValueError("Prompt name must be a non-empty string")

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "prompts/get",
            "params": params,
        }

        return await self.transport.send(request_data)

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available resources asynchronously."""
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "resources/list",
            "params": {},
        }

        result = await self.transport.send(request_data)
        return result if isinstance(result, list) else []

    async def list_roots(self) -> list[dict[str, Any]]:
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "roots/list",
            "params": {},
        }
        result = await self.transport.send(request_data)
        if isinstance(result, dict) and "roots" in result:
            return result["roots"]
        return result if isinstance(result, list) else []

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource asynchronously."""
        if not uri or not isinstance(uri, str):
            raise ValueError("URI must be a non-empty string")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "resources/read",
            "params": {"uri": uri},
        }

        return await self.transport.send(request_data)

    async def close(self) -> None:
        """Close client asynchronously."""
        if self.transport:
            await self.transport.close()
        self._connection_state = ConnectionState.DISCONNECTED

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
