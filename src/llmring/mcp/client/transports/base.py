"""
Abstract base transport interface for MCP client.

Defines the contract that all transport implementations must follow,
enabling pluggable transport layers for different communication protocols.
"""

import abc
from collections.abc import Callable
from enum import Enum
from typing import Any


class ConnectionState(Enum):
    """Connection state enumeration for transport lifecycle."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class Transport(abc.ABC):
    """
    Abstract base class for MCP transport implementations.

    This class defines the interface that all transport implementations
    must follow, enabling the MCP client to work with different communication
    protocols through a unified interface.
    """

    def __init__(self):
        """Initialize the transport."""
        self._state = ConnectionState.DISCONNECTED
        self._onclose: Callable[[], None] | None = None
        self._onerror: Callable[[Exception], None] | None = None
        self._onmessage: Callable[[dict[str, Any]], None] | None = None

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @abc.abstractmethod
    async def start(self) -> None:
        """
        Initialize the transport connection.

        This method should establish the connection and set the state
        to CONNECTED when successful, or ERROR if it fails.
        """
        pass

    @abc.abstractmethod
    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send JSON-RPC message and return response.

        Args:
            message: JSON-RPC message dictionary with keys:
                - jsonrpc: "2.0"
                - id: unique request identifier
                - method: method name
                - params: method parameters

        Returns:
            JSON-RPC response dictionary with keys:
                - jsonrpc: "2.0"
                - id: same as request id
                - result: method result (on success)
                - error: error details (on failure)

        Raises:
            Exception: If transport is not connected or send fails
        """
        pass

    async def send_notification(self, message: dict[str, Any]) -> None:
        """
        Send JSON-RPC notification (no response expected).

        This is optional for transports that don't support notifications.
        Default implementation does nothing.

        Args:
            message: JSON-RPC notification dictionary with keys:
                - jsonrpc: "2.0"
                - method: method name
                - params: method parameters
                - NO id field
        """
        # Default implementation - do nothing
        # Transports that support notifications should override this
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Clean up transport resources.

        This method should gracefully close the connection and
        set the state to DISCONNECTED.
        """
        pass

    # New in MCP 2025-06-18 compliance
    def set_protocol_version(self, version: str) -> None:
        """
        Set negotiated MCP protocol version for transports that need to
        include it in headers (HTTP-based transports).

        Default implementation is a no-op for transports that don't use headers.
        """
        # Optional override in HTTP/SSE transports
        return None

    async def send_response(self, message: dict[str, Any]) -> None:
        """
        Send a JSON-RPC response for server-initiated requests.

        Default implementation raises NotImplementedError to preserve
        compatibility with tests that mock transports without this method.
        """
        raise NotImplementedError("Transport does not support server-initiated responses")

    def set_onclose(self, callback: Callable[[], None] | None) -> None:
        """
        Set callback for when transport disconnects unexpectedly.

        Args:
            callback: Function to call when connection closes unexpectedly
        """
        self._onclose = callback

    def set_onerror(self, callback: Callable[[Exception], None] | None) -> None:
        """
        Set callback for when transport encounters errors.

        Args:
            callback: Function to call when errors occur, receives exception
        """
        self._onerror = callback

    def set_onmessage(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """
        Set callback for incoming server notifications (bidirectional transports).

        Args:
            callback: Function to call when server sends notifications
        """
        self._onmessage = callback

    def _set_state(self, state: ConnectionState) -> None:
        """Internal method to update connection state."""
        self._state = state

    def _handle_close(self) -> None:
        """Internal method to handle unexpected connection closure."""
        self._state = ConnectionState.DISCONNECTED
        if self._onclose:
            try:
                self._onclose()
            except Exception:
                pass  # Don't let callback errors break transport

    def _handle_error(self, error: Exception) -> None:
        """Internal method to handle transport errors."""
        self._state = ConnectionState.ERROR
        if self._onerror:
            try:
                self._onerror(error)
            except Exception:
                pass  # Don't let callback errors break transport

    def _handle_message(self, message: dict[str, Any]) -> None:
        """Internal method to handle incoming server messages."""
        if self._onmessage:
            try:
                self._onmessage(message)
            except Exception:
                pass  # Don't let callback errors break transport
