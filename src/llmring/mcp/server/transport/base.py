"""
Base transport interface for MCP communication.
Defines the common interface that all transport implementations must follow.
"""

import abc
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union

# Type for JSON-RPC messages
JSONRPCMessage = Dict[str, Any]

# Type for message callbacks
MessageCallback = Union[
    Callable[[JSONRPCMessage], None],  # Legacy: message only
    Callable[[JSONRPCMessage, Any], None],  # New: message with context
]


class Transport(abc.ABC):
    """
    Abstract base class for all MCP transport implementations.

    A transport handles the underlying mechanics of how messages
    are sent and received between MCP client and server.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the transport.

        Args:
            logger: Optional logger for transport-specific logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self._message_callback: Optional[MessageCallback] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._close_callback: Optional[Callable[[], None]] = None

    @abc.abstractmethod
    async def start(self) -> bool:
        """
        Start the transport. This method should initialize the
        transport and make it ready to send and receive messages.

        Returns:
            True if the transport was successfully started, False otherwise
        """
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """
        Stop the transport. This method should clean up any resources
        and ensure the transport can no longer send or receive messages.
        """
        pass

    @abc.abstractmethod
    async def send_message(self, message: JSONRPCMessage) -> bool:
        """
        Send a JSON-RPC message through the transport.

        Args:
            message: The JSON-RPC message to send

        Returns:
            True if the message was successfully sent, False otherwise
        """
        pass

    def set_message_callback(self, callback: MessageCallback) -> None:
        """
        Set the callback to be called when a message is received.

        The callback can accept either:
        - Just the message (legacy)
        - Message and context (new)

        Args:
            callback: The callback function
        """
        self._message_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Set the callback to be called when an error occurs.

        Args:
            callback: The callback function
        """
        self._error_callback = callback

    def set_close_callback(self, callback: Callable[[], None]) -> None:
        """
        Set the callback to be called when the transport is closed.

        Args:
            callback: The callback function
        """
        self._close_callback = callback

    def _handle_message(self, message: JSONRPCMessage) -> None:
        """
        Handle a received message by calling the message callback if set.

        This method is for backward compatibility. New code should use
        _handle_message_with_context.

        Args:
            message: The received JSON-RPC message
        """
        self._handle_message_with_context(message, None)

    def _handle_message_with_context(self, message: JSONRPCMessage, context: Any = None) -> None:
        """
        Handle a received message with optional context.

        This method checks the callback signature and calls it appropriately:
        - If callback accepts 2 args, passes message and context
        - If callback accepts 1 arg, passes only message (for backward compatibility)

        Args:
            message: The received JSON-RPC message
            context: Optional context object (e.g., HTTP request, session info)
        """
        if self._message_callback:
            try:
                # Check callback signature
                sig = inspect.signature(self._message_callback)
                params = list(sig.parameters.values())

                # Call with appropriate arguments
                if len(params) >= 2:
                    # New style callback that accepts context
                    self._message_callback(message, context)
                else:
                    # Legacy callback without context
                    self._message_callback(message)
            except Exception as e:
                self.logger.exception(f"Error in message callback: {str(e)}")
                self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle an error by calling the error callback if set.

        Args:
            error: The exception that occurred
        """
        if self._error_callback:
            try:
                self._error_callback(error)
            except Exception as e:
                self.logger.exception(f"Error in error callback: {str(e)}")

    def _handle_close(self) -> None:
        """
        Handle transport closure by calling the close callback if set.
        """
        if self._close_callback:
            try:
                self._close_callback()
            except Exception as e:
                self.logger.exception(f"Error in close callback: {str(e)}")
