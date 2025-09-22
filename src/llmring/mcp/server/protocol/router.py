"""
JSON-RPC router for handling MCP requests.
"""

import logging
from typing import Any, Callable, Dict, Optional

from llmring.mcp.server.protocol.handlers import ProtocolError
from llmring.mcp.server.protocol.json_rpc import JSONRPCError, JSONRPCRequest, JSONRPCResponse


class JSONRPCRouter:
    """
    Routes JSON-RPC requests to appropriate handlers.
    Provides clean separation of concerns for request handling.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the router.

        Args:
            logger: Optional logger for recording events
        """
        self.methods: Dict[str, Callable] = {}
        self.notifications: Dict[str, Callable] = {}
        self.logger = logger or logging.getLogger(__name__)
        self._session_states: Dict[str, bool] = {}

    def register(self, method: str, handler: Callable) -> None:
        """
        Register a handler for a specific JSON-RPC method.

        Args:
            method: The name of the JSON-RPC method
            handler: The handler function for the method
        """
        self.methods[method] = handler
        self.logger.debug(f"Registered handler for method: {method}")

    def unregister(self, method: str) -> bool:
        """
        Unregister a handler for a specific JSON-RPC method.

        Args:
            method: The name of the JSON-RPC method

        Returns:
            True if the method was unregistered, False if not found
        """
        if method in self.methods:
            del self.methods[method]
            self.logger.debug(f"Unregistered handler for method: {method}")
            return True
        return False

    def register_notification(self, method: str, handler: Callable) -> None:
        """
        Register a handler for a specific JSON-RPC notification.

        Args:
            method: The name of the JSON-RPC notification
            handler: The handler function for the notification
        """
        self.notifications[method] = handler
        self.logger.debug(f"Registered notification handler for method: {method}")

    def unregister_notification(self, method: str) -> bool:
        """
        Unregister a notification handler for a specific JSON-RPC method.

        Args:
            method: The name of the JSON-RPC notification

        Returns:
            True if the notification was unregistered, False if not found
        """
        if method in self.notifications:
            del self.notifications[method]
            self.logger.debug(f"Unregistered notification handler for method: {method}")
            return True
        return False

    async def handle_request(
        self, request: JSONRPCRequest, context: Any = None
    ) -> Optional[JSONRPCResponse]:
        """
        Route a JSON-RPC request to the appropriate handler.

        Args:
            request: The JSON-RPC request to handle
            context: Optional context object passed to handlers

        Returns:
            JSON-RPC response or None for notifications
        """
        # Handle notifications (no response expected)
        if request.id is None:
            return await self._handle_notification(request, context)

        # Check if method exists first (before initialization check)
        if request.method not in self.methods:
            self.logger.warning(f"Method not found: {request.method}")
            return JSONRPCResponse(
                id=request.id, error=JSONRPCError.method_not_found(request.method)
            )

        # Get session ID for tracking initialization state
        session_id = self._get_session_id(context)
        is_initialized = self._is_initialized(session_id, context)

        # Check initialization requirements
        if not is_initialized and request.method not in [
            "initialize",
            "ping",
            "shutdown",
        ]:
            self.logger.warning(f"Request before initialization: {request.method}")
            return JSONRPCResponse(
                id=request.id, error=JSONRPCError.not_initialized(request.method)
            )

        # Handle initialize method specially
        if request.method == "initialize":
            if is_initialized:
                self.logger.warning("Server already initialized")
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError.application_error("Server already initialized"),
                )
            # Mark as initialized after handling
            result = await self._handle_method(request, context)
            if result and not result.error:
                self._mark_initialized(session_id, context)
            return result

        # Handle regular methods
        return await self._handle_method(request, context)

    async def _handle_method(self, request: JSONRPCRequest, context: Any = None) -> JSONRPCResponse:
        """Handle a regular JSON-RPC method call."""
        # Method existence already checked in handle_request

        # Call the handler
        try:
            handler = self.methods[request.method]
            self.logger.debug(f"Handling method: {request.method}")
            result = await handler(request.params or {}, context)

            return JSONRPCResponse(id=request.id, result=result)
        except ProtocolError as e:
            # Handle protocol-specific errors with their error codes
            self.logger.warning(f"Protocol error in method {request.method}: {e.message}")
            error = {"code": e.code, "message": e.message}
            if e.data is not None:
                error["data"] = e.data
            return JSONRPCResponse(id=request.id, error=error)
        except ValueError as e:
            # Specific handling for validation errors
            self.logger.warning(f"Invalid parameters for method {request.method}: {str(e)}")
            return JSONRPCResponse(id=request.id, error=JSONRPCError.invalid_params(str(e)))
        except Exception as e:
            # Log the error for debugging
            self.logger.error(f"Error handling method {request.method}: {str(e)}", exc_info=True)
            return JSONRPCResponse(id=request.id, error=JSONRPCError.application_error(str(e)))

    async def _handle_notification(self, request: JSONRPCRequest, context: Any = None) -> None:
        """Handle a JSON-RPC notification (no response)."""
        # Check if notification handler exists
        if request.method not in self.notifications:
            self.logger.warning(f"Notification handler not found: {request.method}")
            return

        # Call the notification handler
        try:
            handler = self.notifications[request.method]
            self.logger.debug(f"Handling notification: {request.method}")
            await handler(request.params or {}, context)
        except Exception as e:
            # Log errors but don't respond for notifications
            self.logger.error(f"Error handling notification {request.method}: {str(e)}")

    async def handle_raw_request(
        self, raw_data: Dict[str, Any], context: Any = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle a raw JSON-RPC request dictionary.

        Args:
            raw_data: The raw JSON-RPC request data
            context: Optional context object passed to handlers

        Returns:
            JSON-RPC response as dictionary or None for notifications
        """
        try:
            request = JSONRPCRequest.from_dict(raw_data)
            response = await self.handle_request(request, context)
            return response.to_dict() if response else None
        except Exception as e:
            # Return error response for failures
            self.logger.error(f"Error processing raw request: {str(e)}")
            request_id = raw_data.get("id") if isinstance(raw_data, dict) else None
            error_response = JSONRPCResponse(
                id=request_id, error=JSONRPCError.internal_error(str(e))
            )
            return error_response.to_dict()

    def _get_session_id(self, context: Any) -> Optional[str]:
        """Extract session ID from context."""
        if context and hasattr(context, "request"):
            if hasattr(context.request, "headers"):
                return context.request.headers.get("x-mcp-session-id")
        return None

    def _is_initialized(self, session_id: Optional[str], context: Any) -> bool:
        """Check if the session is initialized."""
        if session_id:
            return self._session_states.get(session_id, False)
        elif context and hasattr(context, "mcp_initialized"):
            return context.mcp_initialized
        return False

    def _mark_initialized(self, session_id: Optional[str], context: Any) -> None:
        """Mark the session as initialized."""
        if session_id:
            self._session_states[session_id] = True
        elif context:
            context.mcp_initialized = True
