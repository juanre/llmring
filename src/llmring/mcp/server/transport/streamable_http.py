"""
Streamable HTTP transport implementation for MCP.

This transport implements the Streamable HTTP specification which uses a single
endpoint for all communication with server-decided response modes.

Key features:
- Single endpoint architecture
- Server decides between JSON or SSE response modes
- Supports session management (optional)
- Batch request support
- Event resumption via Last-Event-ID
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, AsyncIterator, Callable, Union
from dataclasses import dataclass, field
from types import SimpleNamespace
from enum import Enum

from llmring.mcp.server.transport.base import Transport, JSONRPCMessage

logger = logging.getLogger(__name__)


class ResponseMode(Enum):
    """Server response modes for Streamable HTTP."""

    JSON = "json"
    SSE = "sse"
    ACCEPTED = "accepted"


@dataclass
class EventHistoryEntry:
    """Entry in the event history for resumption support."""

    event_id: int
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime


@dataclass
class Session:
    """Represents an MCP session."""

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    event_counter: int = 0
    event_history: List[EventHistoryEntry] = field(default_factory=list)
    max_history_size: int = 1000

    def add_event(self, event_type: str, data: Dict[str, Any]) -> int:
        """Add an event to history and return its ID."""
        self.event_counter += 1
        event_id = self.event_counter

        entry = EventHistoryEntry(
            event_id=event_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
        )

        self.event_history.append(entry)

        # Trim history if too large
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size :]

        self.last_activity = datetime.now()
        return event_id

    def get_events_after(self, event_id: int) -> List[EventHistoryEntry]:
        """Get all events after the specified event ID."""
        return [e for e in self.event_history if e.event_id > event_id]


class StreamableHTTPTransport(Transport):
    """
    Streamable HTTP transport implementation for MCP.

    This transport implements the MCP Streamable HTTP specification with:
    - Single endpoint for all communication
    - Server-decided response modes (JSON vs SSE)
    - Optional session management
    - Batch request support
    - Event resumption capability
    """

    def __init__(
        self,
        endpoint_path: str = "/mcp",
        enable_sessions: bool = True,
        session_timeout_hours: int = 24,
        streaming_operations: Optional[Set[str]] = None,
        response_mode_decider: Optional[
            Callable[[Dict[str, Any]], ResponseMode]
        ] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Streamable HTTP transport.

        Args:
            endpoint_path: The HTTP endpoint path for MCP communication
            enable_sessions: Whether to enable session management
            session_timeout_hours: Hours before inactive sessions expire
            streaming_operations: Set of method names that should use streaming
            response_mode_decider: Custom function to decide response mode
            logger: Optional logger for transport-specific logging
        """
        super().__init__(logger=logger or logging.getLogger(__name__))
        self.endpoint_path = endpoint_path
        self.enable_sessions = enable_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)

        # Session management
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Agent] = None

        # Default streaming operations
        self.streaming_operations = streaming_operations or {
            "sampling/createMessage",
            "tools/call",
            "resources/read",
            "completion/complete",
        }

        # Response mode decision function
        self.response_mode_decider = (
            response_mode_decider or self._default_response_mode_decider
        )

        self._running = False

    async def start(self) -> bool:
        """Start the HTTP transport."""
        self._running = True

        # Start session cleanup agent if sessions are enabled
        if self.enable_sessions and not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions_loop())

        logger.info(
            f"Streamable HTTP transport started on endpoint {self.endpoint_path}"
        )
        return True

    async def stop(self) -> None:
        """Stop the HTTP transport."""
        self._running = False

        # Cancel cleanup agent
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear sessions
        self.sessions.clear()
        logger.info("Streamable HTTP transport stopped")

    async def send_message(self, message: JSONRPCMessage) -> bool:
        """
        Send a message via the transport.

        Note: In Streamable HTTP, this is used for server-initiated messages
        which would be queued for delivery via SSE streams.

        Args:
            message: The JSON-RPC message to send

        Returns:
            True if the message was queued for delivery
        """
        # This would typically queue the message for delivery via active SSE streams
        # Implementation depends on integration with the web framework
        logger.debug(f"Queueing message for SSE delivery: {message}")
        return True

    def _default_response_mode_decider(self, message: Dict[str, Any]) -> ResponseMode:
        """
        Default logic for deciding response mode based on message characteristics.

        Args:
            message: The JSON-RPC message

        Returns:
            The response mode to use
        """
        method = message.get("method", "")

        # Notifications always return 202 Accepted
        if "id" not in message:
            return ResponseMode.ACCEPTED

        # Check if it's a streaming operation
        if method in self.streaming_operations:
            return ResponseMode.SSE

        # Check for specific parameter patterns that suggest streaming
        params = message.get("params", {})

        # Large resource reads should stream
        if method == "resources/read":
            uri = params.get("uri", "")
            # Simple heuristic: if reading a file, check extension
            if any(uri.endswith(ext) for ext in [".log", ".csv", ".json", ".xml"]):
                return ResponseMode.SSE

        # Tool calls with certain patterns should stream
        if method == "tools/call":
            tool_name = params.get("name", "")
            # Tools with these patterns often benefit from streaming
            if any(
                pattern in tool_name.lower()
                for pattern in ["analyze", "process", "generate", "search"]
            ):
                return ResponseMode.SSE

        # Default to JSON for simple operations
        return ResponseMode.JSON

    async def handle_request(
        self, request: Any
    ) -> Union[Dict[str, Any], AsyncIterator[str], None]:
        """
        Handle an incoming HTTP request.

        This is the main entry point for the web framework integration.

        Args:
            request: The HTTP request object (framework-specific)

        Returns:
            Response data (dict for JSON, async iterator for SSE, None for 202)
        """
        method = self._get_request_method(request)

        if method == "POST":
            return await self._handle_post_request(request)
        elif method == "GET":
            return self._handle_get_request(request)
        elif method == "DELETE":
            return await self._handle_delete_request(request)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    async def _handle_post_request(
        self, request: Any
    ) -> Union[Dict[str, Any], AsyncIterator[str], None]:
        """Handle POST request containing JSON-RPC message(s)."""
        try:
            # Extract and parse body
            body = await self._get_request_body(request)
            data = json.loads(body)

            # Handle batch vs single request
            if isinstance(data, list):
                return await self._handle_batch_request(request, data)
            else:
                return await self._handle_single_request(request, data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error", "data": str(e)},
            }
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
            }

    async def _handle_single_request(
        self, request: Any, message: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncIterator[str], None]:
        """Handle a single JSON-RPC request."""
        # Get or create session if enabled
        session = None
        if self.enable_sessions:
            session_id = self._get_session_id(request)
            if not session_id:
                session_id = self._create_session_id()
                self._set_response_session_id(request, session_id)

            session = self._get_or_create_session(session_id)

        # Create context
        context = self._create_request_context(request, session)

        # Decide response mode
        response_mode = self.response_mode_decider(message)

        if response_mode == ResponseMode.ACCEPTED:
            # Handle notification
            self._handle_message_with_context(message, context)
            return None  # Will be converted to 202 Accepted by framework

        elif response_mode == ResponseMode.JSON:
            # Handle with immediate JSON response
            # Store the response for retrieval
            response_container = {"response": None}

            def capture_response(response_msg: JSONRPCMessage, ctx: Any = None):
                response_container["response"] = response_msg

            # Temporarily override the message callback
            original_callback = self._message_callback
            self._message_callback = capture_response

            try:
                # Process the message
                self._handle_message_with_context(message, context)

                # Wait briefly for response
                await asyncio.sleep(0.1)  # Allow processing

                if response_container["response"]:
                    return response_container["response"]
                else:
                    # No response yet, return error
                    return {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32603,
                            "message": "No response from handler",
                        },
                    }
            finally:
                self._message_callback = original_callback

        else:  # ResponseMode.SSE
            # Handle with SSE streaming
            return self._create_sse_stream(message, context, session)

    async def _handle_batch_request(
        self, request: Any, messages: List[Dict[str, Any]]
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        """Handle a batch of JSON-RPC requests."""
        # Check if any message needs streaming
        needs_streaming = any(
            self.response_mode_decider(msg) == ResponseMode.SSE for msg in messages
        )

        if needs_streaming:
            # If any message needs streaming, use SSE for all
            session = None
            if self.enable_sessions:
                session_id = self._get_session_id(request) or self._create_session_id()
                self._set_response_session_id(request, session_id)
                session = self._get_or_create_session(session_id)

            context = self._create_request_context(request, session)
            return self._create_batch_sse_stream(messages, context, session)
        else:
            # All messages can be handled with JSON
            responses = []

            for message in messages:
                if "id" in message:  # Skip notifications
                    response = await self._handle_single_request(request, message)
                    if response:
                        responses.append(response)

            return responses

    def _handle_get_request(self, request: Any) -> AsyncIterator[str]:
        """Handle GET request for establishing SSE stream."""
        if not self.enable_sessions:
            # Sessions required for GET requests
            raise ValueError("GET requests require session management to be enabled")

        session_id = self._get_session_id(request)
        if not session_id:
            raise ValueError("Session ID required for GET requests")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Invalid session ID")

        # Get Last-Event-ID for resumption
        last_event_id = self._get_last_event_id(request)

        return self._create_notification_stream(session, last_event_id)

    async def _handle_delete_request(self, request: Any) -> None:
        """Handle DELETE request for session termination."""
        if not self.enable_sessions:
            raise ValueError("DELETE requests require session management to be enabled")

        session_id = self._get_session_id(request)
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Terminated session: {session_id}")

    def _create_sse_stream(
        self, message: Dict[str, Any], context: Any, session: Optional[Session]
    ) -> AsyncIterator[str]:
        """Create an SSE stream for a single message response."""

        async def stream():
            # Process the message and collect responses
            response_queue = asyncio.Queue()

            async def queue_response(response_msg: JSONRPCMessage, ctx: Any = None):
                await response_queue.put(response_msg)

            # Temporarily override callback
            original_callback = self._message_callback
            self._message_callback = queue_response

            try:
                # Start processing
                self._handle_message_with_context(message, context)

                # Stream responses
                while True:
                    try:
                        response = await asyncio.wait_for(
                            response_queue.get(), timeout=30.0
                        )

                        # Add to session history if enabled
                        event_id = None
                        if session:
                            event_id = session.add_event("message", response)

                        # Format as SSE
                        yield self._format_sse_event("message", response, event_id)

                        # Check if this was the final response for the request
                        if response.get("id") == message.get("id"):
                            break

                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield ":keepalive\n\n"

            finally:
                self._message_callback = original_callback

        return stream()

    def _create_batch_sse_stream(
        self, messages: List[Dict[str, Any]], context: Any, session: Optional[Session]
    ) -> AsyncIterator[str]:
        """Create an SSE stream for batch message responses."""

        async def stream():
            for message in messages:
                # Process each message in the batch
                async for event in self._create_sse_stream(message, context, session):
                    yield event

        return stream()

    def _create_notification_stream(
        self, session: Session, last_event_id: Optional[int]
    ) -> AsyncIterator[str]:
        """Create an SSE stream for server-initiated notifications."""

        async def stream():
            # First, replay any missed events
            if last_event_id is not None:
                missed_events = session.get_events_after(last_event_id)
                for event in missed_events:
                    yield self._format_sse_event(
                        event.event_type, event.data, event.event_id
                    )

            # Then stream new events (implementation depends on framework)
            # This is a placeholder that sends periodic pings
            while True:
                try:
                    await asyncio.sleep(30)
                    event_id = session.add_event("ping", {})
                    yield self._format_sse_event("ping", {}, event_id)
                except asyncio.CancelledError:
                    break

        return stream()

    def _format_sse_event(
        self, event_type: str, data: Dict[str, Any], event_id: Optional[int] = None
    ) -> str:
        """Format data as an SSE event."""
        lines = []

        if event_id is not None:
            lines.append(f"id: {event_id}")

        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")  # Empty line to end event

        return "\n".join(lines) + "\n"

    def _get_or_create_session(self, session_id: str) -> Session:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)

        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        return session

    def _create_session_id(self) -> str:
        """Create a new session ID."""
        return str(uuid.uuid4())

    def _create_request_context(self, request: Any, session: Optional[Session]) -> Any:
        """Create context object for request handling."""
        return SimpleNamespace(
            request=request,
            session_id=session.session_id if session else None,
            headers=self._get_request_headers(request),
            method=self._get_request_method(request),
            transport="streamable_http",
        )

    async def _cleanup_sessions_loop(self):
        """Background agent to clean up inactive sessions."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check hourly
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_inactive_sessions(self):
        """Remove sessions that have been inactive too long."""
        now = datetime.now()
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            if now - session.last_activity > self.session_timeout:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

    # Framework-specific methods to be overridden

    def _get_request_method(self, request: Any) -> str:
        """Get HTTP method. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_request_method")

    async def _get_request_body(self, request: Any) -> str:
        """Get request body as string. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_request_body")

    def _get_request_headers(self, request: Any) -> Dict[str, str]:
        """Get request headers. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_request_headers")

    def _get_session_id(self, request: Any) -> Optional[str]:
        """Extract session ID from Mcp-Session-Id header."""
        headers = self._get_request_headers(request)
        return headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")

    def _get_last_event_id(self, request: Any) -> Optional[int]:
        """Extract Last-Event-ID from request."""
        headers = self._get_request_headers(request)
        last_event_id = headers.get("last-event-id") or headers.get("Last-Event-ID")
        if last_event_id:
            try:
                return int(last_event_id)
            except ValueError:
                pass
        return None

    def _set_response_session_id(self, request: Any, session_id: str):
        """Set session ID in response headers. Override for specific framework."""
        # This is typically handled by the framework integration
        pass
