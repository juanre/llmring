"""
HTTP+SSE transport implementation for MCP (Legacy).

This transport implements the older HTTP+SSE specification which uses separate
endpoints for messages and SSE streams. It combines HTTP POST for client-to-server
communication with Server-Sent Events (SSE) for server-to-client communication.

Note: This is the legacy HTTP transport. For new implementations, use
StreamableHTTPTransport which implements the newer single-endpoint specification.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace

from llmring.mcp.server.transport.base import Transport, JSONRPCMessage

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SSEConnection:
    """Represents an active SSE connection."""

    session_id: str
    connection_id: str
    send_queue: asyncio.Queue
    last_event_id: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    async def send_event(
        self, event_type: str, data: Dict[str, Any], event_id: Optional[int] = None
    ) -> bool:
        """Send an SSE event to the client."""
        if not self.is_active:
            return False

        if event_id is not None:
            self.last_event_id = event_id
        else:
            self.last_event_id += 1
            event_id = self.last_event_id

        try:
            event_data = {"id": event_id, "event": event_type, "data": json.dumps(data)}
            await self.send_queue.put(event_data)
            return True
        except Exception as e:
            logger.error(f"Failed to send SSE event: {e}")
            self.is_active = False
            return False

    async def close(self):
        """Close the SSE connection."""
        self.is_active = False
        # Send a close signal to the queue
        try:
            await self.send_queue.put(None)
        except:
            pass


@dataclass
class Session:
    """Represents an MCP session with associated SSE connections."""

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    connections: Dict[str, SSEConnection] = field(default_factory=dict)
    last_event_id: int = 0

    def add_connection(self, connection: SSEConnection):
        """Add an SSE connection to this session."""
        self.connections[connection.connection_id] = connection

    def remove_connection(self, connection_id: str):
        """Remove an SSE connection from this session."""
        if connection_id in self.connections:
            del self.connections[connection_id]

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> int:
        """Broadcast an event to all active connections in this session."""
        self.last_event_id += 1
        event_id = self.last_event_id

        active_connections = []
        for connection in self.connections.values():
            if connection.is_active and await connection.send_event(
                event_type, data, event_id
            ):
                active_connections.append(connection)
            elif not connection.is_active:
                # Clean up inactive connections
                await connection.close()

        # Update connections to only include active ones
        self.connections = {conn.connection_id: conn for conn in active_connections}

        return len(active_connections)


class SessionManager:
    """Manages MCP sessions and their associated SSE connections."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Agent] = None

    def create_session(self) -> str:
        """Create a new MCP session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(session_id=session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session by ID."""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        """Remove a session and close all its connections."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Close all connections
            for connection in session.connections.values():
                asyncio.create_task(connection.close())
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")

    async def broadcast_to_session(
        self, session_id: str, event_type: str, data: Dict[str, Any]
    ) -> bool:
        """Broadcast an event to all connections in a specific session."""
        session = self.get_session(session_id)
        if not session:
            return False

        connection_count = await session.broadcast_event(event_type, data)
        logger.debug(
            f"Broadcasted {event_type} to {connection_count} connections in session {session_id}"
        )
        return connection_count > 0

    async def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up inactive sessions older than max_age_hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            # Remove sessions with no active connections that are old
            if not session.connections and session.created_at < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.remove_session(session_id)

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")


class HTTPTransport(Transport):
    """
    Legacy HTTP+SSE transport implementation for MCP.

    This implements the older HTTP+SSE specification which uses:
    - HTTP POST endpoint for client-to-server requests
    - Separate SSE endpoint for server-to-client messages
    - Session-based connection management

    Note: For new implementations, use StreamableHTTPTransport which provides
    a cleaner single-endpoint architecture with server-decided response modes.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        endpoint_path: str = "/mcp",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the HTTP transport.

        Args:
            session_manager: Manager for handling sessions and SSE connections
            endpoint_path: The HTTP endpoint path for MCP communication
            logger: Optional logger for transport-specific logging
        """
        super().__init__(logger=logger or logging.getLogger(__name__))
        self.session_manager = session_manager or SessionManager()
        self.endpoint_path = endpoint_path
        self._running = False
        self._current_session_id: Optional[str] = None

    async def start(self) -> bool:
        """Start the HTTP transport."""
        self._running = True
        logger.info(f"HTTP transport started on endpoint {self.endpoint_path}")
        return True

    async def stop(self) -> None:
        """Stop the HTTP transport."""
        self._running = False
        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            self.session_manager.remove_session(session_id)
        logger.info("HTTP transport stopped")

    async def send_message(self, message: JSONRPCMessage) -> bool:
        """
        Send a message to all active sessions.

        In HTTP transport, this broadcasts the message via SSE to all connected clients.

        Args:
            message: The JSON-RPC message to send

        Returns:
            True if the message was sent to at least one connection
        """
        if not self._running:
            return False

        # Broadcast to all sessions
        sent_count = 0
        for session_id in list(self.session_manager.sessions.keys()):
            if await self.session_manager.broadcast_to_session(
                session_id, "message", message
            ):
                sent_count += 1

        return sent_count > 0

    async def send_to_session(self, session_id: str, message: JSONRPCMessage) -> bool:
        """
        Send a message to a specific session.

        Args:
            session_id: The session to send to
            message: The JSON-RPC message to send

        Returns:
            True if the message was sent successfully
        """
        if not self._running:
            return False

        return await self.session_manager.broadcast_to_session(
            session_id, "message", message
        )

    async def handle_http_request(self, request: Any) -> Dict[str, Any]:
        """
        Handle an incoming HTTP POST request.

        This method should be called by the web framework integration.

        Args:
            request: The HTTP request object (framework-specific)

        Returns:
            Response data to send back
        """
        try:
            # Extract JSON-RPC message from request body
            body = await self._get_request_body(request)
            message = json.loads(body)

            # Get or create session
            session_id = self._get_session_id(request)
            if not session_id:
                session_id = self.session_manager.create_session()

            # Create context with request information
            context = SimpleNamespace(
                request=request,
                session_id=session_id,
                headers=self._get_request_headers(request),
                method="POST",
            )

            # Handle the message with context
            self._handle_message_with_context(message, context)

            # Return acknowledgment
            return {
                "status": "accepted",
                "session_id": session_id,
                "_headers": {"Mcp-Session-Id": session_id},  # Signal to framework
            }

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            return {
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e),
                }
            }
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            return {
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e),
                }
            }

    async def handle_sse_connection(self, request: Any) -> AsyncIterator[str]:
        """
        Handle a Server-Sent Events connection.

        This method should be called by the web framework for GET requests
        to establish an SSE stream.

        Args:
            request: The HTTP request object (framework-specific)

        Yields:
            SSE formatted events
        """
        # Get session ID from request
        session_id = self._get_session_id(request)
        if not session_id:
            yield 'event: error\ndata: {"error": "No session ID provided"}\n\n'
            return

        # Get session
        session = self.session_manager.get_session(session_id)
        if not session:
            yield 'event: error\ndata: {"error": "Invalid session ID"}\n\n'
            return

        # Create connection
        connection_id = str(uuid.uuid4())
        send_queue: asyncio.Queue = asyncio.Queue()
        connection = SSEConnection(
            session_id=session_id,
            connection_id=connection_id,
            send_queue=send_queue,
        )

        # Add to session
        session.add_connection(connection)
        logger.info(
            f"SSE connection {connection_id} established for session {session_id}"
        )

        try:
            # Send initial connection event
            yield 'event: connected\ndata: {"session_id": "' + session_id + '"}\n\n'

            # Stream events
            while connection.is_active:
                try:
                    # Wait for events with timeout
                    event = await asyncio.wait_for(send_queue.get(), timeout=30.0)

                    if event is None:
                        # Close signal
                        break

                    # Format as SSE
                    event_id = event.get("id", "")
                    event_type = event.get("event", "message")
                    event_data = event.get("data", "{}")

                    yield f"id: {event_id}\n"
                    yield f"event: {event_type}\n"
                    yield f"data: {event_data}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ":keepalive\n\n"

        except asyncio.CancelledError:
            logger.info(f"SSE connection {connection_id} cancelled")
        except Exception as e:
            logger.error(f"Error in SSE connection: {e}")
        finally:
            # Clean up
            connection.is_active = False
            session.remove_connection(connection_id)
            logger.info(f"SSE connection {connection_id} closed")

    # Framework-specific methods to be overridden

    async def _get_request_body(self, request: Any) -> str:
        """Get request body as string. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_request_body")

    def _get_request_headers(self, request: Any) -> Dict[str, str]:
        """Get request headers. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_request_headers")

    def _get_session_id(self, request: Any) -> Optional[str]:
        """Extract session ID from request. Override for specific framework."""
        raise NotImplementedError("Subclass must implement _get_session_id")
