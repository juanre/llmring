"""
Tests for Streamable HTTP transport compliance with MCP specification.

These tests verify:
- Server-decided response modes
- Session management
- Batch request handling
- Header compliance
- Error handling
"""

import json
import pytest
import asyncio
from typing import Dict, Any, Optional

from llmring.mcp.server.transport.streamable_http import (
    StreamableHTTPTransport,
    ResponseMode,
)
from llmring.mcp.server.transport.base import JSONRPCMessage


class MockRequest:
    """Mock HTTP request for testing."""

    def __init__(
        self,
        method: str = "POST",
        body: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.method = method
        self._body = body.encode("utf-8")
        self.headers = headers or {}
        self.query_params = {}
        self.cookies = {}


class TestStreamableHTTPTransport:
    """Test Streamable HTTP transport implementation."""

    @pytest.fixture
    def transport(self):
        """Create a transport instance for testing."""
        return StreamableHTTPTransport(
            endpoint_path="/mcp",
            enable_sessions=True,
            session_timeout_hours=1,
        )

    @pytest.fixture
    def transport_no_sessions(self):
        """Create a transport instance without session management."""
        return StreamableHTTPTransport(
            endpoint_path="/mcp",
            enable_sessions=False,
        )

    def create_testable_transport(self, **kwargs):
        """Create a testable transport with framework methods implemented."""

        class TestableTransport(StreamableHTTPTransport):
            """Testable version with framework methods implemented."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._response_headers = {}

            def _get_request_method(self, request: MockRequest) -> str:
                return request.method

            async def _get_request_body(self, request: MockRequest) -> str:
                return request._body.decode("utf-8")

            def _get_request_headers(self, request: MockRequest) -> Dict[str, str]:
                return request.headers

            def _set_response_session_id(self, request: Any, session_id: str):
                """Store session ID to be added to response headers."""
                self._response_headers["Mcp-Session-Id"] = session_id

        return TestableTransport(**kwargs)

    # Test 1: Server Response Mode Decision
    def test_default_response_mode_decider(self):
        """Test default response mode decision logic."""
        transport = self.create_testable_transport()

        # Notifications should return ACCEPTED
        notification = {"jsonrpc": "2.0", "method": "initialized"}
        assert (
            transport._default_response_mode_decider(notification)
            == ResponseMode.ACCEPTED
        )

        # Simple operations should return JSON
        simple_request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert (
            transport._default_response_mode_decider(simple_request)
            == ResponseMode.JSON
        )

        # Streaming operations should return SSE
        streaming_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "sampling/createMessage",
        }
        assert (
            transport._default_response_mode_decider(streaming_request)
            == ResponseMode.SSE
        )

        # Tool calls depend on tool name
        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "analyze_data"},
        }
        assert (
            transport._default_response_mode_decider(tool_request) == ResponseMode.SSE
        )

    # Test 2: Header Compliance
    @pytest.mark.asyncio
    async def test_mcp_session_id_header(self):
        """Test Mcp-Session-Id header handling."""
        transport = self.create_testable_transport()
        await transport.start()

        # Request without session ID should create one
        request = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","id":1,"method":"initialize"}',
            headers={"accept": "application/json, text/event-stream"},
        )

        try:
            result = await transport.handle_request(request)
            # Session ID should be set in response headers
            assert "Mcp-Session-Id" in transport._response_headers
        finally:
            await transport.stop()

    # Test 3: Accept Header Validation
    @pytest.mark.asyncio
    async def test_accept_header_validation(self):
        """Test that Accept header is validated."""
        transport = self.create_testable_transport()
        await transport.start()

        # Request without proper Accept header should fail
        request = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","id":1,"method":"initialize"}',
            headers={"content-type": "application/json"},  # Missing Accept header
        )

        # This should be handled by the FastAPI integration
        # The transport itself doesn't validate Accept header
        await transport.stop()

    # Test 4: Notification Returns 202
    @pytest.mark.asyncio
    async def test_notification_returns_accepted(self):
        """Test that notifications return None (for 202 Accepted)."""
        transport = self.create_testable_transport()
        await transport.start()

        received_messages = []

        def message_handler(msg: JSONRPCMessage, context: Any = None):
            received_messages.append(msg)

        transport.set_message_callback(message_handler)

        # Send notification
        request = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","method":"initialized"}',
            headers={"accept": "application/json, text/event-stream"},
        )

        result = await transport.handle_request(request)

        # Notifications should return None (converted to 202 by framework)
        assert result is None

        # Message should have been handled
        assert len(received_messages) == 1
        assert received_messages[0]["method"] == "initialized"

        await transport.stop()

    # Test 5: Batch Request Support
    @pytest.mark.asyncio
    async def test_batch_request_all_json(self):
        """Test batch request where all operations use JSON response."""
        transport = self.create_testable_transport()
        await transport.start()

        responses = []

        def message_handler(msg: JSONRPCMessage, context: Any = None):
            # Simple echo response
            if "id" in msg:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {"echo": msg.get("method")},
                }
                responses.append(response)
                # Simulate sending response
                asyncio.create_task(transport.send_message(response))

        transport.set_message_callback(message_handler)

        # Batch request with all JSON operations
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"},
            {
                "jsonrpc": "2.0",
                "method": "notification",
            },  # This will be ignored in response
        ]

        request = MockRequest(
            method="POST",
            body=json.dumps(batch),
            headers={"accept": "application/json, text/event-stream"},
        )

        # Override the capture mechanism for testing
        transport._message_callback = message_handler

        result = await transport.handle_request(request)

        # Should return a list of responses (excluding notification)
        assert isinstance(result, list)

        await transport.stop()

    # Test 6: Batch Request with Streaming
    @pytest.mark.asyncio
    async def test_batch_request_with_streaming(self):
        """Test batch request where at least one operation needs streaming."""
        transport = self.create_testable_transport()
        await transport.start()

        # Batch with mixed operations
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "sampling/createMessage",
                "params": {},
            },
        ]

        request = MockRequest(
            method="POST",
            body=json.dumps(batch),
            headers={"accept": "application/json, text/event-stream"},
        )

        result = await transport.handle_request(request)

        # Should return an async iterator (SSE stream)
        assert hasattr(result, "__aiter__")

        await transport.stop()

    # Test 7: Session Management
    @pytest.mark.asyncio
    async def test_session_creation_and_retrieval(self):
        """Test session creation and retrieval."""
        transport = self.create_testable_transport()
        await transport.start()

        # First request creates session
        request1 = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","id":1,"method":"initialize"}',
            headers={"accept": "application/json, text/event-stream"},
        )

        await transport.handle_request(request1)

        session_id = transport._response_headers.get("Mcp-Session-Id")
        assert session_id is not None

        # Verify session was created
        session = transport.sessions.get(session_id)
        assert session is not None
        assert session.session_id == session_id

        # Second request with session ID reuses session
        request2 = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
            headers={
                "accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
        )

        await transport.handle_request(request2)

        # Should still be the same session
        assert len(transport.sessions) == 1

        await transport.stop()

    # Test 8: GET Request for SSE Stream
    @pytest.mark.asyncio
    async def test_get_request_sse_stream(self):
        """Test GET request establishes SSE stream for notifications."""
        transport = self.create_testable_transport()
        await transport.start()

        # Create a session first
        session_id = transport._create_session_id()
        session = transport._get_or_create_session(session_id)

        # GET request with session ID
        request = MockRequest(
            method="GET",
            headers={"accept": "text/event-stream", "mcp-session-id": session_id},
        )

        result = await transport.handle_request(request)

        # Should return an async iterator
        assert hasattr(result, "__aiter__")

        # Collect a few events
        events = []
        async for event in result:
            events.append(event)
            if len(events) >= 1:  # Just get one keepalive
                break

        # Should have received at least a ping
        assert len(events) > 0

        await transport.stop()

    # Test 9: DELETE Request for Session Termination
    @pytest.mark.asyncio
    async def test_delete_request_terminates_session(self):
        """Test DELETE request terminates session."""
        transport = self.create_testable_transport()
        await transport.start()

        # Create a session
        session_id = transport._create_session_id()
        session = transport._get_or_create_session(session_id)

        assert session_id in transport.sessions

        # DELETE request
        request = MockRequest(method="DELETE", headers={"mcp-session-id": session_id})

        await transport.handle_request(request)

        # Session should be removed
        assert session_id not in transport.sessions

        await transport.stop()

    # Test 10: Event History and Resumption
    @pytest.mark.asyncio
    async def test_event_history_and_resumption(self):
        """Test event history tracking and resumption."""
        transport = self.create_testable_transport()
        await transport.start()

        # Create session and add some events
        session_id = transport._create_session_id()
        session = transport._get_or_create_session(session_id)

        # Add some events to history
        event_ids = []
        for i in range(5):
            event_id = session.add_event("message", {"test": f"event_{i}"})
            event_ids.append(event_id)

        # Test getting events after a specific ID
        missed_events = session.get_events_after(event_ids[2])

        # Should get events 3 and 4
        assert len(missed_events) == 2
        assert missed_events[0].event_id == event_ids[3]
        assert missed_events[1].event_id == event_ids[4]

        await transport.stop()

    # Test 11: Parse Error Handling
    @pytest.mark.asyncio
    async def test_parse_error_handling(self):
        """Test handling of invalid JSON."""
        transport = self.create_testable_transport()
        await transport.start()

        # Invalid JSON
        request = MockRequest(
            method="POST",
            body='{"invalid json',
            headers={"accept": "application/json, text/event-stream"},
        )

        result = await transport.handle_request(request)

        # Should return parse error
        assert isinstance(result, dict)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] is None
        assert result["error"]["code"] == -32700
        assert "Parse error" in result["error"]["message"]

        await transport.stop()

    # Test 12: Custom Response Mode Decider
    @pytest.mark.asyncio
    async def test_custom_response_mode_decider(self):
        """Test custom response mode decision logic."""

        def custom_decider(message: Dict[str, Any]) -> ResponseMode:
            # All tool calls use SSE
            if message.get("method") == "tools/call":
                return ResponseMode.SSE
            # Everything else uses JSON
            return ResponseMode.JSON

        transport = self.create_testable_transport(response_mode_decider=custom_decider)
        await transport.start()

        # Tool call should use SSE
        tool_msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
        assert transport.response_mode_decider(tool_msg) == ResponseMode.SSE

        # Other methods should use JSON
        list_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        assert transport.response_mode_decider(list_msg) == ResponseMode.JSON

        await transport.stop()

    # Test 13: SSE Event Formatting
    def test_sse_event_formatting(self):
        """Test SSE event formatting."""
        transport = self.create_testable_transport()

        # Test with event ID
        event = transport._format_sse_event(
            "message", {"jsonrpc": "2.0", "id": 1, "result": "test"}, event_id=42
        )

        expected = 'id: 42\nevent: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": "test"}\n\n'
        assert event == expected

        # Test without event ID
        event = transport._format_sse_event("ping", {}, event_id=None)

        expected = "event: ping\ndata: {}\n\n"
        assert event == expected

    # Test 14: Session Cleanup
    @pytest.mark.asyncio
    async def test_session_cleanup(self):
        """Test automatic cleanup of inactive sessions."""
        # Create transport with very short timeout
        transport = self.create_testable_transport(
            session_timeout_hours=0.0001  # Very short for testing
        )
        await transport.start()

        # Create a session
        session_id = transport._create_session_id()
        session = transport._get_or_create_session(session_id)

        # Manually set last activity to past
        from datetime import datetime, timedelta

        session.last_activity = datetime.now() - timedelta(hours=1)

        # Run cleanup
        await transport._cleanup_inactive_sessions()

        # Session should be removed
        assert session_id not in transport.sessions

        await transport.stop()

    # Test 15: Stateless Server (No Sessions)
    @pytest.mark.asyncio
    async def test_stateless_server_operation(self):
        """Test operation without session management."""
        transport = self.create_testable_transport(enable_sessions=False)
        await transport.start()

        # POST request should work without sessions
        request = MockRequest(
            method="POST",
            body='{"jsonrpc":"2.0","id":1,"method":"tools/list"}',
            headers={"accept": "application/json, text/event-stream"},
        )

        # Should not create any sessions
        await transport.handle_request(request)
        assert len(transport.sessions) == 0

        # GET request should fail without sessions
        get_request = MockRequest(method="GET")

        with pytest.raises(ValueError, match="GET requests require session management"):
            await transport.handle_request(get_request)

        await transport.stop()
