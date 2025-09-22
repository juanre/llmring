"""
Tests for context support in transport and server.
"""

import asyncio
from types import SimpleNamespace

import pytest

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.base import JSONRPCMessage, Transport


class ContextAwareTransport(Transport):
    """Mock transport that provides context."""

    def __init__(self):
        super().__init__()
        self.messages_sent = []
        self.context_data = {"source": "test_transport", "session_id": "test-123"}

    async def start(self) -> bool:
        return True

    async def stop(self) -> None:
        pass

    async def send_message(self, message: JSONRPCMessage) -> bool:
        self.messages_sent.append(message)
        return True

    def simulate_message(self, message: JSONRPCMessage):
        """Simulate receiving a message with context."""
        context = SimpleNamespace(**self.context_data)
        self._handle_message_with_context(message, context)


class TestContextSupport:
    """Test context passing through transport to handlers."""

    @pytest.fixture
    def server(self):
        return MCPServer(name="Test Server", version="1.0.0")

    @pytest.fixture
    def transport(self):
        return ContextAwareTransport()

    @pytest.mark.asyncio
    async def test_context_passed_to_handlers(self, server, transport):
        """Test that transport context is available in handlers."""
        received_contexts = []

        # Register a tool that captures context
        def test_tool(**kwargs):
            return "Tool executed"

        server.register_tool(name="test_tool", handler=test_tool, description="Test tool")

        # Override _create_context to capture all contexts created
        original_create_context = server._create_context

        async def capture_context(message, transport_context=None):
            context = await original_create_context(message, transport_context)
            received_contexts.append(context)
            return context

        server._create_context = capture_context

        # Start server
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport.simulate_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Call tool
            transport.simulate_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "test_tool", "arguments": {}},
                }
            )

            await asyncio.sleep(0.1)

            # Verify context was passed
            assert len(received_contexts) >= 2  # At least for initialize and tools/call

            # Check the tool call context (should be the last one)
            tool_context = received_contexts[-1]
            assert hasattr(tool_context, "transport")
            assert tool_context.transport.source == "test_transport"
            assert tool_context.transport.session_id == "test-123"

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, server):
        """Test that transports without context still work."""
        from tests.mcp.server.test_protocol_compliance import MockTransport

        # Use the original MockTransport that doesn't provide context
        transport = MockTransport()

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Should work without context
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Check response
            response = transport.get_last_message()
            assert response is not None
            assert "result" in response
            assert response["result"]["serverInfo"]["name"] == "Test Server"

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_context_in_error_handling(self, server, transport):
        """Test that context is available during error handling."""
        received_context = None

        # Override error handler to capture context
        original_create_context = server._create_context

        async def capture_context(*args, **kwargs):
            result = await original_create_context(*args, **kwargs)
            nonlocal received_context
            received_context = result
            return result

        server._create_context = capture_context

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Send invalid request (not initialized)
            transport.simulate_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

            await asyncio.sleep(0.1)

            # Context should still be created even for errors
            assert received_context is not None
            assert hasattr(received_context, "transport")

            # Should get error response
            response = transport.messages_sent[-1]
            assert "error" in response
            assert response["error"]["code"] == -32002  # Not initialized

        finally:
            server._running = False
            await server_task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
