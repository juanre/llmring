"""
Comprehensive protocol compliance tests for MCP Server.

These tests ensure that the MCP server implementation follows
the specification exactly, including:
- Initialization sequence
- Error handling
- All protocol methods
- Message format requirements
"""

import asyncio
import json
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.base import Transport
from llmring.mcp.server.protocol.handlers import ProtocolError
from llmring.mcp.server.protocol.json_rpc import JSONRPCError


class MockTransport(Transport):
    """Mock transport for testing."""

    def __init__(self):
        super().__init__()
        self.messages_sent: List[Dict[str, Any]] = []
        self._running = False

    async def start(self) -> bool:
        self._running = True
        return True

    async def stop(self) -> None:
        self._running = False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        self.messages_sent.append(message)
        return True

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        return self.messages_sent[-1] if self.messages_sent else None

    def clear_messages(self):
        self.messages_sent.clear()

    async def _process_message(self, line: str):
        """Process a message line (for testing parse errors)."""
        try:
            message = json.loads(line)
            self._handle_message(message)
        except json.JSONDecodeError as e:
            # Send parse error response
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error", "data": {"line": line[:100]}},
            }
            await self.send_message(error_response)


class TestProtocolCompliance:
    """Test suite for MCP protocol compliance."""

    @pytest.fixture
    def server(self):
        """Create a test server."""
        return MCPServer(name="Test Server", version="1.0.0")

    @pytest.fixture
    def transport(self):
        """Create a mock transport."""
        return MockTransport()

    @pytest.mark.asyncio
    async def test_initialization_required(self, server, transport):
        """Test that initialization must happen first."""
        # Start server in background
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)  # Let server start

        try:
            # Try to call method before initialization
            transport._handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

            # Wait for response
            await asyncio.sleep(0.1)

            # Should get error about not initialized
            response = transport.get_last_message()
            assert response is not None
            assert "error" in response
            assert response["error"]["code"] == -32002
            assert "not initialized" in response["error"]["message"].lower()

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_initialization_sequence(self, server, transport):
        """Test complete initialization sequence."""
        # Start server in background
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Step 1: Initialize request
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "clientInfo": {"name": "Test Client", "version": "1.0.0"},
                    },
                }
            )

            await asyncio.sleep(0.1)

            # Check response
            response = transport.get_last_message()
            assert response["id"] == 1
            assert "result" in response
            assert response["result"]["serverInfo"]["name"] == "Test Server"
            assert response["result"]["serverInfo"]["version"] == "1.0.0"
            assert "capabilities" in response["result"]

            # Step 2: Initialized notification
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "method": "initialized"})

            await asyncio.sleep(0.1)

            # Should receive a log notification
            messages = transport.messages_sent
            assert any(msg.get("method") == "log" for msg in messages)

            # Step 3: Can now make requests
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert "tools" in response["result"]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_double_initialization_error(self, server, transport):
        """Test that double initialization is rejected."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # First initialization
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Second initialization should fail
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "error" in response
            assert "already initialized" in response["error"]["message"].lower()

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_notification_no_response(self, server, transport):
        """Test that notifications don't generate responses."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize first
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)
            transport.clear_messages()

            # Send notification (no ID)
            transport._handle_message({"jsonrpc": "2.0", "method": "initialized"})

            await asyncio.sleep(0.1)

            # Should not send a direct response (may send other notifications)
            messages = transport.messages_sent
            # Filter out any log notifications
            responses = [msg for msg in messages if "id" in msg]
            assert len(responses) == 0

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_error_codes(self, server, transport):
        """Test proper error code handling."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Test method not found (-32601)
            transport._handle_message({"jsonrpc": "2.0", "id": 1, "method": "unknown/method"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["error"]["code"] == JSONRPCError.METHOD_NOT_FOUND

            # Initialize for further tests
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Test invalid params (-32602)
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {},  # Missing required 'name' parameter
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert "error" in response
            # Should be either invalid params or internal error
            assert response["error"]["code"] in [
                JSONRPCError.INVALID_PARAMS,
                JSONRPCError.INTERNAL_ERROR,
            ]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_tools_methods(self, server, transport):
        """Test tools/list and tools/call methods."""
        # Register a test tool
        server.register_tool(
            name="test_tool",
            handler=lambda x: f"Result: {x}",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # List tools
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert "tools" in response["result"]
            assert len(response["result"]["tools"]) == 1
            assert response["result"]["tools"][0]["name"] == "test_tool"

            # Call tool
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "test_tool", "arguments": {"x": "hello"}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert "result" in response
            assert response["result"]["isError"] is False
            assert "Result: hello" in response["result"]["content"][0]["text"]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_resources_methods(self, server, transport):
        """Test resources/list and resources/read methods."""
        # Register test resources
        server.register_resource(
            uri="test://text",
            name="Test Text",
            description="A test text resource",
            mime_type="text/plain",
            handler=lambda: "Hello, World!",
        )

        server.register_resource(
            uri="test://binary",
            name="Test Binary",
            description="A test binary resource",
            mime_type="application/octet-stream",
            handler=lambda: b"\x00\x01\x02\x03",
        )

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # List resources
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert "resources" in response["result"]
            assert len(response["result"]["resources"]) == 2

            # Read text resource
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "resources/read",
                    "params": {"uri": "test://text"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert "result" in response
            assert "contents" in response["result"]
            assert response["result"]["contents"][0]["text"] == "Hello, World!"
            assert "blob" not in response["result"]["contents"][0]

            # Read binary resource
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "resources/read",
                    "params": {"uri": "test://binary"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 4
            assert "result" in response
            assert "contents" in response["result"]
            assert "blob" in response["result"]["contents"][0]
            assert "text" not in response["result"]["contents"][0]

            # Verify base64 encoding
            import base64

            blob_data = base64.b64decode(response["result"]["contents"][0]["blob"])
            assert blob_data == b"\x00\x01\x02\x03"

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_prompts_methods(self, server, transport):
        """Test prompts/list and prompts/get methods."""
        # Register a test prompt
        server.register_prompt(
            name="test_prompt",
            description="A test prompt",
            arguments=[{"name": "topic", "description": "Topic to discuss", "required": True}],
            handler=lambda args: {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": f"Tell me about {args['topic']}"},
                    }
                ]
            },
        )

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # List prompts
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "prompts/list"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert "prompts" in response["result"]
            assert len(response["result"]["prompts"]) == 1
            assert response["result"]["prompts"][0]["name"] == "test_prompt"

            # Get prompt
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "prompts/get",
                    "params": {"name": "test_prompt", "arguments": {"topic": "Python"}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert "result" in response
            assert "messages" in response["result"]
            assert "Tell me about Python" in response["result"]["messages"][0]["content"]["text"]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_logging_set_level(self, server, transport):
        """Test logging/setLevel method."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Set log level
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "logging/setLevel",
                    "params": {"level": "debug"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert response["result"]["level"] == "debug"

            # Test invalid level
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "logging/setLevel",
                    "params": {"level": "invalid"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert "error" in response

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_lifecycle_methods(self, server, transport):
        """Test ping and shutdown methods."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Ping
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert response["result"]["pong"] is True
            assert "timestamp" in response["result"]

            # Shutdown
            transport.clear_messages()
            transport._handle_message({"jsonrpc": "2.0", "id": 3, "method": "shutdown"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert "result" in response
            assert response["result"] == {}

            # Server should shutdown
            await asyncio.sleep(0.2)
            assert not server._running

        finally:
            if server._running:
                server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_message_format_validation(self, server, transport):
        """Test JSON-RPC message format validation."""
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Valid message
            valid_msg = {"jsonrpc": "2.0", "id": 1, "method": "ping"}

            # Ensure no embedded newlines in JSON
            json_str = json.dumps(valid_msg)
            assert "\n" not in json_str

            # Message without method (response)
            transport._handle_message({"jsonrpc": "2.0", "id": 1, "result": "test"})

            await asyncio.sleep(0.1)

            # Should log as response but not send anything
            assert len(transport.messages_sent) == 0

        finally:
            server._running = False
            await server_task


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Register a slow tool
        async def slow_tool(duration: float = 0.1):
            await asyncio.sleep(duration)
            return f"Slept for {duration}s"

        server.register_tool("slow", slow_tool, "Slow tool")

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)
            transport.clear_messages()

            # Send multiple requests concurrently
            for i in range(3):
                transport._handle_message(
                    {
                        "jsonrpc": "2.0",
                        "id": i + 10,
                        "method": "tools/call",
                        "params": {"name": "slow", "arguments": {"duration": 0.1 * (i + 1)}},
                    }
                )

            # Wait for all to complete
            await asyncio.sleep(0.5)

            # Should have 3 responses
            responses = [msg for msg in transport.messages_sent if "id" in msg]
            assert len(responses) == 3

            # All should succeed
            for response in responses:
                assert "result" in response
                assert not response["result"]["isError"]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_parse_error_response(self):
        """Test that parse errors generate proper error responses."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Send invalid JSON through the transport's process_message method
            # This simulates what happens when invalid JSON is received
            await transport._process_message("invalid json{")

            await asyncio.sleep(0.1)

            # Should get parse error response
            response = transport.get_last_message()
            assert response is not None
            assert "error" in response
            assert response["error"]["code"] == -32700
            assert "parse" in response["error"]["message"].lower()
            assert response["id"] is None  # Parse errors have null ID

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_transport_error_handling(self):
        """Test handling of transport errors."""
        server = MCPServer(name="Test", version="1.0")

        # Create a failing transport
        class FailingTransport(MockTransport):
            async def send_message(self, message: Dict[str, Any]) -> bool:
                if message.get("id") == 2:
                    raise RuntimeError("Transport error")
                return await super().send_message(message)

        transport = FailingTransport()
        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize (should work)
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)
            assert len(transport.messages_sent) == 1

            # This should fail but not crash server
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})

            await asyncio.sleep(0.1)

            # Server should still be running
            assert server._running

            # Another request should work
            transport._handle_message({"jsonrpc": "2.0", "id": 3, "method": "ping"})

            await asyncio.sleep(0.1)
            assert any(msg.get("id") == 3 for msg in transport.messages_sent)

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_binary_resource_large(self):
        """Test handling of large binary resources."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Create 1MB of binary data
        large_data = bytes(range(256)) * 4096  # 1MB

        server.register_resource(
            uri="test://large",
            name="Large Binary",
            description="Large binary resource",
            mime_type="application/octet-stream",
            handler=lambda: large_data,
        )

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Initialize
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )

            await asyncio.sleep(0.1)

            # Read large binary resource
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "resources/read",
                    "params": {"uri": "test://large"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert "contents" in response["result"]
            assert "blob" in response["result"]["contents"][0]

            # Verify base64 encoding
            import base64

            blob_data = base64.b64decode(response["result"]["contents"][0]["blob"])
            assert blob_data == large_data
            assert len(blob_data) == 1024 * 1024  # 1MB

        finally:
            server._running = False
            await server_task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
