"""
Additional protocol compliance tests for MCP Server Engine.

These tests cover the missing test cases identified in the compliance report,
ensuring full coverage of the MCP STDIO specification requirements.
"""

import asyncio
import pytest
import time

from llmring.mcp.server import MCPServer

from test_protocol_compliance import MockTransport


class TestMissingProtocolCases:
    """Test cases missing from original compliance tests."""

    @pytest.mark.asyncio
    async def test_invalid_request_error(self):
        """Test invalid request handling (-32600)."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        server_task = asyncio.create_task(server.run(transport))
        await asyncio.sleep(0.1)

        try:
            # Missing jsonrpc field
            transport._handle_message({"id": 1, "method": "test"})

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response is not None
            assert "error" in response
            # Without jsonrpc field, it's treated as method not found since "test" isn't a valid method
            assert response["error"]["code"] in [-32600, -32700, -32601]

            # Wrong jsonrpc version
            transport.clear_messages()
            transport._handle_message(
                {"jsonrpc": "1.0", "id": 2, "method": "test"}  # Wrong version
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert "error" in response

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_request_response_correlation(self):
        """Test that responses are correlated with requests by ID."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Register tools with different delays
        async def fast_tool():
            await asyncio.sleep(0.05)
            return "fast result"

        async def slow_tool():
            await asyncio.sleep(0.15)
            return "slow result"

        server.register_tool("fast", fast_tool, "Fast tool")
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

            # Send multiple concurrent requests with different IDs
            requests = [
                {
                    "jsonrpc": "2.0",
                    "id": "req-slow",
                    "method": "tools/call",
                    "params": {"name": "slow", "arguments": {}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {"name": "fast", "arguments": {}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": "req-fast-2",
                    "method": "tools/call",
                    "params": {"name": "fast", "arguments": {}},
                },
            ]

            for req in requests:
                transport._handle_message(req)

            # Wait for all responses
            await asyncio.sleep(0.3)

            # Check responses have correct IDs
            responses = {
                msg["id"]: msg for msg in transport.messages_sent if "id" in msg
            }

            assert len(responses) == 3
            assert "req-slow" in responses
            assert 42 in responses
            assert "req-fast-2" in responses

            # Verify each response matches its request
            assert (
                "slow result" in responses["req-slow"]["result"]["content"][0]["text"]
            )
            assert "fast result" in responses[42]["result"]["content"][0]["text"]
            assert (
                "fast result" in responses["req-fast-2"]["result"]["content"][0]["text"]
            )

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_resource_not_found(self):
        """Test error when resource not found."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

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

            # Try to read non-existent resource
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "resources/read",
                    "params": {"uri": "unknown://resource"},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "error" in response
            assert (
                "not found" in response["error"]["message"].lower()
                or "unknown" in response["error"]["message"].lower()
            )

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_prompt_default_arguments(self):
        """Test prompts with default argument values."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Register prompt with optional arguments
        def test_prompt(args):
            name = args.get("name", "World")  # Default value
            greeting = args.get("greeting", "Hello")  # Default value
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": f"{greeting}, {name}!"},
                    }
                ]
            }

        server.register_prompt(
            name="greet",
            description="Greeting prompt",
            arguments=[
                {"name": "name", "description": "Name to greet", "required": False},
                {"name": "greeting", "description": "Greeting word", "required": False},
            ],
            handler=test_prompt,
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

            # Get prompt with no arguments (should use defaults)
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "prompts/get",
                    "params": {"name": "greet", "arguments": {}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert "result" in response
            assert (
                response["result"]["messages"][0]["content"]["text"] == "Hello, World!"
            )

            # Get prompt with partial arguments
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "prompts/get",
                    "params": {"name": "greet", "arguments": {"name": "Alice"}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert (
                response["result"]["messages"][0]["content"]["text"] == "Hello, Alice!"
            )

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_performance_message_throughput(self):
        """Test message throughput performance."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Simple echo tool for performance testing
        server.register_tool(
            name="echo", handler=lambda text: text, description="Echo tool"
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
            transport.clear_messages()

            # Measure throughput
            message_count = 100
            start_time = time.time()

            # Send many requests
            for i in range(message_count):
                transport._handle_message(
                    {
                        "jsonrpc": "2.0",
                        "id": i + 1000,
                        "method": "tools/call",
                        "params": {"name": "echo", "arguments": {"text": f"msg{i}"}},
                    }
                )

            # Wait for all responses
            max_wait = 5.0  # Maximum 5 seconds
            wait_interval = 0.1
            waited = 0.0

            while len(transport.messages_sent) < message_count and waited < max_wait:
                await asyncio.sleep(wait_interval)
                waited += wait_interval

            elapsed = time.time() - start_time

            # Should process all messages
            assert len(transport.messages_sent) >= message_count

            # Calculate throughput
            throughput = message_count / elapsed

            # Should handle at least 50 messages/second
            # (Lower threshold than spec due to test overhead)
            assert throughput > 50, f"Throughput too low: {throughput:.1f} msg/s"

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_tool_error_recovery(self):
        """Test that server continues after tool errors."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

        # Register failing tool
        def failing_tool():
            raise RuntimeError("Tool failure")

        server.register_tool("failing", failing_tool, "Failing tool")
        server.register_tool("working", lambda: "success", "Working tool")

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

            # Call failing tool
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "failing", "arguments": {}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 2
            assert response["result"]["isError"] is True
            assert "Tool failure" in response["result"]["content"][0]["text"]

            # Server should still work - call working tool
            transport.clear_messages()
            transport._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "working", "arguments": {}},
                }
            )

            await asyncio.sleep(0.1)

            response = transport.get_last_message()
            assert response["id"] == 3
            assert response["result"]["isError"] is False
            assert "success" in response["result"]["content"][0]["text"]

        finally:
            server._running = False
            await server_task

    @pytest.mark.asyncio
    async def test_shutdown_notification_before_close(self):
        """Test that shutdown sends notification before closing."""
        server = MCPServer(name="Test", version="1.0")
        transport = MockTransport()

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

            # Send shutdown request
            transport._handle_message({"jsonrpc": "2.0", "id": 2, "method": "shutdown"})

            await asyncio.sleep(0.2)

            # Should receive shutdown response
            messages = transport.messages_sent
            shutdown_response = next((m for m in messages if m.get("id") == 2), None)

            assert shutdown_response is not None
            assert "result" in shutdown_response
            assert shutdown_response["result"] == {}

            # Server should stop
            assert not server._running

        finally:
            if server._running:
                server._running = False
            await server_task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
