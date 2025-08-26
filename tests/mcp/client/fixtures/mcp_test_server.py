#!/usr/bin/env python3
"""
Test MCP server using mcp-server-engine as a library.

This server provides test tools, resources, and prompts for integration testing.
"""

import asyncio
import logging
import sys
from typing import Any

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport import StdioTransport

# Configure logging to stderr as per MCP spec
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)


# Test tool implementations
async def test_tool(arg1: str = "default") -> dict[str, Any]:
    """A simple test tool."""
    return {"result": f"Tool called with: {arg1}"}


async def echo_tool(message: str) -> str:
    """Echo the input message."""
    return f"Echo: {message}"


async def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


async def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


async def divide_numbers(a: float, b: float) -> dict[str, Any]:
    """Divide two numbers with error handling."""
    if b == 0:
        raise ValueError("Cannot divide by zero")

    return {"result": a / b, "formatted": f"{a} ÷ {b} = {a/b:.4f}"}


async def greet_user(name: str = "World") -> str:
    """Generate a greeting."""
    return f"Hello, {name}!"


# Resource handlers
async def get_server_status() -> str:
    """Get current server status."""
    import datetime

    return f"""Test Server Status
Time: {datetime.datetime.now().isoformat()}
Available tools: test_tool, echo_tool, add_numbers, multiply_numbers, divide_numbers, greet_user
Status: Running"""


# Prompt handlers
async def generate_test_prompt(name: str = "World") -> str:
    """Generate a test prompt."""
    return f"Hello, {name}! This is a test prompt."


def create_server() -> MCPServer:
    """Create and configure the test MCP server."""
    server = MCPServer(name="Test MCP Server", version="1.0.0")

    # Register test tools
    server.register_tool(
        name="test_tool",
        handler=test_tool,
        description="A simple test tool",
        input_schema={
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Test argument", "default": "default"}
            },
            "required": [],
            "additionalProperties": False,
        },
    )

    server.register_tool(
        name="echo_tool",
        handler=echo_tool,
        description="Echo the input message",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to echo"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    )

    server.register_tool(
        name="add_numbers",
        handler=add_numbers,
        description="Add two numbers together",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )

    server.register_tool(
        name="multiply_numbers",
        handler=multiply_numbers,
        description="Multiply two numbers together",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )

    server.register_tool(
        name="divide_numbers",
        handler=divide_numbers,
        description="Divide two numbers with error handling",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )

    server.register_tool(
        name="greet_user",
        handler=greet_user,
        description="Generate a greeting",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet", "default": "World"}
            },
            "required": [],
            "additionalProperties": False,
        },
    )

    # Register test resources
    server.register_static_resource(
        uri="test://resource1",
        name="Test Resource 1",
        description="A static test resource",
        content="This is test resource 1 content",
        mime_type="text/plain",
    )

    server.register_static_resource(
        uri="test://resource2",
        name="Test Resource 2",
        description="A JSON test resource",
        content='{"data": "Test JSON data", "value": 42}',
        mime_type="application/json",
    )

    server.register_resource(
        uri="server://status",
        name="Server Status",
        description="Real-time server status",
        mime_type="text/plain",
        handler=get_server_status,
    )

    # Register test prompts
    server.register_static_prompt(
        name="test_prompt",
        description="A simple test prompt",
        content="This is a test prompt with parameter: {name}",
        arguments=[{"name": "name", "description": "Name parameter", "required": False}],
    )

    server.register_prompt(
        name="dynamic_test_prompt",
        description="A dynamic test prompt",
        arguments=[{"name": "name", "description": "Name parameter", "required": False}],
        handler=generate_test_prompt,
    )

    return server


async def main():
    """Run the test MCP server."""
    try:
        # Create server
        server = create_server()

        # Create STDIO transport
        transport = StdioTransport()

        logger.info("Test MCP Server starting...")

        # Run server with transport
        await server.run(transport)

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
