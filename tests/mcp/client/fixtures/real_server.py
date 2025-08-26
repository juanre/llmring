"""
Test fixtures for using real mcp-server in tests.

This module provides fixtures that start and manage real mcp-server instances
for integration testing, replacing the need for mocks.
"""

import os
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from llmring.mcp.client import AsyncMCPClient, MCPClient

# Check if mcp-server is available by trying to import it or checking path
MCP_SERVER_AVAILABLE = False
MCP_SERVER_PATH = Path(__file__).parent.parent.parent.parent / "mcp-server"


# Define simple types for testing if mcp-server types aren't available
class Tool:
    def __init__(
        self, name: str, description: str = "", input_schema: dict | None = None, handler=None
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}
        self.handler = handler


class Resource:
    def __init__(
        self,
        uri: str,
        name: str = "",
        description: str = "",
        mime_type: str | None = None,
        mimeType: str | None = None,
        handler=None,
    ):
        self.uri = uri
        self.name = name
        self.description = description
        # Accept both mime_type and mimeType
        self.mime_type = mime_type or mimeType or "text/plain"
        self.mimeType = self.mime_type  # Support both attributes
        self.handler = handler


class Prompt:
    def __init__(
        self, name: str, description: str = "", arguments: list | None = None, handler=None
    ):
        self.name = name
        self.description = description
        self.arguments = arguments or []
        self.handler = handler


class Server:
    def __init__(self, name: str = "test-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools = []
        self.resources = []
        self.prompts = []

    def add_tool(self, tool):
        self.tools.append(tool)

    def add_resource(self, resource):
        self.resources.append(resource)

    def add_prompt(self, prompt):
        self.prompts.append(prompt)


if MCP_SERVER_PATH.exists():
    # Add mcp-server to Python path for imports
    sys.path.insert(0, str(MCP_SERVER_PATH / "src"))
    try:
        # Try importing to verify it works
        import llmring.mcp.server

        MCP_SERVER_AVAILABLE = True
        # Try to import actual types if available
        try:
            from mcp.server import Server as MCPServer
            from mcp.types import Prompt as MCPPrompt
            from mcp.types import Resource as MCPResource
            from mcp.types import Tool as MCPTool

            Tool = MCPTool
            Resource = MCPResource
            Prompt = MCPPrompt
            Server = MCPServer
        except ImportError:
            pass
    except ImportError:
        pass


@pytest.fixture
async def mcp_server_factory():
    """
    Factory fixture for creating MCP server instances with custom configurations.

    Returns:
        A factory function that creates configured MCP servers
    """
    servers = []

    async def _create_server(
        name: str = "test-server",
        version: str = "1.0.0",
        tools: list[Tool] | None = None,
        resources: list[Resource] | None = None,
        prompts: list[Prompt] | None = None,
    ) -> Server:
        """Create a configured MCP server instance."""
        server = Server(name=name, version=version)

        # Add tools
        if tools:
            for tool in tools:
                server.add_tool(tool)

        # Add resources
        if resources:
            for resource in resources:
                server.add_resource(resource)

        # Add prompts
        if prompts:
            for prompt in prompts:
                server.add_prompt(prompt)

        servers.append(server)
        return server

    yield _create_server

    # Cleanup any servers
    for _server in servers:
        # Server cleanup if needed
        pass


@pytest.fixture
async def basic_mcp_server(mcp_server_factory):
    """
    Create a basic MCP server with test tools, resources, and prompts.

    Returns:
        A configured MCP server instance
    """

    # Define test tools
    async def echo_handler(arguments: dict[str, Any]) -> str:
        """Echo the input message."""
        return f"Echo: {arguments.get('message', '')}"

    async def add_handler(arguments: dict[str, Any]) -> str:
        """Add two numbers."""
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return f"Result: {a + b}"

    echo_tool = Tool(
        name="echo",
        description="Echo the input message",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to echo"}},
            "required": ["message"],
        },
        handler=echo_handler,
    )

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        handler=add_handler,
    )

    # Define test resources
    async def test_file_handler(uri: str) -> dict[str, Any]:
        """Read a test file resource."""
        return {
            "contents": [{"type": "text", "text": f"Content of {uri}"}],
            "mimeType": "text/plain",
        }

    test_resource = Resource(
        uri="file:///test.txt", name="Test File", mimeType="text/plain", handler=test_file_handler
    )

    # Define test prompts
    async def greeting_handler(arguments: dict[str, Any]) -> str:
        """Generate a greeting prompt."""
        name = arguments.get("name", "World")
        return f"Hello, {name}! How can I help you today?"

    greeting_prompt = Prompt(
        name="greeting",
        description="Generate a personalized greeting",
        arguments=[
            {
                "name": "name",
                "description": "Name of the person to greet",
                "required": False,
                "default": "World",
            }
        ],
        handler=greeting_handler,
    )

    # Create server with all components
    server = await mcp_server_factory(
        name="test-mcp-server",
        version="1.0.0",
        tools=[echo_tool, add_tool],
        resources=[test_resource],
        prompts=[greeting_prompt],
    )

    return server


@pytest.fixture
async def http_mcp_server(basic_mcp_server) -> AsyncGenerator[dict[str, Any], None]:
    """
    Start an HTTP MCP server for testing.

    Yields:
        Dict with server info including URL and server instance
    """
    server = basic_mcp_server

    # Create HTTP transport
    # TODO: Implement create_http_transport or use a different approach
    pytest.skip("HTTP transport fixture not implemented yet")

    # Start server on a random port
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/mcp", transport.handle_request)

    runner = web.AppRunner(app)
    await runner.setup()

    # Bind to localhost with random port
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()

    # Get the actual port
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://localhost:{port}"

    yield {
        "url": f"{base_url}/mcp",
        "base_url": base_url,
        "server": server,
        "transport": transport,
        "runner": runner,
    }

    # Cleanup
    await runner.cleanup()


@pytest.fixture
def http_mcp_client(http_mcp_server) -> MCPClient:
    """
    Create an HTTP MCP client connected to the test server.

    Returns:
        Configured MCPClient instance
    """
    client = MCPClient.http(http_mcp_server["url"])

    # Initialize the client
    client.initialize()

    yield client

    # Cleanup
    client.close()


@pytest.fixture
async def async_http_mcp_client(http_mcp_server) -> AsyncGenerator[AsyncMCPClient, None]:
    """
    Create an async HTTP MCP client connected to the test server.

    Returns:
        Configured AsyncMCPClient instance
    """
    client = AsyncMCPClient.http(http_mcp_server["url"])

    # Initialize the client
    await client.initialize()

    yield client

    # Cleanup
    await client.close()


@pytest.fixture
def stdio_mcp_server(tmp_path) -> dict[str, Any]:
    """
    Create a stdio-based MCP server subprocess for testing.

    Returns:
        Dict with server process info
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("mcp-server not available")

    # Create a test registry module with sample functions
    registry_script = tmp_path / "test_registry.py"
    registry_script.write_text(
        """
def mcp_echo(args):
    \"\"\"Echo the input message.\"\"\"
    return f"Echo: {args.get('message', '')}"

def mcp_add(args):
    \"\"\"Add two numbers.\"\"\"
    a = args.get('a', 0)
    b = args.get('b', 0)
    return f"The sum of {a} and {b} is {a + b}"
"""
    )

    # Start the mcp-server process using stdio transport
    cmd = ["mcp-server", "--registry", str(registry_script)]

    # Set PYTHONPATH to include mcp-server
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MCP_SERVER_PATH / "src")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,  # Unbuffered
    )

    # Give it a moment to start
    time.sleep(0.5)

    yield {"process": process, "registry": registry_script}

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# Helper functions for tests


def make_jsonrpc_request(
    method: str, params: dict[str, Any] | None = None, id: str | None = None
) -> dict[str, Any]:
    """Create a JSON-RPC request."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
    }

    if params is not None:
        request["params"] = params

    if id is not None:
        request["id"] = id

    return request


def assert_jsonrpc_success(response: dict[str, Any], request_id: str | None = None) -> Any:
    """Assert that a JSON-RPC response is successful and return the result."""
    assert "error" not in response, f"Expected success but got error: {response.get('error')}"
    assert "result" in response, "Missing result in response"

    if request_id is not None:
        assert response.get("id") == request_id, "Response ID mismatch"

    return response["result"]


def assert_jsonrpc_error(
    response: dict[str, Any], expected_code: int | None = None
) -> dict[str, Any]:
    """Assert that a JSON-RPC response is an error and return the error object."""
    assert "error" in response, f"Expected error but got success: {response}"
    assert "result" not in response, "Unexpected result in error response"

    error = response["error"]

    if expected_code is not None:
        assert (
            error.get("code") == expected_code
        ), f"Expected error code {expected_code} but got {error.get('code')}"

    return error


@pytest.fixture
def stdio_mcp_client(stdio_mcp_server) -> MCPClient:
    """
    Create a sync MCP client connected to a STDIO server.

    Returns:
        Configured MCPClient instance
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("mcp-server not available")

    server_info = stdio_mcp_server
    process = server_info["process"]

    # Create a mock stdio URL - the transport factory will handle the process
    # For stdio transport, we need to pass the process directly
    from llmring.mcp.client.transports.stdio import STDIOTransport

    client = MCPClient.stdio(["mcp-server", "--registry", "dummy"])

    # Replace the transport with one connected to our process
    transport = STDIOTransport(["dummy"], allow_unsafe_commands=True)
    transport.process = process
    client.transport = transport

    yield client

    # Cleanup
    client.close()


@pytest.fixture
async def async_stdio_mcp_client(stdio_mcp_server) -> AsyncMCPClient:
    """
    Create an async MCP client connected to a STDIO server.

    Returns:
        Configured AsyncMCPClient instance
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("mcp-server not available")

    server_info = stdio_mcp_server
    process = server_info["process"]

    # Create async client with stdio transport
    from llmring.mcp.client.transports.stdio import STDIOTransport

    client = AsyncMCPClient.stdio(["mcp-server", "--registry", "dummy"])

    # Replace the transport with one connected to our process
    transport = STDIOTransport(["dummy"], allow_unsafe_commands=True)
    transport.process = process
    client.transport = transport

    yield client

    # Cleanup
    await client.close()


@pytest.fixture
def sse_mcp_client():
    """
    SSE client fixture - currently not implemented.

    This fixture returns None to indicate SSE transport is not available.
    Tests should check for None and skip if SSE is required.
    """
    return None
