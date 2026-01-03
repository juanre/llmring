"""
Simplified test fixtures for using real mcp-server in tests.

This module provides fixtures that start real mcp-server subprocesses
for integration testing without requiring the mcp-server Python API.
"""

import json
import os
import subprocess
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from llmring.mcp.client import AsyncMCPClient, MCPClient

MCP_SERVER_AVAILABLE = find_spec("llmring.mcp.server") is not None


@pytest.fixture
def real_stdio_server(tmp_path) -> dict[str, Any]:
    """
    Start a real mcp-server subprocess with stdio transport.

    Returns:
        Dict with process info
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("mcp-server not available at ../mcp-server")

    # Create test functions module
    test_module = tmp_path / "test_functions.py"
    test_module.write_text(
        '''
def mcp_echo(args):
    """Echo the input message."""
    return f"Echo: {args.get('message', '')}"

def mcp_add(args):
    """Add two numbers."""
    a = args.get('a', 0)
    b = args.get('b', 0)
    return f"The sum of {a} and {b} is {a + b}"

def mcp_greet(args):
    """Generate a greeting."""
    name = args.get('name', 'World')
    return f"Hello, {name}!"
'''
    )

    # Start mcp-server
    # Use our library-based test server

    cmd = ["python", str(Path(__file__).parent / "mcp_test_server.py")]

    env = os.environ.copy()

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,
    )

    # Wait for server to start
    time.sleep(0.5)

    # Check if process is still running
    if process.poll() is not None:
        stderr = process.stderr.read()
        raise RuntimeError(f"mcp-server failed to start: {stderr}")

    yield {"process": process, "test_module": test_module}

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture
def stdio_client(real_stdio_server):
    """Create a sync MCP client connected to stdio server."""
    server_info = real_stdio_server
    process = server_info["process"]

    # Create client with stdio URL that will use our server process
    # For stdio transport, we need to pass command info
    client = MCPClient("stdio://localhost")

    # Replace the transport with one that uses our existing process
    # This is a workaround since we already started the server
    from llmring.mcp.client.transports.base import ConnectionState, Transport

    class ProcessWrapperTransport(Transport):
        """Wrapper to use an existing process."""

        def __init__(self, process):
            super().__init__()
            self.process = process
            self._state = ConnectionState.DISCONNECTED

        async def connect(self):
            self._state = ConnectionState.CONNECTED

        async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
            request_id = self._next_id
            self._next_id += 1

            request = {"jsonrpc": "2.0", "method": method, "id": request_id}
            if params:
                request["params"] = params

            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise ConnectionError("Server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                raise Exception(f"RPC Error: {response['error']}")

            return response.get("result")

        async def close(self):
            self._state = ConnectionState.DISCONNECTED

        @property
        def state(self) -> ConnectionState:
            return self._state

    transport = ProcessWrapperTransport(process)
    client.transport = transport

    yield client

    # Cleanup
    client.close()


@pytest.fixture
async def async_stdio_client(real_stdio_server):
    """Create an async MCP client connected to stdio server."""
    server_info = real_stdio_server
    process = server_info["process"]

    # Use the same transport wrapper for async client
    from llmring.mcp.client.transports.base import ConnectionState, Transport

    class ProcessWrapperTransport(Transport):
        """Wrapper to use an existing process."""

        def __init__(self, process):
            super().__init__()
            self.process = process
            self._state = ConnectionState.DISCONNECTED

        async def connect(self):
            self._state = ConnectionState.CONNECTED

        async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
            request_id = self._next_id
            self._next_id += 1

            request = {"jsonrpc": "2.0", "method": method, "id": request_id}
            if params:
                request["params"] = params

            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise ConnectionError("Server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                raise Exception(f"RPC Error: {response['error']}")

            return response.get("result")

        async def close(self):
            self._state = ConnectionState.DISCONNECTED

        @property
        def state(self) -> ConnectionState:
            return self._state

    client = AsyncMCPClient("stdio://localhost")
    transport = ProcessWrapperTransport(process)
    client.transport = transport

    yield client

    # Cleanup
    await client.close()


# Marker for tests requiring real server
pytestmark = pytest.mark.real_server
