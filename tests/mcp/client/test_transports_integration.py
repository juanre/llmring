import os
import socket
import sys
import tempfile
import threading
import time
from contextlib import closing

import pytest

from llmring.mcp.client.mcp_client import AsyncMCPClient, MCPClient
from llmring.mcp.client.transports.streamable_http import (
    StreamableHTTPTransport as ClientStreamHTTP,
)


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_streamable_http_initialize_and_headers():
    # Require fastapi/uvicorn to be importable
    import uvicorn  # noqa: F401

    from llmring.mcp.server import MCPServer
    from llmring.mcp.server.integrations.fastapi_streamable import (
        FastAPIStreamableTransport,
        create_fastapi_app,
    )

    port = _find_free_port()

    # Build FastAPI app and MCP server/transport in-process
    mcp_transport = FastAPIStreamableTransport(enable_sessions=True)
    mcp_server = MCPServer(name="TestServer", version="1.0.0")
    app = create_fastapi_app(mcp_transport, "/mcp", title="Test MCP Server", version="1.0.0")
    mcp_transport.set_message_callback(mcp_server._handle_message)

    # Start uvicorn server in background thread
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = False

    th = threading.Thread(target=server.run, daemon=True)
    th.start()

    # Wait until server is started
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    assert getattr(server, "started", False), "Uvicorn server failed to start"

    try:
        base_url = f"http://127.0.0.1:{port}"
        transport = ClientStreamHTTP(base_url=base_url, endpoint="/mcp", timeout=10.0)
        client = AsyncMCPClient(transport=transport)

        result = await client.initialize()
        assert isinstance(result, dict)

        roots = await client.list_roots()
        assert isinstance(roots, list)
        # Exercise another method to confirm end-to-end connectivity
        tools = await client.list_tools()
        assert isinstance(tools, list)
    finally:
        # Shutdown server
        server.should_exit = True
        th.join(timeout=5)


def test_stdio_initialize_and_roots():
    # Create temporary script that runs a minimal MCP STDIO server
    server_code = (
        "import asyncio\n"
        "from llmring.mcp.server import MCPServer\n"
        "from llmring.mcp.server.transport.stdio import StdioServerTransport\n"
        "async def main():\n"
        "    server = MCPServer(name='TestSTDIO', version='1.0.0')\n"
        "    transport = StdioServerTransport()\n"
        "    await server.run(transport)\n"
        "asyncio.run(main())\n"
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
        f.write(server_code)
        tmp_path = f.name

    cmd = [sys.executable, tmp_path]

    client = MCPClient.stdio(command=cmd, timeout=20.0, allow_unsafe_commands=True)
    try:
        with client:
            result = client.initialize()
            assert isinstance(result, dict)
            roots = client.list_roots()
            assert isinstance(roots, list)
            tools = client.list_tools()
            assert isinstance(tools, list)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
