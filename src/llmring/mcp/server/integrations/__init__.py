"""
MCP Server Engine integrations with web frameworks.
"""

try:
    from llmring.mcp.server.integrations.fastapi_streamable import (
        FastAPIStreamableTransport,
        create_mcp_endpoint,
        setup_mcp_routes,
        create_fastapi_app,
    )

    __all__ = [
        "FastAPIStreamableTransport",
        "create_mcp_endpoint",
        "setup_mcp_routes",
        "create_fastapi_app",
    ]
except ImportError:
    # FastAPI integration not available
    __all__ = []
