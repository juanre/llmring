"""
FastAPI integration for MCP Streamable HTTP transport.

This module provides a FastAPI-specific implementation of the Streamable HTTP transport.
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llmring.mcp.server.transport.streamable_http import StreamableHTTPTransport


class FastAPIStreamableTransport(StreamableHTTPTransport):
    """FastAPI-specific implementation of Streamable HTTP transport."""

    def __init__(self, **kwargs):
        """Initialize with FastAPI-specific settings."""
        super().__init__(**kwargs)
        self._response_headers: Dict[str, str] = {}

    def _get_request_method(self, request: Request) -> str:
        """Get HTTP method from FastAPI request."""
        return request.method

    async def _get_request_body(self, request: Request) -> str:
        """Get request body from FastAPI request."""
        body = await request.body()
        return body.decode("utf-8")

    def _get_request_headers(self, request: Request) -> Dict[str, str]:
        """Get headers from FastAPI request."""
        return dict(request.headers)

    def _set_response_session_id(self, request: Any, session_id: str):
        """Store session ID to be added to response headers."""
        self._response_headers["Mcp-Session-Id"] = session_id

    async def create_response(self, request: Request) -> Response:
        """
        Create a FastAPI response from the transport's handle_request result.

        This method handles the conversion from the transport's return values
        to appropriate FastAPI response types.
        """
        # Clear response headers
        self._response_headers.clear()

        # Check Accept header
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" not in accept_header and "application/json" not in accept_header:
            raise HTTPException(
                status_code=406,
                detail="Accept header must include 'application/json' and/or 'text/event-stream'",
            )

        # Enforce MCP-Protocol-Version header after initialization if present in session
        # Note: In full implementation, you would check/track negotiated version per session.

        # Handle the request
        result = await self.handle_request(request)

        if result is None:
            # 202 Accepted for notifications
            return Response(status_code=202, headers=self._response_headers)

        elif isinstance(result, dict):
            # JSON response
            return JSONResponse(content=result, headers=self._response_headers)

        else:
            # SSE streaming response
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    **self._response_headers,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                },
            )


def create_mcp_endpoint(transport: FastAPIStreamableTransport, endpoint_path: str = "/mcp"):
    """
    Create FastAPI endpoint handlers for MCP Streamable HTTP.

    Args:
        transport: The configured transport instance
        endpoint_path: The endpoint path (default: /mcp)

    Returns:
        A tuple of (post_handler, get_handler, delete_handler)
    """

    async def handle_post(
        request: Request,
        mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    ):
        """Handle POST requests to the MCP endpoint."""
        return await transport.create_response(request)

    async def handle_get(
        request: Request,
        mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
        last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
    ):
        """Handle GET requests for SSE streams."""
        if not transport.enable_sessions:
            raise HTTPException(
                status_code=400,
                detail="GET requests require session management to be enabled",
            )

        if not mcp_session_id:
            raise HTTPException(
                status_code=400,
                detail="Mcp-Session-Id header required for GET requests",
            )

        return await transport.create_response(request)

    async def handle_delete(
        request: Request,
        mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    ):
        """Handle DELETE requests for session termination."""
        if not transport.enable_sessions:
            raise HTTPException(
                status_code=400,
                detail="DELETE requests require session management to be enabled",
            )

        if not mcp_session_id:
            raise HTTPException(
                status_code=400,
                detail="Mcp-Session-Id header required for DELETE requests",
            )

        await transport.handle_request(request)
        return Response(status_code=204)  # No Content

    return handle_post, handle_get, handle_delete


def setup_mcp_routes(app, transport: FastAPIStreamableTransport, endpoint_path: str = "/mcp"):
    """
    Setup MCP routes on a FastAPI application.

    Args:
        app: FastAPI application instance
        transport: The configured transport instance
        endpoint_path: The endpoint path (default: /mcp)
    """
    post_handler, get_handler, delete_handler = create_mcp_endpoint(transport, endpoint_path)

    app.post(endpoint_path)(post_handler)
    app.get(endpoint_path)(get_handler)
    app.delete(endpoint_path)(delete_handler)

    # Add startup/shutdown handlers
    # Use lifespan context if possible; fall back to on_event for legacy support
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):  # type: ignore
        await transport.start()
        try:
            yield
        finally:
            await transport.stop()

    try:
        app.router.lifespan_context = lifespan  # type: ignore[attr-defined]
    except Exception:

        @app.on_event("startup")
        async def startup():
            await transport.start()

        @app.on_event("shutdown")
        async def shutdown():
            await transport.stop()


def create_fastapi_app(
    transport: FastAPIStreamableTransport, endpoint_path: str = "/mcp", **fastapi_kwargs
) -> FastAPI:
    """
    Create a FastAPI app configured with MCP routes and lifespan handlers.

    Prefer this factory to avoid deprecation warnings from on_event handlers.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):  # type: ignore
        await transport.start()
        try:
            yield
        finally:
            await transport.stop()

    app = FastAPI(lifespan=lifespan, **fastapi_kwargs)
    setup_mcp_routes(app, transport, endpoint_path)
    return app
