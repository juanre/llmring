"""
Base HTTP client for all LLMRing HTTP operations.

This module provides a unified base class for HTTP operations,
consolidating the various HTTP client implementations across the codebase.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    """
    Base HTTP client that provides common functionality for all HTTP operations.

    This class consolidates the HTTP client implementations
    and provides a consistent interface for HTTP operations.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        auth_header_name: str = "Authorization",
        auth_header_prefix: str = "Bearer",
    ):
        """
        Initialize the base HTTP client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            headers: Additional headers to include
            auth_header_name: Name of the auth header (default: Authorization)
            auth_header_prefix: Prefix for auth header value (default: Bearer)
        """
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Setup headers
        self.headers = headers or {}
        if api_key:
            if auth_header_prefix:
                self.headers[auth_header_name] = f"{auth_header_prefix} {api_key}"
            else:
                self.headers[auth_header_name] = api_key

            # Also add X-API-Key for compatibility
            self.headers["X-API-Key"] = api_key

        # Setup content type
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @asynccontextmanager
    async def session(self):
        """Context manager for a session that ensures proper cleanup."""
        try:
            yield self
        finally:
            await self.close()

    async def request(
        self,
        method: str,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            json: JSON data to send
            params: Query parameters
            headers: Additional headers for this request
            **kwargs: Additional arguments for httpx

        Returns:
            HTTP response

        Raises:
            httpx.HTTPStatusError: For HTTP errors
        """
        # Merge headers
        request_headers = dict(self.headers)
        if headers:
            request_headers.update(headers)

        # Make request
        try:
            response = await self.client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=request_headers,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP {e.response.status_code} error for {method} {path}: " f"{e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Request failed for {method} {path}: {e}")
            raise

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a GET request and return JSON response.

        Args:
            path: API endpoint path
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            Response data as dictionary
        """
        response = await self.request("GET", path, params=params, **kwargs)
        return response.json()

    async def post(
        self,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a POST request and return JSON response.

        Args:
            path: API endpoint path
            json: JSON data to send
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            Response data as dictionary
        """
        response = await self.request("POST", path, json=json, params=params, **kwargs)
        return response.json()

    async def put(
        self,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a PUT request and return JSON response.

        Args:
            path: API endpoint path
            json: JSON data to send
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            Response data as dictionary
        """
        response = await self.request("PUT", path, json=json, params=params, **kwargs)
        return response.json()

    async def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], bool]:
        """
        Make a DELETE request.

        Args:
            path: API endpoint path
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            Response data or True if successful
        """
        response = await self.request("DELETE", path, params=params, **kwargs)

        # Check if response has content
        if response.content:
            try:
                return response.json()
            except Exception:
                # Response might not be valid JSON, return success status
                return True
        return True

    async def patch(
        self,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a PATCH request and return JSON response.

        Args:
            path: API endpoint path
            json: JSON data to send
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            Response data as dictionary
        """
        response = await self.request("PATCH", path, json=json, params=params, **kwargs)
        return response.json()

    async def stream(
        self,
        method: str,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Make a streaming request.

        Args:
            method: HTTP method
            path: API endpoint path
            json: JSON data to send
            params: Query parameters
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        async with self.client.stream(
            method=method,
            url=path,
            json=json,
            params=params,
            **kwargs,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk
