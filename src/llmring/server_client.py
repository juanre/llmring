"""Client utilities for interacting with llmring-server.

This module provides utilities for server communication, but does NOT include
alias sync functionality as aliases are purely local per source-of-truth v3.8.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ServerClient:
    """Client for communicating with llmring-server or llmring-api."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the server client.
        
        Args:
            base_url: Base URL of the server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # Setup HTTP client
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key  # Alternative header format
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
    
    async def post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the server.
        
        Args:
            path: API endpoint path
            json: JSON data to send
            
        Returns:
            Response data as dictionary
        """
        try:
            response = await self.client.post(path, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the server.
        
        Args:
            path: API endpoint path
            params: Optional query parameters
            
        Returns:
            Response data as dictionary
        """
        try:
            response = await self.client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()