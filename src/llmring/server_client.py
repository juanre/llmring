"""Client utilities for interacting with llmring-server.

This module provides utilities for server communication, but does NOT include
alias sync functionality as aliases are purely local per source-of-truth v3.8.
"""

from __future__ import annotations

import logging
from typing import Optional

from llmring.net.http_base import BaseHTTPClient

logger = logging.getLogger(__name__)


class ServerClient(BaseHTTPClient):
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
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    # The base class provides all the needed methods:
    # - post(path, json) -> Dict[str, Any]
    # - get(path, params) -> Dict[str, Any]
    # - put(path, json) -> Dict[str, Any]
    # - delete(path) -> Union[Dict[str, Any], bool]
    # - close() -> None
    # - __aenter__ and __aexit__ for context manager support

    # Any additional server-specific methods can be added here
