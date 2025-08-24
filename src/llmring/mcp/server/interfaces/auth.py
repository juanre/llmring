"""
Authentication provider interface for MCP Server Engine.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AuthProvider(ABC):
    """Abstract authentication provider interface."""

    @abstractmethod
    async def authenticate_request(self, request: Any) -> Optional[str]:
        """
        Authenticate an incoming request.

        Args:
            request: The incoming request object (transport-specific)

        Returns:
            User/session ID if authenticated, None otherwise
        """
        pass

    @abstractmethod
    async def authenticate_context(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Authenticate from a context dictionary.

        Args:
            context: Context dictionary with auth information

        Returns:
            User/session ID if authenticated, None otherwise
        """
        pass

    @abstractmethod
    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """
        Get permissions for an authenticated user.

        Args:
            user_id: The authenticated user ID

        Returns:
            Dictionary of permissions
        """
        pass
