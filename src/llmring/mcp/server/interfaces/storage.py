"""
Storage provider interface for MCP Server Engine.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Tool:
    """Tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[str] = None  # Module path or inline function


@dataclass
class Prompt:
    """Prompt definition."""

    name: str
    description: str
    arguments: List[Dict[str, Any]]
    template: str


@dataclass
class Resource:
    """Resource definition."""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    content: Optional[str] = None


class StorageProvider(ABC):
    """Abstract storage provider interface."""

    @abstractmethod
    async def get_tools(
        self, user_id: str, project_id: Optional[str] = None
    ) -> List[Tool]:
        """
        Get tools available to a user.

        Args:
            user_id: The authenticated user ID
            project_id: Optional project ID for filtering

        Returns:
            List of available tools
        """
        pass

    @abstractmethod
    async def get_prompts(
        self, user_id: str, project_id: Optional[str] = None
    ) -> List[Prompt]:
        """
        Get prompts available to a user.

        Args:
            user_id: The authenticated user ID
            project_id: Optional project ID for filtering

        Returns:
            List of available prompts
        """
        pass

    @abstractmethod
    async def get_resources(
        self, user_id: str, project_id: Optional[str] = None
    ) -> List[Resource]:
        """
        Get resources available to a user.

        Args:
            user_id: The authenticated user ID
            project_id: Optional project ID for filtering

        Returns:
            List of available resources
        """
        pass

    @abstractmethod
    async def execute_tool(
        self, user_id: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool with given arguments.

        Args:
            user_id: The authenticated user ID
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    async def read_resource(self, user_id: str, uri: str) -> str:
        """
        Read a resource's content.

        Args:
            user_id: The authenticated user ID
            uri: Resource URI

        Returns:
            Resource content
        """
        pass

    @abstractmethod
    async def get_prompt(
        self, user_id: str, name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get a prompt with arguments filled in.

        Args:
            user_id: The authenticated user ID
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Prompt messages in MCP format
        """
        pass
