"""
Resource registry for MCP Server Engine.
Provides management of resources that can be exposed via MCP.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional


class ResourceRegistry:
    """
    Registry for MCP resources.
    Provides management of resources that can be read via MCP.
    """

    def __init__(self):
        """Initialize an empty resource registry."""
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "text/plain",
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register a resource with the MCP registry.

        Args:
            uri: The URI of the resource (e.g., "file:///path/to/file")
            name: Human-readable name of the resource
            description: Description of the resource
            mime_type: MIME type of the resource content
            handler: Optional function to handle resource reading
        """
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description,
            "mimeType": mime_type,
        }

        if handler:
            self.handlers[uri] = handler

    def register_static_resource(
        self,
        uri: str,
        name: str,
        description: str,
        content: str,
        mime_type: str = "text/plain",
    ) -> None:
        """
        Register a static resource with fixed content.

        Args:
            uri: The URI of the resource
            name: Human-readable name of the resource
            description: Description of the resource
            content: Static content of the resource
            mime_type: MIME type of the resource content
        """

        async def static_handler() -> str:
            return content

        self.register_resource(uri, name, description, mime_type, static_handler)

    def get_resource_info(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Get resource information by URI.

        Args:
            uri: The URI of the resource

        Returns:
            Resource information or None if not found
        """
        return self.resources.get(uri)

    def get_handler(self, uri: str) -> Optional[Callable]:
        """
        Get the handler function for a resource.

        Args:
            uri: The URI of the resource

        Returns:
            Handler function or None if not found
        """
        return self.handlers.get(uri)

    async def read_resource(self, uri: str) -> str:
        """
        Read the content of a resource.

        Args:
            uri: The URI of the resource

        Returns:
            Resource content as string

        Raises:
            ValueError: If resource not found or handler fails
        """
        handler = self.get_handler(uri)
        if not handler:
            raise ValueError(f"Resource not found: {uri}")

        try:
            if inspect.iscoroutinefunction(handler):
                return await handler()
            else:
                return handler()
        except Exception as e:
            raise ValueError(f"Error reading resource {uri}: {str(e)}")

    def list_resources(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered resources.
        Returns MCP-compliant format with pagination support.

        Args:
            cursor: Optional cursor for pagination

        Returns:
            MCP-compliant resources list response
        """
        resources = list(self.resources.values())

        # Pagination not implemented as typical use cases involve < 100 resources
        return {"resources": resources, "nextCursor": None}

    def unregister_resource(self, uri: str) -> bool:
        """
        Unregister a resource from the registry.

        Args:
            uri: The URI of the resource

        Returns:
            True if the resource was found and removed, False otherwise
        """
        if uri in self.resources:
            del self.resources[uri]

            if uri in self.handlers:
                del self.handlers[uri]

            return True
        return False

    def clear(self) -> None:
        """Clear all registered resources."""
        self.resources.clear()
        self.handlers.clear()

    def get_all_resources(self) -> List[Dict[str, Any]]:
        """
        Get all resources in MCP format.

        Returns:
            List of resource definitions
        """
        return list(self.resources.values())
