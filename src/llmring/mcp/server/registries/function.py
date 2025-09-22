"""
Function registry for MCP Server Engine.
Provides in-memory storage and management of registered functions.
"""

from typing import Any, Callable, Dict, List, Optional


class FunctionRegistry:
    """
    Registry for MCP functions.
    Provides in-memory storage and management of functions that can be called via MCP.
    """

    def __init__(self):
        """Initialize an empty function registry."""
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, Optional[Dict[str, Any]]] = {}
        self.descriptions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        func: Callable,
        schema: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a function with the MCP registry.

        Args:
            name: The name of the function
            func: The function to register
            schema: Optional JSON schema for function arguments
            description: Optional description of the function
        """
        self.functions[name] = func
        self.schemas[name] = schema

        # Use provided description or function docstring or default
        if description:
            self.descriptions[name] = description
        elif func.__doc__:
            self.descriptions[name] = func.__doc__.strip()
        else:
            self.descriptions[name] = f"Function {name}"

    def get_function(self, name: str) -> Optional[Callable]:
        """
        Get a registered function by name.

        Args:
            name: The name of the function

        Returns:
            The registered function or None if not found
        """
        return self.functions.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a registered function.

        Args:
            name: The name of the function

        Returns:
            The JSON schema for the function or None if not found
        """
        return self.schemas.get(name)

    def get_description(self, name: str) -> Optional[str]:
        """
        Get the description for a registered function.

        Args:
            name: The name of the function

        Returns:
            The description for the function or None if not found
        """
        return self.descriptions.get(name)

    def list_functions(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered functions with their schemas and descriptions.
        Returns MCP-compliant format with pagination support.

        Args:
            cursor: Optional cursor for pagination

        Returns:
            MCP-compliant tools list response
        """
        tools = [
            {
                "name": name,
                "description": self.descriptions.get(name, f"Function {name}"),
                "inputSchema": self.schemas.get(name),
            }
            for name in self.functions.keys()
        ]

        # Pagination not implemented as typical use cases involve < 100 functions
        return {"tools": tools, "nextCursor": None}

    def unregister(self, name: str) -> bool:
        """
        Unregister a function from the registry.

        Args:
            name: The name of the function

        Returns:
            True if the function was found and removed, False otherwise
        """
        if name in self.functions:
            del self.functions[name]

            if name in self.schemas:
                del self.schemas[name]

            if name in self.descriptions:
                del self.descriptions[name]

            return True
        return False

    def clear(self) -> None:
        """Clear all registered functions."""
        self.functions.clear()
        self.schemas.clear()
        self.descriptions.clear()

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools in MCP format.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": name,
                "description": self.descriptions.get(name, f"Function {name}"),
                "inputSchema": self.schemas.get(name),
            }
            for name in self.functions.keys()
        ]
