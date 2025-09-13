"""
Prompt registry for MCP Server Engine.
Provides management of prompt templates that can be exposed via MCP.
"""

from typing import Any, Callable, Dict, List, Optional
import inspect


class PromptRegistry:
    """
    Registry for MCP prompts.
    Provides management of prompt templates that can be retrieved via MCP.
    """

    def __init__(self):
        """Initialize an empty prompt registry."""
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register a prompt template with the MCP registry.

        Args:
            name: The name of the prompt
            description: Description of the prompt
            arguments: List of argument definitions for the prompt
            handler: Optional function to handle prompt generation
        """
        self.prompts[name] = {
            "name": name,
            "description": description,
            "arguments": arguments or [],
        }

        if handler:
            self.handlers[name] = handler

    def register_static_prompt(
        self,
        name: str,
        description: str,
        content: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Register a static prompt with fixed content.

        Args:
            name: The name of the prompt
            description: Description of the prompt
            content: Static content of the prompt
            arguments: List of argument definitions for the prompt
        """

        async def static_handler(args: Dict[str, Any]) -> Dict[str, Any]:
            # Simple template substitution for static prompts
            formatted_content = content
            for key, value in args.items():
                formatted_content = formatted_content.replace(f"{{{key}}}", str(value))

            return {
                "description": description,
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": formatted_content},
                    }
                ],
            }

        self.register_prompt(name, description, arguments, static_handler)

    def get_prompt_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt information by name.

        Args:
            name: The name of the prompt

        Returns:
            Prompt information or None if not found
        """
        return self.prompts.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """
        Get the handler function for a prompt.

        Args:
            name: The name of the prompt

        Returns:
            Handler function or None if not found
        """
        return self.handlers.get(name)

    async def get_prompt(
        self, name: str, arguments: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get a prompt with the given arguments.

        Args:
            name: The name of the prompt
            arguments: Arguments to pass to the prompt handler

        Returns:
            Prompt response with messages

        Raises:
            ValueError: If prompt not found or handler fails
        """
        handler = self.get_handler(name)
        if not handler:
            raise ValueError(f"Prompt not found: {name}")

        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(arguments or {})
            else:
                return handler(arguments or {})
        except Exception as e:
            raise ValueError(f"Error generating prompt {name}: {str(e)}")

    def list_prompts(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered prompts.
        Returns MCP-compliant format with pagination support.

        Args:
            cursor: Optional cursor for pagination

        Returns:
            MCP-compliant prompts list response
        """
        prompts = list(self.prompts.values())

        # Pagination not implemented as typical use cases involve < 100 prompts
        return {"prompts": prompts, "nextCursor": None}

    def unregister_prompt(self, name: str) -> bool:
        """
        Unregister a prompt from the registry.

        Args:
            name: The name of the prompt

        Returns:
            True if the prompt was found and removed, False otherwise
        """
        if name in self.prompts:
            del self.prompts[name]

            if name in self.handlers:
                del self.handlers[name]

            return True
        return False

    def clear(self) -> None:
        """Clear all registered prompts."""
        self.prompts.clear()
        self.handlers.clear()

    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all prompts in MCP format.

        Returns:
            List of prompt definitions
        """
        return list(self.prompts.values())
