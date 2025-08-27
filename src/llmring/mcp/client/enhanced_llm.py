"""
Enhanced LLM Interface for MCP Client - Fixed Version

This module provides an LLM-compatible interface that modules can use to interact
with the MCP client as if it were a smart LLM with tool capabilities.

This version is fully database-agnostic and uses HTTP endpoints exclusively.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing

from llmring.mcp.client.mcp_client import AsyncMCPClient
from llmring.mcp.client.info_service import create_info_service
from llmring.mcp.http_client import MCPHttpClient
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager


def create_enhanced_llm(
    llm_model: str = "anthropic:claude-3-haiku-20240307",
    llmring_server_url: str | None = None,
    mcp_server_url: str | None = None,
    origin: str = "enhanced-llm",
    user_id: str | None = None,
    api_key: str | None = None,
) -> "EnhancedLLM":
    """
    Factory function to create an EnhancedLLM instance.
    
    Args:
        llm_model: The underlying LLM model to use
        llmring_server_url: LLMRing server URL for persistence
        mcp_server_url: Optional MCP server URL for additional tools
        origin: Origin identifier for usage tracking
        user_id: Default user ID for requests
        api_key: Optional API key for LLMRing server
        
    Returns:
        Configured EnhancedLLM instance
    """
    return EnhancedLLM(
        llm_model=llm_model,
        llmring_server_url=llmring_server_url,
        mcp_server_url=mcp_server_url,
        origin=origin,
        user_id=user_id,
        api_key=api_key,
    )


@dataclass
class ToolDefinition:
    """Definition of a tool that can be registered with the enhanced LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable
    module_name: str | None = None


class EnhancedLLM:
    """
    Enhanced LLM interface that combines LLM capabilities with MCP tool execution.
    
    This version is fully database-agnostic and uses HTTP endpoints for all persistence.
    """

    def __init__(
        self,
        llm_model: str = "anthropic:claude-3-haiku-20240307",
        llmring_server_url: str | None = None,
        mcp_server_url: str | None = None,
        origin: str = "enhanced-llm",
        user_id: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the Enhanced LLM.

        Args:
            llm_model: The underlying LLM model to use
            llmring_server_url: LLMRing server URL for persistence
            mcp_server_url: Optional MCP server URL for additional tools
            origin: Origin identifier for usage tracking
            user_id: Default user ID for requests
            api_key: Optional API key for LLMRing server
        """
        self.llm_model = llm_model
        self.origin = origin
        self.default_user_id = user_id or "enhanced-llm-user"
        self.api_key = api_key
        
        # Initialize LLM service
        self.llmring = LLMRing(origin=origin)
        
        # Initialize HTTP client for MCP operations
        self.http_client = MCPHttpClient(
            base_url=llmring_server_url,
            api_key=api_key,
        )
        
        # Initialize conversation manager
        self.conversation_manager = AsyncConversationManager(
            llmring_server_url=llmring_server_url,
            api_key=api_key,
        )
        
        # Initialize MCP client if server URL provided
        self.mcp_client: AsyncMCPClient | None = None
        if mcp_server_url:
            # Parse URL to determine transport type
            if mcp_server_url.startswith("ws://") or mcp_server_url.startswith("wss://"):
                self.mcp_client = AsyncMCPClient.websocket(mcp_server_url)
            elif mcp_server_url.startswith("stdio://"):
                # Extract command from URL
                command = mcp_server_url.replace("stdio://", "").split()
                self.mcp_client = AsyncMCPClient.stdio(command)
            else:
                # Default to HTTP
                self.mcp_client = AsyncMCPClient.http(mcp_server_url)
        
        # Create info service for system information
        self.info_service = create_info_service()
        
        # Registered tools from modules
        self.registered_tools: dict[str, ToolDefinition] = {}
        
        # Current conversation context
        self.current_conversation_id: str | None = None
        self.conversation_history: list[Message] = []

    async def initialize(self) -> None:
        """Initialize the MCP client if configured."""
        if self.mcp_client:
            await self.mcp_client.initialize()
            
    async def close(self) -> None:
        """Clean up resources."""
        if self.mcp_client:
            await self.mcp_client.close()
        await self.http_client.close()

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable,
        module_name: str | None = None,
    ) -> None:
        """
        Register a tool that can be used by the LLM.
        
        Args:
            name: Tool name
            description: Tool description for the LLM
            parameters: JSON schema for tool parameters
            handler: Async function to execute the tool
            module_name: Optional module name for grouping
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if name in self.registered_tools:
            raise ValueError(f"Tool '{name}' is already registered")
            
        self.registered_tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            module_name=module_name,
        )
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if it wasn't registered
        """
        if name in self.registered_tools:
            del self.registered_tools[name]
            return True
        return False
    
    def list_registered_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools.
        
        Returns:
            List of tool descriptions with name, description, module_name, and parameters
        """
        tools = []
        for tool_def in self.registered_tools.values():
            tools.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "module_name": tool_def.module_name,
                "parameters": tool_def.parameters,
            })
        return tools

    async def _get_available_tools(self) -> list[dict[str, Any]]:
        """Get all available tools (registered + MCP)."""
        tools = []
        
        # Add registered tools
        for tool_def in self.registered_tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                }
            })
        
        # Add MCP tools if client is available
        if self.mcp_client:
            try:
                mcp_tools = await self.mcp_client.list_tools()
                for tool in mcp_tools:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": f"mcp_{tool['name']}",
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema", {}),
                        }
                    })
            except Exception as e:
                # Log but don't fail
                print(f"Failed to get MCP tools: {e}")
        
        return tools

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Check if it's a registered tool
        if tool_name in self.registered_tools:
            tool_def = self.registered_tools[tool_name]
            # Check if handler is async
            import asyncio
            import inspect
            
            if inspect.iscoroutinefunction(tool_def.handler):
                return await tool_def.handler(**arguments)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: tool_def.handler(**arguments))
        
        # Check if it's an MCP tool
        if tool_name.startswith("mcp_") and self.mcp_client:
            actual_name = tool_name[4:]  # Remove "mcp_" prefix
            return await self.mcp_client.call_tool(actual_name, arguments)
        
        raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _execute_registered_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a registered tool by name and handle errors.
        
        Args:
            tool_name: Name of the registered tool to execute
            arguments: Tool arguments
            
        Returns:
            Result dict with either success result or error message
        """
        try:
            if tool_name not in self.registered_tools:
                return {"error": f"Tool '{tool_name}' is not registered"}
            
            tool_def = self.registered_tools[tool_name]
            # Check if handler is async
            import asyncio
            import inspect
            
            if inspect.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**arguments)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool_def.handler(**arguments))
            
            return {"result": result}
            
        except Exception as e:
            return {"error": str(e)}

    async def chat(
        self,
        messages: list[Message | dict[str, Any]],
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a conversation to the enhanced LLM and get a response.
        
        This method is compatible with the standard LLMRing interface.
        
        Args:
            messages: List of conversation messages
            user_id: User ID for tracking
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with content and usage information
        """
        # Convert messages to Message objects if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(Message(**msg))
            else:
                formatted_messages.append(msg)
        
        # Get available tools
        tools = await self._get_available_tools()
        
        # Create LLM request
        request = LLMRequest(
            model=self.llm_model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            user_id=user_id or self.default_user_id,
            **kwargs,
        )
        
        # Send to LLM
        response = await self.llmring.chat(request)
        
        # Handle tool calls if present
        if response.tool_calls:
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                try:
                    # Parse arguments if they're a string
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    
                    result = await self._execute_tool(
                        tool_call["function"]["name"],
                        args,
                    )
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "content": f"Error executing tool: {e}",
                    })
            
            # Add tool results to messages and get final response
            # Parse tool call arguments to ensure they're dicts
            parsed_tool_calls = []
            for tc in response.tool_calls:
                parsed_tc = tc.copy()
                if "function" in parsed_tc and "arguments" in parsed_tc["function"]:
                    args = parsed_tc["function"]["arguments"]
                    if isinstance(args, str):
                        parsed_tc["function"]["arguments"] = json.loads(args)
                parsed_tool_calls.append(parsed_tc)
            
            formatted_messages.append(Message(
                role="assistant",
                content=response.content or "",
                tool_calls=parsed_tool_calls,
            ))
            
            for tool_result in tool_results:
                formatted_messages.append(Message(
                    role="tool",
                    content=tool_result["content"],
                    tool_call_id=tool_result["tool_call_id"],
                ))
            
            # Get final response after tool execution
            final_request = LLMRequest(
                model=self.llm_model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id or self.default_user_id,
                **kwargs,
            )
            response = await self.llmring.chat(final_request)
        
        # Store in conversation history if we have a conversation
        if self.current_conversation_id:
            try:
                await self.conversation_manager.add_message(
                    conversation_id=self.current_conversation_id,
                    user_id=user_id or self.default_user_id,
                    role="assistant",
                    content=response.content or "",
                    metadata={
                        "model": self.llm_model,
                        "usage": response.usage,
                    },
                )
            except Exception as e:
                # Log but don't fail
                print(f"Failed to store message: {e}")
        
        return response

    async def create_conversation(
        self,
        title: str | None = None,
        system_prompt: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            system_prompt: System prompt
            user_id: User ID
            
        Returns:
            Conversation ID
        """
        conversation_id = await self.conversation_manager.create_conversation(
            user_id=user_id or self.default_user_id,
            title=title,
            system_prompt=system_prompt,
            model=self.llm_model,
        )
        self.current_conversation_id = conversation_id
        return conversation_id

    async def load_conversation(
        self,
        conversation_id: str,
        user_id: str | None = None,
    ) -> None:
        """
        Load an existing conversation.
        
        Args:
            conversation_id: Conversation ID to load
            user_id: User ID for verification
        """
        conversation = await self.conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id or self.default_user_id,
        )
        
        if conversation:
            self.current_conversation_id = conversation_id
            self.conversation_history = conversation.messages
        else:
            raise ValueError(f"Conversation {conversation_id} not found")

    async def list_conversations(
        self,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        List recent conversations.
        
        Args:
            user_id: User ID
            limit: Maximum number of conversations
            
        Returns:
            List of conversation summaries
        """
        return await self.conversation_manager.list_conversations(
            user_id=user_id or self.default_user_id,
            limit=limit,
        )
    
    async def get_usage_stats(
        self,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get usage statistics for the user.
        
        Args:
            user_id: User ID (uses default if not provided)
            
        Returns:
            Dictionary with usage statistics
        """
        # Placeholder implementation - would need server endpoint
        return {
            "user_id": user_id or self.default_user_id,
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "note": "Usage statistics not yet implemented - requires server endpoint",
        }