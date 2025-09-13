"""
Enhanced LLM Interface for MCP Client - Fixed Version

This module provides an LLM-compatible interface that modules can use to interact
with the MCP client as if it were a smart LLM with tool capabilities.

This version is fully database-agnostic and uses HTTP endpoints exclusively.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator


from llmring.schemas import LLMRequest, LLMResponse, Message, StreamChunk
from llmring.service import LLMRing
from llmring.exceptions import (
    MCPError,
    MCPToolError,
    ServerConnectionError,
    ConversationError,
    ToolExecutionError,
    FileProcessingError,
    NetworkError,
)

from llmring.mcp.client.mcp_client import AsyncMCPClient
from llmring.mcp.client.info_service import create_info_service
from llmring.mcp.http_client import MCPHttpClient
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.client.file_utils import (
    process_file_from_source as process_file_util,
    guess_content_type_from_bytes,
    fetch_file_from_url,
    decode_base64_file,
    create_file_content,
)
from llmring.mcp.client.streaming_handler import StreamingToolHandler

logger = logging.getLogger(__name__)


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

        # Initialize streaming handler for tool support
        self.streaming_handler = StreamingToolHandler(self)
        if mcp_server_url:
            # Parse URL to determine transport type
            if mcp_server_url.startswith("ws://") or mcp_server_url.startswith(
                "wss://"
            ):
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
            tools.append(
                {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "module_name": tool_def.module_name,
                    "parameters": tool_def.parameters,
                }
            )
        return tools

    async def _get_available_tools(self) -> list[dict[str, Any]]:
        """Get all available tools (registered + MCP)."""
        tools = []

        # Add registered tools
        for tool_def in self.registered_tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.parameters,
                    },
                }
            )

        # Add MCP tools if client is available
        if self.mcp_client:
            try:
                mcp_tools = await self.mcp_client.list_tools()
                for tool in mcp_tools:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": f"mcp_{tool['name']}",
                                "description": tool.get("description", ""),
                                "parameters": tool.get("inputSchema", {}),
                            },
                        }
                    )
            except (MCPError, ServerConnectionError, NetworkError) as e:
                # Log but don't fail
                logger.warning(f"Failed to get MCP tools: {e}")
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error getting MCP tools: {e}")

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
            if inspect.iscoroutinefunction(tool_def.handler):
                return await tool_def.handler(**arguments)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: tool_def.handler(**arguments)
                )

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
            if inspect.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**arguments)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: tool_def.handler(**arguments)
                )

            return {"result": result}

        except (ToolExecutionError, MCPToolError) as e:
            return {"error": f"Tool execution failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error in tool execution: {e}"}

    async def chat(
        self,
        messages: list[Message | dict[str, Any]],
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
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
                # Make a copy to avoid mutating the original
                msg = msg.copy()
                # Check for attachments in the message
                attachments = msg.pop("attachments", None)

                # If we have attachments, process them into the message content
                if attachments:
                    # Convert attachments to content parts
                    content_parts = []

                    # Add text content first if present
                    if msg.get("content"):
                        content_parts.append({"type": "text", "text": msg["content"]})

                    # Process each attachment
                    for attachment in attachments:
                        if attachment.get("type") == "file" and attachment.get("data"):
                            # Convert file data to base64 for the API
                            file_data = attachment["data"]
                            if isinstance(file_data, bytes):
                                content_type = attachment.get(
                                    "content_type", "application/octet-stream"
                                )
                                base64_url = create_file_content(
                                    file_data, content_type, as_base64=True
                                )
                                # Extract just the base64 part for compatibility
                                if base64_url.startswith("data:"):
                                    base64_data = (
                                        base64_url.split(",")[1]
                                        if "," in base64_url
                                        else base64_url
                                    )
                                else:
                                    base64_data = base64_url
                            else:
                                base64_data = file_data

                            # Add as image content (for vision-capable models)
                            content_type = attachment.get("content_type", "")
                            if content_type.startswith("image/"):
                                # Use OpenAI format which providers will convert
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{content_type};base64,{base64_data}"
                                        },
                                    }
                                )
                            else:
                                # For non-image files, add as text description
                                content_parts.append(
                                    {
                                        "type": "text",
                                        "text": f"[File: {attachment.get('filename', 'unknown')} ({content_type})]",
                                    }
                                )

                    # Update message content with structured parts
                    msg["content"] = content_parts

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
            stream=stream,
            **kwargs,
        )

        # Handle streaming
        if stream:
            if tools:
                # Streaming with tools - use the handler
                return self.streaming_handler.handle_streaming_with_tools(
                    request,
                    formatted_messages,
                )
            else:
                # Streaming without tools - pass through directly
                return await self.llmring.chat(request)

        # Send to LLM (non-streaming)
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
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result)
                            if not isinstance(result, str)
                            else result,
                        }
                    )
                except (ToolExecutionError, MCPToolError) as e:
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "content": f"Tool execution failed: {e}",
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "content": f"Unexpected error executing tool: {e}",
                        }
                    )

            # Add tool results to messages and get final response
            # Ensure tool call arguments are strings for the message history
            parsed_tool_calls = []
            for tc in response.tool_calls:
                parsed_tc = tc.copy()
                if "function" in parsed_tc and "arguments" in parsed_tc["function"]:
                    args = parsed_tc["function"]["arguments"]
                    # Ensure arguments are strings for the API
                    if not isinstance(args, str):
                        parsed_tc["function"]["arguments"] = json.dumps(args)
                parsed_tool_calls.append(parsed_tc)

            formatted_messages.append(
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=parsed_tool_calls,
                )
            )

            for tool_result in tool_results:
                formatted_messages.append(
                    Message(
                        role="tool",
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"],
                    )
                )

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
            except (ConversationError, ServerConnectionError) as e:
                # Log but don't fail
                logger.warning(f"Failed to store message: {e}")
            except Exception as e:
                # Handle unexpected storage errors
                logger.error(f"Unexpected error storing message: {e}")

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

    # File Processing Methods

    def _guess_content_type_from_bytes(self, file_bytes: bytes) -> str:
        """Guess content type from file bytes using consolidated utility."""
        return guess_content_type_from_bytes(file_bytes)

    async def _fetch_file_from_url(self, url: str) -> tuple[bytes, str]:
        """Fetch a file from a URL using consolidated utility."""
        try:
            file_bytes = fetch_file_from_url(url)
            content_type = guess_content_type_from_bytes(file_bytes)
            return file_bytes, content_type
        except (NetworkError, FileProcessingError) as e:
            raise FileProcessingError(
                f"Failed to fetch file from URL: {e}", details={"url": url}
            )
        except Exception as e:
            raise FileProcessingError(
                f"Unexpected error fetching file: {e}", details={"url": url}
            )

    async def _decode_base64_file(
        self, base64_data: str, content_type: str | None = None
    ) -> tuple[bytes, str]:
        """Decode base64 file data using consolidated utility."""
        return decode_base64_file(base64_data)

    async def process_file_from_source(
        self,
        source_type: str,
        source_data: str,
        filename: str = "unknown",
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a file from different sources (upload, url, base64).

        Args:
            source_type: Type of source ("upload", "url", "base64")
            source_data: Source-specific data:
                - "upload": file path to read from storage
                - "url": URL to fetch from
                - "base64": base64 encoded file data
            filename: Original filename
            content_type: Optional content type hint

        Returns:
            File attachment dictionary ready for LLM processing
        """
        try:
            if source_type == "upload":
                # Use consolidated file utility for local files
                result = process_file_util(source_data, source_type="path")
                file_bytes = result["content"]
                content_type = content_type or result["content_type"]

            elif source_type == "url":
                # Fetch file from URL
                file_bytes, detected_content_type = await self._fetch_file_from_url(
                    source_data
                )
                content_type = content_type or detected_content_type

            elif source_type == "base64":
                # Decode base64 data
                file_bytes, detected_content_type = await self._decode_base64_file(
                    source_data, content_type
                )
                content_type = content_type or detected_content_type

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            return {
                "type": "file",
                "filename": filename,
                "content_type": content_type,
                "data": file_bytes,
                "source_type": source_type,
                "source_data": source_data,
            }

        except FileProcessingError:
            raise  # Re-raise file processing errors as-is
        except Exception as e:
            raise FileProcessingError(
                f"Failed to process file from {source_type}: {e}",
                details={"source_type": source_type, "filename": filename},
            )

    # Information Service Methods

    def get_available_providers(self) -> list[dict[str, Any]]:
        """
        Get information about all available LLM providers.

        Returns:
            List of provider information dictionaries
        """
        # Use the info service to get provider information
        if self.info_service:
            providers = self.info_service.get_available_providers()
            # Convert to dict format
            return [
                {
                    "name": provider.name,
                    "description": provider.description,
                    "capabilities": provider.capabilities,
                    "status": provider.status,
                }
                for provider in providers
            ]
        else:
            # Default providers if info service not available
            return [
                {
                    "name": "anthropic",
                    "description": "Anthropic Claude models",
                    "capabilities": ["chat", "tools", "vision"],
                    "status": "available",
                },
                {
                    "name": "openai",
                    "description": "OpenAI GPT models",
                    "capabilities": ["chat", "tools", "vision"],
                    "status": "available",
                },
            ]

    async def get_transparency_report(
        self, user_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get a comprehensive transparency report for a user.

        Args:
            user_id: User identifier (uses default if not provided)

        Returns:
            Complete transparency report including all information
        """
        user_id = user_id or self.default_user_id

        # Gather information for the report
        from datetime import datetime, UTC

        report = {
            "user_id": user_id,
            "llm_model": self.llm_model,
            "origin": self.origin,
            "registered_tools": self.list_registered_tools(),
            "mcp_connected": self.mcp_client is not None,
            "current_conversation_id": self.current_conversation_id,
            "available_providers": self.get_available_providers(),
            "usage_stats": await self.get_usage_stats(user_id),
            "enhanced_llm_config": {
                "model": self.llm_model,
                "default_model": self.llm_model,
                "origin": self.origin,
                "registered_tools": self.list_registered_tools(),
                "mcp_enabled": self.mcp_client is not None,
                "mcp_server_connected": self.mcp_client is not None,
            },
            "generated_at": datetime.now(UTC).isoformat(),
            "data_storage": self.get_data_storage_info(),
            "user_data_summary": self.get_user_data_summary(user_id),
        }

        # Add MCP server information if connected
        if self.mcp_client:
            try:
                mcp_tools = await self.mcp_client.list_tools()
                report["mcp_tools"] = [
                    {"name": tool.get("name"), "description": tool.get("description")}
                    for tool in mcp_tools
                ]
            except (MCPError, ServerConnectionError) as e:
                logger.warning(f"Failed to get MCP tools for transparency report: {e}")
                report["mcp_tools"] = []
            except Exception as e:
                logger.error(
                    f"Unexpected error getting MCP tools for transparency report: {e}"
                )
                report["mcp_tools"] = []

        return report

    def get_data_storage_info(self) -> dict[str, Any]:
        """
        Get information about data storage and privacy.

        Returns:
            Information about how data is stored and managed
        """
        return {
            "storage_type": "http_api",
            "database_connection": False,
            "persistence_url": self.http_client.base_url
            if hasattr(self, "http_client") and self.http_client
            else None,
            "conversation_storage": "server" if self.conversation_manager else "none",
            "privacy_notes": "All data is stored via HTTP API endpoints. No local database access.",
            "mcp_client_tables": [
                "conversations",
                "messages",
                "tool_calls",
                "usage",
            ],  # Logical tables via API
            "llm_service_tables": [
                "providers",
                "models",
                "usage_stats",
            ],  # LLM service logical tables
        }

    def get_user_data_summary(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Get a summary of user data stored in the system.

        Args:
            user_id: User ID (uses default if not provided)

        Returns:
            Summary of user data
        """
        user_id = user_id or self.default_user_id

        return {
            "user_id": user_id,
            "origin": self.origin,
            "has_conversations": self.current_conversation_id is not None,
            "conversation_count": "unknown",  # Would need API endpoint
            "message_count": "unknown",  # Would need API endpoint
            "tool_usage_count": len(self.registered_tools),
            "data_retention": "Server-managed",
            "export_available": True,
            "deletion_available": True,
        }
