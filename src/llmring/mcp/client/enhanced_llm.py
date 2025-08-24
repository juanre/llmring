"""
Enhanced LLM Interface for MCP Client

This module provides an LLM-compatible interface that modules can use to interact
with the MCP client as if it were a smart LLM with tool capabilities.

Modules can:
1. Register their own tools
2. Send the same messages they would send to an LLM
3. Get responses that may include tool usage from both registered tools and MCP servers
"""

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from llmbridge.file_utils import create_file_content, get_file_mime_type
from llmbridge.schemas import LLMRequest, LLMResponse, Message
from llmbridge.service import LLMBridge

from llmring.mcp.server.client.client import AsyncMCPClient
from llmring.mcp.server.client.db import create_mcp_db_manager
from llmring.mcp.server.client.info_service import create_info_service


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

    This class provides an LLM-compatible interface that modules can use to:
    - Register their own tools
    - Send conversation messages (system + user/assistant messages)
    - Get intelligent responses that may use both registered tools and MCP server tools
    - Track usage and costs like a regular LLM
    """

    def __init__(
        self,
        llm_model: str = "anthropic:claude-3-haiku-20240307",
        db_connection_string: str | None = None,
        mcp_server_url: str | None = None,
        origin: str = "enhanced-llm",
        user_id: str | None = None,
    ):
        """
        Initialize the Enhanced LLM.

        Args:
            llm_model: The underlying LLM model to use
            db_connection_string: Database connection for conversation storage
            mcp_server_url: Optional MCP server URL for additional tools
            origin: Origin identifier for usage tracking
            user_id: Default user ID for requests
        """
        self.llm_model = llm_model
        self.origin = origin
        self.default_user_id = user_id or "enhanced-llm-user"

        # Initialize LLM service
        self.llmbridge = LLMBridge(
            db_connection_string=db_connection_string,
            origin=origin,
            enable_db_logging=bool(db_connection_string),
        )

        # Initialize MCP client if server URL provided
        self.mcp_client = None
        if mcp_server_url:
            self.mcp_client = AsyncMCPClient.http(mcp_server_url)

        # Registry for module-registered tools
        self.registered_tools: dict[str, ToolDefinition] = {}

        # Database for conversation storage (optional)
        self.db_manager = None
        if db_connection_string:
            self.db_manager = create_mcp_db_manager(db_connection_string)

        # Initialize info service for transparency
        self.info_service = create_info_service(
            llmbridge=self.llmbridge,
            db_connection_string=db_connection_string,
            origin=origin,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable,
        module_name: str | None = None,
    ) -> None:
        """
        Register a tool that the enhanced LLM can use.

        Args:
            name: Tool name (must be unique)
            description: Description of what the tool does
            parameters: JSON schema for tool parameters
            handler: Function to call when tool is invoked
            module_name: Optional module name for organization

        Example:
            enhanced_llm.register_tool(
                name="calculate",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                },
                handler=my_calculator_function
            )
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
            name: Name of tool to unregister

        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self.registered_tools:
            del self.registered_tools[name]
            return True
        return False

    def list_registered_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "module_name": tool.module_name,
            }
            for tool in self.registered_tools.values()
        ]

    async def _process_file_attachments(
        self, attachments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Process file attachments and convert them to LLM-compatible format.

        Args:
            attachments: List of file attachment dictionaries with format:
                {
                    "type": "file",
                    "filename": "document.pdf",
                    "content_type": "application/pdf",
                    "data": b"raw_file_bytes",
                    "parameter_name": "pdf_file",  # Which agent parameter this came from
                    "url": "optional_source_url"   # If fetched from URL
                }

        Returns:
            List of content parts for LLM message
        """
        content_parts = []

        for attachment in attachments:
            try:
                file_data = attachment.get("data")
                content_type = attachment.get("content_type", "application/octet-stream")
                filename = attachment.get("filename", "unknown")

                if not file_data:
                    continue

                # Convert bytes to base64 if needed
                if isinstance(file_data, bytes):
                    base64_data = base64.b64encode(file_data).decode("utf-8")
                elif isinstance(file_data, str):
                    # Assume it's already base64 encoded
                    base64_data = file_data
                else:
                    continue

                # Use universal file interface that works for all file types
                try:
                    file_content = create_file_content(
                        base64_data,
                        f"Please analyze this {filename} file.",
                        content_type,
                    )
                    content_parts.extend(file_content)
                except Exception as file_error:
                    # Fallback for unsupported file types
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"Unable to process {content_type} file: {filename}. Error: {file_error!s}",
                        }
                    )

            except Exception as e:
                # If file processing fails, add an error note
                content_parts.append(
                    {
                        "type": "text",
                        "text": f"Error processing file attachment {attachment.get('filename', 'unknown')}: {e!s}",
                    }
                )

        return content_parts

    async def _fetch_file_from_url(self, url: str) -> tuple[bytes, str]:
        """
        Fetch a file from a URL.

        Args:
            url: URL to fetch file from

        Returns:
            Tuple of (file_bytes, content_type)
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "application/octet-stream")
            # Clean up content type (remove charset etc.)
            if ";" in content_type:
                content_type = content_type.split(";")[0].strip()

            return response.content, content_type

    async def _decode_base64_file(
        self, base64_data: str, content_type: str | None = None
    ) -> tuple[bytes, str]:
        """
        Decode base64 file data.

        Args:
            base64_data: Base64 encoded file data
            content_type: Optional content type, will try to guess if not provided

        Returns:
            Tuple of (file_bytes, content_type)
        """
        try:
            file_bytes = base64.b64decode(base64_data)
            if not content_type:
                # Try to guess content type from first few bytes (magic numbers)
                content_type = self._guess_content_type_from_bytes(file_bytes)
            return file_bytes, content_type
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

    def _guess_content_type_from_bytes(self, file_bytes: bytes) -> str:
        """
        Guess content type from file magic numbers.

        Args:
            file_bytes: Raw file bytes

        Returns:
            Guessed content type
        """
        if len(file_bytes) < 4:
            return "application/octet-stream"

        # Check common file signatures
        if file_bytes.startswith(b"\x89PNG"):
            return "image/png"
        elif file_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif file_bytes.startswith(b"GIF8"):
            return "image/gif"
        elif file_bytes.startswith(b"RIFF") and b"WEBP" in file_bytes[:12]:
            return "image/webp"
        elif file_bytes.startswith(b"%PDF"):
            return "application/pdf"
        elif file_bytes.startswith(b"PK\x03\x04"):
            # ZIP-based formats (could be DOCX, etc.)
            return "application/zip"
        else:
            return "application/octet-stream"

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
                # Read file from local storage
                with open(source_data, "rb") as f:
                    file_bytes = f.read()
                if not content_type:
                    content_type = get_file_mime_type(source_data)

            elif source_type == "url":
                # Fetch file from URL
                file_bytes, detected_content_type = await self._fetch_file_from_url(source_data)
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

        except Exception as e:
            raise ValueError(f"Failed to process file from {source_type}: {e}")

    async def chat(
        self,
        messages: list[Message | dict[str, Any]],
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a conversation to the enhanced LLM and get a response.

        This is the main interface that modules use - it's compatible with
        regular LLM interfaces but adds tool execution capabilities.

        Args:
            messages: List of conversation messages (system, user, assistant)
            user_id: User ID for tracking (uses default if not provided)
            temperature: LLM temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters passed to LLM

        Returns:
            LLMResponse with content and usage information

        Example:
            response = await enhanced_llm.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Calculate 15 * 23 and tell me about it."}
            ])
        """
        user_id = user_id or self.default_user_id

        # Convert dict messages to Message objects and process file attachments
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Check for file attachments in this message
                if "attachments" in msg:
                    attachments = msg.pop("attachments")  # Remove from message dict

                    # Process the file attachments
                    file_content_parts = await self._process_file_attachments(attachments)

                    # If we have file attachments, modify the message content
                    if file_content_parts:
                        original_content = msg.get("content", "")

                        # Create structured content with text and file attachments
                        if isinstance(original_content, str):
                            # Start with text content
                            content_parts = (
                                [{"type": "text", "text": original_content}]
                                if original_content
                                else []
                            )
                            # Add file content parts
                            content_parts.extend(file_content_parts)
                            msg["content"] = content_parts
                        elif isinstance(original_content, list):
                            # Content is already structured, append file parts
                            msg["content"] = original_content + file_content_parts

                # Create Message object (without attachments field)
                normalized_messages.append(Message(**msg))
            else:
                normalized_messages.append(msg)

        # Prepare tools for LLM (both registered tools and MCP tools)
        tools = await self._prepare_tools()

        # Create LLM request
        request = LLMRequest(
            messages=normalized_messages,
            model=self.llm_model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Get initial LLM response
        response = await self.llmbridge.chat(request, id_at_origin=user_id)

        # Handle tool calls if present
        if response.tool_calls:
            # Execute tools and continue conversation
            response = await self._handle_tool_calls(
                normalized_messages,
                response,
                user_id,
                temperature,
                max_tokens,
                **kwargs,
            )

        return response

    async def _prepare_tools(self) -> list[dict[str, Any]]:
        """Prepare tool definitions for the LLM."""
        tools = []

        # Add registered tools
        for tool in self.registered_tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )

        # Add MCP server tools if available
        if self.mcp_client:
            try:
                # Get tools from MCP server
                await self.mcp_client.connect()
                mcp_tools = await self.mcp_client.list_tools()

                for mcp_tool in mcp_tools.tools:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": f"mcp_{mcp_tool.name}",  # Prefix to avoid conflicts
                                "description": mcp_tool.description
                                or f"MCP tool: {mcp_tool.name}",
                                "parameters": mcp_tool.inputSchema
                                or {"type": "object", "properties": {}},
                            },
                        }
                    )
            except Exception:
                # MCP server not available, continue without MCP tools
                pass

        return tools

    async def _handle_tool_calls(
        self,
        original_messages: list[Message],
        initial_response: LLMResponse,
        user_id: str,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs,
    ) -> LLMResponse:
        """Handle tool calls and continue the conversation."""

        # Execute all requested tools
        tool_results = []
        for tool_call in initial_response.tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            try:
                # Check if it's a registered tool
                if tool_name in self.registered_tools:
                    result = await self._execute_registered_tool(tool_name, arguments)
                elif tool_name.startswith("mcp_") and self.mcp_client:
                    # MCP tool (remove prefix)
                    mcp_tool_name = tool_name[4:]  # Remove "mcp_" prefix
                    result = await self._execute_mcp_tool(mcp_tool_name, arguments)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({"tool_call_id": tool_call["id"], "result": result})

            except Exception as e:
                tool_results.append({"tool_call_id": tool_call["id"], "result": {"error": str(e)}})

        # Build new conversation with tool results
        new_messages = original_messages.copy()

        # Add assistant message with tool calls
        # Note: For now, we'll just include the content and handle tool results directly
        # The LLM service will handle the tool call formatting internally
        new_messages.append(
            Message(
                role="assistant",
                content=initial_response.content or "I'll help you with that.",
            )
        )

        # Add tool result messages as user messages with context
        tool_results_text = []
        for tool_result in tool_results:
            result_data = tool_result["result"]
            if "result" in result_data:
                tool_results_text.append(f"Tool result: {result_data['result']}")
            elif "error" in result_data:
                tool_results_text.append(f"Tool error: {result_data['error']}")

        if tool_results_text:
            combined_results = "\\n".join(tool_results_text)
            new_messages.append(
                Message(
                    role="user",
                    content=f"Based on the tool results: {combined_results}\\n\\nPlease provide a comprehensive response to my original question.",
                )
            )

        # Get final response from LLM
        final_request = LLMRequest(
            messages=new_messages,
            model=self.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        final_response = await self.llmbridge.chat(final_request, id_at_origin=user_id)

        # Combine usage statistics
        if hasattr(initial_response, "usage") and hasattr(final_response, "usage"):
            final_response.usage = {
                "prompt_tokens": initial_response.usage.get("prompt_tokens", 0)
                + final_response.usage.get("prompt_tokens", 0),
                "completion_tokens": initial_response.usage.get("completion_tokens", 0)
                + final_response.usage.get("completion_tokens", 0),
                "total_tokens": initial_response.usage.get("total_tokens", 0)
                + final_response.usage.get("total_tokens", 0),
            }

        return final_response

    async def _execute_registered_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a registered tool."""
        tool = self.registered_tools[tool_name]

        try:
            # Call the tool handler
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)

            return {"result": result, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _execute_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP server tool."""
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            return {"result": result.content, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def get_usage_stats(
        self, user_id: str | None = None, days: int = 30
    ) -> dict[str, Any] | None:
        """
        Get usage statistics for the enhanced LLM.

        Args:
            user_id: User ID to get stats for (uses default if not provided)
            days: Number of days to look back

        Returns:
            Usage statistics or None if not available
        """
        user_id = user_id or self.default_user_id

        if self.llmbridge:
            stats = await self.llmbridge.get_usage_stats(user_id, days=days)
            if stats:
                return {
                    "total_calls": stats.total_calls,
                    "total_tokens": stats.total_tokens,
                    "total_cost": float(stats.total_cost),
                    "avg_cost_per_call": float(stats.avg_cost_per_call),
                    "most_used_model": stats.most_used_model,
                    "success_rate": float(stats.success_rate),
                    "avg_response_time_ms": stats.avg_response_time_ms,
                }
        return None

    async def close(self) -> None:
        """Clean up resources."""
        if self.llmbridge:
            await self.llmbridge.close()
        if self.mcp_client:
            await self.mcp_client.disconnect()

    # Transparency and Information Methods

    def get_available_providers(self) -> list[dict[str, Any]]:
        """
        Get information about all available LLM providers.

        Returns:
            List of provider information dictionaries
        """
        providers = self.info_service.get_available_providers()
        return [self.info_service.to_dict(provider) for provider in providers]

    def get_models_for_provider(self, provider: str) -> list[dict[str, Any]]:
        """
        Get detailed information about models for a specific provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')

        Returns:
            List of model information dictionaries
        """
        models = self.info_service.get_models_for_provider(provider)
        return [self.info_service.to_dict(model) for model in models]

    def get_model_cost_info(self, model_identifier: str) -> dict[str, Any] | None:
        """
        Get cost information for a specific model.

        Args:
            model_identifier: Either "provider:model" or just "model"

        Returns:
            Dictionary with cost information or None if not found
        """
        return self.info_service.get_model_cost_info(model_identifier)

    async def get_enhanced_usage_stats(
        self, user_id: str | None = None, days: int = 30
    ) -> dict[str, Any] | None:
        """
        Get comprehensive usage statistics including tool usage.

        Args:
            user_id: User ID to get stats for (uses default if not provided)
            days: Number of days to look back

        Returns:
            Comprehensive usage statistics or None if not available
        """
        user_id = user_id or self.default_user_id
        # info_service.get_usage_stats is synchronous; call directly
        usage_stats = self.info_service.get_usage_stats(user_id, days=days)

        if usage_stats:
            stats_dict = self.info_service.to_dict(usage_stats)
            # Add enhanced LLM specific information
            stats_dict["enhanced_llm_info"] = {
                "registered_tools_count": len(self.registered_tools),
                "mcp_server_connected": self.mcp_client is not None,
                "origin": self.origin,
                "default_model": self.llm_model,
            }
            return stats_dict

        return None

    def get_data_storage_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about what data is stored where.

        Returns:
            Dictionary describing all data storage locations and policies
        """
        storage_info = self.info_service.get_data_storage_info()
        return self.info_service.to_dict(storage_info)

    def get_user_data_summary(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Get a summary of all data stored for a specific user.

        Args:
            user_id: User identifier (uses default if not provided)

        Returns:
            Dictionary summarizing all stored user data
        """
        user_id = user_id or self.default_user_id
        return self.info_service.get_user_data_summary(user_id)

    async def get_transparency_report(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Get a comprehensive transparency report for a user.

        Args:
            user_id: User identifier (uses default if not provided)

        Returns:
            Complete transparency report including all information
        """
        user_id = user_id or self.default_user_id

        return {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "enhanced_llm_config": {
                "origin": self.origin,
                "default_model": self.llm_model,
                "registered_tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "module_name": tool.module_name,
                    }
                    for tool in self.registered_tools.values()
                ],
                "mcp_server_connected": self.mcp_client is not None,
            },
            "available_providers": self.get_available_providers(),
            "usage_statistics": await self.get_enhanced_usage_stats(user_id),
            "data_storage": self.get_data_storage_info(),
            "user_data_summary": self.get_user_data_summary(user_id),
        }


# Convenience function for quick setup
def create_enhanced_llm(
    llm_model: str = "anthropic:claude-3-haiku-20240307",
    db_connection_string: str | None = None,
    mcp_server_url: str | None = None,
    origin: str = "enhanced-llm",
    user_id: str | None = None,
) -> EnhancedLLM:
    """
    Create an Enhanced LLM instance with sensible defaults.

    Args:
        llm_model: The LLM model to use
        db_connection_string: Optional database for conversation storage
        mcp_server_url: Optional MCP server for additional tools
        origin: Origin identifier for usage tracking
        user_id: Default user ID

    Returns:
        Configured EnhancedLLM instance
    """
    return EnhancedLLM(
        llm_model=llm_model,
        db_connection_string=db_connection_string,
        mcp_server_url=mcp_server_url,
        origin=origin,
        user_id=user_id,
    )
