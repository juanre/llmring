"""Integration layer between MCP client and the LLMRing service."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from llmring.mcp.http_client import MCPHttpClient
from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


class MCPLLMRingIntegration:
    """
    Integration layer that connects MCP with LLMRing and llmring-server.

    This class:
    - Uses LLMRing for actual LLM calls
    - Uses llmring-server API for all persistence
    - Maintains no database dependencies
    """

    def __init__(
        self,
        origin: str = "mcp",
        llmring_server_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the integration layer.

        Args:
            origin: Origin identifier for tracking
            llmring_server_url: URL of llmring-server for persistence
            api_key: API key for llmring-server
        """
        self.origin = origin

        # Initialize LLMRing for LLM calls
        self.llmring = LLMRing(origin=origin)

        # HTTP client for llmring-server API
        self.http_client = MCPHttpClient(
            base_url=llmring_server_url,
            api_key=api_key,
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        conversation_id: Optional[UUID] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat request via LLMRing.

        Args:
            messages: List of message dictionaries
            model: Model alias or identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            conversation_id: Optional conversation ID for tracking
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        formatted_messages = [Message.model_validate(message) for message in messages]

        # Create LLMRequest
        request = LLMRequest(
            model=model or "mcp_agent",  # Use alias instead of hardcoded model
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Make the LLM call
        response = await self.llmring.chat(request)

        # If we have a conversation ID, log to llmring-server
        if conversation_id:
            await self._log_to_server(conversation_id, messages, response)

        return response

    async def _log_to_server(
        self,
        conversation_id: UUID,
        messages: List[Dict[str, Any]],
        response: LLMResponse,
    ):
        """Log messages and response to llmring-server."""
        try:
            # Log user message (last in the list)
            if messages:
                user_message = messages[-1]
                await self.http_client.add_message(
                    conversation_id=conversation_id,
                    role=user_message.get("role", "user"),
                    content=user_message.get("content", ""),
                )

            # Log assistant response
            if response.content:
                await self.http_client.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response.content,
                )
        except Exception as e:
            logger.warning(f"Failed to log to server: {e}")

    async def create_conversation(
        self,
        title: str,
        system_prompt: Optional[str] = None,
        model_alias: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> UUID:
        """
        Create a new conversation via llmring-server.

        Args:
            title: Conversation title
            system_prompt: System prompt
            model_alias: Model to use
            project_id: Project ID

        Returns:
            Conversation ID
        """
        return await self.http_client.create_conversation(
            title=title,
            system_prompt=system_prompt,
            model_alias=model_alias,
            project_id=project_id,
        )

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation from llmring-server.

        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        return await self.http_client.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit,
        )

    async def register_mcp_server(
        self,
        name: str,
        url: str,
        transport_type: str = "http",
        auth_config: Optional[Dict[str, Any]] = None,
        project_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Register an MCP server via llmring-server.

        Args:
            name: Server name
            url: Server URL
            transport_type: Transport type
            auth_config: Auth configuration
            project_id: Optional project ID

        Returns:
            Server data with ID
        """
        return await self.http_client.register_server(
            name=name,
            url=url,
            transport_type=transport_type,
            auth_config=auth_config,
            project_id=project_id,
        )

    async def get_mcp_tools(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all registered MCP tools via llmring-server.

        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID

        Returns:
            List of tool definitions
        """
        return await self.http_client.list_tools(
            server_id=server_id,
            project_id=project_id,
        )

    async def record_mcp_tool_execution(
        self,
        tool_id: UUID,
        input: Dict[str, Any],
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        conversation_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Record an MCP tool execution via llmring-server.

        Note: This records execution metadata for observability.
        Actual tool execution should happen via AsyncMCPClient.

        Args:
            tool_id: Tool ID
            input: Tool input
            output: Tool output (if execution completed)
            error: Error message (if execution failed)
            duration_ms: Execution duration in milliseconds
            conversation_id: Optional conversation ID

        Returns:
            Execution record with ID and metadata
        """
        return await self.http_client.record_tool_execution(
            tool_id=tool_id,
            input=input,
            output=output,
            error=error,
            duration_ms=duration_ms,
            conversation_id=conversation_id,
        )

    async def get_mcp_resources(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all registered MCP resources via llmring-server.

        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID

        Returns:
            List of resource definitions
        """
        return await self.http_client.list_resources(
            server_id=server_id,
            project_id=project_id,
        )

    async def get_mcp_prompts(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all registered MCP prompts via llmring-server.

        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID

        Returns:
            List of prompt definitions
        """
        return await self.http_client.list_prompts(
            server_id=server_id,
            project_id=project_id,
        )

    async def close(self):
        """Clean up resources."""
        await self.http_client.close()
