"""HTTP client for llmring-server MCP endpoints - Refactored Version.

This module provides a clean HTTP interface for MCP operations,
using the unified BaseHTTPClient for consistency.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

from llmring.net.http_base import BaseHTTPClient

logger = logging.getLogger(__name__)


class MCPHttpClient(BaseHTTPClient):
    """HTTP client for llmring-server MCP endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the MCP HTTP client.

        Args:
            base_url: Base URL of llmring-server (defaults to env or localhost)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        base_url = base_url or os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
        api_key = api_key or os.getenv("LLMRING_API_KEY")

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    # ============= Server Management =============

    async def register_server(
        self,
        name: str,
        url: str,
        transport_type: str = "http",
        auth_config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        project_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Register an MCP server."""
        return await self.post(
            "/api/v1/mcp/servers",
            json={
                "name": name,
                "url": url,
                "transport_type": transport_type,
                "auth_config": auth_config,
                "capabilities": capabilities,
                "project_id": str(project_id) if project_id else None,
            },
        )

    async def list_servers(
        self,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP servers."""
        params = {"is_active": is_active}
        if project_id:
            params["project_id"] = str(project_id)

        return await self.get("/api/v1/mcp/servers", params=params)

    async def get_server(self, server_id: UUID) -> Dict[str, Any]:
        """Get an MCP server by ID."""
        return await self.get(f"/api/v1/mcp/servers/{server_id}")

    async def update_server(
        self,
        server_id: UUID,
        name: Optional[str] = None,
        url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update an MCP server."""
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if url is not None:
            update_data["url"] = url
        if auth_config is not None:
            update_data["auth_config"] = auth_config
        if capabilities is not None:
            update_data["capabilities"] = capabilities
        if is_active is not None:
            update_data["is_active"] = is_active

        return await self.put(f"/api/v1/mcp/servers/{server_id}", json=update_data)

    async def delete_server(self, server_id: UUID) -> bool:
        """Delete an MCP server."""
        await self.delete(f"/api/v1/mcp/servers/{server_id}")
        return True

    async def refresh_server_capabilities(
        self,
        server_id: UUID,
        tools: List[Dict[str, Any]],
        resources: List[Dict[str, Any]],
        prompts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Refresh server capabilities."""
        return await self.post(
            f"/api/v1/mcp/servers/{server_id}/refresh",
            json={
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
            },
        )

    # ============= Tool Management =============

    async def list_tools(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP tools."""
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)

        return await self.get("/api/v1/mcp/tools", params=params)

    async def get_tool(self, tool_id: UUID) -> Dict[str, Any]:
        """Get an MCP tool by ID."""
        return await self.get(f"/api/v1/mcp/tools/{tool_id}")

    async def get_tool_by_name(
        self,
        name: str,
        server_id: Optional[UUID] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get an MCP tool by name."""
        tools = await self.list_tools(server_id=server_id)
        for tool in tools:
            if tool["name"] == name:
                return tool
        return None

    async def execute_tool(
        self,
        tool_id: UUID,
        input: Dict[str, Any],
        conversation_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Execute an MCP tool."""
        return await self.post(
            f"/api/v1/mcp/tools/{tool_id}/execute",
            json={
                "tool_id": str(tool_id),
                "input": input,
                "conversation_id": str(conversation_id) if conversation_id else None,
            },
        )

    async def get_tool_history(
        self,
        tool_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get tool execution history."""
        return await self.get(
            f"/api/v1/mcp/tools/{tool_id}/history",
            params={"limit": limit},
        )

    # ============= Resource Management =============

    async def list_resources(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP resources."""
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)

        return await self.get("/api/v1/mcp/resources", params=params)

    async def get_resource(self, resource_id: UUID) -> Dict[str, Any]:
        """Get an MCP resource by ID."""
        return await self.get(f"/api/v1/mcp/resources/{resource_id}")

    async def get_resource_content(self, resource_id: UUID) -> Dict[str, Any]:
        """Get resource content."""
        return await self.get(f"/api/v1/mcp/resources/{resource_id}/content")

    # ============= Prompt Management =============

    async def list_prompts(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP prompts."""
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)

        return await self.get("/api/v1/mcp/prompts", params=params)

    async def get_prompt(self, prompt_id: UUID) -> Dict[str, Any]:
        """Get an MCP prompt by ID."""
        return await self.get(f"/api/v1/mcp/prompts/{prompt_id}")

    async def render_prompt(
        self,
        prompt_id: UUID,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Render a prompt with arguments."""
        return await self.post(
            f"/api/v1/mcp/prompts/{prompt_id}/render",
            json=arguments,
        )

    # ============= Conversation Management =============

    async def create_conversation(
        self,
        title: str,
        system_prompt: Optional[str] = None,
        model_alias: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> UUID:
        """Create a new conversation."""
        data = await self.post(
            "/api/v1/conversations",
            json={
                "title": title,
                "system_prompt": system_prompt,
                "model_alias": model_alias,
                "project_id": project_id,
            },
        )
        return UUID(data["id"])

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to a conversation."""
        return await self.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "role": role,
                "content": content,
                "metadata": metadata,
            },
        )

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        return await self.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"limit": limit},
        )
