"""HTTP client for llmring-server MCP endpoints.

This module provides a clean HTTP interface for MCP operations,
allowing llmring to remain database-agnostic while persisting
all data through llmring-server's REST API.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class MCPHttpClient:
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
        self.base_url = base_url or os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("LLMRING_API_KEY")
        self.timeout = timeout
        
        # Configure headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
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
        """Register an MCP server.
        
        Args:
            name: Server name
            url: Server URL
            transport_type: Transport type (stdio, http, websocket)
            auth_config: Authentication configuration
            capabilities: Server capabilities
            project_id: Optional project ID
            
        Returns:
            Server data with ID
        """
        response = await self.client.post(
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
        response.raise_for_status()
        return response.json()
    
    async def list_servers(
        self,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP servers.
        
        Args:
            project_id: Filter by project ID
            is_active: Filter by active status
            
        Returns:
            List of servers
        """
        params = {"is_active": is_active}
        if project_id:
            params["project_id"] = str(project_id)
        
        response = await self.client.get("/api/v1/mcp/servers", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_server(self, server_id: UUID) -> Dict[str, Any]:
        """Get an MCP server by ID.
        
        Args:
            server_id: Server ID
            
        Returns:
            Server data
        """
        response = await self.client.get(f"/api/v1/mcp/servers/{server_id}")
        response.raise_for_status()
        return response.json()
    
    async def update_server(
        self,
        server_id: UUID,
        name: Optional[str] = None,
        url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update an MCP server.
        
        Args:
            server_id: Server ID
            name: New name
            url: New URL
            auth_config: New auth configuration
            capabilities: New capabilities
            is_active: New active status
            
        Returns:
            Updated server data
        """
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
        
        response = await self.client.put(
            f"/api/v1/mcp/servers/{server_id}",
            json=update_data,
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_server(self, server_id: UUID) -> bool:
        """Delete an MCP server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if deleted
        """
        response = await self.client.delete(f"/api/v1/mcp/servers/{server_id}")
        response.raise_for_status()
        return True
    
    async def refresh_server_capabilities(
        self,
        server_id: UUID,
        tools: List[Dict[str, Any]],
        resources: List[Dict[str, Any]],
        prompts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Refresh server capabilities.
        
        Args:
            server_id: Server ID
            tools: List of tools
            resources: List of resources
            prompts: List of prompts
            
        Returns:
            Updated capabilities
        """
        response = await self.client.post(
            f"/api/v1/mcp/servers/{server_id}/refresh",
            json={
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # ============= Tool Management =============
    
    async def list_tools(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP tools.
        
        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID
            is_active: Filter by active status
            
        Returns:
            List of tools
        """
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)
        
        response = await self.client.get("/api/v1/mcp/tools", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_tool(self, tool_id: UUID) -> Dict[str, Any]:
        """Get an MCP tool by ID.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Tool data
        """
        response = await self.client.get(f"/api/v1/mcp/tools/{tool_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_tool_by_name(
        self,
        name: str,
        server_id: Optional[UUID] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get an MCP tool by name.
        
        Args:
            name: Tool name
            server_id: Optional server ID to filter by
            
        Returns:
            Tool data or None
        """
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
        """Execute an MCP tool.
        
        Args:
            tool_id: Tool ID
            input: Tool input
            conversation_id: Optional conversation ID
            
        Returns:
            Execution result
        """
        response = await self.client.post(
            f"/api/v1/mcp/tools/{tool_id}/execute",
            json={
                "tool_id": str(tool_id),
                "input": input,
                "conversation_id": str(conversation_id) if conversation_id else None,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def get_tool_history(
        self,
        tool_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get tool execution history.
        
        Args:
            tool_id: Tool ID
            limit: Maximum results
            
        Returns:
            List of executions
        """
        response = await self.client.get(
            f"/api/v1/mcp/tools/{tool_id}/history",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()
    
    # ============= Resource Management =============
    
    async def list_resources(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP resources.
        
        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID
            is_active: Filter by active status
            
        Returns:
            List of resources
        """
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)
        
        response = await self.client.get("/api/v1/mcp/resources", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_resource(self, resource_id: UUID) -> Dict[str, Any]:
        """Get an MCP resource by ID.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            Resource data
        """
        response = await self.client.get(f"/api/v1/mcp/resources/{resource_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_resource_content(self, resource_id: UUID) -> Dict[str, Any]:
        """Get resource content.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            Resource content
        """
        response = await self.client.get(f"/api/v1/mcp/resources/{resource_id}/content")
        response.raise_for_status()
        return response.json()
    
    # ============= Prompt Management =============
    
    async def list_prompts(
        self,
        server_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """List MCP prompts.
        
        Args:
            server_id: Filter by server ID
            project_id: Filter by project ID
            is_active: Filter by active status
            
        Returns:
            List of prompts
        """
        params = {"is_active": is_active}
        if server_id:
            params["server_id"] = str(server_id)
        if project_id:
            params["project_id"] = str(project_id)
        
        response = await self.client.get("/api/v1/mcp/prompts", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_prompt(self, prompt_id: UUID) -> Dict[str, Any]:
        """Get an MCP prompt by ID.
        
        Args:
            prompt_id: Prompt ID
            
        Returns:
            Prompt data
        """
        response = await self.client.get(f"/api/v1/mcp/prompts/{prompt_id}")
        response.raise_for_status()
        return response.json()
    
    async def render_prompt(
        self,
        prompt_id: UUID,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Render a prompt with arguments.
        
        Args:
            prompt_id: Prompt ID
            arguments: Prompt arguments
            
        Returns:
            Rendered prompt
        """
        response = await self.client.post(
            f"/api/v1/mcp/prompts/{prompt_id}/render",
            json=arguments,
        )
        response.raise_for_status()
        return response.json()
    
    # ============= Conversation Management =============
    
    async def create_conversation(
        self,
        title: str,
        system_prompt: Optional[str] = None,
        model_alias: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> UUID:
        """Create a new conversation.
        
        Args:
            title: Conversation title
            system_prompt: System prompt
            model_alias: Model to use
            project_id: Project ID
            
        Returns:
            Conversation ID
        """
        response = await self.client.post(
            "/api/v1/conversations",
            json={
                "title": title,
                "system_prompt": system_prompt,
                "model_alias": model_alias,
                "project_id": project_id,
            },
        )
        response.raise_for_status()
        data = response.json()
        return UUID(data["id"])
    
    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message data
        """
        response = await self.client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "role": role,
                "content": content,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return
            
        Returns:
            List of messages
        """
        response = await self.client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()