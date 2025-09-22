"""
MCP Protocol handlers for standard methods.

Implements handlers for all required MCP protocol methods including:
- Initialization (initialize/initialized)
- Tools (list/call)
- Resources (list/read)
- Prompts (list/get)
- Logging (setLevel)
- Lifecycle (ping/shutdown)
"""

import base64
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional

from llmring.mcp.server.protocol.json_rpc import JSONRPCError

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    """Protocol error with JSON-RPC error code."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class ProtocolHandlers:
    """Default handlers for MCP protocol methods."""

    def __init__(self, server_name: str = "MCP Server", server_version: str = "1.0.0"):
        """
        Initialize protocol handlers.

        Args:
            server_name: Name of the server
            server_version: Version of the server
        """
        self.server_name = server_name
        self.server_version = server_version
        self.initialized = False
        self.client_info: Optional[Dict[str, Any]] = None
        self.protocol_version: Optional[str] = None
        self.shutdown_requested = False

    def check_initialized(self, method: str) -> None:
        """Check if server is initialized, raise error if not."""
        # Allow initialize and shutdown without being initialized
        if method in ["initialize", "shutdown"]:
            return

        if not self.initialized:
            raise ProtocolError(
                -32002,  # Server not initialized
                "Server not initialized",
                {"hint": "Call initialize first"},
            )

    async def handle_initialize(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the initialize method."""
        if self.initialized:
            raise ProtocolError(JSONRPCError.INVALID_REQUEST, "Server already initialized")

        # Extract client info
        self.client_info = params.get("clientInfo", {})
        self.protocol_version = params.get("protocolVersion", "2025-06-18")

        # Mark as initialized
        self.initialized = True

        logger.info(
            f"Initialized with client: {self.client_info.get('name', 'Unknown')} "
            f"v{self.client_info.get('version', 'Unknown')}"
        )

        return {
            "protocolVersion": self.protocol_version,
            "serverInfo": {"name": self.server_name, "version": self.server_version},
            "capabilities": {
                "tools": {"list": True, "call": True},
                "resources": {"list": True, "read": True, "subscribe": False},
                "prompts": {"list": True, "get": True},
                "roots": {"list": True},
                "logging": {"setLevel": True, "message": True},
            },
        }

    async def handle_initialized(self, params: Dict[str, Any], context: SimpleNamespace) -> None:
        """Handle initialized notification."""
        logger.info("Client completed initialization")
        return None  # No response for notifications

    async def handle_ping(self, params: Dict[str, Any], context: SimpleNamespace) -> Dict[str, Any]:
        """Handle the ping method for connection health check."""
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def handle_shutdown(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested by client")
        self.shutdown_requested = True
        return {}  # Empty response before shutdown

    async def handle_logging_set_level(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle logging/setLevel request."""
        level = params.get("level", "info").upper()

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }

        if level not in level_map:
            raise ProtocolError(
                JSONRPCError.INVALID_PARAMS,
                f"Invalid log level: {level}",
                {"validLevels": list(level_map.keys())},
            )

        # Set global log level
        logging.getLogger().setLevel(level_map[level])
        logger.info(f"Log level set to: {level}")

        return {"level": level.lower()}

    async def handle_tools_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the tools/list method."""
        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            return {"tools": []}

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        # Get tools from storage
        tools = await storage.get_tools(user_id)

        # Convert to MCP format
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in tools
            ]
        }

    async def handle_tools_call(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the tools/call method."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if not tool_name:
            raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Tool name is required")

        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, "Storage provider not available")

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        try:
            # Execute the tool
            result = await storage.execute_tool(user_id, tool_name, tool_args)

            # Convert result to MCP-compliant format
            content_text = self._format_tool_result(result)

            return {
                "content": [{"type": "text", "text": content_text}],
                "isError": False,
            }

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            # Return error in MCP-compliant format
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            }

    async def handle_resources_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the resources/list method."""
        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            return {"resources": []}

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        # Get resources from storage
        resources = await storage.get_resources(user_id)

        # Convert to MCP format
        return {
            "resources": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mime_type,
                }
                for resource in resources
            ]
        }

    async def handle_resources_read(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the resources/read method."""
        uri = params.get("uri")
        if not uri:
            raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Resource URI is required")

        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, "Storage provider not available")

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        try:
            # Read the resource
            content = await storage.read_resource(user_id, uri)

            # Get resource metadata
            resources = await storage.get_resources(user_id)
            resource = next((r for r in resources if r.uri == uri), None)

            if not resource:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, f"Resource not found: {uri}")

            mime_type = resource.mime_type

            # Handle binary vs text content
            if isinstance(content, bytes):
                # Binary content - base64 encode
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": mime_type,
                            "blob": base64.b64encode(content).decode("utf-8"),
                        }
                    ]
                }
            else:
                # Text content
                return {"contents": [{"uri": uri, "mimeType": mime_type, "text": str(content)}]}

        except ProtocolError:
            raise
        except Exception as e:
            logger.error(f"Resource read error: {e}", exc_info=True)
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, f"Failed to read resource: {str(e)}")

    async def handle_prompts_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the prompts/list method."""
        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            return {"prompts": []}

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        # Get prompts from storage
        prompts = await storage.get_prompts(user_id)

        # Convert to MCP format
        return {
            "prompts": [
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": prompt.arguments,
                }
                for prompt in prompts
            ]
        }

    async def handle_prompts_get(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle the prompts/get method."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Prompt name is required")

        # Get storage provider from context
        storage = getattr(context, "storage_provider", None)
        if not storage:
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, "Storage provider not available")

        # Get user ID from context (anonymous if no auth)
        user_id = getattr(context, "user_id", "anonymous")

        try:
            # Get the prompt with arguments
            result = await storage.get_prompt(user_id, name, arguments)

            # Get prompt metadata for description
            prompts = await storage.get_prompts(user_id)
            prompt = next((p for p in prompts if p.name == name), None)

            if not prompt:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, f"Prompt not found: {name}")

            return {
                "description": prompt.description,
                "messages": result.get("messages", []),
            }

        except ProtocolError:
            raise
        except Exception as e:
            logger.error(f"Prompt generation error: {e}", exc_info=True)
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, f"Failed to generate prompt: {str(e)}")

    def _format_tool_result(self, result: Any) -> str:
        """
        Format tool execution result into text content.

        Args:
            result: The result from function execution

        Returns:
            Formatted string representation of the result
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, (dict, list)):
            import json

            return json.dumps(result, indent=2)
        else:
            return str(result)
