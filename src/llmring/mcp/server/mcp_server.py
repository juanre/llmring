"""
Main MCP Server implementation.

Provides a transport-agnostic MCP server that can work with any transport layer
(STDIO, WebSocket, HTTP) and properly implements the MCP protocol.
"""

import asyncio
import logging
import signal
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from llmring.mcp.server.interfaces import AuthProvider, MCPMiddleware, StorageProvider
from llmring.mcp.server.protocol import JSONRPCError, JSONRPCRouter
from llmring.mcp.server.protocol.handlers import ProtocolError, ProtocolHandlers
from llmring.mcp.server.registries import FunctionRegistry, PromptRegistry, ResourceRegistry
from llmring.mcp.server.transport.base import Transport

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Core MCP Server implementation with pluggable auth, storage, and transport.

    This server implements the full MCP protocol and can work with any transport
    layer (STDIO, WebSocket, HTTP). It properly handles initialization, error
    responses, and all standard MCP methods.
    """

    def __init__(
        self,
        name: str = "MCP Server",
        version: str = "1.0.0",
        auth_provider: Optional[AuthProvider] = None,
        storage_provider: Optional[StorageProvider] = None,
        middleware: Optional[List[MCPMiddleware]] = None,
    ):
        """
        Initialize the MCP server.

        Args:
            name: Name of the MCP server
            version: Version of the MCP server
            auth_provider: Optional authentication provider
            storage_provider: Storage provider for tools/resources/prompts
            middleware: Optional list of middleware to apply
        """
        self.name = name
        self.version = version
        self.auth_provider = auth_provider
        self.storage_provider = storage_provider
        self.middleware = middleware or []

        # Initialize protocol components
        self.handlers = ProtocolHandlers(name, version)
        self.router = JSONRPCRouter()

        # Initialize registries for local storage (if no storage provider)
        self.function_registry = FunctionRegistry()
        self.resource_registry = ResourceRegistry()
        self.prompt_registry = PromptRegistry()

        # Register protocol methods
        self._register_protocol_methods()

        # Transport will be set when running
        self.transport: Optional[Transport] = None
        self._running = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            if self.transport:
                # Use thread-safe method to schedule shutdown
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(lambda: asyncio.create_task(self.shutdown()))

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _register_protocol_methods(self):
        """Register all MCP protocol methods with the router."""
        # Initialization
        self.router.register("initialize", self._handle_initialize)
        self.router.register_notification("initialized", self._handle_initialized)

        # Tools
        self.router.register("tools/list", self._handle_tools_list)
        self.router.register("tools/call", self._handle_tools_call)

        # Resources
        self.router.register("resources/list", self._handle_resources_list)
        self.router.register("resources/read", self._handle_resources_read)

        # Prompts
        self.router.register("prompts/list", self._handle_prompts_list)
        self.router.register("prompts/get", self._handle_prompts_get)

        # Roots
        self.router.register("roots/list", self._handle_roots_list)

        # Logging
        self.router.register("logging/setLevel", self._handle_logging_set_level)

        # Lifecycle
        self.router.register("ping", self._handle_ping)
        self.router.register("shutdown", self._handle_shutdown)

    async def run(self, transport: Transport) -> None:
        """
        Run the server with the given transport.

        Args:
            transport: The transport to use for communication
        """
        self.transport = transport
        self._running = True

        # Set message handler
        transport.set_message_callback(self._handle_message)
        transport.set_error_callback(self._handle_transport_error)
        transport.set_close_callback(self._handle_transport_close)

        try:
            # Start transport
            if not await transport.start():
                raise RuntimeError("Failed to start transport")

            logger.info(f"{self.name} v{self.version} started")

            # Keep running until shutdown
            while self._running and not self.handlers.shutdown_requested:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the server."""
        if not self._running:
            return

        logger.info("Shutting down server")
        self._running = False

        # Don't send shutdown notification here - it interferes with shutdown response
        # The shutdown response should be sent before this method is called

        # Stop transport
        if self.transport:
            await self.transport.stop()

        logger.info("Server shutdown complete")

    def _handle_message(self, message: Dict[str, Any], context: Any = None) -> None:
        """
        Handle incoming message from transport.

        This is called synchronously by the transport, so we create a agent
        to handle it asynchronously.

        Args:
            message: The JSON-RPC message
            context: Optional context from transport (e.g., HTTP request)
        """
        asyncio.create_task(self._process_message(message, context))

    async def _process_message(
        self, message: Dict[str, Any], transport_context: Any = None
    ) -> None:
        """Process an incoming message asynchronously.

        Args:
            message: The JSON-RPC message to process
            transport_context: Optional context from transport
        """
        try:
            # Create context for this request
            context = await self._create_context(message, transport_context)

            # Check if it's a request or notification
            msg_id = message.get("id")
            method = message.get("method")

            if not method:
                # It's a response to our request (not typical for servers)
                logger.debug(f"Received response: {message}")
                return

            # Skip initialization check here - let router handle it
            # The router now uses context.mcp_initialized which is synchronized

            # Handle the request
            response = await self.router.handle_raw_request(message, context)

            # Send response if it's a request (not a notification)
            if msg_id is not None and response is not None:
                await self.transport.send_message(response)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            if message.get("id") is not None:
                await self._send_error(
                    message["id"],
                    JSONRPCError.INTERNAL_ERROR,
                    "Internal server error",
                    {"error": str(e)},
                )

    def _handle_transport_error(self, error: Exception) -> None:
        """Handle transport error."""
        logger.error(f"Transport error: {error}")

    def _handle_transport_close(self) -> None:
        """Handle transport close."""
        logger.info("Transport closed")
        asyncio.create_task(self.shutdown())

    async def _create_context(
        self, message: Dict[str, Any], transport_context: Any = None
    ) -> SimpleNamespace:
        """Create context for request handlers.

        Args:
            message: The JSON-RPC message
            transport_context: Optional context from transport

        Returns:
            Context object with all necessary data for handlers
        """
        context = SimpleNamespace()

        # Include transport context if provided
        if transport_context:
            context.transport = transport_context

        # Add storage provider
        context.storage_provider = self.storage_provider

        # Add registries (for local storage)
        context.function_registry = self.function_registry
        context.resource_registry = self.resource_registry
        context.prompt_registry = self.prompt_registry

        # Add authentication context if available
        if self.auth_provider:
            auth_context = await self.auth_provider.authenticate_context(message)
            context.user_id = auth_context.get("user_id", "anonymous")
            context.permissions = auth_context.get("permissions", [])
        else:
            # No auth - anonymous user
            context.user_id = "anonymous"
            context.permissions = ["*"]  # All permissions

        # Add logger
        context.logger = logger

        # Add initialization state
        context.mcp_initialized = self.handlers.initialized

        return context

    async def _send_error(
        self,
        msg_id: Union[str, int],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> None:
        """Send an error response."""
        error_response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

        if data is not None:
            error_response["error"]["data"] = data

        await self.transport.send_message(error_response)

    async def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a notification (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method}

        if params is not None:
            notification["params"] = params

        await self.transport.send_message(notification)

    # Protocol method handlers

    async def _handle_initialize(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle initialize method."""
        try:
            return await self.handlers.handle_initialize(params, context)
        except ProtocolError as e:
            raise JSONRPCError(e.code, e.message, e.data)

    async def _handle_initialized(self, params: Dict[str, Any], context: SimpleNamespace) -> None:
        """Handle initialized notification."""
        await self.handlers.handle_initialized(params, context)

        # Send initial notifications
        await self._send_notification(
            "log", {"level": "info", "message": f"{self.name} ready to serve requests"}
        )

        return None  # No response for notifications

    async def _handle_tools_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle tools/list method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_tools_list(params, context)

        # Otherwise use local registry
        tools = []
        for name in self.function_registry.functions:
            tools.append(
                {
                    "name": name,
                    "description": self.function_registry.get_description(name) or "",
                    "inputSchema": self.function_registry.get_schema(name) or {"type": "object"},
                }
            )

        return {"tools": tools}

    async def _handle_tools_call(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle tools/call method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_tools_call(params, context)

        # Otherwise use local registry
        try:
            tool_name = params.get("name")
            if not tool_name:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Tool name is required")

            func = self.function_registry.get_function(tool_name)
            if not func:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, f"Tool not found: {tool_name}")

            # Get tool schema and validate arguments
            tool_args = params.get("arguments", {})
            schema = self.function_registry.get_schema(tool_name)

            if schema:
                # Validate arguments against schema
                validation_error = self._validate_tool_arguments(tool_args, schema)
                if validation_error:
                    raise ProtocolError(JSONRPCError.INVALID_PARAMS, validation_error)

            # Execute function
            import inspect

            if inspect.iscoroutinefunction(func):
                result = await func(**tool_args)
            else:
                result = func(**tool_args)

            # Format result
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict) and "content" in result:
                content = result["content"]
            else:
                import json

                content = [{"type": "text", "text": json.dumps(result, indent=2)}]

            return {"content": content, "isError": False}

        except ProtocolError:
            raise
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            }

    async def _handle_resources_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle resources/list method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_resources_list(params, context)

        # Otherwise use local registry
        resources = []
        for uri, info in self.resource_registry.resources.items():
            resources.append(
                {
                    "uri": uri,
                    "name": info.get("name", uri),
                    "description": info.get("description", ""),
                    "mimeType": info.get("mimeType", "text/plain"),
                }
            )

        return {"resources": resources}

    async def _handle_resources_read(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle resources/read method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_resources_read(params, context)

        # Otherwise use local registry
        try:
            uri = params.get("uri")
            if not uri:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Resource URI is required")

            content = await self.resource_registry.read_resource(uri)
            info = self.resource_registry.get_resource_info(uri)

            if not info:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, f"Resource not found: {uri}")

            # Handle binary vs text
            import base64

            if isinstance(content, bytes):
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": info.get("mimeType", "application/octet-stream"),
                            "blob": base64.b64encode(content).decode("utf-8"),
                        }
                    ]
                }
            else:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": info.get("mimeType", "text/plain"),
                            "text": str(content),
                        }
                    ]
                }

        except ProtocolError:
            raise
        except Exception as e:
            logger.error(f"Resource read error: {e}", exc_info=True)
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, f"Failed to read resource: {str(e)}")

    async def _handle_prompts_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle prompts/list method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_prompts_list(params, context)

        # Otherwise use local registry
        prompts = []
        for name, info in self.prompt_registry.prompts.items():
            prompts.append(
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "arguments": info.get("arguments", []),
                }
            )

        return {"prompts": prompts}

    async def _handle_prompts_get(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle prompts/get method."""
        # If we have a storage provider, use it
        if context.storage_provider:
            return await self.handlers.handle_prompts_get(params, context)

        # Otherwise use local registry
        try:
            name = params.get("name")
            if not name:
                raise ProtocolError(JSONRPCError.INVALID_PARAMS, "Prompt name is required")

            arguments = params.get("arguments", {})
            result = await self.prompt_registry.get_prompt(name, arguments)

            return result

        except ProtocolError:
            raise
        except Exception as e:
            logger.error(f"Prompt generation error: {e}", exc_info=True)
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, f"Failed to generate prompt: {str(e)}")

    async def _handle_roots_list(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle roots/list method.

        Returns a list of roots. For now, return a simple default root representing
        the server's working context.
        """
        try:
            roots = [
                {
                    "uri": "mcp://root",
                    "name": self.name,
                    "description": "Default root",
                }
            ]
            return {"roots": roots}
        except Exception as e:
            logger.error(f"Error listing roots: {e}")
            raise ProtocolError(JSONRPCError.INTERNAL_ERROR, f"Failed to list roots: {str(e)}")

    async def _handle_logging_set_level(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle logging/setLevel method."""
        return await self.handlers.handle_logging_set_level(params, context)

    async def _handle_ping(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle ping method."""
        return await self.handlers.handle_ping(params, context)

    async def _handle_shutdown(
        self, params: Dict[str, Any], context: SimpleNamespace
    ) -> Dict[str, Any]:
        """Handle shutdown method."""
        result = await self.handlers.handle_shutdown(params, context)

        # Schedule shutdown after sending response
        asyncio.create_task(self.shutdown())

        return result

    # Public API for registering tools, resources, and prompts locally

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool with the server.

        Args:
            name: Name of the tool
            handler: Function to handle tool execution
            description: Description of the tool
            input_schema: JSON Schema for tool input
        """
        schema = input_schema or {"type": "object"}
        self.function_registry.register(name, handler, schema, description)

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "text/plain",
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register a resource with the server.

        Args:
            uri: URI of the resource
            name: Name of the resource
            description: Description of the resource
            mime_type: MIME type of the resource
            handler: Function to get resource content
        """
        self.resource_registry.register_resource(uri, name, description, mime_type, handler)

    def register_static_resource(
        self,
        uri: str,
        name: str,
        description: str,
        content: str,
        mime_type: str = "text/plain",
    ) -> None:
        """
        Register a static resource with the server.

        Args:
            uri: URI of the resource
            name: Name of the resource
            description: Description of the resource
            content: Static content of the resource
            mime_type: MIME type of the resource
        """
        self.resource_registry.register_static_resource(uri, name, description, content, mime_type)

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register a prompt with the server.

        Args:
            name: Name of the prompt
            description: Description of the prompt
            arguments: List of argument definitions
            handler: Function to generate prompt
        """
        self.prompt_registry.register_prompt(name, description, arguments, handler)

    def register_static_prompt(
        self,
        name: str,
        description: str,
        content: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Register a static prompt with the server.

        Args:
            name: Name of the prompt
            description: Description of the prompt
            content: Static content template
            arguments: List of argument definitions
        """
        self.prompt_registry.register_static_prompt(name, description, content, arguments)

    def _validate_tool_arguments(
        self, arguments: Dict[str, Any], schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate tool arguments against a JSON Schema.

        Args:
            arguments: The arguments to validate
            schema: The JSON Schema to validate against

        Returns:
            Error message if validation fails, None if validation passes
        """
        # Check if schema is an object type
        if schema.get("type") != "object":
            return None

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_allowed = schema.get("additionalProperties", True)

        # Check required properties
        for prop in required:
            if prop not in arguments:
                return f"Missing required parameter: {prop}"

        # Check each provided argument
        for arg_name, arg_value in arguments.items():
            if arg_name not in properties and not additional_allowed:
                return f"Unexpected parameter: {arg_name}"

            if arg_name in properties:
                prop_schema = properties[arg_name]
                error = self._validate_value(arg_value, prop_schema, arg_name)
                if error:
                    return error

        return None

    def _validate_value(self, value: Any, schema: Dict[str, Any], path: str) -> Optional[str]:
        """
        Validate a single value against a schema.

        Args:
            value: The value to validate
            schema: The schema for this value
            path: The property path for error messages

        Returns:
            Error message if validation fails, None if validation passes
        """
        expected_type = schema.get("type")

        if expected_type == "string":
            if not isinstance(value, str):
                return f"Parameter '{path}' must be a string, got {type(value).__name__}"
        elif expected_type == "number":
            if not isinstance(value, (int, float)):
                return f"Parameter '{path}' must be a number, got {type(value).__name__}"
        elif expected_type == "integer":
            if not isinstance(value, int):
                return f"Parameter '{path}' must be an integer, got {type(value).__name__}"
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                return f"Parameter '{path}' must be a boolean, got {type(value).__name__}"
        elif expected_type == "array":
            if not isinstance(value, list):
                return f"Parameter '{path}' must be an array, got {type(value).__name__}"
            # Validate array items if schema is provided
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(value):
                    error = self._validate_value(item, items_schema, f"{path}[{i}]")
                    if error:
                        return error
        elif expected_type == "object":
            if not isinstance(value, dict):
                return f"Parameter '{path}' must be an object, got {type(value).__name__}"
            # Could add nested object validation here if needed
        elif expected_type == "null":
            if value is not None:
                return f"Parameter '{path}' must be null, got {type(value).__name__}"

        return None
