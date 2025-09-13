"""
Enhanced MCP client with LLM sampling capabilities.

This module provides MCP clients that can handle server-initiated LLM sampling requests
by integrating with the existing LLM service and providers.
"""

import logging
from typing import Any

from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

from llmring.mcp.client.mcp_client import AsyncMCPClient, MCPClient

logger = logging.getLogger(__name__)


class MCPClientWithLLM(MCPClient):
    """
    MCP client with LLM sampling capabilities for server-initiated requests.

    This client can handle sampling requests from MCP servers by routing them
    to configured LLM providers.
    """

    def __init__(
        self,
        base_url: str,
        llmring: LLMRing | None = None,
        default_model: str | None = None,
        sampling_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the MCP client with LLM capabilities.

        Args:
            base_url: The base URL of the MCP server
            llmring: Optional LLM service instance (will create default if not provided)
            default_model: Default model to use for sampling requests
            sampling_config: Configuration for sampling behavior
            **kwargs: Additional arguments passed to base MCPClient
        """
        super().__init__(base_url, **kwargs)

        # Initialize LLM service
        self.llmring = llmring or LLMRing(origin="mcp-llm-client")
        self.default_model = default_model
        self.sampling_config = sampling_config or {}

        # Sampling configuration
        self.max_sampling_tokens = self.sampling_config.get("max_tokens", 1000)
        self.default_temperature = self.sampling_config.get("temperature", 0.7)
        self.enable_sampling = self.sampling_config.get("enabled", True)
        self.allowed_models = self.sampling_config.get(
            "allowed_models", None
        )  # None = all models allowed

        logger.info(
            f"Initialized MCP client with LLM sampling (enabled: {self.enable_sampling})"
        )

        # Register server-initiated sampling handler
        self.register_method_handler(
            "sampling/createMessage", self._sampling_create_message
        )

    def _sampling_create_message(self, params: dict[str, Any]) -> dict[str, Any]:
        # Adapter to use existing handle_sampling_request signature
        request = {"id": "0", "method": "sampling/createMessage", "params": params}
        response = self.handle_sampling_request(request)
        if "error" in response:
            # Raise to let dispatcher wrap into JSON-RPC error
            raise RuntimeError(response["error"].get("message", "Sampling error"))
        return response.get("result", {})

    def handle_sampling_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle server-initiated LLM sampling request.

        Args:
            request: The sampling request from the server

        Returns:
            JSON-RPC response with the LLM result or error
        """
        if not self.enable_sampling:
            return {
                "error": {
                    "code": -32601,
                    "message": "Sampling disabled",
                    "data": "LLM sampling is disabled in client configuration",
                }
            }

        try:
            # Extract request parameters
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method != "sampling/createMessage":
                return {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found",
                        "data": f"Unsupported sampling method: {method}",
                    },
                }

            # Validate and extract parameters
            messages_data = params.get("messages", [])
            if not messages_data:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Messages array is required",
                    },
                }

            # Convert messages to Message objects
            messages = []
            for msg_data in messages_data:
                if (
                    not isinstance(msg_data, dict)
                    or "role" not in msg_data
                    or "content" not in msg_data
                ):
                    return {
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params",
                            "data": "Each message must have 'role' and 'content' fields",
                        },
                    }
                messages.append(
                    Message(role=msg_data["role"], content=msg_data["content"])
                )

            # Extract sampling parameters
            model = params.get("model", self.default_model)
            if not model:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Model must be specified in request or default_model in client config",
                    },
                }

            # Check if model is allowed
            if self.allowed_models and model not in self.allowed_models:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Forbidden model",
                        "data": f"Model '{model}' is not in allowed models list",
                    },
                }

            temperature = params.get("temperature", self.default_temperature)
            max_tokens = params.get("max_tokens", self.max_sampling_tokens)

            # Validate max_tokens limit
            if max_tokens > self.max_sampling_tokens:
                max_tokens = self.max_sampling_tokens
                logger.warning(
                    f"Requested max_tokens capped at {self.max_sampling_tokens}"
                )

            # Create LLM request
            llm_request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=params.get("response_format"),
                tools=params.get("tools"),
                tool_choice=params.get("tool_choice"),
            )

            # Execute LLM request synchronously
            response = self._run_async(self.llmring.chat(llm_request))

            # Format response according to MCP sampling specification
            result = {
                "id": request_id,
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": response.content},
                    "model": model,
                    "stopReason": response.finish_reason or "max_tokens",
                },
            }

            # Add usage information if available
            if response.usage:
                result["result"]["usage"] = {
                    "inputTokens": response.usage.get("prompt_tokens", 0),
                    "outputTokens": response.usage.get("completion_tokens", 0),
                    "totalTokens": response.usage.get("total_tokens", 0),
                }

            return result

        except Exception as e:
            logger.error(f"Error in sampling request: {e}")
            return {
                "id": request.get("id"),
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
            }


class AsyncMCPClientWithLLM(AsyncMCPClient):
    """
    Async MCP client with LLM sampling capabilities for server-initiated requests.

    This client can handle sampling requests from MCP servers by routing them
    to configured LLM providers.
    """

    def __init__(
        self,
        base_url: str,
        llmring: LLMRing | None = None,
        default_model: str | None = None,
        sampling_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the async MCP client with LLM capabilities.

        Args:
            base_url: The base URL of the MCP server
            llmring: Optional LLM service instance (will create default if not provided)
            default_model: Default model to use for sampling requests
            sampling_config: Configuration for sampling behavior
            **kwargs: Additional arguments passed to base AsyncMCPClient
        """
        super().__init__(base_url, **kwargs)

        # Initialize LLM service
        self.llmring = llmring or LLMRing(origin="mcp-llm-client")
        self.default_model = default_model
        self.sampling_config = sampling_config or {}

        # Sampling configuration
        self.max_sampling_tokens = self.sampling_config.get("max_tokens", 1000)
        self.default_temperature = self.sampling_config.get("temperature", 0.7)
        self.enable_sampling = self.sampling_config.get("enabled", True)
        self.allowed_models = self.sampling_config.get(
            "allowed_models", None
        )  # None = all models allowed

        logger.info(
            f"Initialized async MCP client with LLM sampling (enabled: {self.enable_sampling})"
        )

    async def handle_sampling_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle server-initiated LLM sampling request.

        Args:
            request: The sampling request from the server

        Returns:
            JSON-RPC response with the LLM result or error
        """
        if not self.enable_sampling:
            return {
                "error": {
                    "code": -32601,
                    "message": "Sampling disabled",
                    "data": "LLM sampling is disabled in client configuration",
                }
            }

        try:
            # Extract request parameters
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method != "sampling/createMessage":
                return {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found",
                        "data": f"Unsupported sampling method: {method}",
                    },
                }

            # Validate and extract parameters
            messages_data = params.get("messages", [])
            if not messages_data:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Messages array is required",
                    },
                }

            # Convert messages to Message objects
            messages = []
            for msg_data in messages_data:
                if (
                    not isinstance(msg_data, dict)
                    or "role" not in msg_data
                    or "content" not in msg_data
                ):
                    return {
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params",
                            "data": "Each message must have 'role' and 'content' fields",
                        },
                    }
                messages.append(
                    Message(role=msg_data["role"], content=msg_data["content"])
                )

            # Extract sampling parameters
            model = params.get("model", self.default_model)
            if not model:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Model must be specified in request or default_model in client config",
                    },
                }

            # Check if model is allowed
            if self.allowed_models and model not in self.allowed_models:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Forbidden model",
                        "data": f"Model '{model}' is not in allowed models list",
                    },
                }

            temperature = params.get("temperature", self.default_temperature)
            max_tokens = params.get("max_tokens", self.max_sampling_tokens)

            # Validate max_tokens limit
            if max_tokens > self.max_sampling_tokens:
                max_tokens = self.max_sampling_tokens
                logger.warning(
                    f"Requested max_tokens capped at {self.max_sampling_tokens}"
                )

            # Create LLM request
            llm_request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=params.get("response_format"),
                tools=params.get("tools"),
                tool_choice=params.get("tool_choice"),
            )

            # Execute LLM request
            response = await self.llmring.chat(llm_request)

            # Format response according to MCP sampling specification
            result = {
                "id": request_id,
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": response.content},
                    "model": model,
                    "stopReason": response.finish_reason or "max_tokens",
                },
            }

            # Add usage information if available
            if response.usage:
                result["result"]["usage"] = {
                    "inputTokens": response.usage.get("prompt_tokens", 0),
                    "outputTokens": response.usage.get("completion_tokens", 0),
                    "totalTokens": response.usage.get("total_tokens", 0),
                }

            return result

        except Exception as e:
            logger.error(f"Error in sampling request: {e}")
            return {
                "id": request.get("id"),
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
            }
