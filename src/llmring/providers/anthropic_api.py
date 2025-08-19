"""
Anthropic Claude API provider implementation using the official SDK.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
import asyncio
from llmring.net.retry import retry_async
from llmring.net.circuit_breaker import CircuitBreaker
from anthropic.types import Message as AnthropicMessage
# Note: do not call load_dotenv() in library code; handle in app entrypoints
from llmring.base import BaseLLMProvider
from llmring.schemas import LLMResponse, Message



class AnthropicProvider(BaseLLMProvider):
    """Implementation of Anthropic Claude API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "claude-3-7-sonnet-20250219",
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Optional base URL for the API
            model: Default model to use
        """
        super().__init__(api_key=api_key)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_model = model

        if not self.api_key:
            raise ValueError("Anthropic API key must be provided")

        # Initialize the client with the SDK
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(api_key=self.api_key, base_url=base_url)

        # List of officially supported models
        self.supported_models = [
            # Claude 3.7 models
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet",  # Latest version without date
            # Claude 3.5 models
            "claude-3-5-sonnet-20241022-v2",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet",  # Latest version without date
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku",  # Latest version without date
            # Claude 3 models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        self._breaker = CircuitBreaker()

    def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by Anthropic.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise
        """
        # Strip provider prefix if present
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # Check for exact match
        if model in self.supported_models:
            return True

        # For models without specific version, try to find a match with the base name
        if "-202" not in model:  # If no date in the model name
            for supported_model in self.supported_models:
                if supported_model.startswith(model + "-"):
                    return True

            # Handle claude-3.5 -> claude-3-5 conversion
            if "claude-3.5" in model:
                normalized_model = model.replace("claude-3.5", "claude-3-5")
                for supported_model in self.supported_models:
                    if (
                        supported_model.startswith(normalized_model + "-")
                        or supported_model == normalized_model
                    ):
                        return True

        return False

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported Anthropic model names.

        Returns:
            List of supported model names
        """
        return self.supported_models.copy()

    def get_default_model(self) -> str:
        """
        Get the default model to use.

        Returns:
            Default model name
        """
        return self.default_model

    async def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        cache: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Send a chat request to the Anthropic Claude API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "claude-3-opus-20240229")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter

        Returns:
            LLM response
        """
        # Strip provider prefix if present (e.g., "anthropic:claude-3-sonnet" -> "claude-3-sonnet")
        original_model = model
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # For models without version, find the latest version
        if "-202" not in model:  # If no date in model name
            for supported_model in self.supported_models:
                if supported_model.startswith(model + "-"):
                    model = supported_model
                    break

        # Verify model is supported
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {original_model}")

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                # Handle function/tool responses properly
                if (
                    msg.role == "assistant"
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    # Convert tool calls from our format to Anthropic's format
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})

                    for tool_call in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": tool_call["function"]["arguments"],
                            }
                        )

                    anthropic_messages.append({"role": "assistant", "content": content})
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    # Tool responses in Anthropic format
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.tool_call_id,
                                    "content": msg.content,
                                }
                            ],
                        }
                    )
                else:
                    # Regular messages
                    content = msg.content
                    # Convert content to list format if it's a string
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        # Ensure the list is in the correct format
                        formatted_content = []
                        for item in content:
                            if isinstance(item, str):
                                formatted_content.append({"type": "text", "text": item})
                            elif isinstance(item, dict):
                                # Handle image content
                                if item.get("type") == "image_url":
                                    # Convert OpenAI format to Anthropic format
                                    image_data = item["image_url"]["url"]
                                    # Extract base64 data from data URL
                                    if image_data.startswith("data:"):
                                        media_type, base64_data = (
                                            image_data.split(";")[0].split(":")[1],
                                            image_data.split(",")[1],
                                        )
                                        formatted_content.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": base64_data,
                                                },
                                            }
                                        )
                                    else:
                                        # URL image
                                        formatted_content.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "url",
                                                    "url": image_data,
                                                },
                                            }
                                        )
                                # Handle document content (universal format)
                                elif item.get("type") == "document":
                                    # Universal document format -> Anthropic document format (same!)
                                    formatted_content.append(item)
                                else:
                                    formatted_content.append(item)
                        content = formatted_content

                    anthropic_messages.append({"role": msg.role, "content": content})

        # Build the request parameters
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature or 0.7,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            request_params["system"] = system_message

        # Handle tools if provided
        if tools:
            anthropic_tools = []
            for tool in tools:
                # Handle both OpenAI/unit test format and direct format
                if "function" in tool:
                    # OpenAI/unit test format: {"type": "function", "function": {"name": "...", "parameters": {...}}}
                    func = tool["function"]
                    anthropic_tool = {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                else:
                    # Direct format: {"name": "...", "input_schema": {...}}
                    anthropic_tool = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get(
                            "input_schema",
                            tool.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        ),
                    }
                anthropic_tools.append(anthropic_tool)
            request_params["tools"] = anthropic_tools

            # Handle tool choice
            if tool_choice:
                if tool_choice == "auto":
                    request_params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "any":
                    request_params["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    request_params["tool_choice"] = {"type": "none"}
                elif isinstance(tool_choice, dict):
                    # Specific tool choice
                    request_params["tool_choice"] = tool_choice

        # Handle response format for JSON outputs
        if response_format:
            if (
                response_format.get("type") == "json_object"
                or response_format.get("type") == "json"
            ):
                # While Anthropic doesn't have a direct JSON mode like OpenAI,
                # we can use structured output through function calling or clear instructions
                json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any explanatory text before or after the JSON."

                if system_message:
                    request_params["system"] = system_message + json_instruction
                else:
                    request_params["system"] = json_instruction

                # If a schema is provided, add it to the instructions
                if response_format.get("schema"):
                    schema_instruction = f"\n\nThe JSON must conform to this schema: {json.dumps(response_format['schema'])}"
                    request_params["system"] += schema_instruction

        # Make the API call using the SDK
        try:
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.messages.create(**request_params), timeout=timeout_s
                )

            breaker_key = f"anthropic:{model}"
            if not await self._breaker.allow(breaker_key):
                raise Exception("Anthropic circuit open for model")
            response: AnthropicMessage = await retry_async(_do_call)
            await self._breaker.record_success(breaker_key)
        except Exception as e:
            try:
                await self._breaker.record_failure(f"anthropic:{model}")
            except Exception:
                pass
            # Handle specific Anthropic errors with more context
            error_msg = str(e)
            if "API key" in error_msg.lower():
                raise ValueError(
                    f"Anthropic API authentication failed: {error_msg}"
                ) from e
            elif "rate limit" in error_msg.lower():
                raise ValueError(
                    f"Anthropic API rate limit exceeded: {error_msg}"
                ) from e
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ValueError(f"Anthropic model not available: {error_msg}") from e
            else:
                # Re-raise SDK exceptions with our standard format
                raise Exception(f"Anthropic API error: {error_msg}") from e

        # Extract the content from the response
        content = ""
        tool_calls = []
        finish_reason = response.stop_reason

        # Handle different content types in the response
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                # Convert Anthropic tool calls to our format
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "type": "function",
                        "function": {
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input),
                        },
                    }
                )

        # Prepare the response
        llm_response = LLMResponse(
            content=content.strip() if content else "",
            model=model,
            usage={
                "prompt_tokens": int(response.usage.input_tokens),
                "completion_tokens": int(response.usage.output_tokens),
                "total_tokens": int(
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            },
            finish_reason=finish_reason,
        )

        # Add tool calls if present
        if tool_calls:
            llm_response.tool_calls = tool_calls

        return llm_response

    def get_token_count(self, text: str) -> int:
        """
        Get an estimated token count for the text.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            # Try to use Anthropic's tokenizer if available
            from anthropic.tokenizer import count_tokens

            return count_tokens(text)
        except ImportError:
            # Fall back to rough estimation - around 4 characters per token
            return len(text) // 4 + 1

    async def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get capabilities for a specific Anthropic model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model capabilities
        """
        # Known capabilities for Anthropic models as of 2025
        capabilities_map = {
            # Claude 4 models (2025)
            "claude-4-opus": {
                "max_context": 200000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-4-sonnet": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            # Claude 3.7 models
            "claude-3-7-sonnet-20250219": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-7-sonnet": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            # Claude 3.5 models
            "claude-3-5-sonnet-20241022": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-5-sonnet-20241022-v2": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-5-sonnet-20240620": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-5-sonnet": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-5-haiku-20241022": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-5-haiku": {
                "max_context": 200000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            # Claude 3 models
            "claude-3-opus-20240229": {
                "max_context": 200000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-sonnet-20240229": {
                "max_context": 200000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
            "claude-3-haiku-20240307": {
                "max_context": 200000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "anthropic",
            },
        }

        return capabilities_map.get(model_name)
