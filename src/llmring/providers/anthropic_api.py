"""
Anthropic Claude API provider implementation using the official SDK.
"""

import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage

# Note: do not call load_dotenv() in library code; handle in app entrypoints
from llmring.base import BaseLLMProvider, ProviderCapabilities, ProviderConfig
from llmring.exceptions import (
    CircuitBreakerError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from llmring.net.circuit_breaker import CircuitBreaker
from llmring.net.retry import retry_async
from llmring.schemas import LLMResponse, Message, StreamChunk


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
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ProviderAuthenticationError(
                "Anthropic API key must be provided", provider="anthropic"
            )

        # Create config for base class
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            default_model=model,
            timeout_seconds=float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60")),
        )
        super().__init__(config)

        # Store for backward compatibility
        self.api_key = api_key
        self.default_model = model

        # Initialize the client with the SDK
        # Include beta header for prompt caching (still needed as of 2025)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

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

    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities
        """
        return ProviderCapabilities(
            provider_name="anthropic",
            supported_models=self.supported_models.copy(),
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            supports_documents=True,  # Native document support
            supports_json_mode=False,  # No native JSON mode, but can be prompted
            supports_caching=True,  # Anthropic has prompt caching
            max_context_window=200000,  # Claude 3 models have 200K context
            default_model=self.default_model,
        )

    async def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Union[LLMResponse, AsyncIterator[StreamChunk]]:
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
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            stream: Whether to stream the response

        Returns:
            LLM response or async iterator of stream chunks if streaming
        """
        if stream:
            return self._stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                json_response=json_response,
                cache=cache,
                extra_params=extra_params,
            )

        return await self._chat_non_streaming(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            json_response=json_response,
            cache=cache,
            extra_params=extra_params,
        )

    async def _stream_chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response from Anthropic."""
        # Process model name
        original_model = model
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # Find latest version for models without date
        if "-202" not in model:
            for supported_model in self.supported_models:
                if supported_model.startswith(model + "-"):
                    model = supported_model
                    break

        # Verify model is supported
        if not self.validate_model(model):
            raise ModelNotFoundError(
                f"Unsupported model: {original_model}",
                provider="anthropic",
                model_name=original_model,
            )

        # Prepare messages and system prompt
        anthropic_messages, system_message, system_cache_control = (
            self._prepare_messages(messages)
        )

        # Build request parameters
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature or 0.7,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }

        if system_message:
            # Add system message with cache control if available
            if system_cache_control:
                request_params["system"] = [
                    {
                        "type": "text",
                        "text": system_message,
                        "cache_control": system_cache_control,
                    }
                ]
            else:
                request_params["system"] = system_message

        # Handle tools if provided
        if tools:
            request_params["tools"] = self._prepare_tools(tools)
            if tool_choice:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)

        # Apply extra parameters if provided
        if extra_params:
            request_params.update(extra_params)

        # Handle JSON response format
        if json_response or (
            response_format and response_format.get("type") in ["json_object", "json"]
        ):
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only."
            # Preserve cache control structure if present
            if isinstance(request_params.get("system"), list):
                # System is a list with cache control, append to text
                request_params["system"][0]["text"] += json_instruction
            elif request_params.get("system"):
                # System is a string, just append
                request_params["system"] += json_instruction
            else:
                # No system message yet
                request_params["system"] = json_instruction.strip()

        # Make streaming API call
        try:
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            stream = await asyncio.wait_for(
                self.client.messages.create(**request_params), timeout=timeout_s
            )

            # Process the stream
            accumulated_content = ""
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        accumulated_content += event.delta.text
                        yield StreamChunk(
                            delta=event.delta.text,
                            model=model,
                            finish_reason=None,
                        )
                elif event.type == "message_delta":
                    # Final event with usage information
                    if hasattr(event, "usage"):
                        usage_dict = {
                            "prompt_tokens": event.usage.input_tokens,
                            "completion_tokens": event.usage.output_tokens,
                            "total_tokens": event.usage.input_tokens
                            + event.usage.output_tokens,
                        }
                        # Add cache-related usage if available
                        if hasattr(event.usage, "cache_creation_input_tokens"):
                            usage_dict["cache_creation_input_tokens"] = (
                                event.usage.cache_creation_input_tokens
                            )
                        if hasattr(event.usage, "cache_read_input_tokens"):
                            usage_dict["cache_read_input_tokens"] = (
                                event.usage.cache_read_input_tokens
                            )

                        yield StreamChunk(
                            delta="",
                            model=model,
                            finish_reason=event.stop_reason
                            if hasattr(event, "stop_reason")
                            else "stop",
                            usage=usage_dict if event.usage else None,
                        )

        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            error_msg = str(e)
            if (
                "api key" in error_msg.lower()
                or "authentication_error" in error_msg.lower()
                or "x-api-key" in error_msg.lower()
            ):
                raise ProviderAuthenticationError(
                    f"Anthropic API authentication failed: {error_msg}",
                    provider="anthropic",
                ) from e
            elif "rate limit" in error_msg.lower():
                raise ProviderRateLimitError(
                    f"Anthropic API rate limit exceeded: {error_msg}",
                    provider="anthropic",
                ) from e
            else:
                raise ProviderResponseError(
                    f"Anthropic API error: {error_msg}", provider="anthropic"
                ) from e

    def _prepare_messages(
        self, messages: List[Message]
    ) -> tuple[List[Dict], Optional[str], Optional[Dict]]:
        """Convert messages to Anthropic format and extract system message with cache control."""
        anthropic_messages = []
        system_message = None
        system_cache_control = None

        for msg in messages:
            if msg.role == "system":
                system_message = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                # Check for cache control in system message metadata
                if (
                    hasattr(msg, "metadata")
                    and msg.metadata
                    and "cache_control" in msg.metadata
                ):
                    system_cache_control = msg.metadata["cache_control"]
            else:
                # Handle tool calls and responses
                if (
                    msg.role == "assistant"
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
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
                    content = self._format_message_content(msg.content)

                    # Add cache control to the last content block if present in metadata
                    if (
                        hasattr(msg, "metadata")
                        and msg.metadata
                        and "cache_control" in msg.metadata
                    ):
                        if content and isinstance(content, list) and len(content) > 0:
                            # Add cache_control to the last content block
                            content[-1]["cache_control"] = msg.metadata["cache_control"]

                    anthropic_messages.append({"role": msg.role, "content": content})

        return anthropic_messages, system_message, system_cache_control

    def _format_message_content(self, content: Any) -> List[Dict]:
        """Format message content to Anthropic's expected format."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            formatted = []
            for item in content:
                if isinstance(item, str):
                    formatted.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Convert OpenAI format to Anthropic
                        image_data = item["image_url"]["url"]
                        if image_data.startswith("data:"):
                            media_type, base64_data = (
                                image_data.split(";")[0].split(":")[1],
                                image_data.split(",")[1],
                            )
                            formatted.append(
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
                            formatted.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": image_data},
                                }
                            )
                    elif item.get("type") == "document":
                        # Anthropic supports documents directly
                        formatted.append(item)
                    else:
                        formatted.append(item)
            return formatted
        else:
            return [{"type": "text", "text": str(content)}]

    def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if "function" in tool:
                # OpenAI format
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            else:
                # Direct format
                anthropic_tools.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get(
                            "input_schema",
                            tool.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        ),
                    }
                )
        return anthropic_tools

    def _prepare_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert tool choice to Anthropic format."""
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "any":
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif isinstance(tool_choice, dict):
            return tool_choice
        else:
            return {"type": "auto"}

    async def _chat_non_streaming(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Non-streaming chat implementation."""
        # Process model name
        original_model = model
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # Find latest version for models without date
        if "-202" not in model:
            for supported_model in self.supported_models:
                if supported_model.startswith(model + "-"):
                    model = supported_model
                    break

        # Verify model is supported
        if not self.validate_model(model):
            raise ModelNotFoundError(
                f"Unsupported model: {original_model}",
                provider="anthropic",
                model_name=original_model,
            )

        # Convert messages to Anthropic format using _prepare_messages
        anthropic_messages, system_message, system_cache_control = (
            self._prepare_messages(messages)
        )

        # Build the request parameters
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature or 0.7,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            # Add system message with cache control if available
            if system_cache_control:
                request_params["system"] = [
                    {
                        "type": "text",
                        "text": system_message,
                        "cache_control": system_cache_control,
                    }
                ]
            else:
                request_params["system"] = system_message

        # Handle tools if provided
        if tools:
            request_params["tools"] = self._prepare_tools(tools)
            if tool_choice:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)

        # Apply extra parameters if provided
        if extra_params:
            request_params.update(extra_params)

        # Handle JSON response format
        if json_response or (
            response_format and response_format.get("type") in ["json_object", "json"]
        ):
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only."
            # Preserve cache control structure if present
            if isinstance(request_params.get("system"), list):
                # System is a list with cache control, append to text
                request_params["system"][0]["text"] += json_instruction
            elif request_params.get("system"):
                # System is a string, just append
                request_params["system"] += json_instruction
            else:
                # No system message yet
                request_params["system"] = json_instruction.strip()

        # Make the API call using the SDK
        try:
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.messages.create(**request_params), timeout=timeout_s
                )

            breaker_key = f"anthropic:{model}"
            if not await self._breaker.allow(breaker_key):
                raise CircuitBreakerError(
                    "Anthropic circuit breaker is open - too many recent failures",
                    provider="anthropic",
                )
            response: AnthropicMessage = await retry_async(_do_call)
            await self._breaker.record_success(breaker_key)
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            try:
                await self._breaker.record_failure(f"anthropic:{model}")
            except Exception:
                pass
            # Handle specific Anthropic errors with more context
            error_msg = str(e)
            if (
                "api key" in error_msg.lower()
                or "authentication_error" in error_msg.lower()
                or "x-api-key" in error_msg.lower()
            ):
                raise ProviderAuthenticationError(
                    f"Anthropic API authentication failed: {error_msg}",
                    provider="anthropic",
                ) from e
            elif "rate limit" in error_msg.lower():
                raise ProviderRateLimitError(
                    f"Anthropic API rate limit exceeded: {error_msg}",
                    provider="anthropic",
                ) from e
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(
                    f"Anthropic model not available: {error_msg}",
                    provider="anthropic",
                    model_name=model,
                ) from e
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Anthropic API request timed out: {error_msg}",
                    provider="anthropic",
                ) from e
            else:
                # Re-raise SDK exceptions with our standard format
                raise ProviderResponseError(
                    f"Anthropic API error: {error_msg}", provider="anthropic"
                ) from e

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

        # Prepare the response with cache-aware usage
        usage_dict = {
            "prompt_tokens": int(response.usage.input_tokens),
            "completion_tokens": int(response.usage.output_tokens),
            "total_tokens": int(
                response.usage.input_tokens + response.usage.output_tokens
            ),
        }

        # Add cache-related usage if available
        # The Anthropic SDK returns these as separate fields
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_input_tokens"] = (
                response.usage.cache_creation_input_tokens
            )
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage_dict["cache_read_input_tokens"] = (
                response.usage.cache_read_input_tokens
            )

        # Also check for cache_creation detail object which has ephemeral token info
        if hasattr(response.usage, "cache_creation") and response.usage.cache_creation:
            cache_creation = response.usage.cache_creation
            if hasattr(cache_creation, "ephemeral_5m_input_tokens"):
                usage_dict["cache_creation_5m_tokens"] = (
                    cache_creation.ephemeral_5m_input_tokens
                )
            if hasattr(cache_creation, "ephemeral_1h_input_tokens"):
                usage_dict["cache_creation_1h_tokens"] = (
                    cache_creation.ephemeral_1h_input_tokens
                )

        llm_response = LLMResponse(
            content=content.strip() if content else "",
            model=model,
            usage=usage_dict,
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
