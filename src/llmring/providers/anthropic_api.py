"""
Anthropic Claude API provider implementation using the official SDK.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage

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
from llmring.providers.base_mixin import ProviderLoggingMixin, RegistryModelSelectorMixin
from llmring.registry import RegistryClient
from llmring.schemas import LLMResponse, Message, StreamChunk

# Note: do not call load_dotenv() in library code; handle in app entrypoints

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider, RegistryModelSelectorMixin, ProviderLoggingMixin):
    """Implementation of Anthropic Claude API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
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

        # Initialize registry client BEFORE mixin init
        self._registry_client = RegistryClient()

        # Now initialize mixins that may use the registry client
        RegistryModelSelectorMixin.__init__(self)
        ProviderLoggingMixin.__init__(self, "anthropic")

        # Store for backward compatibility
        self.api_key = api_key
        self.default_model = model  # Will be derived from registry if None

        # Initialize the client with the SDK
        # Include beta header for prompt caching (still needed as of 2025)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        self._cached_models = None  # Will be populated from registry
        self._breaker = CircuitBreaker()

    async def get_default_model(self) -> str:
        """
        Get the default model to use, derived from registry if not specified.

        Returns:
            Default model name
        """
        if self.default_model:
            return self.default_model

        # Derive from registry using policy-based selection
        try:
            if self._registry_client:
                registry_models = await self._registry_client.fetch_current_models("anthropic")
                if registry_models:
                    # Extract model names from registry models
                    models = [m.model_name for m in registry_models]
                else:
                    models = []
            else:
                models = []

            if models:
                # Use registry-based selection with Anthropic-specific cost range
                selected_model = await self.select_default_from_registry(
                    provider_name="anthropic",
                    available_models=models,
                    cost_range=(0.1, 20.0),  # Anthropic's typical range
                    fallback_model=None,  # No hardcoded fallback
                )
                self.default_model = selected_model
                self.log_info(f"Derived default model from registry: {selected_model}")
                return selected_model

        except Exception as e:
            self.log_warning(f"Could not derive default model from registry: {e}")

        # No hardcoded fallback - require explicit model specification
        raise ValueError(
            "Could not determine default model from registry. "
            "Please specify a model explicitly or check your API configuration."
        )

    # Legacy method removed - now using RegistryModelSelectorMixin.select_default_from_registry()

    async def aclose(self) -> None:
        """Clean up provider resources."""
        if hasattr(self, "client") and self.client:
            await self.client.close()

    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities
        """
        # Get models from registry if available
        supported_models = []
        if self._registry_client:
            try:
                registry_models = await self._registry_client.fetch_current_models("anthropic")
                if registry_models:
                    supported_models = [m.model_name for m in registry_models]
            except Exception:
                pass  # Registry unavailable

        return ProviderCapabilities(
            provider_name="anthropic",
            supported_models=supported_models,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            supports_documents=True,  # Native document support
            supports_json_mode=False,  # No native JSON mode, but can be prompted
            supports_caching=True,  # Anthropic has prompt caching
            max_context_window=200000,  # Claude 3 models have 200K context
            default_model=await self.get_default_model(),
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
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # Note: Model name normalization removed - use exact model names from registry

        # Log warning if model not in registry (but don't block)
        try:
            registry_models = await self._registry_client.fetch_current_models("anthropic")
            if not any(m.model_name == model and m.is_active for m in registry_models):
                logger.warning(f"Model '{model}' not found in registry, proceeding anyway")
        except Exception:
            pass  # Registry unavailable, continue anyway

        # Prepare messages and system prompt
        anthropic_messages, system_message, system_cache_control = self._prepare_messages(messages)

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
                            "total_tokens": event.usage.input_tokens + event.usage.output_tokens,
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
                            finish_reason=(
                                event.stop_reason if hasattr(event, "stop_reason") else "stop"
                            ),
                            usage=usage_dict if event.usage else None,
                        )

        except Exception as e:
            # Already wrapped? Just re-raise
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # Handle known SDK exceptions
            from anthropic import AuthenticationError, RateLimitError

            if isinstance(e, AuthenticationError):
                raise ProviderAuthenticationError(
                    "Authentication failed",
                    provider="anthropic",
                    original=e,
                ) from e
            elif isinstance(e, RateLimitError):
                raise ProviderRateLimitError(
                    "Rate limit exceeded",
                    provider="anthropic",
                    retry_after=getattr(e, "retry_after", None),
                    original=e,
                ) from e

            # Unknown error - wrap minimally
            raise ProviderResponseError(
                "Streaming error",
                provider="anthropic",
                original=e,
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
                system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Check for cache control in system message metadata
                if hasattr(msg, "metadata") and msg.metadata and "cache_control" in msg.metadata:
                    system_cache_control = msg.metadata["cache_control"]
            else:
                # Handle tool calls and responses
                if msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tool_call in msg.tool_calls:
                        # Standard interface: arguments are JSON strings from providers
                        arguments = tool_call["function"]["arguments"]

                        # Parse JSON string to dict for Anthropic's input field
                        if isinstance(arguments, str):
                            try:
                                input_dict = json.loads(arguments)
                            except (json.JSONDecodeError, TypeError):
                                raise ValueError(
                                    f"Tool call arguments must be valid JSON: {arguments}"
                                )
                        elif isinstance(arguments, dict):
                            # Already a dict, use as-is
                            input_dict = arguments
                        else:
                            raise ValueError(
                                f"Tool call arguments must be JSON string or dict, got {type(arguments)}"
                            )

                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": input_dict,
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
                            tool.get("parameters", {"type": "object", "properties": {}}),
                        ),
                    }
                )
        return anthropic_tools

    def _prepare_tool_choice(self, tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
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
        if model.lower().startswith("anthropic:"):
            model = model.split(":", 1)[1]

        # Note: Model name normalization removed - use exact model names from registry

        # Log warning if model not in registry (but don't block)
        try:
            registry_models = await self._registry_client.fetch_current_models("anthropic")
            if not any(m.model_name == model and m.is_active for m in registry_models):
                logger.warning(f"Model '{model}' not found in registry, proceeding anyway")
        except Exception:
            pass  # Registry unavailable, continue anyway

        # Convert messages to Anthropic format using _prepare_messages
        anthropic_messages, system_message, system_cache_control = self._prepare_messages(messages)

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
            # Already wrapped? Just re-raise
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # Record failure (non-blocking)
            try:
                await self._breaker.record_failure(f"anthropic:{model}")
            except Exception:
                pass

            # Handle Anthropic SDK specific exceptions
            from anthropic import (
                AuthenticationError,
                BadRequestError,
                NotFoundError,
                RateLimitError,
            )

            from llmring.net.retry import RetryError

            # Check for RetryError wrapper first
            if isinstance(e, RetryError):
                root = e.__cause__ if hasattr(e, "__cause__") else e

                # Timeout after retries
                if isinstance(root, asyncio.TimeoutError):
                    raise ProviderTimeoutError(
                        f"Request timed out after {getattr(e, 'attempts', 0)} attempts",
                        provider="anthropic",
                        original=e,
                    ) from e

                # Anthropic SDK errors after retries
                if isinstance(root, NotFoundError):
                    raise ModelNotFoundError(
                        f"Model '{model}' not found",
                        provider="anthropic",
                        model_name=model,
                        original=e,
                    ) from e
                elif isinstance(root, AuthenticationError):
                    raise ProviderAuthenticationError(
                        "Authentication failed",
                        provider="anthropic",
                        original=e,
                    ) from e
                elif isinstance(root, RateLimitError):
                    raise ProviderRateLimitError(
                        "Rate limit exceeded",
                        provider="anthropic",
                        retry_after=getattr(root, "retry_after", None),
                        original=e,
                    ) from e

                # Generic retry exhaustion
                raise ProviderResponseError(
                    f"Request failed after {getattr(e, 'attempts', 0)} attempts",
                    provider="anthropic",
                    original=e,
                ) from e

            # Direct Anthropic SDK exceptions (no retry wrapper)
            if isinstance(e, NotFoundError):
                raise ModelNotFoundError(
                    f"Model '{model}' not found",
                    provider="anthropic",
                    model_name=model,
                    original=e,
                ) from e
            elif isinstance(e, AuthenticationError):
                raise ProviderAuthenticationError(
                    "Authentication failed",
                    provider="anthropic",
                    original=e,
                ) from e
            elif isinstance(e, RateLimitError):
                raise ProviderRateLimitError(
                    "Rate limit exceeded",
                    provider="anthropic",
                    retry_after=getattr(e, "retry_after", None),
                    original=e,
                ) from e
            elif isinstance(e, BadRequestError):
                # BadRequest often means model issue for Anthropic
                if "model" in str(e).lower():
                    raise ModelNotFoundError(
                        f"Model '{model}' not available",
                        provider="anthropic",
                        model_name=model,
                        original=e,
                    ) from e
                raise ProviderResponseError(
                    "Bad request",
                    provider="anthropic",
                    original=e,
                ) from e

            # Direct timeout (no retry)
            if isinstance(e, asyncio.TimeoutError):
                raise ProviderTimeoutError(
                    "Request timed out",
                    provider="anthropic",
                    original=e,
                ) from e

            # Unknown error - wrap minimally
            raise ProviderResponseError(
                "Unexpected error",
                provider="anthropic",
                original=e,
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
            "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),
        }

        # Add cache-related usage if available
        # The Anthropic SDK returns these as separate fields
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_input_tokens"] = response.usage.cache_creation_input_tokens
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage_dict["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

        # Also check for cache_creation detail object which has ephemeral token info
        if hasattr(response.usage, "cache_creation") and response.usage.cache_creation:
            cache_creation = response.usage.cache_creation
            if hasattr(cache_creation, "ephemeral_5m_input_tokens"):
                usage_dict["cache_creation_5m_tokens"] = cache_creation.ephemeral_5m_input_tokens
            if hasattr(cache_creation, "ephemeral_1h_input_tokens"):
                usage_dict["cache_creation_1h_tokens"] = cache_creation.ephemeral_1h_input_tokens

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
