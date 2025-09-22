"""
Ollama API provider implementation using the official SDK.
"""

import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ollama import AsyncClient, ResponseError

from llmring.base import BaseLLMProvider, ProviderCapabilities, ProviderConfig
from llmring.exceptions import CircuitBreakerError, ProviderResponseError, ProviderTimeoutError
from llmring.net.circuit_breaker import CircuitBreaker
from llmring.net.retry import retry_async
from llmring.providers.base_mixin import ProviderLoggingMixin, RegistryModelSelectorMixin
from llmring.registry import RegistryClient
from llmring.schemas import LLMResponse, Message, StreamChunk


class OllamaProvider(BaseLLMProvider, RegistryModelSelectorMixin, ProviderLoggingMixin):
    """Implementation of Ollama API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,  # Not used for Ollama, included for API compatibility
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Ollama provider.

        Args:
            api_key: Not used for Ollama (included for API compatibility)
            base_url: Base URL for the Ollama API server
            model: Default model to use
        """
        # Get base URL from parameter or environment
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # Create config for base class (no API key needed for Ollama)
        config = ProviderConfig(
            api_key=None,
            base_url=base_url,
            default_model=model,
            timeout_seconds=float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60")),
        )
        super().__init__(config)

        # Initialize registry client BEFORE mixin init
        self._registry_client = RegistryClient()

        # Now initialize mixins that may use the registry client
        RegistryModelSelectorMixin.__init__(self)
        ProviderLoggingMixin.__init__(self, "ollama")

        # Store for backward compatibility
        self.base_url = base_url
        self.default_model = model  # Will be derived from registry if None

        # Initialize the client with the SDK
        self.client = AsyncClient(host=base_url)

        # Registry client already initialized before mixins

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
            # For Ollama, try registry first, then fall back to actual available models
            models = []
            if self._registry_client:
                try:
                    registry_models = await self._registry_client.fetch_current_models("ollama")
                    if registry_models:
                        models = [m.model_name for m in registry_models]
                except Exception:
                    pass

            # If no models from registry, try to get available models from Ollama
            if not models:
                try:
                    models = await self.get_available_models()
                except Exception:
                    pass

            if models:
                # Use registry-based selection with Ollama-specific cost range (usually free local models)
                selected_model = await self.select_default_from_registry(
                    provider_name="ollama",
                    available_models=models,
                    cost_range=(0.0, 0.1),  # Ollama models are typically free/local
                    fallback_model=None,  # No hardcoded fallback
                )
                self.default_model = selected_model
                self.log_info(f"Derived default model from registry: {selected_model}")
                return selected_model

        except Exception as e:
            self.log_warning(f"Could not derive default model from registry: {e}")

        # For Ollama, we can be more lenient since models are local
        # Just use the first available model if any
        if models:
            self.default_model = models[0]
            self.log_warning(f"Using first available Ollama model: {self.default_model}")
            return self.default_model

        # No models available at all
        raise ValueError(
            "No Ollama models available. Please install a model first using 'ollama pull'."
        )

    async def aclose(self) -> None:
        """Clean up provider resources."""
        # Ollama AsyncClient doesn't have a close method
        pass

    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities
        """
        # Get current models for capabilities
        # For Ollama, use actual available models since it's local
        try:
            supported_models = await self.get_available_models()
        except Exception:
            supported_models = []

        return ProviderCapabilities(
            provider_name="ollama",
            supported_models=supported_models,
            supports_streaming=True,
            supports_tools=False,  # Ollama doesn't support function calling yet
            supports_vision=True,  # Some models like llava support vision
            supports_audio=False,
            supports_documents=False,
            supports_json_mode=True,  # Via format parameter
            supports_caching=False,
            max_context_window=32768,  # Varies by model
            default_model=await self.get_default_model(),
        )

    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens (estimated)
        """
        # Rough estimate: ~4 characters per token for English text
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """
        Get list of models available in the local Ollama instance.

        Returns:
            List of available model names
        """
        try:
            # Use the Ollama SDK to list models
            response = await self.client.list()

            # Normalize to dict access
            if isinstance(response, dict):
                raw_models = response.get("models", [])
                models = []
                for m in raw_models:
                    name = m.get("name") or m.get("model")
                    if name:
                        models.append(name)
                return models

            # Fallback: object with attribute access
            models = []
            if hasattr(response, "models"):
                for model in response.models:
                    model_name = getattr(model, "name", None) or getattr(model, "model", "")
                    if model_name:
                        models.append(model_name)
            return models
        except Exception:
            # If we can't get the list, return empty list for graceful degradation
            return []

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
        Send a chat request to the Ollama API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "llama3.3")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            tools: Optional list of tools (implemented through prompt engineering)
            tool_choice: Optional tool choice parameter (implemented through prompt engineering)

        Returns:
            LLM response or async iterator of stream chunks if streaming
        """
        # Implement real streaming using Ollama SDK
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
        """Real streaming implementation using Ollama SDK."""
        # Strip provider prefix if present
        if model.lower().startswith("ollama:"):
            model = model.split(":", 1)[1]

        # Validate model (warn but don't fail if not in registry)
        # For Ollama, just check if model is available locally
        try:
            available = await self.get_available_models()
            if model not in available:
                # Check with flexible matching (base names)
                base_model = model.split(":")[0]
                if not any(base_model in m for m in available):
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Model '{model}' not found locally, proceeding anyway")
        except Exception:
            pass  # Can't check, continue anyway

        # Convert messages to Ollama format (includes tool handling)
        ollama_messages = []
        for msg in messages:
            # Handle different message types
            if msg.role == "system":
                # System messages become assistant messages in Ollama
                ollama_messages.append({"role": "assistant", "content": f"System: {msg.content}"})
            elif msg.role in ["user", "assistant"]:
                ollama_messages.append({"role": msg.role, "content": str(msg.content)})

        # Handle tools through prompt engineering (Ollama doesn't have native function calling)
        if tools:
            tools_prompt = self._create_tools_prompt(tools)
            if ollama_messages and ollama_messages[0]["role"] == "assistant":
                ollama_messages[0]["content"] += f"\n\n{tools_prompt}"
            else:
                ollama_messages.insert(0, {"role": "assistant", "content": tools_prompt})

        # Handle JSON response format
        if json_response or (
            response_format and response_format.get("type") in ["json_object", "json"]
        ):
            json_instruction = "\n\nIMPORTANT: Respond only with valid JSON."
            if ollama_messages:
                ollama_messages[-1]["content"] += json_instruction

        # Build request parameters for streaming
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request_params = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,  # Enable real streaming
            "options": options,
        }

        # Apply extra parameters
        # Supports Ollama-specific options: mirostat, penalty_*, num_ctx, seed, etc.
        # Can be passed as {"options": {"seed": 123}} or {"seed": 123}
        if extra_params:
            if "options" in extra_params:
                # Merge with existing options
                options.update(extra_params["options"])
                request_params["options"] = options
            else:
                # Apply directly to request
                request_params.update(extra_params)

        try:
            float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            key = f"ollama:{model}"
            if not await self._breaker.allow(key):
                raise CircuitBreakerError(
                    "Ollama circuit breaker is open - too many recent failures",
                    provider="ollama",
                )

            # Use real streaming API (returns async generator directly)
            stream_response = self.client.chat(**request_params)
            await self._breaker.record_success(key)

            # Process the streaming response
            accumulated_content = ""
            async for chunk in stream_response:
                if hasattr(chunk, "message") and chunk.message:
                    delta_content = chunk.message.get("content", "")
                    if delta_content:
                        accumulated_content += delta_content
                        yield StreamChunk(
                            delta=delta_content,
                            model=model,
                            finish_reason=None,
                        )

                # Check if this is the final chunk
                if hasattr(chunk, "done") and chunk.done:
                    # Final chunk with usage estimation
                    yield StreamChunk(
                        delta="",
                        model=model,
                        finish_reason="stop",
                        usage={
                            "prompt_tokens": self.get_token_count(str(ollama_messages)),
                            "completion_tokens": self.get_token_count(accumulated_content),
                            "total_tokens": self.get_token_count(str(ollama_messages))
                            + self.get_token_count(accumulated_content),
                        },
                    )

        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            await self._breaker.record_failure(key)
            error_msg = str(e)

            if "connect" in error_msg.lower():
                raise ProviderResponseError(
                    f"Failed to connect to Ollama at http://localhost:11434: {error_msg}",
                    provider="ollama",
                ) from e
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Ollama request timed out: {error_msg}", provider="ollama"
                ) from e
            else:
                raise ProviderResponseError(f"Ollama error: {error_msg}", provider="ollama") from e

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
        # Strip provider prefix if present
        if model.lower().startswith("ollama:"):
            model = model.split(":", 1)[1]

        # Note: We're more lenient with model validation for Ollama
        # since models are user-installed locally
        # For Ollama, just check if model is available locally
        try:
            available = await self.get_available_models()
            if model not in available:
                # Check with flexible matching (base names)
                base_model = model.split(":")[0]
                if not any(base_model in m for m in available):
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Model '{model}' not found locally, proceeding anyway")
        except Exception:
            pass  # Can't check, continue anyway

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        # Options for Ollama
        options = {}

        if temperature is not None:
            options["temperature"] = temperature

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Add performance options for faster responses
        options.update(
            {
                "top_k": 10,  # Limit vocabulary to top 10 tokens for faster sampling
                "top_p": 0.9,  # Use nucleus sampling for faster generation
                "repeat_penalty": 1.1,  # Light repeat penalty for speed
            }
        )

        # Handle tools through prompt engineering (Ollama doesn't natively support tools)
        if tools:
            tools_str = json.dumps(tools, indent=2)
            tool_instruction = (
                "\n\nYou have access to the following tools. When using a tool, respond with JSON "
                f'in the format {{"name": "tool_name", "arguments": {{...}}}}:\n{tools_str}\n'
            )

            # Add to system message if present, otherwise add to the last user message
            system_msg_idx = None
            for i, msg in enumerate(ollama_messages):
                if msg["role"] == "system":
                    system_msg_idx = i
                    break

            if system_msg_idx is not None:
                ollama_messages[system_msg_idx]["content"] += tool_instruction
            elif ollama_messages and ollama_messages[-1]["role"] == "user":
                ollama_messages[-1]["content"] += tool_instruction

        # Handle JSON response format
        format_param = None
        if response_format:
            if (
                response_format.get("type") == "json_object"
                or response_format.get("type") == "json"
            ):
                # Instruct Ollama to format response as JSON
                format_param = "json"

                # If a schema is provided, we can add it to the system message
                # or the last user message to guide the model
                if response_format.get("schema"):
                    schema_str = json.dumps(response_format["schema"], indent=2)
                    schema_instruction = f"\n\nPlease format your response as JSON that conforms to this schema:\n{schema_str}"

                    # Find a system message or use the last user message
                    system_msg_idx = None
                    for i, msg in enumerate(ollama_messages):
                        if msg["role"] == "system":
                            system_msg_idx = i
                            break

                    if system_msg_idx is not None:
                        ollama_messages[system_msg_idx]["content"] += schema_instruction
                    elif ollama_messages and ollama_messages[-1]["role"] == "user":
                        ollama_messages[-1]["content"] += schema_instruction

        try:
            # Use the Ollama SDK's chat method with a total deadline and retries
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            # Build request parameters
            request_params = {
                "model": model,
                "messages": ollama_messages,
                "stream": False,
                "options": options,
            }

            if format_param:
                request_params["format"] = format_param

            # Apply extra parameters if provided
            if extra_params:
                request_params.update(extra_params)

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.chat(**request_params),
                    timeout=timeout_s,
                )

            key = f"ollama:{model}"
            if not await self._breaker.allow(key):
                raise CircuitBreakerError(
                    "Ollama circuit breaker is open - too many recent failures",
                    provider="ollama",
                )
            response = await retry_async(_do_call)
            await self._breaker.record_success(key)

            # Extract the response content
            content = response["message"]["content"]

            # Parse for function calls if tools were provided
            tool_calls = None
            if tools and "```json" in content:
                # Try to extract function call from JSON code blocks
                try:
                    # Find JSON blocks
                    start_idx = content.find("```json")
                    if start_idx != -1:
                        json_text = content[start_idx + 7 :]
                        end_idx = json_text.find("```")
                        if end_idx != -1:
                            json_text = json_text[:end_idx].strip()
                            tool_data = json.loads(json_text)

                            # Basic structure expected: {"name": "...", "arguments": {...}}
                            if isinstance(tool_data, dict) and "name" in tool_data:
                                tool_calls = [
                                    {
                                        "id": f"call_{hash(json_text) & 0xFFFFFFFF:x}",  # Generate a deterministic ID
                                        "type": "function",
                                        "function": {
                                            "name": tool_data["name"],
                                            "arguments": json.dumps(tool_data.get("arguments", {})),
                                        },
                                    }
                                ]
                except (json.JSONDecodeError, KeyError):
                    # If extraction fails, just return the text response
                    pass

            # Get usage information
            eval_count = response.get("eval_count", 0)
            prompt_eval_count = response.get("prompt_eval_count", 0)

            usage = {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval_count + eval_count,
            }

            # Prepare the response
            llm_response = LLMResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason="stop",  # Ollama doesn't provide detailed finish reasons
            )

            # Add tool calls if present
            if tool_calls:
                llm_response.tool_calls = tool_calls

            return llm_response

        except ResponseError as e:
            # Ollama SDK ResponseError - wrap minimally with context preservation
            await self._breaker.record_failure(f"ollama:{model}")

            # Let the error message guide categorization, but don't over-analyze
            error_msg = str(getattr(e, "error", e))
            if "model" in error_msg.lower():
                from llmring.exceptions import ModelNotFoundError

                raise ModelNotFoundError(
                    f"Model '{model}' not available",
                    provider="ollama",
                    model_name=model,
                    original=e,
                ) from e

            # Default to generic provider error with full context
            raise ProviderResponseError(
                "Ollama API error",
                provider="ollama",
                original=e,
            ) from e
        except Exception as e:
            # Already wrapped? Just re-raise
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # Record failure (non-blocking)
            try:
                await self._breaker.record_failure(f"ollama:{model}")
            except Exception:
                pass

            # Handle RetryError - check the root cause
            from llmring.net.retry import RetryError

            if isinstance(e, RetryError):
                root = e.__cause__ if hasattr(e, "__cause__") else e

                # Timeout after retries
                if isinstance(root, asyncio.TimeoutError):
                    raise ProviderTimeoutError(
                        f"Request timed out after {getattr(e, 'attempts', 0)} attempts",
                        provider="ollama",
                        original=e,
                    ) from e

                # Ollama ResponseError after retries
                if "ResponseError" in type(root).__name__:
                    error_msg = str(getattr(root, "error", root))
                    if "model" in error_msg.lower():
                        from llmring.exceptions import ModelNotFoundError

                        raise ModelNotFoundError(
                            f"Model '{model}' not available",
                            provider="ollama",
                            model_name=model,
                            original=e,
                        ) from e

                # Generic retry exhaustion
                raise ProviderResponseError(
                    f"Request failed after {getattr(e, 'attempts', 0)} attempts",
                    provider="ollama",
                    original=e,
                ) from e

            # Direct timeout (no retry)
            if isinstance(e, asyncio.TimeoutError):
                raise ProviderTimeoutError(
                    "Request timed out",
                    provider="ollama",
                    original=e,
                ) from e

            # Connection errors
            if isinstance(e, (ConnectionError, OSError)):
                raise ProviderResponseError(
                    f"Cannot connect to Ollama at {self.base_url}",
                    provider="ollama",
                    original=e,
                ) from e

            # Unknown error - wrap minimally
            raise ProviderResponseError(
                "Unexpected error",
                provider="ollama",
                original=e,
            ) from e
