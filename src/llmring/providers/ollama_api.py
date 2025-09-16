"""
Ollama API provider implementation using the official SDK.
"""

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ollama import AsyncClient, ResponseError

from llmring.base import BaseLLMProvider, ProviderCapabilities, ProviderConfig
from llmring.exceptions import (
    CircuitBreakerError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from llmring.net.circuit_breaker import CircuitBreaker
from llmring.net.retry import retry_async
from llmring.registry import RegistryClient
from llmring.schemas import LLMResponse, Message, StreamChunk


class OllamaProvider(BaseLLMProvider):
    """Implementation of Ollama API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[
            str
        ] = None,  # Not used for Ollama, included for API compatibility
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
        base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

        # Create config for base class (no API key needed for Ollama)
        config = ProviderConfig(
            api_key=None,
            base_url=base_url,
            default_model=model,
            timeout_seconds=float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60")),
        )
        super().__init__(config)

        # Store for backward compatibility
        self.base_url = base_url
        self.default_model = model  # Will be derived from registry if None

        # Initialize the client with the SDK
        self.client = AsyncClient(host=base_url)

        # Registry client for model validation
        self._registry_client = RegistryClient()

        self._breaker = CircuitBreaker()

    async def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by Ollama using registry lookup.
        This is a best-effort check since Ollama can support any
        model that's installed locally.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise
        """
        # Strip provider prefix if present
        if model.lower().startswith("ollama:"):
            model = model.split(":", 1)[1]

        try:
            # Try registry validation
            return await self._registry_client.validate_model("ollama", model)
        except Exception as e:
            # If registry is unavailable, log warning and allow gracefully
            # Ollama supports any locally installed model - no hardcoded validation
            import logging
            logging.warning(f"Registry unavailable for Ollama model validation, allowing gracefully: {e}")
            return True


    async def get_supported_models(self) -> List[str]:
        """
        Get list of supported Ollama model names from registry.
        Note: This returns common models, but Ollama can support
        any model that's installed locally.

        Returns:
            List of supported model names
        """
        try:
            # Try to fetch from registry
            models = await self._registry_client.fetch_current_models("ollama")
            return [model.model_name for model in models if model.is_active]
        except Exception as e:
            # Registry unavailable - return empty list for graceful degradation
            import logging
            logging.warning(f"Registry unavailable for supported models, returning empty list: {e}")
            return []

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
            models = await self.get_supported_models()
            if models:
                # Use registry-based selection policy (no hardcoded preferences)
                selected_model = await self._select_default_from_registry(models)
                self.default_model = selected_model
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Ollama: Derived default model from registry: {selected_model}")
                return selected_model

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not derive default model from registry: {e}")

        # Ultimate fallback - use a sensible default when registry is unavailable
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("No default model available from registry, using fallback: llama3")
        self.default_model = "llama3"  # Reasonable fallback
        return self.default_model

    async def _select_default_from_registry(self, available_models: List[str]) -> str:
        """
        Select default model from registry using policy (no hardcoded preferences).

        Args:
            available_models: List of available model names from registry

        Returns:
            Selected default model
        """
        # Policy: Select based on registry metadata if available
        try:
            registry_models = await self._registry_client.fetch_current_models("ollama")
            active_models = [m for m in registry_models if m.is_active and m.model_name in available_models]

            if active_models:
                # Select most balanced: good capabilities + popularity indicators
                scored_models = []
                for model in active_models:
                    score = 0

                    # Prefer models with good capabilities (Ollama doesn't have function calling yet)
                    if model.supports_vision:
                        score += 5

                    # Prefer models with larger context windows
                    if model.max_input_tokens and model.max_input_tokens >= 32000:
                        score += 4

                    # Prefer more recent models (if added_date is available)
                    if model.added_date:
                        import datetime
                        days_since_added = (datetime.datetime.now(datetime.timezone.utc) - model.added_date).days
                        if days_since_added < 180:  # Less than 6 months old
                            score += 5

                    # For Ollama, prefer stable versions over experimental ones
                    model_name_lower = model.model_name.lower()
                    if ":latest" in model_name_lower or not ":" in model_name_lower:
                        score += 3  # Stable/latest versions
                    if "llama" in model_name_lower:  # Popular model family
                        score += 2

                    scored_models.append((model.model_name, score))

                if scored_models:
                    # Select highest scoring model
                    best_model = max(scored_models, key=lambda x: x[1])
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Ollama: Selected default model '{best_model[0]}' with score {best_model[1]} from registry analysis")
                    return best_model[0]

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not use registry metadata for Ollama selection: {e}")

        # Fallback: first available model
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Ollama: Fallback to first available model: {available_models[0]}")
        return available_models[0]

    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities
        """
        # Get current models for capabilities
        supported_models = await self.get_supported_models()

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
                    model_name = getattr(model, "name", None) or getattr(
                        model, "model", ""
                    )
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
        if not await self.validate_model(model):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Model '{model}' not found in registry, proceeding anyway"
            )

        # Convert messages to Ollama format (includes tool handling)
        ollama_messages = []
        for msg in messages:
            # Handle different message types
            if msg.role == "system":
                # System messages become assistant messages in Ollama
                ollama_messages.append(
                    {"role": "assistant", "content": f"System: {msg.content}"}
                )
            elif msg.role in ["user", "assistant"]:
                ollama_messages.append({"role": msg.role, "content": str(msg.content)})

        # Handle tools through prompt engineering (Ollama doesn't have native function calling)
        if tools:
            tools_prompt = self._create_tools_prompt(tools)
            if ollama_messages and ollama_messages[0]["role"] == "assistant":
                ollama_messages[0]["content"] += f"\n\n{tools_prompt}"
            else:
                ollama_messages.insert(
                    0, {"role": "assistant", "content": tools_prompt}
                )

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
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

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
                            "completion_tokens": self.get_token_count(
                                accumulated_content
                            ),
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
                raise ProviderResponseError(
                    f"Ollama error: {error_msg}", provider="ollama"
                ) from e

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
        if not await self.validate_model(model):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Model '{model}' not found in registry, proceeding anyway"
            )

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
                                            "arguments": json.dumps(
                                                tool_data.get("arguments", {})
                                            ),
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
            # Handle Ollama-specific errors
            error_msg = str(e.error)
            if "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Ollama API request timed out: {error_msg}", provider="ollama"
                ) from e
            else:
                raise ProviderResponseError(
                    f"Ollama API error: {error_msg}", provider="ollama"
                ) from e
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            try:
                await self._breaker.record_failure(f"ollama:{model}")
            except Exception:
                pass
            # Handle general errors
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Failed to connect to Ollama at {self.base_url}: {error_msg}",
                    provider="ollama",
                ) from e
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                raise ProviderResponseError(
                    f"Failed to connect to Ollama at {self.base_url}: {error_msg}",
                    provider="ollama",
                ) from e
            else:
                raise ProviderResponseError(
                    f"Ollama error: {error_msg}", provider="ollama"
                ) from e
