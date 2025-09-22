"""
Google Gemini API provider implementation using the official SDK.
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from google import genai
from google.genai import types

from llmring.base import BaseLLMProvider, ProviderCapabilities, ProviderConfig
from llmring.exceptions import (
    CircuitBreakerError,
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


class GoogleProvider(BaseLLMProvider, RegistryModelSelectorMixin, ProviderLoggingMixin):
    """Implementation of Google Gemini API provider using the official google-genai library."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        project_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Google Gemini provider.

        Args:
            api_key: Google API key
            base_url: Optional base URL for the API (not used for Google)
            project_id: Google Cloud project ID (optional, for some use cases)
            model: Default model to use
        """
        # Get API key from parameter or environment
        api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GOOGLE_GEMINI_API_KEY", "")
        )
        if not api_key:
            raise ProviderAuthenticationError(
                "Google API key must be provided (GEMINI_API_KEY or GOOGLE_API_KEY)",
                provider="google",
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
        ProviderLoggingMixin.__init__(self, "google")

        # Store for backward compatibility
        self.api_key = api_key
        self.project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID", "")
        self.default_model = model  # Will be derived from registry if None

        # Initialize the client
        self.client = genai.Client(api_key=api_key)

        # Registry client already initialized before mixins

        # Note: Model name mapping removed - rely on registry for model availability
        self._breaker = CircuitBreaker()

    def _convert_content_to_google_format(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> Union[str, List[types.Part]]:
        """
        Convert OpenAI-style content format to Google genai format.

        Args:
            content: Either a string or list of content objects (OpenAI format)

        Returns:
            String for text-only, or list of types.Part for mixed content
        """
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue

            content_type = item.get("type", "")

            if content_type == "text":
                text_content = item.get("text", "")
                if text_content:
                    parts.append(types.Part(text=text_content))

            elif content_type == "image_url":
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url", "")

                # Handle data URL format: data:image/png;base64,<data>
                if url.startswith("data:"):
                    try:
                        # Extract mime type and base64 data
                        header, data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Decode base64 data
                        image_data = base64.b64decode(data)

                        # Create Google-style image part
                        parts.append(
                            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data))
                        )
                    except (ValueError, IndexError):
                        # Skip invalid image data
                        continue

            elif content_type == "document":
                # Handle universal document format
                source = item.get("source", {})
                if source.get("type") == "base64":
                    try:
                        mime_type = source.get("media_type", "application/pdf")
                        base64_data = source.get("data", "")

                        # Decode base64 data
                        document_data = base64.b64decode(base64_data)

                        # Create Google-style document part using inlineData
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(mime_type=mime_type, data=document_data)
                            )
                        )
                    except (ValueError, base64.binascii.Error):
                        # Skip invalid document data
                        continue

        return parts if parts else str(content)

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
                registry_models = await self._registry_client.fetch_current_models("google")
                if registry_models:
                    # Extract model names from registry models
                    models = [m.model_name for m in registry_models]
                else:
                    models = []
            else:
                models = []

            if models:
                # Use registry-based selection with Google-specific cost range
                selected_model = await self.select_default_from_registry(
                    provider_name="google",
                    available_models=models,
                    cost_range=(0.05, 2.0),  # Google's typical range
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
        # Google SDK doesn't use httpx client directly
        pass

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
                registry_models = await self._registry_client.fetch_current_models("google")
                if registry_models:
                    supported_models = [m.model_name for m in registry_models if m.is_active]
            except Exception:
                supported_models = []

        return ProviderCapabilities(
            provider_name="google",
            supported_models=supported_models,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,  # Gemini models support audio
            supports_documents=True,  # Gemini models support PDFs
            supports_json_mode=True,  # Via response_mime_type
            supports_caching=False,
            max_context_window=1000000,  # Gemini 1.5 Pro has 1M context
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

    # NOTE: Removed unused _convert_type_to_gemini method to reduce dead code

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
        Send a chat request to the Google Gemini API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gemini-2.5-pro")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            tools: Optional list of tools (not fully supported by Gemini yet)
            tool_choice: Optional tool choice parameter (not fully supported by Gemini yet)
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            stream: Whether to stream the response

        Returns:
            LLM response or async iterator of stream chunks if streaming
        """
        # Implement real streaming using Google SDK
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
        """Real streaming implementation using Google SDK."""
        # Process model name (remove provider prefix)
        if model.lower().startswith("google:") or model.lower().startswith("gemini:"):
            model = model.split(":", 1)[1]

        # Note: Model name normalization removed - use exact model names from registry

        # Log warning if model not in registry (but don't block)
        try:
            registry_models = await self._registry_client.fetch_current_models("google")
            if not any(m.model_name == model and m.is_active for m in registry_models):
                logger.warning(f"Model '{model}' not found in registry, proceeding anyway")
        except Exception:
            pass  # Registry unavailable, continue anyway

        # Extract system message and build conversation
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append(msg)

        # Build generation config
        config_params = {}

        if system_message:
            config_params["system_instruction"] = system_message

        if temperature is not None:
            config_params["temperature"] = temperature

        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        # Handle JSON response format
        if json_response or (
            response_format and response_format.get("type") in ["json_object", "json"]
        ):
            config_params["response_mime_type"] = "application/json"

        # Apply extra parameters
        if extra_params:
            config_params.update(extra_params)

        # Handle tools using native Google function calling
        google_tools = None
        if tools:
            google_tools = []
            for tool in tools:
                # Convert OpenAI/universal format to Google format
                if "function" in tool:
                    # OpenAI format: {"type": "function", "function": {"name": "...", "parameters": {...}}}
                    func = tool["function"]
                    tool_name = func["name"]
                    tool_description = func.get("description", "")
                    tool_parameters = func.get("parameters", {})
                else:
                    # Direct format: {"name": "...", "parameters": {...}}
                    tool_name = tool["name"]
                    tool_description = tool.get("description", "")
                    tool_parameters = tool.get("parameters", {})

                # Create Google FunctionDeclaration
                google_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool_name,
                                description=tool_description,
                                parameters=tool_parameters,
                            )
                        ]
                    )
                )

        # Add tools to config if present
        if google_tools:
            config_params["tools"] = google_tools

            # Handle tool_choice if provided
            if tool_choice:
                if tool_choice == "auto":
                    # Google default behavior
                    pass
                elif tool_choice == "none":
                    # Disable function calling
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="NONE")
                    )
                elif tool_choice == "any" or tool_choice == "required":
                    # Force function calling
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    )
                elif isinstance(tool_choice, dict) and "function" in tool_choice:
                    # Specific function choice - not directly supported by Google
                    # Fall back to ANY mode with the available tools
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    )

        config = types.GenerateContentConfig(**config_params)

        # Convert conversation messages to Google format
        google_messages = []
        for msg in conversation_messages:
            if msg.role in ["user", "assistant"]:
                # Handle different content types
                if isinstance(msg.content, str):
                    google_messages.append(
                        types.Content(
                            role="user" if msg.role == "user" else "model",
                            parts=[types.Part(text=msg.content)],
                        )
                    )
                elif isinstance(msg.content, list):
                    # Handle multimodal content
                    parts = []
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part(text=item))
                        elif isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(types.Part(text=item["text"]))
                            elif item.get("type") == "image_url":
                                # Convert image to Google format
                                image_data = item["image_url"]["url"]
                                if image_data.startswith("data:"):
                                    # Extract base64 data
                                    media_type, base64_data = (
                                        image_data.split(";")[0].split(":")[1],
                                        image_data.split(",")[1],
                                    )
                                    parts.append(
                                        types.Part(
                                            inline_data=types.Blob(
                                                mime_type=media_type,
                                                data=base64.b64decode(base64_data),
                                            )
                                        )
                                    )

                    if parts:
                        google_messages.append(
                            types.Content(
                                role="user" if msg.role == "user" else "model",
                                parts=parts,
                            )
                        )

        try:
            key = f"google:{model}"
            if not await self._breaker.allow(key):
                raise CircuitBreakerError(
                    "Google circuit breaker is open - too many recent failures",
                    provider="google",
                )

            # Start Google SDK streaming (sync generator)
            stream_response = self.client.models.generate_content_stream(
                model=model,
                contents=google_messages,
                config=config,
            )
            await self._breaker.record_success(key)

            # True streaming: iterate sync generator in a background thread and
            # forward chunks through an asyncio queue
            loop = asyncio.get_event_loop()
            import threading

            queue: asyncio.Queue = asyncio.Queue(maxsize=100)
            _SENTINEL = object()
            stop_event = threading.Event()

            def _producer():
                try:
                    for _chunk in stream_response:
                        if stop_event.is_set():
                            break
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, _chunk)
                        except RuntimeError:
                            # Loop is closed, exit gracefully
                            break
                except Exception as _e:
                    # Forward error to consumer if loop still running
                    if not stop_event.is_set():
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, ("__error__", _e))
                        except RuntimeError:
                            pass  # Loop is closed
                finally:
                    if not stop_event.is_set():
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                        except RuntimeError:
                            pass  # Loop is closed

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            accumulated_content = ""
            tool_calls: List[Dict[str, Any]] = []

            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple) and item and item[0] == "__error__":
                        e = item[1]
                        await self._breaker.record_failure(key)
                        error_msg = self._extract_error_message(e)
                        if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                            raise ProviderAuthenticationError(
                                f"Google API authentication failed: {error_msg}",
                                provider="google",
                            ) from e
                        elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                            raise ProviderRateLimitError(
                                f"Google API rate limit exceeded: {error_msg}",
                                provider="google",
                            ) from e
                        elif "timeout" in error_msg.lower():
                            raise ProviderTimeoutError(
                                f"Google API timeout: {error_msg}", provider="google"
                            ) from e
                        elif "cancelled" in error_msg.lower():
                            raise ProviderTimeoutError(
                                "Google API request timed out or was cancelled",
                                provider="google",
                            ) from e
                        elif (
                            "not found" in error_msg.lower() and "model" in error_msg.lower()
                        ) or "not supported" in error_msg.lower():
                            from llmring.exceptions import ModelNotFoundError

                            raise ModelNotFoundError(
                                f"Google model not available: {error_msg}",
                                provider="google",
                                model_name=model,
                            ) from e
                        else:
                            raise ProviderResponseError(
                                f"Google API error: {error_msg}", provider="google"
                            ) from e

                    chunk = item
                    if chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]

                        if hasattr(candidate, "content") and candidate.content:
                            # Extract text and function calls from content parts
                            chunk_text = ""
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    chunk_text += part.text
                                elif hasattr(part, "function_call") and part.function_call:
                                    function_call = part.function_call
                                    tool_calls.append(
                                        {
                                            "id": f"call_{len(tool_calls)}",
                                            "type": "function",
                                            "function": {
                                                "name": function_call.name,
                                                "arguments": (
                                                    json.dumps(function_call.args)
                                                    if function_call.args
                                                    else "{}"
                                                ),
                                            },
                                        }
                                    )

                            if chunk_text:
                                accumulated_content += chunk_text
                                yield StreamChunk(
                                    delta=chunk_text,
                                    model=model,
                                    finish_reason=None,
                                )

                        # Finish reason indicates end of stream
                        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                            finish_reason = str(candidate.finish_reason).lower()
                            yield StreamChunk(
                                delta="",
                                model=model,
                                finish_reason=finish_reason,
                                tool_calls=tool_calls if tool_calls else None,
                                usage={
                                    "prompt_tokens": self.get_token_count(str(google_messages)),
                                    "completion_tokens": self.get_token_count(accumulated_content),
                                    "total_tokens": self.get_token_count(str(google_messages))
                                    + self.get_token_count(accumulated_content),
                                },
                            )
                            break
            finally:
                # Signal the producer thread to stop
                stop_event.set()

        except Exception as e:
            # Already wrapped? Just re-raise
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # Record failure (non-blocking)
            try:
                await self._breaker.record_failure(key)
            except Exception:
                pass

            # Check for timeout
            if isinstance(e, asyncio.TimeoutError):
                raise ProviderTimeoutError(
                    "Request timed out",
                    provider="google",
                    original=e,
                ) from e

            # Parse error message for categorization
            error_msg = self._extract_error_message(e)
            if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ProviderAuthenticationError(
                    "Authentication failed",
                    provider="google",
                    original=e,
                ) from e
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise ProviderRateLimitError(
                    "Rate limit exceeded",
                    provider="google",
                    original=e,
                ) from e

            # Unknown error - wrap minimally
            raise ProviderResponseError(
                "Unexpected error",
                provider="google",
                original=e,
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
        if model.lower().startswith("google:"):
            model = model.split(":", 1)[1]

        # Log warning if model not in registry (but don't block)
        try:
            registry_models = await self._registry_client.fetch_current_models("google")
            if not any(m.model_name == model and m.is_active for m in registry_models):
                logger.warning(f"Model '{model}' not found in registry, proceeding anyway")
        except Exception:
            pass  # Registry unavailable, continue anyway

        # Use model name as provided (no hardcoded mapping)
        api_model = model

        # Extract system message and build conversation history
        system_message = None
        conversation_messages = []
        history = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append(msg)

        # Prepare config
        config_params = {}

        if system_message:
            config_params["system_instruction"] = system_message

        if temperature is not None:
            config_params["temperature"] = temperature

        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        # Handle tools using native Google function calling
        google_tools = None
        if tools:
            google_tools = []
            for tool in tools:
                # Convert OpenAI/universal format to Google format
                if "function" in tool:
                    # OpenAI format: {"type": "function", "function": {"name": "...", "parameters": {...}}}
                    func = tool["function"]
                    tool_name = func["name"]
                    tool_description = func.get("description", "")
                    tool_parameters = func.get("parameters", {})
                else:
                    # Direct format: {"name": "...", "parameters": {...}}
                    tool_name = tool["name"]
                    tool_description = tool.get("description", "")
                    tool_parameters = tool.get("parameters", {})

                # Create Google FunctionDeclaration
                google_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool_name,
                                description=tool_description,
                                parameters=tool_parameters,
                            )
                        ]
                    )
                )

        # Handle JSON response format (schema injection handled by service adapter)
        if response_format:
            if (
                response_format.get("type") == "json_object"
                or response_format.get("type") == "json"
            ):
                config_params["response_mime_type"] = "application/json"

        # Apply extra parameters if provided
        if extra_params:
            config_params.update(extra_params)

        # Add tools to config if present
        if google_tools:
            config_params["tools"] = google_tools

            # Handle tool_choice if provided
            if tool_choice:
                if tool_choice == "auto":
                    # Google default behavior
                    pass
                elif tool_choice == "none":
                    # Disable function calling
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="NONE")
                    )
                elif tool_choice == "any" or tool_choice == "required":
                    # Force function calling
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    )
                elif isinstance(tool_choice, dict) and "function" in tool_choice:
                    # Specific function choice - not directly supported by Google
                    # Fall back to ANY mode with the available tools
                    config_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    )

        config = types.GenerateContentConfig(**config_params) if config_params else None

        # Execute in thread pool since google-genai is synchronous
        loop = asyncio.get_event_loop()

        try:
            # For single user message, we can use generate_content directly
            if len(conversation_messages) == 1 and conversation_messages[0].role == "user":
                msg = conversation_messages[0]

                # Convert content to Google format
                converted_content = self._convert_content_to_google_format(msg.content)

                # Run synchronous operation in thread pool
                total_timeout = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

                async def _do_call():
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.client.models.generate_content(
                                model=api_model,
                                contents=converted_content,
                                config=config,
                            ),
                        ),
                        timeout=total_timeout,
                    )

                key = f"google:{api_model}"
                if not await self._breaker.allow(key):
                    raise CircuitBreakerError(
                        "Google circuit breaker is open - too many recent failures",
                        provider="google",
                    )
                response = await retry_async(_do_call)
                await self._breaker.record_success(key)

                response_text = response.text

            else:
                # For multi-turn conversations, construct proper history
                # Split conversation into history and current message
                if conversation_messages:
                    # Build history from all messages except the last user message
                    current_message = None
                    history_messages = []

                    # Find the last user message
                    for i in reversed(range(len(conversation_messages))):
                        if conversation_messages[i].role == "user":
                            current_message = conversation_messages[i]
                            history_messages = conversation_messages[:i]
                            break

                    if not current_message:
                        # No user message found, treat the last message as the query
                        current_message = conversation_messages[-1]
                        history_messages = conversation_messages[:-1]

                    # Convert history to google-genai format
                    for msg in history_messages:
                        if msg.role == "user":
                            converted_content = self._convert_content_to_google_format(msg.content)
                            if isinstance(converted_content, str):
                                parts = [types.Part(text=converted_content)]
                            else:
                                parts = converted_content
                            history.append(types.Content(role="user", parts=parts))
                        elif msg.role == "assistant":
                            history.append(
                                types.Content(role="model", parts=[types.Part(text=msg.content)])
                            )

                    # Create chat with history and send the current message
                    def _run_chat():
                        chat = self.client.chats.create(
                            model=api_model, config=config, history=history
                        )
                        converted_content = self._convert_content_to_google_format(
                            current_message.content
                        )
                        return chat.send_message(converted_content)

                    # Run the chat in thread pool
                    total_timeout = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

                    async def _do_chat():
                        return await asyncio.wait_for(
                            loop.run_in_executor(None, _run_chat), timeout=total_timeout
                        )

                    key = f"google:{api_model}"
                    if not await self._breaker.allow(key):
                        raise CircuitBreakerError(
                            "Google circuit breaker is open - too many recent failures",
                            provider="google",
                        )
                    response = await retry_async(_do_chat)
                    await self._breaker.record_success(key)
                    response_text = response.text
                else:
                    # No messages? This shouldn't happen but handle gracefully
                    raise ProviderResponseError("No messages provided for chat", provider="google")
        except Exception as e:
            # Already wrapped? Just re-raise
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # Record failure (non-blocking)
            try:
                await self._breaker.record_failure(f"google:{model}")
            except Exception:
                pass

            # Handle RetryError
            from llmring.net.retry import RetryError

            if isinstance(e, RetryError):
                root = e.__cause__ if hasattr(e, "__cause__") else e

                # Timeout after retries
                if isinstance(root, asyncio.TimeoutError):
                    raise ProviderTimeoutError(
                        f"Request timed out after {getattr(e, 'attempts', 0)} attempts",
                        provider="google",
                        original=e,
                    ) from e

                # Generic retry exhaustion
                raise ProviderResponseError(
                    f"Request failed after {getattr(e, 'attempts', 0)} attempts",
                    provider="google",
                    original=e,
                ) from e

            # Direct timeout (no retry)
            if isinstance(e, asyncio.TimeoutError):
                raise ProviderTimeoutError(
                    "Request timed out",
                    provider="google",
                    original=e,
                ) from e

            # Parse error message for categorization
            error_msg = self._extract_error_message(e)
            if "model" in error_msg.lower() and (
                "not found" in error_msg.lower() or "not supported" in error_msg.lower()
            ):
                from llmring.exceptions import ModelNotFoundError

                raise ModelNotFoundError(
                    f"Model '{model}' not available",
                    provider="google",
                    model_name=model,
                    original=e,
                ) from e
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise ProviderRateLimitError(
                    "Rate limit exceeded",
                    provider="google",
                    original=e,
                ) from e
            elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ProviderAuthenticationError(
                    "Authentication failed",
                    provider="google",
                    original=e,
                ) from e

            # Unknown error - wrap minimally
            raise ProviderResponseError(
                "Unexpected error",
                provider="google",
                original=e,
            ) from e

        # Parse for native function calls from Google response
        tool_calls = None
        if tools and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                tool_calls = []
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        # Handle native function calls
                        function_call = part.function_call
                        tool_calls.append(
                            {
                                "id": f"call_{len(tool_calls)}",  # Google doesn't provide IDs
                                "type": "function",
                                "function": {
                                    "name": function_call.name,
                                    "arguments": (
                                        json.dumps(function_call.args)
                                        if function_call.args
                                        else "{}"
                                    ),
                                },
                            }
                        )

                # If no tool calls found, reset to None
                if not tool_calls:
                    tool_calls = None

        # Simple usage tracking (google-genai doesn't provide detailed token counts)
        usage = {
            "prompt_tokens": self.get_token_count("\n".join([str(m.content) for m in messages])),
            "completion_tokens": self.get_token_count(response_text or ""),
            "total_tokens": 0,  # Will be calculated below
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        # Prepare the response
        llm_response = LLMResponse(
            content=response_text or "",
            model=model,  # Return the original model name
            usage=usage,
            finish_reason="stop",  # google-genai doesn't provide this
        )

        # Add tool calls if present
        if tool_calls:
            llm_response.tool_calls = tool_calls

        return llm_response

    def _extract_error_message(self, exc: BaseException) -> str:
        """Extract a meaningful error message from an exception chain.

        Prefers deeper cause/context messages; handles timeouts explicitly; avoids
        returning the generic 'unknown error' sentinel.
        """
        seen = set()

        def _walk(e: BaseException) -> str:
            if e is None or id(e) in seen:
                return ""
            seen.add(id(e))

            # Prefer deeper causes/contexts first
            cause = getattr(e, "__cause__", None)
            ctx = getattr(e, "__context__", None)
            deeper = _walk(cause) or _walk(ctx)
            if deeper and deeper.strip() and deeper.strip().lower() != "unknown error":
                return deeper

            # Explicit timeout handling
            try:
                import asyncio as _asyncio  # local import to avoid cycles
            except Exception:
                _asyncio = None

            if (
                hasattr(_asyncio, "TimeoutError") and isinstance(e, _asyncio.TimeoutError)
            ) or isinstance(e, TimeoutError):
                return "Request timed out"
            if hasattr(_asyncio, "CancelledError") and isinstance(e, _asyncio.CancelledError):
                return "Request was cancelled (likely due to timeout)"

            # Use this exception's own message if meaningful
            msg = (str(e) or "").strip()
            if msg and msg.lower() != "unknown error":
                return msg

            # Fallback to type name
            return f"{type(e).__name__} (no message available)"

        return _walk(exc)
