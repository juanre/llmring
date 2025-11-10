# ABOUTME: Google Gemini provider implementation.
# ABOUTME: Handles chat, streaming, and multimodal content for Gemini models.
"""
Google Gemini API provider implementation using the official SDK.
"""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

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
from llmring.providers.error_handler import ProviderErrorHandler
from llmring.registry import RegistryClient
from llmring.schemas import FileMetadata, FileUploadResponse, LLMResponse, Message, StreamChunk
from llmring.utils import strip_provider_prefix

# Note: do not call load_dotenv() in library code; handle in app entrypoints

logger = logging.getLogger(__name__)


@dataclass
class UploadedFileInfo:
    """Track Google uploaded files for expiration management."""

    file_name: str  # Google file name (e.g., "files/abc123")
    expiration_time: datetime  # When file expires (48h from upload)
    local_path: Optional[str]  # Original file path for re-upload (None for file-like objects)


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

        self._registry_client = RegistryClient()
        RegistryModelSelectorMixin.__init__(self)
        ProviderLoggingMixin.__init__(self, "google")
        self.api_key = api_key
        self.project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID", "")
        self.default_model = model

        # Initialize the client
        self.client = genai.Client(api_key=api_key)

        # File upload tracking for expiration management
        self._uploaded_files: Dict[str, UploadedFileInfo] = {}

        self._breaker = CircuitBreaker()
        self._error_handler = ProviderErrorHandler("google", self._breaker)

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

        try:
            if self._registry_client:
                registry_models = await self._registry_client.fetch_current_models("google")
                if registry_models:
                    models = [m.model_name for m in registry_models]
                else:
                    models = []
            else:
                models = []

            if models:
                selected_model = await self.select_default_from_registry(
                    provider_name="google",
                    available_models=models,
                    cost_range=(0.05, 2.0),
                    fallback_model=None,
                )
                self.default_model = selected_model
                self.log_info(f"Derived default model from registry: {selected_model}")
                return selected_model

        except Exception as e:
            self.log_warning(f"Could not derive default model from registry: {e}")

        raise ValueError(
            "Could not determine default model from registry. "
            "Please specify a model explicitly or check your API configuration."
        )

    async def aclose(self) -> None:
        """Clean up provider resources."""
        # Google SDK doesn't use httpx client directly
        pass

    def _should_use_single_turn(self, messages: List[Message]) -> bool:
        """
        Determine if we can use single-turn generation (simpler path).

        Args:
            messages: List of conversation messages

        Returns:
            True if single-turn path can be used
        """
        return len(messages) == 1 and messages[0].role == "user"

    def _split_conversation(
        self, messages: List[Message]
    ) -> tuple[List[Message], Optional[Message]]:
        """
        Split conversation into history and current message.

        Args:
            messages: List of conversation messages

        Returns:
            Tuple of (history_messages, current_message)
        """
        if not messages:
            return [], None

        # Last message is current, rest is history
        return messages[:-1], messages[-1]

    def _parse_tool_response(self, content: Any) -> Dict[str, Any]:
        """
        Parse tool response content into a dictionary suitable for FunctionResponse.

        Args:
            content: The tool response content (str, dict, or other)

        Returns:
            Dictionary with the tool response
        """
        if isinstance(content, str):
            try:
                response_obj = json.loads(content)
                # Ensure response_obj is a dict (it might be a list or primitive)
                if not isinstance(response_obj, dict):
                    response_obj = {"result": response_obj}
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it as a result
                response_obj = {"result": content}
        elif isinstance(content, dict):
            response_obj = content
        else:
            # For any other type, convert to string and wrap
            response_obj = {"result": str(content)}

        return response_obj

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

    # Token counting not used for Google provider; usage is taken from SDK when available.

    async def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Send a chat request to the Google Gemini API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gemini-2.5-pro")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            reasoning_tokens: Token budget for reasoning models' internal thinking (ignored)
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration with cached_content name
            files: Optional list of cached_content IDs (Google uses cache mechanism for files)
            extra_params: Provider-specific parameters

        Returns:
            LLM response with complete generated content

        Note:
            Google uses cached_content for file handling. The files parameter should contain
            cached_content IDs (starting with "cachedContents/"). Use the cache parameter
            with {"cached_content": "cachedContents/..."} or pass files directly.
        """
        # Google uses cached_content for files, merge files into cache if provided
        merged_cache = cache or {}
        if files:
            # Use the first file as cached_content (Google's file mechanism)
            if len(files) > 1:
                logger.warning(
                    "Google provider only supports one cached_content at a time. Using first file: %s, ignoring others: %s",
                    files[0],
                    files[1:],
                )
            merged_cache["cached_content"] = files[0]

        return await self._chat_non_streaming(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_tokens=reasoning_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            json_response=json_response,
            cache=merged_cache if merged_cache else None,
            extra_params=extra_params,
        )

    async def chat_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a streaming chat request to the Google Gemini API.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gemini-2.5-pro")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            reasoning_tokens: Token budget for reasoning models' internal thinking (ignored)
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration with cached_content name
            files: Optional list of cached_content IDs (Google uses cache mechanism for files)
            extra_params: Provider-specific parameters

        Returns:
            Async iterator of stream chunks

        Example:
            >>> async for chunk in provider.chat_stream(messages, model="gemini-2.5-pro"):
            ...     print(chunk.content, end="", flush=True)
        """
        # Google uses cached_content for files, merge files into cache if provided
        merged_cache = cache or {}
        if files:
            # Use the first file as cached_content (Google's file mechanism)
            if len(files) > 1:
                logger.warning(
                    "Google provider only supports one cached_content at a time. Using first file: %s, ignoring others: %s",
                    files[0],
                    files[1:],
                )
            merged_cache["cached_content"] = files[0]

        return self._stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_tokens=reasoning_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            json_response=json_response,
            cache=merged_cache if merged_cache else None,
            extra_params=extra_params,
        )

    async def _stream_chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Real streaming implementation using Google SDK."""
        # reasoning_tokens is ignored for Google models
        # Process model name (remove provider prefix)
        model = strip_provider_prefix(model, "google")
        model = strip_provider_prefix(model, "gemini")

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

        # Convert conversation messages to Google format, including function calls and responses
        google_messages = []
        tool_name_by_id: Dict[str, str] = {}
        for msg in conversation_messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    parts = [types.Part(text=msg.content)]
                elif isinstance(msg.content, list):
                    parts = []
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part(text=item))
                        elif isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(types.Part(text=item.get("text", "")))
                            elif item.get("type") == "image_url":
                                image_data = item.get("image_url", {}).get("url", "")
                                if image_data.startswith("data:"):
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
                else:
                    parts = [types.Part(text=str(msg.content))]

                google_messages.append(types.Content(role="user", parts=parts))

            elif msg.role == "assistant":
                parts: List[types.Part] = []
                if isinstance(msg.content, str) and msg.content:
                    parts.append(types.Part(text=msg.content))
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part(text=item))
                        elif isinstance(item, dict) and item.get("type") == "text":
                            parts.append(types.Part(text=item.get("text", "")))

                if getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls or []:
                        name = call.get("function", {}).get("name")
                        args = call.get("function", {}).get("arguments")
                        if isinstance(args, str):
                            try:
                                args_dict = json.loads(args)
                            except json.JSONDecodeError:
                                args_dict = {}
                        elif isinstance(args, dict):
                            args_dict = args
                        else:
                            args_dict = {}

                        call_id = call.get("id")
                        if call_id and name:
                            tool_name_by_id[call_id] = name

                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(name=name or "", args=args_dict)
                            )
                        )

                google_messages.append(
                    types.Content(role="model", parts=parts or [types.Part(text="")])
                )

            elif msg.role == "tool":
                # Map tool outputs to function_response parts on user role
                response_obj = self._parse_tool_response(msg.content)

                call_id = getattr(msg, "tool_call_id", None)
                func_name = tool_name_by_id.get(call_id or "", "")

                google_messages.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=func_name or "",
                                    response=response_obj,
                                )
                            )
                        ],
                    )
                )

        try:
            key = f"google:{model}"
            if not await self._breaker.allow(key):
                raise CircuitBreakerError(
                    "Google circuit breaker is open - too many recent failures",
                    provider="google",
                )

            # Why: Google's SDK doesn't have native async support for streaming.
            # The generate_content_stream() method returns a synchronous generator.
            # We use threading to avoid blocking the event loop.
            stream_response = self.client.models.generate_content_stream(
                model=model,
                contents=google_messages,
                config=config,
            )
            await self._breaker.record_success(key)

            # Why: We run the sync generator in a thread pool and use a queue to
            # forward chunks to the async iterator. This allows the event loop to
            # remain responsive while consuming the sync stream.
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
                            # Map usage metadata on final streamed response if present
                            usage_payload = None
                            um = getattr(chunk, "usage_metadata", None) or getattr(
                                chunk, "usageMetadata", None
                            )
                            if um is not None:
                                prompt = getattr(um, "prompt_token_count", None) or getattr(
                                    um, "promptTokens", None
                                )
                                candidates_tok = getattr(
                                    um, "candidates_token_count", None
                                ) or getattr(um, "candidatesTokens", None)
                                total = getattr(um, "total_token_count", None) or getattr(
                                    um, "totalTokens", None
                                )
                                usage_payload = {
                                    "prompt_tokens": prompt,
                                    "completion_tokens": candidates_tok,
                                    "total_tokens": total,
                                }
                            yield StreamChunk(
                                delta="",
                                model=model,
                                finish_reason=finish_reason,
                                tool_calls=tool_calls if tool_calls else None,
                                usage=usage_payload,
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
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Non-streaming chat implementation."""
        # reasoning_tokens is ignored for Google models
        # Strip provider prefix if present
        model = strip_provider_prefix(model, "google")
        model = strip_provider_prefix(model, "gemini")

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
            if self._should_use_single_turn(conversation_messages):
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

            else:
                # For multi-turn conversations, construct proper history
                # Split conversation into history and current message
                history_messages, current_message = self._split_conversation(conversation_messages)

                # Convert history to google-genai format (include tool calls and responses)
                tool_name_by_id_hist: Dict[str, str] = {}
                for msg in history_messages:
                    if msg.role == "user":
                        converted_content = self._convert_content_to_google_format(msg.content)
                        if isinstance(converted_content, str):
                            parts = [types.Part(text=converted_content)]
                        else:
                            parts = converted_content
                        history.append(types.Content(role="user", parts=parts))
                    elif msg.role == "assistant":
                        parts: List[types.Part] = []
                        if isinstance(msg.content, str) and msg.content:
                            parts.append(types.Part(text=msg.content))
                        elif isinstance(msg.content, list):
                            for item in msg.content:
                                if isinstance(item, str):
                                    parts.append(types.Part(text=item))
                                elif isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(types.Part(text=item.get("text", "")))

                        # Add function_call parts if present
                        if getattr(msg, "tool_calls", None):
                            for call in msg.tool_calls or []:
                                name = call.get("function", {}).get("name")
                                args = call.get("function", {}).get("arguments")
                                if isinstance(args, str):
                                    try:
                                        args_dict = json.loads(args)
                                    except json.JSONDecodeError:
                                        args_dict = {}
                                elif isinstance(args, dict):
                                    args_dict = args
                                else:
                                    args_dict = {}

                                call_id = call.get("id")
                                if call_id and name:
                                    tool_name_by_id_hist[call_id] = name

                                parts.append(
                                    types.Part(
                                        function_call=types.FunctionCall(
                                            name=name or "", args=args_dict
                                        )
                                    )
                                )

                        history.append(
                            types.Content(role="model", parts=parts or [types.Part(text="")])
                        )

                    elif msg.role == "tool":
                        # Map tool outputs into function_response on the user role
                        response_obj = self._parse_tool_response(msg.content)

                        call_id = getattr(msg, "tool_call_id", None)
                        func_name = tool_name_by_id_hist.get(call_id or "", "")
                        history.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        function_response=types.FunctionResponse(
                                            name=func_name or "",
                                            response=response_obj,
                                        )
                                    )
                                ],
                            )
                        )

                # Create chat with history and send the current message
                def _run_chat():
                    chat = self.client.chats.create(model=api_model, config=config, history=history)
                    # Send current message. If it's a tool response, send function_response parts
                    if current_message.role == "tool":
                        # Reconstruct function_response part for current message
                        resp_obj = self._parse_tool_response(current_message.content)

                        call_id_cur = getattr(current_message, "tool_call_id", None)
                        # Try to find function name from previously seen calls
                        func_name_cur = ""
                        try:
                            func_name_cur = tool_name_by_id_hist.get(call_id_cur or "", "")
                        except Exception:
                            func_name_cur = ""

                        parts = [
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=func_name_cur or "",
                                    response=resp_obj,
                                )
                            )
                        ]
                        return chat.send_message(parts)
                    else:
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
        except Exception as e:
            await self._error_handler.handle_error(e, model)

        # Parse for native function calls from Google response
        tool_calls = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        tool_calls = tool_calls or []
                        tool_calls.append(
                            {
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": getattr(function_call, "name", ""),
                                    "arguments": (
                                        json.dumps(getattr(function_call, "args", {}))
                                        if getattr(function_call, "args", None) is not None
                                        else "{}"
                                    ),
                                },
                            }
                        )

        # Build content text from parts to avoid SDK non-text warning
        response_text_accum = ""
        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate0 = response.candidates[0]
                if hasattr(candidate0, "content") and candidate0.content:
                    for _part in candidate0.content.parts:
                        if hasattr(_part, "text") and _part.text:
                            response_text_accum += _part.text
        except Exception:
            response_text_accum = ""

        # Usage from SDK if available (usage_metadata)
        usage = None
        um = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        if um is not None:
            # Map common fields (names differ by SDK)
            prompt = getattr(um, "prompt_token_count", None) or getattr(um, "promptTokens", None)
            candidates = getattr(um, "candidates_token_count", None) or getattr(
                um, "candidatesTokens", None
            )
            total = getattr(um, "total_token_count", None) or getattr(um, "totalTokens", None)
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": candidates,
                "total_tokens": total,
            }

        # Prepare the response
        llm_response = LLMResponse(
            content=response_text_accum or "",
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

    async def upload_file(
        self,
        file: Union[str, Path, BinaryIO],
        purpose: str = "analysis",
        filename: Optional[str] = None,
        **kwargs,
    ) -> FileUploadResponse:
        """
        Upload file to Google File API (supports binary files up to 2GB).

        Args:
            file: File path, Path object, or file-like object
            purpose: Purpose of the file (not used by Google, for API consistency)
            filename: Optional filename (required for file-like objects)
            **kwargs: Additional provider-specific parameters

        Returns:
            FileUploadResponse with file_id, size, etc.

        Raises:
            FileSizeError: If file exceeds 2GB limit
            FileNotFoundError: If file path doesn't exist
            ProviderResponseError: If upload fails
        """
        from llmring.exceptions import FileSizeError

        # Google File API max size is 2GB
        MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

        # Handle file path or file-like object
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                raise FileSizeError(
                    f"File size {file_size} bytes exceeds Google limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="google",
                    filename=str(file_path),
                )

            actual_filename = filename or file_path.name
            local_path = str(file_path.absolute())

            # Upload using genai SDK
            try:
                # Run synchronous SDK call in thread pool
                loop = asyncio.get_event_loop()

                def _upload():
                    return self.client.files.upload(file=str(file_path))

                uploaded_file = await loop.run_in_executor(None, _upload)
            except Exception as e:
                await self._error_handler.handle_error(e, "files")

        else:
            # File-like object
            if filename is None:
                raise ValueError("filename parameter is required for file-like objects")

            # Read content to check size
            current_pos = file.tell() if hasattr(file, "tell") else 0
            file_content = file.read()
            file_size = len(file_content)

            # Reset file position if possible
            if hasattr(file, "seek"):
                file.seek(current_pos)

            # Check size
            if file_size > MAX_FILE_SIZE:
                raise FileSizeError(
                    f"File size {file_size} bytes exceeds Google limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="google",
                    filename=filename,
                )

            actual_filename = filename
            local_path = None  # Can't re-upload file-like objects

            # Upload using genai SDK
            try:
                loop = asyncio.get_event_loop()

                def _upload():
                    # Reset position again before upload
                    if hasattr(file, "seek"):
                        file.seek(current_pos)
                    return self.client.files.upload(file=file)

                uploaded_file = await loop.run_in_executor(None, _upload)
            except Exception as e:
                await self._error_handler.handle_error(e, "files")

        # Parse expiration time
        try:
            expiration_str = uploaded_file.expiration_time
            if isinstance(expiration_str, str):
                expiration_time = datetime.fromisoformat(expiration_str.replace("Z", "+00:00"))
            else:
                # Might be a datetime object already
                expiration_time = expiration_str
        except Exception:
            # Fallback: 48 hours from now
            from datetime import timedelta, timezone

            expiration_time = datetime.now(timezone.utc) + timedelta(hours=48)

        # Store metadata for expiration tracking
        self._uploaded_files[uploaded_file.name] = UploadedFileInfo(
            file_name=uploaded_file.name,
            expiration_time=expiration_time,
            local_path=local_path,
        )

        # Parse created_at time
        try:
            create_time_str = uploaded_file.create_time
            if isinstance(create_time_str, str):
                created_at = datetime.fromisoformat(create_time_str.replace("Z", "+00:00"))
            else:
                created_at = create_time_str
        except Exception:
            created_at = datetime.now()

        # Return standardized response
        return FileUploadResponse(
            file_id=uploaded_file.name,
            provider="google",
            filename=actual_filename,
            size_bytes=(
                int(uploaded_file.size_bytes) if hasattr(uploaded_file, "size_bytes") else file_size
            ),
            created_at=created_at,
            purpose=purpose,
            metadata={
                "mime_type": (
                    uploaded_file.mime_type if hasattr(uploaded_file, "mime_type") else None
                ),
                "expiration_time": str(expiration_time),
                "state": (
                    uploaded_file.state.name
                    if hasattr(uploaded_file, "state") and uploaded_file.state
                    else "ACTIVE"
                ),
            },
        )

    async def list_files(
        self, purpose: Optional[str] = None, limit: int = 100
    ) -> "List[FileMetadata]":
        """
        List context caches (Google's file list equivalent).

        Args:
            purpose: Optional filter by purpose (ignored for Google, always "cache")
            limit: Maximum number of caches to return (default 100)

        Returns:
            List of FileMetadata objects representing caches
        """
        try:
            # List caches using Google SDK
            # Run synchronous SDK call in thread pool
            loop = asyncio.get_event_loop()

            def _list_caches():
                return self.client.caches.list()

            cache_list = await loop.run_in_executor(None, _list_caches)

            # Parse response
            # Google SDK returns a Pager object that can be iterated
            files = []
            for cache in cache_list:
                # Extract metadata
                create_time = cache.create_time
                if isinstance(create_time, str):
                    create_time = datetime.fromisoformat(create_time.replace("Z", "+00:00"))

                files.append(
                    FileMetadata(
                        file_id=cache.name,  # e.g., "cachedContents/abc123"
                        provider="google",
                        filename=cache.name.split("/")[-1],  # Use cache ID as filename
                        size_bytes=0,  # Google doesn't expose cache size
                        created_at=create_time,
                        purpose="cache",
                        status="ready",  # Caches are always ready once created
                        metadata={
                            "model": cache.model,
                            "expire_time": str(cache.expire_time) if cache.expire_time else None,
                        },
                    )
                )

                # Limit results
                if len(files) >= limit:
                    break

            return files

        except Exception as e:
            await self._error_handler.handle_error(e, "caching")

    async def get_file(self, file_id: str) -> "FileMetadata":
        """
        Get cache metadata (Google's get file equivalent).

        Args:
            file_id: Cache ID to retrieve (format: "cachedContents/abc123")

        Returns:
            FileMetadata object
        """
        try:
            # Get cache using Google SDK
            # Run synchronous SDK call in thread pool
            loop = asyncio.get_event_loop()

            def _get_cache():
                return self.client.caches.get(name=file_id)

            cache = await loop.run_in_executor(None, _get_cache)

            # Parse response
            create_time = cache.create_time
            if isinstance(create_time, str):
                create_time = datetime.fromisoformat(create_time.replace("Z", "+00:00"))

            return FileMetadata(
                file_id=cache.name,
                provider="google",
                filename=cache.name.split("/")[-1],  # Use cache ID as filename
                size_bytes=0,  # Google doesn't expose cache size
                created_at=create_time,
                purpose="cache",
                status="ready",
                metadata={
                    "model": cache.model,
                    "expire_time": str(cache.expire_time) if cache.expire_time else None,
                },
            )

        except Exception as e:
            await self._error_handler.handle_error(e, "caching")

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete cache (Google's delete file equivalent).

        Args:
            file_id: Cache ID to delete (format: "cachedContents/abc123")

        Returns:
            True on success
        """
        try:
            # Delete cache using Google SDK
            # Run synchronous SDK call in thread pool
            loop = asyncio.get_event_loop()

            def _delete_cache():
                self.client.caches.delete(name=file_id)
                return True

            return await loop.run_in_executor(None, _delete_cache)

        except Exception as e:
            await self._error_handler.handle_error(e, "caching")
