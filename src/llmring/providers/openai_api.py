"""OpenAI provider implementation for GPT models. Handles chat, streaming, vision, reasoning tokens, and function calling."""

import asyncio
import base64
import copy
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types import FilePurpose
from openai.types.chat import ChatCompletion

from llmring.base import (
    TIMEOUT_UNSET,
    BaseLLMProvider,
    ProviderCapabilities,
    ProviderConfig,
    TimeoutSetting,
    resolve_timeout_config,
)
from llmring.exceptions import (
    CircuitBreakerError,
    FileSizeError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from llmring.net.circuit_breaker import CircuitBreaker
from llmring.net.retry import retry_async
from llmring.net.safe_fetcher import SafeFetchError
from llmring.net.safe_fetcher import fetch_bytes as safe_fetch_bytes
from llmring.providers.base_mixin import ProviderLoggingMixin, RegistryModelSelectorMixin
from llmring.providers.error_handler import ProviderErrorHandler
from llmring.registry import RegistryClient
from llmring.schemas import FileMetadata, FileUploadResponse, LLMResponse, Message, StreamChunk
from llmring.utils import strip_provider_prefix
from llmring.validation import InputValidator


class OpenAIProvider(BaseLLMProvider, RegistryModelSelectorMixin, ProviderLoggingMixin):
    """Implementation of OpenAI API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: TimeoutSetting = TIMEOUT_UNSET,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for the API
            model: Default model to use
        """
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderAuthenticationError("OpenAI API key must be provided", provider="openai")

        # Create config for base class
        config = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            default_model=model or "",
            timeout_seconds=resolve_timeout_config(
                timeout, os.getenv("LLMRING_PROVIDER_TIMEOUT_S")
            ),
        )
        super().__init__(config)

        self._registry_client = RegistryClient()
        RegistryModelSelectorMixin.__init__(self)
        ProviderLoggingMixin.__init__(self, "openai")
        self.api_key = api_key
        self.default_model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._breaker = CircuitBreaker()
        self._error_handler = ProviderErrorHandler("openai", self._breaker)

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
                registry_models = await self._registry_client.fetch_current_models("openai")
                if registry_models:
                    models = [m.model_name for m in registry_models]
                    selected_model = await self.select_default_from_registry(
                        provider_name="openai",
                        available_models=models,
                        cost_range=(0.1, 5.0),
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
        if hasattr(self, "client") and self.client:
            await self.client.close()

    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities
        """
        supported_models = []
        if self._registry_client:
            try:
                registry_models = await self._registry_client.fetch_current_models("openai")
                if registry_models:
                    supported_models = [m.model_name for m in registry_models]
            except Exception:
                pass  # Registry unavailable

        return ProviderCapabilities(
            provider_name="openai",
            supported_models=supported_models,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            supports_documents=True,
            supports_json_mode=True,
            supports_caching=False,
            max_context_window=128000,
            default_model=self.config.default_model or self.default_model or "gpt-4o-mini",
        )

    def _contains_pdf_content(self, messages: List[Message]) -> bool:
        """
        Check if any message contains PDF document content.

        Args:
            messages: List of messages to check

        Returns:
            True if PDF content is found, False otherwise
        """
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "document":
                        source = part.get("source", {})
                        media_type = source.get("media_type", "")
                        if media_type == "application/pdf":
                            return True
        return False

    def _extract_pdf_content_and_text(self, messages: List[Message]) -> tuple[List[bytes], str]:
        """
        Extract PDF content and combine all text content from messages.

        Args:
            messages: List of messages to process

        Returns:
            Tuple of (pdf_data_list, combined_text)
        """
        pdf_data_list = []
        text_parts = []

        for msg in messages:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "document":
                            source = part.get("source", {})
                            if (
                                source.get("type") == "base64"
                                and source.get("media_type") == "application/pdf"
                            ):
                                # Safely decode base64 with size validation
                                base64_data = source.get("data", "")
                                pdf_data = InputValidator.safe_decode_base64(
                                    base64_data, "PDF document"
                                )
                                pdf_data_list.append(pdf_data)
                        elif isinstance(part, str):
                            text_parts.append(part)

        combined_text = " ".join(text_parts)
        return pdf_data_list, combined_text

    async def _process_with_pdf_upload(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Process messages containing PDFs using Chat Completions API with file upload."""
        # Extract PDF data and text from messages
        pdf_data_list, combined_text = self._extract_pdf_content_and_text(messages)
        if not pdf_data_list:
            raise ProviderResponseError("No PDF content found in messages", provider="openai")
        if not combined_text.strip():
            combined_text = "Please analyze this PDF document and provide a summary."

        uploaded_files: List[Dict[str, str]] = []
        try:
            for i, pdf_data in enumerate(pdf_data_list):
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix=f"_document_{i}.pdf", delete=False, mode="wb"
                )
                tmp_file.write(pdf_data)
                tmp_file.flush()
                tmp_file.close()  # Close before reopening to avoid Windows file locking issues

                # Open in binary read mode for upload
                with open(tmp_file.name, "rb") as f:
                    # PDFs use 'user_data' purpose for Chat Completions API file type
                    file_obj = await self.client.files.create(file=f, purpose="user_data")
                    if not file_obj or not file_obj.id:
                        raise ProviderResponseError(
                            f"Failed to upload PDF file {i + 1}: no file ID returned",
                            provider="openai",
                        )
                    uploaded_files.append({"file_id": file_obj.id, "temp_path": tmp_file.name})

            # Build Chat Completions message with file references
            content_parts: List[Dict[str, Any]] = []
            for info in uploaded_files:
                content_parts.append({"type": "file", "file": {"file_id": info["file_id"]}})
            content_parts.append({"type": "text", "text": combined_text})

            request_params: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": content_parts}],
            }

            if temperature is not None:
                request_params["temperature"] = temperature
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Apply extra parameters if provided
            if extra_params:
                request_params.update(extra_params)

            resp = await self._await_with_timeout(
                self.client.chat.completions.create(**request_params),
                timeout,
            )

            if resp.choices:
                response_content = resp.choices[0].message.content or ""
                finish_reason = resp.choices[0].finish_reason
            else:
                response_content = ""
                finish_reason = "stop"
            usage = self._map_responses_usage(resp.usage)
            return LLMResponse(
                content=response_content,
                model=model,
                usage=usage,
                finish_reason=finish_reason,
            )
        finally:
            # Cleanup uploaded files
            tasks = []
            for info in uploaded_files:
                tasks.append(self.client.files.delete(info["file_id"]))
                try:
                    os.unlink(info["temp_path"])
                except OSError:
                    pass
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception:
                    pass

    async def _get_model_config(self, model: str) -> Dict[str, Any]:
        """
        Get model configuration from registry.

        Args:
            model: Model name (without provider prefix)

        Returns:
            Dictionary with model configuration:
            - is_reasoning: Whether model uses reasoning tokens
            - min_recommended_reasoning_tokens: Minimum recommended token budget
            - api_endpoint: Preferred API endpoint (chat/responses)
        """
        try:
            if self._registry_client:
                models = await self._registry_client.fetch_current_models("openai")
                model_info = next((m for m in models if m.model_name == model), None)

                if model_info:
                    # Only use fallback default for reasoning models
                    min_reasoning_tokens = model_info.min_recommended_reasoning_tokens
                    if model_info.is_reasoning_model and min_reasoning_tokens is None:
                        min_reasoning_tokens = 2000

                    return {
                        "is_reasoning": model_info.is_reasoning_model,
                        "min_recommended_reasoning_tokens": min_reasoning_tokens,
                        "api_endpoint": model_info.api_endpoint,
                    }
        except Exception as e:
            self.log_debug(f"Could not fetch model config from registry: {e}")

        # Fallback to string matching for backward compatibility
        is_reasoning = model.startswith(("o1", "o3", "gpt-5"))
        return {
            "is_reasoning": is_reasoning,
            "min_recommended_reasoning_tokens": 2000 if is_reasoning else None,
            "api_endpoint": None,
        }

    async def _chat_via_responses(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Handle o1* models using the Responses API.
        """
        # Convert messages to Responses input preserving roles/content
        responses_input = self._messages_to_responses_input(messages)

        try:
            request_params = {
                "model": model,
                "input": responses_input,
            }

            # temperature and max tokens support may vary; pass only if provided
            if temperature is not None:
                request_params["temperature"] = temperature
            # Only pass max_tokens if provided
            if max_tokens is not None:
                request_params["max_output_tokens"] = max_tokens

            # Apply extra parameters if provided
            if extra_params:
                request_params.update(extra_params)

            resp = await self._await_with_timeout(
                self.client.responses.create(**request_params), timeout
            )
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            error_msg = str(e)
            if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ProviderAuthenticationError(
                    f"OpenAI API authentication failed: {error_msg}", provider="openai"
                ) from e
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise ProviderRateLimitError(
                    f"OpenAI API rate limit exceeded: {error_msg}", provider="openai"
                ) from e
            elif "model" in error_msg.lower() and (
                "not found" in error_msg.lower() or "does not exist" in error_msg.lower()
            ):
                raise ModelNotFoundError(
                    f"OpenAI model not available: {error_msg}",
                    provider="openai",
                    model_name=model,
                ) from e
            else:
                raise ProviderResponseError(
                    f"OpenAI API error: {error_msg}", provider="openai"
                ) from e

        # Try to get plain text; fallback to stringified output
        content_text: str
        if hasattr(resp, "output_text") and resp.output_text is not None:
            content_text = resp.output_text
        else:
            try:
                content_text = str(resp)
            except Exception:
                content_text = ""

        # Provider usage
        usage: Optional[Dict[str, Any]] = self._map_responses_usage(getattr(resp, "usage", None))

        return LLMResponse(
            content=content_text,
            model=model,
            usage=usage,
            finish_reason="stop",
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
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from OpenAI.

        Args:
            messages: List of messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter

        Yields:
            Stream chunks from the response
        """
        # Strip provider prefix if present
        model = strip_provider_prefix(model, "openai")

        # Log warning if model not found in registry (but don't block)
        # Note: Alias resolution happens at service layer, not here
        try:
            registry_client = self._registry_client
            if registry_client is not None:
                models = await registry_client.fetch_current_models("openai")
                model_found = any(m.model_name == model and m.is_active for m in models)
            else:
                model_found = True

            if not model_found:
                logging.getLogger(__name__).warning(
                    f"Model '{model}' not found in registry, proceeding anyway"
                )
        except Exception:
            pass  # Registry unavailable, continue anyway

        # o1 models and PDF processing don't support streaming yet
        if model.startswith("o1") or self._contains_pdf_content(messages):
            # Fall back to non-streaming for these cases
            response = await self._chat_non_streaming(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_tokens=reasoning_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                extra_params=extra_params,
                timeout=timeout,
            )
            # Return the full response as a single chunk
            yield StreamChunk(
                delta=response.content,
                model=response.model,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
            return

        # Convert messages to OpenAI format (reuse existing logic)
        openai_messages = await self._prepare_openai_messages(messages)

        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "stream": True,  # Enable streaming
        }

        # Only include temperature if explicitly provided (not None)
        if temperature is not None:
            request_params["temperature"] = temperature

        # Handle token limits with registry-based reasoning model detection
        if max_tokens:
            model_config = await self._get_model_config(model)

            if model_config["is_reasoning"]:
                # For reasoning models, separate reasoning and output tokens
                reasoning_budget = (
                    reasoning_tokens
                    if reasoning_tokens is not None
                    else model_config["min_recommended_reasoning_tokens"]
                )
                total_tokens = reasoning_budget + max_tokens
                request_params["max_completion_tokens"] = total_tokens

                self.log_debug(
                    f"Reasoning model '{model}': reasoning={reasoning_budget}, "
                    f"output={max_tokens}, total={total_tokens}"
                )
            else:
                # Non-reasoning models use max_tokens directly
                request_params["max_tokens"] = max_tokens

        # Handle response format
        if response_format:
            if response_format.get("type") == "json_object":
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json":
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                # Support OpenAI's JSON schema format
                json_schema_format: Dict[str, Any] = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format["strict"]
                request_params["response_format"] = json_schema_format
            else:
                request_params["response_format"] = response_format

        # Handle tools if provided
        if tools:
            openai_tools = self._prepare_tools(tools)
            request_params["tools"] = openai_tools

            if tool_choice is not None:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)

        # Apply extra parameters if provided
        if extra_params:
            request_params.update(extra_params)

        # Make the streaming API call
        try:
            # Enable streaming and include usage on the final chunk
            request_params["stream"] = True
            request_params["stream_options"] = {"include_usage": True}

            stream = await self._await_with_timeout(
                self.client.chat.completions.create(**request_params),
                timeout,
            )

            # Process the stream
            accumulated_content = ""
            accumulated_tool_calls = {}

            # OpenAI streaming behavior with include_usage=True:
            # 1. Content chunks arrive with delta.content
            # 2. A chunk with finish_reason signals end of content
            # 3. A SEPARATE final chunk arrives with usage data (after finish_reason)
            # 4. This final usage chunk has empty choices[] array
            #
            # We must capture usage regardless of choices[] and yield it AFTER
            # the stream completes to ensure we don't lose usage data.
            final_usage = None
            final_finish_reason = None
            final_tool_calls = None

            async for chunk in stream:
                # OpenAI sends usage on a final chunk with empty choices[] when
                # include_usage=True. Capture it regardless of choices.
                if getattr(chunk, "usage", None) is not None:
                    final_usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Handle content streaming
                    if choice.delta and choice.delta.content:
                        accumulated_content += choice.delta.content
                        yield StreamChunk(
                            delta=choice.delta.content,
                            model=model,
                            finish_reason=choice.finish_reason,
                        )

                    # Handle tool call streaming
                    if choice.delta and choice.delta.tool_calls:
                        for tool_call_delta in choice.delta.tool_calls:
                            idx = tool_call_delta.index

                            # Initialize tool call if new
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Accumulate tool call deltas
                            if tool_call_delta.id:
                                accumulated_tool_calls[idx]["id"] = tool_call_delta.id

                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    accumulated_tool_calls[idx]["function"][
                                        "name"
                                    ] = tool_call_delta.function.name

                                if tool_call_delta.function.arguments:
                                    accumulated_tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tool_call_delta.function.arguments

                    # Capture finish_reason for the final chunk (yielded after stream ends)
                    if choice.finish_reason:
                        final_finish_reason = choice.finish_reason
                        if accumulated_tool_calls:
                            final_tool_calls = [
                                accumulated_tool_calls[idx]
                                for idx in sorted(accumulated_tool_calls.keys())
                            ]

            # Yield final chunk with usage after stream completes
            # OpenAI sends usage on a separate chunk after finish_reason
            if final_finish_reason:
                yield StreamChunk(
                    delta="",
                    model=model,
                    finish_reason=final_finish_reason,
                    tool_calls=final_tool_calls,
                    usage=final_usage,
                )
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            error_msg = str(e)
            if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ProviderAuthenticationError(
                    f"OpenAI API authentication failed: {error_msg}", provider="openai"
                ) from e
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise ProviderRateLimitError(
                    f"OpenAI API rate limit exceeded: {error_msg}", provider="openai"
                ) from e
            else:
                raise ProviderResponseError(
                    f"OpenAI API error: {error_msg}", provider="openai"
                ) from e

    async def _prepare_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        for msg in messages:
            # Handle special message types
            if hasattr(msg, "tool_calls") and msg.role == "assistant":
                # Assistant message with tool calls
                message_dict = {
                    "role": msg.role,
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    message_dict["tool_calls"] = msg.tool_calls
                openai_messages.append(message_dict)
            elif hasattr(msg, "tool_call_id") and msg.role == "tool":
                # Tool response messages
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                # Regular messages (system, user, assistant)
                if isinstance(msg.content, str):
                    openai_messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )
                elif isinstance(msg.content, list):
                    # Handle multimodal content (text and images)
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, str):
                            content_parts.append({"type": "text", "text": part})
                        elif isinstance(part, dict):
                            if part.get("type") == "text":
                                content_parts.append(copy.deepcopy(part))
                            elif part.get("type") == "image_url":
                                content_parts.append(copy.deepcopy(part))
                            elif part.get("type") == "document":
                                # OpenAI doesn't support document content blocks
                                source = part.get("source", {})
                                media_type = source.get("media_type", "unknown")
                                content_parts.append(
                                    {
                                        "type": "text",
                                        "text": f"[Document file of type {media_type} was provided but OpenAI doesn't support document processing. Please use Anthropic Claude or Google Gemini for document analysis.]",
                                    }
                                )
                            else:
                                # Unknown content type
                                content_parts.append(
                                    {
                                        "type": "text",
                                        "text": f"[Unsupported content type: {part.get('type', 'unknown')}]",
                                    }
                                )
                    openai_messages.append(
                        {
                            "role": msg.role,
                            "content": content_parts,
                        }
                    )
                else:
                    openai_messages.append(
                        {
                            "role": msg.role,
                            "content": str(msg.content),
                        }
                    )

        # Optional: inline remote images using safe fetcher if enabled
        if os.getenv("LLMRING_INLINE_REMOTE_IMAGES", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            for message in openai_messages:
                if isinstance(message, dict) and isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image_url"
                            and isinstance(part.get("image_url"), dict)
                        ):
                            url = part["image_url"].get("url")
                            if isinstance(url, str) and url.startswith(("http://", "https://")):
                                try:
                                    data, mime = await safe_fetch_bytes(url)
                                    b64 = base64.b64encode(data).decode("utf-8")
                                    part["image_url"]["url"] = f"data:{mime};base64,{b64}"
                                except (SafeFetchError, Exception):
                                    # Leave URL as-is if fetch fails
                                    pass

        return openai_messages

    def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        openai_tools = []
        for tool in tools:
            # Check if tool is already in OpenAI format
            if "type" in tool and tool["type"] == "function" and "function" in tool:
                # Already in OpenAI format, use as-is
                openai_tools.append(tool)
            else:
                # Convert from simplified format to OpenAI format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                openai_tools.append(openai_tool)
        return openai_tools

    def _prepare_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Convert tool choice to OpenAI format."""
        if isinstance(tool_choice, str):
            return tool_choice
        elif isinstance(tool_choice, dict):
            # Convert our format to OpenAI's format
            if "function" in tool_choice:
                return {
                    "type": "function",
                    "function": {"name": tool_choice["function"]},
                }
            else:
                return tool_choice
        return tool_choice

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
        timeout: TimeoutSetting = TIMEOUT_UNSET,
    ) -> LLMResponse:
        """
        Send a chat request to the OpenAI API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gpt-4o")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            files: Optional list of file_ids (not supported in Chat Completions API)
            extra_params: Provider-specific parameters

        Returns:
            LLM response with complete generated content

        Notes:
            When files are provided, this method uses the OpenAI Responses API
            with attachments + file_search to enable document grounding.
        """
        resolved_timeout = self._resolve_timeout_value(timeout)

        # If files are present, route through Responses API with file attachments
        if files:
            return await self._chat_with_files_responses_api(
                messages=messages,
                model=model,
                files=files,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                extra_params=extra_params,
                timeout=resolved_timeout,
            )

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
            cache=cache,
            extra_params=extra_params,
            timeout=resolved_timeout,
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
        extra_params: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        timeout: TimeoutSetting = TIMEOUT_UNSET,
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a streaming chat request to the OpenAI API.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gpt-4o")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            files: Optional list of file_ids (not supported in Chat Completions API)
            extra_params: Provider-specific parameters

        Returns:
            Async iterator of stream chunks

        Example:
            >>> async for chunk in provider.chat_stream(messages, model="gpt-4o"):
            ...     print(chunk.delta, end="", flush=True)
        """
        resolved_timeout = self._resolve_timeout_value(timeout)
        return self._chat_stream_iterator(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_tokens=reasoning_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            extra_params=extra_params,
            files=files,
            timeout=resolved_timeout,
        )

    async def _chat_stream_iterator(
        self,
        *,
        messages: List[Message],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        reasoning_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        extra_params: Optional[Dict[str, Any]],
        files: Optional[List[str]],
        timeout: Optional[float],
    ) -> AsyncIterator[StreamChunk]:
        # If files are present, stream via Responses API with attachments
        if files:
            async for chunk in self._stream_with_files_responses_api(
                messages=messages,
                model=model,
                files=files,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                extra_params=extra_params,
                timeout=timeout,
            ):
                yield chunk
            return

        async for chunk in self._stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_tokens=reasoning_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            extra_params=extra_params,
            timeout=timeout,
        ):
            yield chunk

    def _flatten_messages_to_input_text(self, messages: List[Message]) -> str:
        """Flatten a list of llmring messages into a single input text string."""
        parts: List[str] = []
        for msg in messages:
            role = msg.role
            content_str = ""
            if isinstance(msg.content, str):
                content_str = msg.content
            elif isinstance(msg.content, list):
                text_bits: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        text_bits.append(item)
                    elif isinstance(item, dict):
                        t = item.get("text") if item.get("type") == "text" else None
                        if isinstance(t, str):
                            text_bits.append(t)
                content_str = " ".join(text_bits)
            else:
                content_str = str(msg.content)
            parts.append(f"{role}: {content_str}")
        return "\n".join(parts)

    def _map_responses_usage(self, usage_obj: Any) -> Optional[Dict[str, Any]]:
        """Normalize OpenAI Responses/Chat usage object to a dict or None."""
        if usage_obj is None:
            return None
        try:
            prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage_obj, "input_tokens", None)
            completion_tokens = getattr(usage_obj, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage_obj, "output_tokens", None)
            total_tokens = getattr(usage_obj, "total_tokens", None)
            if total_tokens is None and (
                prompt_tokens is not None or completion_tokens is not None
            ):
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        except Exception:
            return None

    def _messages_to_responses_input(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert llmring Message[] into OpenAI Responses input messages preserving roles
        and multimodal content where possible.

        - role mapping: system/user/assistant preserved; developer maps to system (if present upstream);
          tool messages are mapped to user text for context.
        - content mapping:
          - str -> pass through as string
          - {type: text, text} -> input_text
          - {type: image_url, image_url: {url}} -> input_image
          - other types are ignored here (PDF/documents are handled by separate PDF path)
        """
        responses_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.role
            # Map llmring 'tool' role to a user message with textual content
            if role == "tool":
                role = "user"

            entry: Dict[str, Any] = {"role": role, "type": "message"}

            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content_items: List[Dict[str, Any]] = []
                text_accum: List[str] = []
                for part in msg.content:
                    if isinstance(part, str):
                        text_accum.append(part)
                    elif isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            text_accum.append(part.get("text", ""))
                        elif ptype == "image_url" and isinstance(part.get("image_url"), dict):
                            url = part["image_url"].get("url")
                            if isinstance(url, str) and url:
                                content_items.append({"type": "input_image", "image_url": url})
                        # Skip 'document' here (handled by dedicated PDF path)
                # If we accumulated text, add as input_text first
                if text_accum:
                    content_items.insert(0, {"type": "input_text", "text": " ".join(text_accum)})

                # Choose content representation
                content = content_items if content_items else " ".join(text_accum)
                if content == "":
                    content = None
            else:
                content = str(msg.content)

            if content is None:
                continue

            entry["content"] = content
            responses_messages.append(entry)

        return responses_messages

    async def _chat_with_files_responses_api(
        self,
        messages: List[Message],
        model: str,
        files: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Handle chat with uploaded files using OpenAI Responses API + file_search attachments.
        """
        # Strip provider prefix if present
        model = strip_provider_prefix(model, "openai")

        # Convert messages to Responses input and attach files to last user message
        responses_input = self._messages_to_responses_input(messages)

        # Find last user message, else create one
        user_idx = None
        for i in range(len(responses_input) - 1, -1, -1):
            if responses_input[i].get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            responses_input.append({"role": "user", "type": "message", "content": []})
            user_idx = len(responses_input) - 1

        # Ensure content is a list
        if isinstance(responses_input[user_idx].get("content"), str):
            text_val = responses_input[user_idx]["content"]
            responses_input[user_idx]["content"] = [{"type": "input_text", "text": text_val}]
        elif responses_input[user_idx].get("content") is None:
            responses_input[user_idx]["content"] = []

        # Append files: inline text files as input_text; others as input_file
        for fid in files:
            try:
                content_resp = await self.client.files.content(fid)
                content_resp_any: Any = content_resp
                file_bytes = (
                    content_resp_any.read()
                    if hasattr(content_resp_any, "read")
                    else bytes(content_resp_any)
                )
            except Exception:
                file_bytes = b""
            # If looks like PDF, use input_file; otherwise inline as text
            if file_bytes.startswith(b"%PDF"):
                responses_input[user_idx]["content"].append({"type": "input_file", "file_id": fid})
            else:
                try:
                    text = (
                        file_bytes.decode("utf-8", errors="replace")
                        if file_bytes
                        else "[File content unavailable]"
                    )
                except Exception:
                    text = "[Binary file content omitted]"
                responses_input[user_idx]["content"].append({"type": "input_text", "text": text})

        # Build request
        request_params: Dict[str, Any] = {
            "model": model,
            "input": responses_input,
        }

        # Map common parameters
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens

        # Handle response format for Responses API
        if response_format:
            if response_format.get("type") in {"json_object", "json"}:
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                json_schema_format: Dict[str, Any] = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format["strict"]
                request_params["response_format"] = json_schema_format
            else:
                request_params["response_format"] = response_format

        # Include user tools if provided (file inputs do not require file_search)
        if tools:
            request_params["tools"] = self._prepare_tools(tools)
            if tool_choice is not None:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)

        if extra_params:
            request_params.update(extra_params)

        try:
            resp = await self._await_with_timeout(
                self.client.responses.create(**request_params),
                timeout,
            )
        except Exception as e:
            # Fallback: if input_file is not accepted (e.g., certain text files),
            # inline file contents as input_text and retry once.
            try:
                # Clone input to avoid mutating original in error paths
                retry_input = [dict(m) for m in responses_input]
                # Ensure last user message
                user_idx2 = None
                for i in range(len(retry_input) - 1, -1, -1):
                    if retry_input[i].get("role") == "user":
                        user_idx2 = i
                        break
                if user_idx2 is None:
                    retry_input.append({"role": "user", "type": "message", "content": []})
                    user_idx2 = len(retry_input) - 1
                # Ensure list content
                if isinstance(retry_input[user_idx2].get("content"), str):
                    text_val = retry_input[user_idx2]["content"]
                    retry_input[user_idx2]["content"] = [{"type": "input_text", "text": text_val}]
                elif retry_input[user_idx2].get("content") is None:
                    retry_input[user_idx2]["content"] = []

                # Fetch and inline file contents
                for fid in files:
                    try:
                        content_resp = await self.client.files.content(fid)
                        # content_resp is an httpx response-like; get bytes
                        content_resp_any: Any = content_resp
                        file_bytes = (
                            content_resp_any.read()
                            if hasattr(content_resp_any, "read")
                            else bytes(content_resp_any)
                        )
                        try:
                            text = file_bytes.decode("utf-8", errors="replace")
                        except Exception:
                            text = "[Binary file content omitted]"
                        retry_input[user_idx2]["content"].append(
                            {
                                "type": "input_text",
                                "text": text,
                            }
                        )
                    except Exception:
                        # If content retrieval fails, skip inlining for that file
                        pass

                retry_params = dict(request_params)
                retry_params["input"] = retry_input
                resp = await self._await_with_timeout(
                    self.client.responses.create(**retry_params),
                    timeout,
                )
            except Exception:
                await self._error_handler.handle_error(e, model)

        # Extract text
        content_text: str
        if hasattr(resp, "output_text") and resp.output_text is not None:
            content_text = resp.output_text
        else:
            try:
                content_text = str(resp)
            except Exception:
                content_text = ""

        # Normalize usage fields into llmring format
        usage = self._map_responses_usage(getattr(resp, "usage", None))

        return LLMResponse(
            content=content_text,
            model=model,
            usage=usage,
            finish_reason="stop",
        )

    async def _stream_with_files_responses_api(
        self,
        messages: List[Message],
        model: str,
        files: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via Responses API using file attachments and file_search."""
        # Strip provider prefix if present
        model = strip_provider_prefix(model, "openai")

        # Convert messages to Responses input and attach files
        responses_input = self._messages_to_responses_input(messages)
        # Find last user message
        user_idx = None
        for i in range(len(responses_input) - 1, -1, -1):
            if responses_input[i].get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            responses_input.append({"role": "user", "type": "message", "content": []})
            user_idx = len(responses_input) - 1

        # Ensure list content
        if isinstance(responses_input[user_idx].get("content"), str):
            text_val = responses_input[user_idx]["content"]
            responses_input[user_idx]["content"] = [{"type": "input_text", "text": text_val}]
        elif responses_input[user_idx].get("content") is None:
            responses_input[user_idx]["content"] = []

        for fid in files:
            try:
                content_resp = await self.client.files.content(fid)
                content_resp_any: Any = content_resp
                file_bytes = (
                    content_resp_any.read()
                    if hasattr(content_resp_any, "read")
                    else bytes(content_resp_any)
                )
            except Exception:
                file_bytes = b""
            if file_bytes.startswith(b"%PDF"):
                responses_input[user_idx]["content"].append({"type": "input_file", "file_id": fid})
            else:
                try:
                    text = (
                        file_bytes.decode("utf-8", errors="replace")
                        if file_bytes
                        else "[File content unavailable]"
                    )
                except Exception:
                    text = "[Binary file content omitted]"
                responses_input[user_idx]["content"].append({"type": "input_text", "text": text})

        request_params: Dict[str, Any] = {
            "model": model,
            "input": responses_input,
        }
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens
        if response_format:
            if response_format.get("type") in {"json_object", "json"}:
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                json_schema_format: Dict[str, Any] = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format["strict"]
                request_params["response_format"] = json_schema_format
            else:
                request_params["response_format"] = response_format
        if tools:
            request_params["tools"] = self._prepare_tools(tools)
            if tool_choice is not None:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)
        if extra_params:
            request_params.update(extra_params)

        # Stream using the SDK stream context manager
        try:
            async with self._context_with_timeout(
                self.client.responses.stream(**request_params), timeout
            ) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype.endswith("response.output_text.delta"):
                        delta = getattr(event, "delta", None)
                        if delta:
                            yield StreamChunk(delta=delta, model=model)
                    elif etype.endswith("response.error"):
                        err = getattr(event, "error", None)
                        msg = (
                            getattr(err, "message", "OpenAI streaming error")
                            if err
                            else "OpenAI streaming error"
                        )
                        raise ProviderResponseError(msg, provider="openai")

                # Completed; fetch final response for usage
                final = await stream.get_final_response()
                usage = None
                u = getattr(final, "usage", None)
                usage = self._map_responses_usage(u)
                # Emit a final empty chunk with usage/finish
                yield StreamChunk(delta="", model=model, finish_reason="stop", usage=usage)
        except Exception as e:
            await self._error_handler.handle_error(e, model)

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
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Send a non-streaming chat request to the OpenAI API.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gpt-4o")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter

        Returns:
            LLM response
        """
        # Strip provider prefix if present
        model = strip_provider_prefix(model, "openai")

        # Log warning if model not found in registry (but don't block)
        # Note: Alias resolution happens at service layer, not here
        try:
            registry_client = self._registry_client
            if registry_client is not None:
                models = await registry_client.fetch_current_models("openai")
                model_found = any(m.model_name == model and m.is_active for m in models)
            else:
                model_found = True

            if not model_found:
                logging.getLogger(__name__).warning(
                    f"Model '{model}' not found in registry, proceeding anyway"
                )
        except Exception:
            pass  # Registry unavailable, continue anyway

        # Route o1* models via Responses API
        if model.startswith("o1"):
            if tools or response_format or tool_choice is not None:
                raise ProviderResponseError(
                    "OpenAI o1 models do not support tools or custom response formats",
                    provider="openai",
                )
            return await self._chat_via_responses(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_params=extra_params,
                timeout=timeout,
            )

        # Check if messages contain PDF content - if so, route to Responses API path
        if self._contains_pdf_content(messages):
            # Tools and response_format are not supported in the Responses+file_search PDF path
            if tools or response_format:
                raise ProviderResponseError(
                    "Tools and custom response formats are not supported when processing PDFs with OpenAI (Responses API + file_search).",
                    provider="openai",
                )

            return await self._process_with_pdf_upload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_params=extra_params,
                timeout=timeout,
            )

        # Convert messages to OpenAI format using helper method
        openai_messages = await self._prepare_openai_messages(messages)

        # Build the request parameters
        request_params = {
            "model": model,
            "messages": openai_messages,
        }

        # Only include temperature if explicitly provided (not None)
        if temperature is not None:
            request_params["temperature"] = temperature

        # Handle token limits with registry-based reasoning model detection
        if max_tokens:
            model_config = await self._get_model_config(model)

            if model_config["is_reasoning"]:
                # For reasoning models, separate reasoning and output tokens
                reasoning_budget = (
                    reasoning_tokens
                    if reasoning_tokens is not None
                    else model_config["min_recommended_reasoning_tokens"]
                )
                total_tokens = reasoning_budget + max_tokens
                request_params["max_completion_tokens"] = total_tokens

                self.log_debug(
                    f"Reasoning model '{model}': reasoning={reasoning_budget}, "
                    f"output={max_tokens}, total={total_tokens}"
                )
            else:
                # Non-reasoning models use max_tokens directly
                request_params["max_tokens"] = max_tokens

        # Handle response format
        if response_format:
            if response_format.get("type") == "json_object":
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json":
                # Map our generic "json" to OpenAI's "json_object"
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                # Support OpenAI's JSON schema format
                json_schema_format: Dict[str, Any] = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format["strict"]
                request_params["response_format"] = json_schema_format
            else:
                request_params["response_format"] = response_format

        # Handle tools if provided
        if tools:
            request_params["tools"] = self._prepare_tools(tools)

            # Handle tool choice
            if tool_choice is not None:
                request_params["tool_choice"] = self._prepare_tool_choice(tool_choice)

        # Apply extra parameters if provided
        if extra_params:
            request_params.update(extra_params)

        # Make the API call using the SDK
        try:

            async def _do_call():
                return await self._await_with_timeout(
                    self.client.chat.completions.create(**request_params), timeout
                )

            # Circuit breaker key per model
            breaker_key = f"openai:{model}"
            if not await self._breaker.allow(breaker_key):
                raise CircuitBreakerError(
                    "OpenAI circuit breaker is open - too many recent failures",
                    provider="openai",
                )

            response: ChatCompletion = await retry_async(_do_call)
            await self._breaker.record_success(breaker_key)

            # Extract the content from the response
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = None

            # Handle tool calls if present
            if choice.message.tool_calls:
                tool_calls = []
                for tc in choice.message.tool_calls:
                    tc_any: Any = tc
                    tc_function = getattr(tc_any, "function", None)
                    if tc_function is None:
                        continue
                    tool_calls.append(
                        {
                            "id": getattr(tc_any, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(tc_function, "name", ""),
                                "arguments": getattr(tc_function, "arguments", ""),
                            },
                        }
                    )

            # Prepare the response
            llm_response = LLMResponse(
                content=content,
                model=model,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=choice.finish_reason,
            )

            # Add tool calls if present
            if tool_calls:
                llm_response.tool_calls = tool_calls

            return llm_response

        except Exception as e:
            await self._error_handler.handle_error(e, model)

    async def upload_file(
        self,
        file: Union[str, Path, BinaryIO],
        purpose: str = "analysis",
        filename: Optional[str] = None,
        **kwargs,
    ) -> FileUploadResponse:
        """
        Upload file to OpenAI Files API.

        Args:
            file: File path, Path object, or file-like object
            purpose: Purpose of the file (e.g., "analysis", "code_execution", "assistant")
            filename: Optional filename (required for file-like objects)
            **kwargs: Additional provider-specific parameters

        Returns:
            FileUploadResponse with file_id, size, etc.

        Raises:
            FileSizeError: If file exceeds 512MB limit
            ProviderAuthenticationError: If authentication fails
            ProviderResponseError: If upload fails
        """
        # OpenAI max file size is 512MB for assistants
        MAX_FILE_SIZE = 512 * 1024 * 1024  # 512MB in bytes

        # Map llmring purpose to OpenAI purpose
        # OpenAI uses "assistants" for most file purposes
        openai_purpose_map: dict[str, FilePurpose] = {
            "code_execution": "assistants",
            "assistant": "assistants",
            "analysis": "assistants",
            "cache": "assistants",
        }
        openai_purpose = openai_purpose_map.get(purpose, "assistants")

        # Handle file path or file-like object
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                raise FileSizeError(
                    f"File size {file_size} bytes exceeds OpenAI limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="openai",
                    filename=str(file_path),
                )

            # Upload file using OpenAI SDK
            with open(file_path, "rb") as f:
                file_obj = await self.client.files.create(file=f, purpose=openai_purpose)

            actual_filename = filename or file_path.name
            actual_size = file_size
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
                    f"File size {file_size} bytes exceeds OpenAI limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="openai",
                    filename=filename,
                )

            # Upload using SDK - it expects a file-like object
            if hasattr(file, "seek"):
                file.seek(current_pos)
            file_obj = await self.client.files.create(file=file, purpose=openai_purpose)

            actual_filename = filename
            actual_size = file_size

        try:
            # Parse response
            return FileUploadResponse(
                file_id=file_obj.id,
                provider="openai",
                filename=actual_filename,
                size_bytes=actual_size,
                created_at=datetime.fromtimestamp(file_obj.created_at),
                purpose=purpose,
                metadata={
                    "openai_purpose": openai_purpose,
                    "status": file_obj.status if hasattr(file_obj, "status") else "uploaded",
                },
            )

        except Exception as e:
            await self._error_handler.handle_error(e, "files")

    async def list_files(
        self, purpose: Optional[str] = None, limit: int = 100
    ) -> List[FileMetadata]:
        """
        List uploaded files.

        Args:
            purpose: Optional filter by purpose (OpenAI doesn't filter on API side,
                     so we filter client-side)
            limit: Maximum number of files to return (default 100)

        Returns:
            List of FileMetadata objects
        """
        try:
            # OpenAI SDK returns a paginated list
            response = await self.client.files.list()

            # Parse response
            files = []
            for file_obj in response.data:
                # OpenAI returns "purpose" field directly
                file_purpose = file_obj.purpose

                # Map OpenAI purpose back to llmring purpose
                # (This is a best-effort mapping since OpenAI doesn't have fine-grained purposes)
                llmring_purpose_map = {
                    "assistants": "assistant",
                    "fine-tune": "fine-tune",
                    "batch": "batch",
                }
                mapped_purpose = llmring_purpose_map.get(file_purpose, file_purpose)

                # Filter by purpose if requested
                if purpose and mapped_purpose != purpose:
                    continue

                files.append(
                    FileMetadata(
                        file_id=file_obj.id,
                        provider="openai",
                        filename=file_obj.filename,
                        size_bytes=file_obj.bytes,
                        created_at=datetime.fromtimestamp(file_obj.created_at),
                        purpose=mapped_purpose,
                        status="uploaded",  # OpenAI doesn't expose status in list
                        metadata={
                            "openai_purpose": file_purpose,
                        },
                    )
                )

                # Limit results
                if len(files) >= limit:
                    break

            return files

        except Exception as e:
            await self._error_handler.handle_error(e, "files")

    async def get_file(self, file_id: str) -> FileMetadata:
        """
        Get file metadata.

        Args:
            file_id: File ID to retrieve

        Returns:
            FileMetadata object
        """
        try:
            file_obj = await self.client.files.retrieve(file_id=file_id)

            # Map OpenAI purpose back to llmring purpose
            file_purpose = file_obj.purpose
            llmring_purpose_map = {
                "assistants": "assistant",
                "fine-tune": "fine-tune",
                "batch": "batch",
            }
            mapped_purpose = llmring_purpose_map.get(file_purpose, file_purpose)

            return FileMetadata(
                file_id=file_obj.id,
                provider="openai",
                filename=file_obj.filename,
                size_bytes=file_obj.bytes,
                created_at=datetime.fromtimestamp(file_obj.created_at),
                purpose=mapped_purpose,
                status="uploaded",  # OpenAI doesn't expose detailed status
                metadata={
                    "openai_purpose": file_purpose,
                },
            )

        except Exception as e:
            await self._error_handler.handle_error(e, "files")

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete uploaded file.

        Args:
            file_id: File ID to delete

        Returns:
            True on success
        """
        try:
            await self.client.files.delete(file_id=file_id)
            return True

        except Exception as e:
            await self._error_handler.handle_error(e, "files")
