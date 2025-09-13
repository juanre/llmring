"""
OpenAI API provider implementation using the official SDK.
"""

import asyncio
import base64
import copy
import os
import tempfile
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

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

# Note: do not call load_dotenv() in library code; handle in app entrypoints
from llmring.net.safe_fetcher import SafeFetchError
from llmring.net.safe_fetcher import fetch_bytes as safe_fetch_bytes
from llmring.schemas import LLMResponse, Message, StreamChunk


class OpenAIProvider(BaseLLMProvider):
    """Implementation of OpenAI API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
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
            raise ProviderAuthenticationError(
                "OpenAI API key must be provided", provider="openai"
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
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # List of officially supported models (as of early 2025)
        self.supported_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
        ]
        # Simple circuit breaker per model
        self._breaker = CircuitBreaker()

    def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by OpenAI.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise
        """
        # Strip provider prefix if present
        if model.lower().startswith("openai:"):
            model = model.split(":", 1)[1]

        return model in self.supported_models

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported OpenAI model names.

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
            provider_name="openai",
            supported_models=self.supported_models.copy(),
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,  # GPT-4o models support audio
            supports_documents=True,  # Via Responses API file_search
            supports_json_mode=True,
            supports_caching=False,  # OpenAI doesn't have native caching like Anthropic
            max_context_window=128000,  # GPT-4o context window
            default_model=self.default_model,
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

    def _extract_pdf_content_and_text(
        self, messages: List[Message]
    ) -> tuple[List[bytes], str]:
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
                                pdf_data = base64.b64decode(source.get("data", ""))
                                pdf_data_list.append(pdf_data)
                        elif isinstance(part, str):
                            text_parts.append(part)

        combined_text = " ".join(text_parts)
        return pdf_data_list, combined_text

    async def _process_with_responses_file_search(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Process messages containing PDFs using OpenAI's Responses API with file upload.

        Note: This implementation uploads PDFs directly and processes them with the model.
        It does not use vector stores or file_search for RAG functionality.
        """
        # Extract PDF data and text from messages
        pdf_data_list, combined_text = self._extract_pdf_content_and_text(messages)
        if not pdf_data_list:
            raise ProviderResponseError(
                "No PDF content found in messages", provider="openai"
            )
        if not combined_text.strip():
            combined_text = "Please analyze this PDF document and provide a summary."

        uploaded_files: List[Dict[str, str]] = []
        try:
            # Upload PDFs
            for i, pdf_data in enumerate(pdf_data_list):
                with tempfile.NamedTemporaryFile(
                    suffix=f"_document_{i}.pdf", delete=False
                ) as tmp_file:
                    tmp_file.write(pdf_data)
                    tmp_file.flush()
                    with open(tmp_file.name, "rb") as f:
                        # PDFs must use 'assistants' purpose for Responses input_file
                        file_obj = await self.client.files.create(
                            file=f, purpose="assistants"
                        )
                        uploaded_files.append(
                            {"file_id": file_obj.id, "temp_path": tmp_file.name}
                        )

            # Build Responses API input using input_file items (direct file processing)
            content_items: List[Dict[str, Any]] = []
            content_items.append({"type": "input_text", "text": combined_text})
            for info in uploaded_files:
                content_items.append({"type": "input_file", "file_id": info["file_id"]})

            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            request_params = {
                "model": model,
                "input": [{"role": "user", "content": content_items}],
            }

            if temperature is not None:
                request_params["temperature"] = temperature
            if max_tokens is not None:
                request_params["max_output_tokens"] = max_tokens

            # Apply extra parameters if provided
            if extra_params:
                request_params.update(extra_params)

            resp = await asyncio.wait_for(
                self.client.responses.create(**request_params),
                timeout=timeout_s,
            )

            response_content = (
                resp.output_text if hasattr(resp, "output_text") else str(resp)
            )
            estimated_usage = {
                "prompt_tokens": self.get_token_count(combined_text),
                "completion_tokens": self.get_token_count(response_content or ""),
                "total_tokens": self.get_token_count(combined_text)
                + self.get_token_count(response_content or ""),
            }
            return LLMResponse(
                content=response_content or "",
                model=model,
                usage=estimated_usage,
                finish_reason="stop",
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

    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens (estimated)
        """
        try:
            import tiktoken  # type: ignore

            # Use a safe encoding for token counting
            encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            return len(encoder.encode(text))
        except ImportError:
            # Fallback to rough estimate: ~4 characters per token for English text
            return len(text) // 4

    async def _chat_via_responses(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Handle o1* models using the Responses API.
        """
        # Flatten conversation into an input string preserving roles
        parts: List[str] = []
        for msg in messages:
            role = msg.role
            content_str = ""
            if isinstance(msg.content, str):
                content_str = msg.content
            elif isinstance(msg.content, list):
                # Join text parts; ignore non-text for o1
                text_bits: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        text_bits.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_bits.append(item.get("text", ""))
                content_str = " ".join(text_bits)
            else:
                content_str = str(msg.content)
            parts.append(f"{role}: {content_str}")

        input_text = "\n".join(parts)

        try:
            request_params = {
                "model": model,
                "input": input_text,
            }

            # temperature and max tokens support may vary; pass only if provided
            if temperature is not None:
                request_params["temperature"] = temperature
            if max_tokens is not None:
                request_params["max_output_tokens"] = max_tokens

            # Apply extra parameters if provided
            if extra_params:
                request_params.update(extra_params)

            resp = await self.client.responses.create(**request_params)
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
                "not found" in error_msg.lower()
                or "does not exist" in error_msg.lower()
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

        estimated_usage = {
            "prompt_tokens": self.get_token_count(input_text),
            "completion_tokens": self.get_token_count(content_text),
            "total_tokens": self.get_token_count(input_text)
            + self.get_token_count(content_text),
        }

        return LLMResponse(
            content=content_text,
            model=model,
            usage=estimated_usage,
            finish_reason="stop",
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
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from OpenAI.

        Args:
            messages: List of messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter

        Yields:
            Stream chunks from the response
        """
        # Strip provider prefix if present
        if model.lower().startswith("openai:"):
            model = model.split(":", 1)[1]

        # Verify model is supported
        if not self.validate_model(model):
            raise ModelNotFoundError(
                f"Unsupported model: {model}", provider="openai", model_name=model
            )

        # o1 models and PDF processing don't support streaming yet
        if model.startswith("o1") or self._contains_pdf_content(messages):
            # Fall back to non-streaming for these cases
            response = await self._chat_non_streaming(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                extra_params=extra_params,
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
        request_params = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature or 0.7,
            "stream": True,  # Enable streaming
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        # Handle response format
        if response_format:
            if response_format.get("type") == "json_object":
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json":
                request_params["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                # Support OpenAI's JSON schema format
                json_schema_format = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format[
                        "strict"
                    ]
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
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=timeout_s,
            )

            # Process the stream
            accumulated_content = ""
            accumulated_tool_calls = {}

            async for chunk in stream:
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
                                    accumulated_tool_calls[idx]["function"]["name"] = (
                                        tool_call_delta.function.name
                                    )

                                if tool_call_delta.function.arguments:
                                    accumulated_tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tool_call_delta.function.arguments

                    # Handle finish
                    if choice.finish_reason:
                        # Convert accumulated tool calls to list
                        tool_calls_list = None
                        if accumulated_tool_calls:
                            tool_calls_list = [
                                accumulated_tool_calls[idx]
                                for idx in sorted(accumulated_tool_calls.keys())
                            ]

                        # Final chunk with usage information and tool calls
                        yield StreamChunk(
                            delta="",
                            model=model,
                            finish_reason=choice.finish_reason,
                            tool_calls=tool_calls_list,
                            usage={
                                "prompt_tokens": self.get_token_count(
                                    str(openai_messages)
                                ),
                                "completion_tokens": self.get_token_count(
                                    accumulated_content
                                ),
                                "total_tokens": self.get_token_count(
                                    str(openai_messages)
                                )
                                + self.get_token_count(accumulated_content),
                            },
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

    async def _prepare_openai_messages(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
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
                if isinstance(message, dict) and isinstance(
                    message.get("content"), list
                ):
                    for part in message["content"]:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image_url"
                            and isinstance(part.get("image_url"), dict)
                        ):
                            url = part["image_url"].get("url")
                            if isinstance(url, str) and url.startswith(
                                ("http://", "https://")
                            ):
                                try:
                                    data, mime = await safe_fetch_bytes(url)
                                    b64 = base64.b64encode(data).decode("utf-8")
                                    part["image_url"]["url"] = (
                                        f"data:{mime};base64,{b64}"
                                    )
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
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
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
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Union[LLMResponse, AsyncIterator[StreamChunk]]:
        """
        Send a chat request to the OpenAI API using the official SDK.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gpt-4o")
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
                extra_params=extra_params,
            )
        else:
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
        """
        Send a non-streaming chat request to the OpenAI API.

        Args:
            messages: List of messages
            model: Model to use (e.g., "gpt-4o")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            tools: Optional list of tools
            tool_choice: Optional tool choice parameter

        Returns:
            LLM response
        """
        # Strip provider prefix if present
        if model.lower().startswith("openai:"):
            model = model.split(":", 1)[1]

        # Verify model is supported
        if not self.validate_model(model):
            raise ModelNotFoundError(
                f"Unsupported model: {model}", provider="openai", model_name=model
            )

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
            )

        # Check if messages contain PDF content - if so, route to Assistants API
        if self._contains_pdf_content(messages):
            # Tools and response_format are not supported in the Responses+file_search PDF path
            if tools or response_format:
                raise ProviderResponseError(
                    "Tools and custom response formats are not supported when processing PDFs with OpenAI (Responses API + file_search).",
                    provider="openai",
                )

            return await self._process_with_responses_file_search(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_params=extra_params,
            )

        # Convert messages to OpenAI format using helper method
        openai_messages = await self._prepare_openai_messages(messages)

        # Build the request parameters
        request_params = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature or 0.7,
        }

        if max_tokens:
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
                json_schema_format = {"type": "json_schema"}
                if "json_schema" in response_format:
                    json_schema_format["json_schema"] = response_format["json_schema"]
                if response_format.get("strict") is not None:
                    json_schema_format["json_schema"]["strict"] = response_format[
                        "strict"
                    ]
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
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.chat.completions.create(**request_params),
                    timeout=timeout_s,
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
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            # record failure for breaker
            try:
                breaker_key = f"openai:{model}"
                await self._breaker.record_failure(breaker_key)
            except Exception:
                pass
            # Handle specific OpenAI errors with more context
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
                "not found" in error_msg.lower()
                or "does not exist" in error_msg.lower()
            ):
                raise ModelNotFoundError(
                    f"OpenAI model not available: {error_msg}",
                    provider="openai",
                    model_name=model,
                ) from e
            elif "context length" in error_msg.lower() or "token" in error_msg.lower():
                raise ProviderResponseError(
                    f"OpenAI token limit exceeded: {error_msg}", provider="openai"
                ) from e
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"OpenAI API request timed out: {error_msg}", provider="openai"
                ) from e
            else:
                # Re-raise SDK exceptions with our standard format
                raise ProviderResponseError(
                    f"OpenAI API error: {error_msg}", provider="openai"
                ) from e

        # Extract the content from the response
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = None

        # Handle tool calls if present
        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
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
