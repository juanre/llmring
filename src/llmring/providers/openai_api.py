"""
OpenAI API provider implementation using the official SDK.
"""

import asyncio
import base64
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
import copy
import warnings

from dotenv import load_dotenv
from llmring.net.safe_fetcher import fetch_bytes as safe_fetch_bytes, SafeFetchError
from llmring.net.retry import retry_async, RetryError
from llmring.net.circuit_breaker import CircuitBreaker
from llmring.base import BaseLLMProvider
from llmring.model_refresh.models import ModelInfo
from llmring.schemas import LLMResponse, Message
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# Load environment variables from .env file
load_dotenv()


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
        super().__init__(api_key=api_key, base_url=base_url)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_model = model

        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")

        # Initialize the client with the SDK
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

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
    ) -> LLMResponse:
        """
        Process messages containing PDFs using OpenAI's Responses API with file_search vector stores.
        """
        # Extract PDF data and text from messages
        pdf_data_list, combined_text = self._extract_pdf_content_and_text(messages)
        if not pdf_data_list:
            raise ValueError("No PDF content found in messages")
        if not combined_text.strip():
            combined_text = "Please analyze this PDF document and provide a summary."

        uploaded_files: List[Dict[str, str]] = []
        try:
            # Upload PDFs
            for i, pdf_data in enumerate(pdf_data_list):
                with tempfile.NamedTemporaryFile(suffix=f"_document_{i}.pdf", delete=False) as tmp_file:
                    tmp_file.write(pdf_data)
                    tmp_file.flush()
                    with open(tmp_file.name, "rb") as f:
                        # PDFs must use 'assistants' purpose for Responses input_file
                        file_obj = await self.client.files.create(file=f, purpose="assistants")
                        uploaded_files.append({"file_id": file_obj.id, "temp_path": tmp_file.name})

            # Build Responses API input using input_file items (no RAG/vector store)
            content_items: List[Dict[str, Any]] = []
            content_items.append({"type": "input_text", "text": combined_text})
            for info in uploaded_files:
                content_items.append({"type": "input_file", "file_id": info["file_id"]})

            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))
            resp = await asyncio.wait_for(
                self.client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": content_items}],
                    **({"temperature": temperature} if temperature is not None else {}),
                    **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
                ),
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
            resp = await self.client.responses.create(
                model=model,
                input=input_text,
                # temperature and max tokens support may vary; pass only if provided
                **({"temperature": temperature} if temperature is not None else {}),
                **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
            )
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ValueError(
                    f"OpenAI API authentication failed: {error_msg}"
                ) from e
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise ValueError(f"OpenAI API rate limit exceeded: {error_msg}") from e
            elif "model" in error_msg.lower() and (
                "not found" in error_msg.lower() or "does not exist" in error_msg.lower()
            ):
                raise ValueError(f"OpenAI model not available: {error_msg}") from e
            else:
                raise Exception(f"OpenAI API error: {error_msg}") from e

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
            "total_tokens": self.get_token_count(input_text) + self.get_token_count(content_text),
        }

        return LLMResponse(
            content=content_text,
            model=model,
            usage=estimated_usage,
            finish_reason="stop",
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
    ) -> LLMResponse:
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

        Returns:
            LLM response
        """
        # Strip provider prefix if present
        if model.lower().startswith("openai:"):
            model = model.split(":", 1)[1]

        # Verify model is supported
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")

        # Route o1* models via Responses API
        if model.startswith("o1"):
            if tools or response_format or tool_choice is not None:
                raise ValueError("OpenAI o1 models do not support tools or custom response formats")
            return await self._chat_via_responses(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Check if messages contain PDF content - if so, route to Assistants API
        if self._contains_pdf_content(messages):
            # Tools and response_format are not supported in the Responses+file_search PDF path
            if tools or response_format:
                raise ValueError(
                    "Tools and custom response formats are not supported when processing PDFs with OpenAI (Responses API + file_search)."
                )

            return await self._process_with_responses_file_search(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Convert messages to OpenAI format
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
                                # Convert to a text description instead
                                source = part.get("source", {})
                                media_type = source.get("media_type", "unknown")
                                content_parts.append(
                                    {
                                        "type": "text",
                                        "text": f"[Document file of type {media_type} was provided but OpenAI doesn't support document processing. Please use Anthropic Claude or Google Gemini for document analysis.]",
                                    }
                                )
                            else:
                                # Unknown content type - convert to text description
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
        if os.getenv("LLMRING_INLINE_REMOTE_IMAGES", "false").lower() in {"1", "true", "yes", "on"}:
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
                                    # Leave URL as-is if fetch fails or not allowed
                                    pass

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
            else:
                request_params["response_format"] = response_format

        # Handle tools if provided
        if tools:
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
            request_params["tools"] = openai_tools

            # Handle tool choice
            if tool_choice is not None:
                if isinstance(tool_choice, str):
                    request_params["tool_choice"] = tool_choice
                elif isinstance(tool_choice, dict):
                    # Convert our format to OpenAI's format
                    if "function" in tool_choice:
                        request_params["tool_choice"] = {
                            "type": "function",
                            "function": {"name": tool_choice["function"]},
                        }
                    else:
                        request_params["tool_choice"] = tool_choice

        # Make the API call using the SDK
        try:
            timeout_s = float(os.getenv("LLMRING_PROVIDER_TIMEOUT_S", "60"))

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.chat.completions.create(**request_params), timeout=timeout_s
                )

            # Circuit breaker key per model
            breaker_key = f"openai:{model}"
            if not await self._breaker.allow(breaker_key):
                raise Exception("OpenAI circuit open for model")

            response: ChatCompletion = await retry_async(_do_call)
            await self._breaker.record_success(breaker_key)
        except Exception as e:
            # record failure for breaker
            try:
                breaker_key = f"openai:{model}"
                await self._breaker.record_failure(breaker_key)
            except Exception:
                pass
            # Handle specific OpenAI errors with more context
            error_msg = str(e)
            if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ValueError(
                    f"OpenAI API authentication failed: {error_msg}"
                ) from e
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise ValueError(f"OpenAI API rate limit exceeded: {error_msg}") from e
            elif "model" in error_msg.lower() and (
                "not found" in error_msg.lower()
                or "does not exist" in error_msg.lower()
            ):
                raise ValueError(f"OpenAI model not available: {error_msg}") from e
            elif "context length" in error_msg.lower() or "token" in error_msg.lower():
                raise ValueError(f"OpenAI token limit exceeded: {error_msg}") from e
            else:
                # Re-raise SDK exceptions with our standard format
                raise Exception(f"OpenAI API error: {error_msg}") from e

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

    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from OpenAI API.

        Returns:
            List of discovered models with capabilities
        """
        try:
            # Call OpenAI's models endpoint
            models_response = await self.client.models.list()
            discovered = []

            for model in models_response.data:
                # Only include chat models (filter out others like embeddings, etc.)
                if self._is_chat_model(model.id):
                    model_info = ModelInfo(
                        provider="openai",
                        model_name=model.id,
                        display_name=self._get_display_name(model.id),
                        description=f"OpenAI {model.id} model",
                        source="api",
                        raw_data={"api_response": model.model_dump()},
                    )

                    # Add known capabilities for recognized models
                    capabilities = await self.get_model_capabilities(model.id)
                    if capabilities:
                        model_info.max_context = capabilities.get("max_context")
                        model_info.max_output_tokens = capabilities.get(
                            "max_output_tokens"
                        )
                        model_info.supports_vision = capabilities.get(
                            "supports_vision", False
                        )
                        model_info.supports_function_calling = capabilities.get(
                            "supports_function_calling", False
                        )
                        model_info.supports_json_mode = capabilities.get(
                            "supports_json_mode", False
                        )
                        model_info.supports_parallel_tool_calls = capabilities.get(
                            "supports_parallel_tool_calls", False
                        )
                        model_info.tool_call_format = capabilities.get(
                            "tool_call_format"
                        )

                    discovered.append(model_info)

            return discovered

        except Exception as e:
            # Fallback to static model list if API fails
            import logging

            logging.warning(
                f"OpenAI model discovery failed: {str(e)}, falling back to static list"
            )
            return await super().discover_models()

    async def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get capabilities for a specific OpenAI model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model capabilities
        """
        # Known capabilities for OpenAI models as of 2025
        capabilities_map = {
            "gpt-4o": {
                "max_context": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "gpt-4o-mini": {
                "max_context": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "gpt-4o-2024-08-06": {
                "max_context": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "gpt-4-turbo": {
                "max_context": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "gpt-4": {
                "max_context": 8192,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "gpt-3.5-turbo": {
                "max_context": 16385,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "tool_call_format": "openai",
            },
            "o1": {
                "max_context": 200000,
                "max_output_tokens": 100000,
                "supports_vision": False,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": None,
            },
            "o1-mini": {
                "max_context": 128000,
                "max_output_tokens": 65536,
                "supports_vision": False,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": None,
            },
        }

        return capabilities_map.get(model_name)

    def _is_chat_model(self, model_id: str) -> bool:
        """Check if a model is a chat/completion model."""
        # Filter out non-chat models
        exclude_patterns = [
            "whisper",
            "tts",
            "dall-e",
            "embedding",
            "text-embedding",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
            "code-davinci",
            "moderation",
        ]

        model_lower = model_id.lower()
        for pattern in exclude_patterns:
            if pattern in model_lower:
                return False

        return True

    def _get_display_name(self, model_id: str) -> str:
        """Get a friendly display name for a model."""
        display_names = {
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
            "gpt-4o-2024-08-06": "GPT-4o (2024-08-06)",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-4": "GPT-4",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "o1": "OpenAI o1",
            "o1-mini": "OpenAI o1-mini",
        }

        return display_names.get(model_id, model_id.replace("-", " ").title())
