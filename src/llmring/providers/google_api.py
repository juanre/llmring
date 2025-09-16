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

logger = logging.getLogger(__name__)

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

# Note: do not call load_dotenv() in library code; handle in app entrypoints
from llmring.net.retry import retry_async
from llmring.registry import RegistryClient
from llmring.schemas import LLMResponse, Message, StreamChunk


class GoogleProvider(BaseLLMProvider):
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

        # Store for backward compatibility
        self.api_key = api_key
        self.project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID", "")
        self.default_model = model  # Will be derived from registry if None

        # Initialize the client
        self.client = genai.Client(api_key=api_key)

        # Registry client for model validation
        self._registry_client = RegistryClient()


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
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type, data=image_data
                                )
                            )
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
                                inline_data=types.Blob(
                                    mime_type=mime_type, data=document_data
                                )
                            )
                        )
                    except (ValueError, base64.binascii.Error):
                        # Skip invalid document data
                        continue

        return parts if parts else str(content)

    async def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by Google using registry lookup.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise
        """
        # Strip provider prefix if present
        if model.lower().startswith("google:"):
            model = model.split(":", 1)[1]

        try:
            # Try registry validation
            return await self._registry_client.validate_model("google", model)
        except Exception as e:
            # If registry is unavailable, log warning and allow gracefully
            import logging
            logging.warning(f"Registry unavailable for model validation, allowing gracefully: {e}")
            return True

    async def get_supported_models(self) -> List[str]:
        """
        Get list of supported Google model names from registry.

        Returns:
            List of supported model names
        """
        try:
            # Try to fetch from registry
            models = await self._registry_client.fetch_current_models("google")
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
                logger.info(f"Google: Derived default model from registry: {selected_model}")
                return selected_model

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not derive default model from registry: {e}")

        # Ultimate fallback - use a sensible default when registry is unavailable
        logger.warning("No default model available from registry, using fallback: gemini-1.5-flash")
        self.default_model = "gemini-1.5-flash"  # Reasonable fallback
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
            registry_models = await self._registry_client.fetch_current_models("google")
            active_models = [m for m in registry_models if m.is_active and m.model_name in available_models]

            if active_models:
                # Select most balanced: reasonable cost + good capabilities + recent
                scored_models = []
                for model in active_models:
                    score = 0
                    # Prefer models with moderate cost (not cheapest, not most expensive)
                    input_cost = model.dollars_per_million_tokens_input or 0
                    if 0.01 <= input_cost <= 10.0:  # Balanced cost range for Google
                        score += 10

                    # Prefer models with good capabilities
                    if model.supports_function_calling:
                        score += 5
                    if model.supports_vision:
                        score += 3

                    # Prefer larger context models
                    if model.max_input_tokens and model.max_input_tokens >= 1000000:  # 1M+ context
                        score += 4

                    # Prefer more recent models (if added_date is available)
                    if model.added_date:
                        import datetime
                        days_since_added = (datetime.datetime.now(datetime.timezone.utc) - model.added_date).days
                        if days_since_added < 180:  # Less than 6 months old
                            score += 5

                    scored_models.append((model.model_name, score))

                if scored_models:
                    # Select highest scoring model
                    best_model = max(scored_models, key=lambda x: x[1])
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Google: Selected default model '{best_model[0]}' with score {best_model[1]} from registry analysis")
                    return best_model[0]

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not use registry metadata for Google selection: {e}")

        # Fallback: first available model
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Google: Fallback to first available model: {available_models[0]}")
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

    def _convert_type_to_gemini(self, json_type: str) -> types.Type:
        """
        Convert JSON schema type to Gemini Type enum.

        Args:
            json_type: JSON schema type string

        Returns:
            Gemini Type enum value
        """
        type_mapping = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
        }
        return type_mapping.get(json_type.lower(), types.Type.STRING)

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
        original_model = model
        if model.lower().startswith("google:") or model.lower().startswith("gemini:"):
            model = model.split(":", 1)[1]

        # Note: Model name normalization removed - use exact model names from registry

        # Validate model (warn but don't fail if not in registry)
        if not await self.validate_model(model):
            logger.warning(
                f"Model '{model}' not found in registry, proceeding anyway"
            )

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

            # Use real streaming API (Google SDK returns sync generator, need to wrap)
            # Tools are now passed via config, not as a separate parameter
            stream_response = self.client.models.generate_content_stream(
                model=model,
                contents=google_messages,
                config=config,
            )
            await self._breaker.record_success(key)

            # Process the streaming response (wrap sync iteration in thread executor)
            accumulated_content = ""
            tool_calls = []

            # Convert sync generator to async generator to avoid blocking event loop

            def _iterate_chunks():
                """Iterate over sync generator in thread."""
                chunks = []
                for chunk in stream_response:
                    chunks.append(chunk)
                return chunks

            # Get all chunks in thread executor
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, _iterate_chunks)

            for chunk in chunks:
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]

                    if hasattr(candidate, "content") and candidate.content:
                        # Extract text and function calls from content parts
                        chunk_text = ""
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                chunk_text += part.text
                            elif hasattr(part, "function_call") and part.function_call:
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

                        if chunk_text:
                            accumulated_content += chunk_text
                            yield StreamChunk(
                                delta=chunk_text,
                                model=model,
                                finish_reason=None,
                            )

                    # Check for finish
                    if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                        finish_reason = str(candidate.finish_reason).lower()

                        # Final chunk with usage estimation and tool calls
                        yield StreamChunk(
                            delta="",
                            model=model,
                            finish_reason=finish_reason,
                            tool_calls=tool_calls if tool_calls else None,
                            usage={
                                "prompt_tokens": self.get_token_count(
                                    str(google_messages)
                                ),
                                "completion_tokens": self.get_token_count(
                                    accumulated_content
                                ),
                                "total_tokens": self.get_token_count(
                                    str(google_messages)
                                )
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

            if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ProviderAuthenticationError(
                    f"Google API authentication failed: {error_msg}", provider="google"
                ) from e
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise ProviderRateLimitError(
                    f"Google API rate limit exceeded: {error_msg}", provider="google"
                ) from e
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Google API timeout: {error_msg}", provider="google"
                ) from e
            else:
                raise ProviderResponseError(
                    f"Google API error: {error_msg}", provider="google"
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

        # Verify model is supported (warn but don't fail if not in registry)
        if not await self.validate_model(model):
            logger.warning(
                f"Model '{model}' not found in registry, proceeding anyway"
            )

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

        # Handle JSON response format
        if response_format:
            if (
                response_format.get("type") == "json_object"
                or response_format.get("type") == "json"
            ):
                config_params["response_mime_type"] = "application/json"

                # If a schema is provided, we need to convert it to the format expected by google-genai
                if response_format.get("schema"):
                    schema_str = json.dumps(response_format["schema"], indent=2)
                    existing_instruction = config_params.get("system_instruction", "")
                    config_params["system_instruction"] = (
                        f"{existing_instruction}\n\nResponse must follow this JSON schema:\n{schema_str}".strip()
                    )

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
            if (
                len(conversation_messages) == 1
                and conversation_messages[0].role == "user"
            ):
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
                            converted_content = self._convert_content_to_google_format(
                                msg.content
                            )
                            if isinstance(converted_content, str):
                                parts = [types.Part(text=converted_content)]
                            else:
                                parts = converted_content
                            history.append(types.Content(role="user", parts=parts))
                        elif msg.role == "assistant":
                            history.append(
                                types.Content(
                                    role="model", parts=[types.Part(text=msg.content)]
                                )
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
                    raise ProviderResponseError(
                        "No messages provided for chat", provider="google"
                    )
        except Exception as e:
            # If it's already a typed LLMRing exception, just re-raise it
            from llmring.exceptions import LLMRingError

            if isinstance(e, LLMRingError):
                raise

            error_msg = str(e)
            # Handle rate limiting with exponential backoff
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                # Wait a bit before re-raising to allow for retry at higher level
                await asyncio.sleep(1)
                raise ProviderRateLimitError(
                    f"Google Gemini API rate limit exceeded: {error_msg}",
                    provider="google",
                ) from e
            elif "api key" in error_msg.lower():
                raise ProviderAuthenticationError(
                    f"Google API authentication failed: {error_msg}", provider="google"
                ) from e
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(
                    f"Google API request timed out: {error_msg}", provider="google"
                ) from e
            else:
                # Re-raise SDK exceptions with our standard format
                raise ProviderResponseError(
                    f"Google Gemini API error: {error_msg}", provider="google"
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
            "prompt_tokens": self.get_token_count(
                "\n".join([str(m.content) for m in messages])
            ),
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
