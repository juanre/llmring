"""
Ollama API provider implementation using the official SDK.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from llmring.base import BaseLLMProvider
import asyncio
from llmring.model_refresh.models import ModelInfo
from llmring.schemas import LLMResponse, Message
from ollama import AsyncClient, ResponseError
from llmring.net.retry import retry_async
from llmring.net.circuit_breaker import CircuitBreaker


class OllamaProvider(BaseLLMProvider):
    """Implementation of Ollama API provider using the official SDK."""

    def __init__(
        self,
        api_key: Optional[
            str
        ] = None,  # Not used for Ollama, included for API compatibility
        base_url: Optional[str] = None,
        model: str = "llama3.3:latest",
    ):
        """
        Initialize the Ollama provider.

        Args:
            api_key: Not used for Ollama (included for API compatibility)
            base_url: Base URL for the Ollama API server
            model: Default model to use
        """
        super().__init__(api_key=None, base_url=base_url)
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.default_model = model

        # Initialize the client with the SDK
        self.client = AsyncClient(host=self.base_url)

        # List of commonly supported models
        self.supported_models = [
            "llama3.3:latest",
            "llama3.3",
            "llama3.2",
            "llama3.1",
            "llama3",
            "mistral",
            "mixtral",
            "codellama",
            "phi3",
            "gemma2",
            "gemma",
            "qwen2.5",
            "qwen",
        ]
        self._breaker = CircuitBreaker()

    def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by Ollama.
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

        # Check if it's exactly in our supported models list
        if model in self.supported_models:
            return True

        # Allow any model that starts with a known model name
        for known_model in self.supported_models:
            if model.startswith(known_model):
                return True

        # Check for common Ollama model patterns but reject non-Ollama models
        ollama_patterns = [
            r"^llama\d*",  # llama, llama2, llama3, etc.
            r"^mistral",  # mistral variations
            r"^mixtral",  # mixtral variations
            r"^codellama",  # codellama variations
            r"^phi\d*",  # phi, phi3, etc.
            r"^gemma",  # gemma variations
            r"^qwen",  # qwen variations
            r"^deepseek",  # deepseek variations
            r"^[a-z][a-z0-9\-_]*[:]",  # custom models with tags
        ]

        # Only allow models that match Ollama patterns and look valid
        if re.match(r"^[a-zA-Z0-9\-_:.]+$", model):
            for pattern in ollama_patterns:
                if re.match(pattern, model.lower()):
                    return True

        return False

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported Ollama model names.
        Note: This returns common models, but Ollama can support
        any model that's installed locally.

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
            # If we can't get the list, return our supported models list
            return self.supported_models.copy()

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
            LLM response
        """
        # Strip provider prefix if present
        if model.lower().startswith("ollama:"):
            model = model.split(":", 1)[1]

        # Note: We're more lenient with model validation for Ollama
        # since models are user-installed locally
        if not self.validate_model(model):
            raise ValueError(f"Invalid model name format: {model}")

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

            async def _do_call():
                return await asyncio.wait_for(
                    self.client.chat(
                        model=model,
                        messages=ollama_messages,
                        stream=False,
                        options=options,
                        format=format_param,
                    ),
                    timeout=timeout_s,
                )

            key = f"ollama:{model}"
            if not await self._breaker.allow(key):
                raise Exception("Ollama circuit open for model")
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
                                        "id": f"call_{hash(json_text) & 0xffffffff:x}",  # Generate a deterministic ID
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
            raise Exception(f"Ollama API error: {e.error}")
        except Exception as e:
            try:
                await self._breaker.record_failure(f"ollama:{model}")
            except Exception:
                pass
            # Handle general errors
            raise Exception(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")

    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from Ollama API.

        Returns:
            List of discovered models with capabilities
        """
        try:
            # Call Ollama's list endpoint
            models_response = await self.client.list()
            discovered = []

            for model in models_response["models"]:
                model_name = model["name"]

                model_info = ModelInfo(
                    provider="ollama",
                    model_name=model_name,
                    display_name=self._get_display_name(model_name),
                    description=f"Ollama {model_name} model",
                    source="api",
                    raw_data={"api_response": model},
                    cost_per_token_input=None,  # Ollama is free
                    cost_per_token_output=None,  # Ollama is free
                )

                # Add model-specific capabilities if available
                capabilities = await self.get_model_capabilities(model_name)
                if capabilities:
                    model_info.max_context = capabilities.get("max_context")
                    model_info.max_output_tokens = capabilities.get("max_output_tokens")
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
                    model_info.tool_call_format = capabilities.get("tool_call_format")

                # Extract additional metadata from Ollama response
                if "details" in model:
                    details = model["details"]
                    if "parameter_size" in details:
                        model_info.description += f" ({details['parameter_size']})"
                    if "quantization_level" in details:
                        model_info.description += f" - {details['quantization_level']}"

                discovered.append(model_info)

            return discovered

        except Exception as e:
            # Fallback to static model list if API fails
            import logging

            logging.warning(
                f"Ollama model discovery failed: {str(e)}, falling back to static list"
            )
            return await super().discover_models()

    async def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get capabilities for a specific Ollama model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model capabilities
        """
        # Extract base model name (remove tags like :latest)
        base_model = model_name.split(":")[0].lower()

        # Known capabilities for popular Ollama models
        capabilities_map = {
            # Llama models
            "llama3.3": {
                "max_context": 131072,
                "max_output_tokens": 131072,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            "llama3.2": {
                "max_context": 131072,
                "max_output_tokens": 131072,
                "supports_vision": True,  # Some variants support vision
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            "llama3.1": {
                "max_context": 131072,
                "max_output_tokens": 131072,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            "llama3": {
                "max_context": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            # Mistral models
            "mistral": {
                "max_context": 32768,
                "max_output_tokens": 32768,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            "mixtral": {
                "max_context": 32768,
                "max_output_tokens": 32768,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            # Code models
            "codellama": {
                "max_context": 16384,
                "max_output_tokens": 16384,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            # Other models
            "phi3": {
                "max_context": 4096,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": None,
            },
            "gemma2": {
                "max_context": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": None,
            },
            "gemma": {
                "max_context": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": False,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": None,
            },
            "qwen2.5": {
                "max_context": 32768,
                "max_output_tokens": 32768,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
            "qwen": {
                "max_context": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_json_mode": False,
                "supports_parallel_tool_calls": False,
                "tool_call_format": "ollama",
            },
        }

        return capabilities_map.get(base_model)

    def _get_display_name(self, model_id: str) -> str:
        """Get a friendly display name for a model."""
        # Extract base name and version info
        parts = model_id.split(":")
        base_name = parts[0]
        tag = parts[1] if len(parts) > 1 else "latest"

        # Common display name mappings
        display_names = {
            "llama3.3": "Llama 3.3",
            "llama3.2": "Llama 3.2",
            "llama3.1": "Llama 3.1",
            "llama3": "Llama 3",
            "mistral": "Mistral",
            "mixtral": "Mixtral",
            "codellama": "Code Llama",
            "phi3": "Phi-3",
            "gemma2": "Gemma 2",
            "gemma": "Gemma",
            "qwen2.5": "Qwen 2.5",
            "qwen": "Qwen",
        }

        display_name = display_names.get(base_name, base_name.title())

        if tag != "latest":
            display_name += f" ({tag})"

        return display_name
