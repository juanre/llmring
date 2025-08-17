"""
Base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from llmring.model_refresh.models import ModelInfo
from llmring.schemas import LLMResponse, Message


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider
            base_url: Optional base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
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
        Send a chat request to the LLM provider.

        Args:
            messages: List of messages in the conversation
            model: The model to use
            temperature: Optional temperature parameter
            max_tokens: Optional max tokens parameter
            response_format: Optional response format parameters
            tools: Optional list of tools to make available
            tool_choice: Optional tool choice configuration

        Returns:
            LLM response object
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by this provider.

        Args:
            model: Model name to check

        Returns:
            True if the model is supported, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names for this provider.

        Returns:
            List of supported model names
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model for this provider.

        Returns:
            Default model name
        """
        pass

    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from the provider's API.

        Returns:
            List of discovered models with capabilities and metadata.
            Default implementation falls back to static model list.
        """
        # Default implementation for providers without API discovery
        supported_models = self.get_supported_models()
        discovered = []

        for model_name in supported_models:
            model_info = ModelInfo(
                provider=self._get_provider_name(),
                model_name=model_name,
                display_name=model_name,
                source="static",
            )

            # Try to get additional capabilities if implemented
            capabilities = await self.get_model_capabilities(model_name)
            if capabilities:
                model_info.max_context = capabilities.get("max_context")
                model_info.max_output_tokens = capabilities.get("max_output_tokens")
                model_info.supports_vision = capabilities.get("supports_vision", False)
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

            discovered.append(model_info)

        return discovered

    async def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get capabilities for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model capabilities, or None if not available
        """
        # Default implementation returns None - subclasses can override
        return None

    def _get_provider_name(self) -> str:
        """
        Get the provider name for this provider.

        Returns:
            Provider name (e.g., "openai", "anthropic")
        """
        # Extract provider name from class name
        class_name = self.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name:
            return "anthropic"
        elif "google" in class_name:
            return "google"
        elif "ollama" in class_name:
            return "ollama"
        else:
            return "unknown"

    async def generate_response(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the LLM (convenience method).

        Args:
            system_message: System message to send
            user_message: User message to send

        Returns:
            Generated response as a string
        """
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=user_message),
        ]
        response = await self.chat(messages, model=self.get_default_model())
        return response.content


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
        """
        Create an LLM provider based on the provider type.

        Args:
            provider_type: Type of provider to create
            **kwargs: Additional arguments to pass to the provider

        Returns:
            LLM provider instance

        Raises:
            ValueError: If the provider type is not supported
        """
        provider_type = provider_type.lower()

        if provider_type == "openai":
            from llmring.providers.openai_api import OpenAIProvider

            return OpenAIProvider(**kwargs)

        elif provider_type == "anthropic":
            from llmring.providers.anthropic_api import AnthropicProvider

            return AnthropicProvider(**kwargs)

        elif provider_type == "google":
            from llmring.providers.google_api import GoogleProvider

            return GoogleProvider(**kwargs)

        elif provider_type == "ollama":
            from llmring.providers.ollama_api import OllamaProvider

            return OllamaProvider(**kwargs)

        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
