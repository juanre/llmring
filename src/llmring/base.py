"""
Base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

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
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Check if the model is supported by this provider.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.

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
