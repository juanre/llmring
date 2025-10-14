"""
Base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from llmring.schemas import LLMResponse, Message, StreamChunk


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    default_model: Optional[str] = Field(None, description="Default model to use")
    timeout_seconds: float = Field(60.0, description="Request timeout in seconds")


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        """
        Initialize the LLM provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @abstractmethod
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
    ) -> LLMResponse:
        """
        Send a chat request to the LLM provider.

        Args:
            messages: List of messages in the conversation
            model: Model identifier to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format specification
            tools: Optional list of tools/functions available
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            extra_params: Provider-specific parameters to pass through

        Returns:
            LLM response with complete generated content
        """
        pass

    @abstractmethod
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
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a streaming chat request to the LLM provider.

        Args:
            messages: List of messages in the conversation
            model: Model identifier to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (output tokens)
            reasoning_tokens: Token budget for reasoning models' internal thinking
            response_format: Optional response format specification
            tools: Optional list of tools/functions available
            tool_choice: Optional tool choice parameter
            json_response: Optional flag to request JSON response
            cache: Optional cache configuration
            extra_params: Provider-specific parameters to pass through

        Returns:
            Async iterator of stream chunks

        Example:
            >>> async for chunk in provider.chat_stream(messages, model="gpt-4"):
            ...     print(chunk.content, end="", flush=True)
        """
        pass

    @abstractmethod
    async def get_capabilities(self) -> "ProviderCapabilities":
        """
        Get the capabilities of this provider.

        Returns:
            Provider capabilities including supported models and features
        """
        pass

    async def aclose(self) -> None:
        """
        Clean up provider resources.

        This should be called when the provider is no longer needed to properly
        close any open connections (e.g., httpx clients).
        """
        pass  # Default implementation does nothing


class ProviderCapabilities(BaseModel):
    """Capabilities of an LLM provider."""

    provider_name: str = Field(..., description="Name of the provider")
    supported_models: List[str] = Field(..., description="List of supported model IDs")
    supports_streaming: bool = Field(True, description="Whether streaming is supported")
    supports_tools: bool = Field(True, description="Whether function calling is supported")
    supports_vision: bool = Field(False, description="Whether image inputs are supported")
    supports_audio: bool = Field(False, description="Whether audio inputs are supported")
    supports_documents: bool = Field(False, description="Whether document inputs are supported")
    supports_json_mode: bool = Field(False, description="Whether JSON mode is supported")
    supports_caching: bool = Field(False, description="Whether prompt caching is supported")
    max_context_window: Optional[int] = Field(None, description="Maximum context window size")
    default_model: str = Field(..., description="Default model for this provider")
