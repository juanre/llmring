"""Type definitions for the LLM Service API.

This module defines data types used by the LLM Service API.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about an LLM model.
    
    Attributes:
        provider: The model provider (e.g., 'openai', 'anthropic', 'google', 'ollama')
        model_name: The canonical model name as used by the provider
        supported: Whether the model is supported by the provider
        is_default: Whether this is the default model for the provider
    """
    
    provider: str = Field(..., description="Model provider")
    model_name: str = Field(..., description="Canonical model name")
    supported: bool = Field(..., description="Whether model is supported")
    is_default: Optional[bool] = Field(None, description="Whether this is the default model")


class ProviderInfo(BaseModel):
    """Information about a model provider.
    
    Attributes:
        provider: Provider identifier (e.g., 'openai', 'anthropic')
        has_api_key: Whether an API key is configured for this provider
        models: List of supported model names
    """
    
    provider: str = Field(..., description="Provider identifier")
    has_api_key: bool = Field(..., description="Whether API key is configured")
    models: List[str] = Field(default_factory=list, description="Supported models")


class ChatRequest(BaseModel):
    """Chat completion request.
    
    Attributes:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model identifier (e.g., 'openai:gpt-4')
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        response_format: Optional response format specification
        tools: Optional list of tools/functions
        tool_choice: Optional tool selection strategy
    """
    
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    model: str = Field(..., description="Model identifier")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate")
    response_format: Optional[Dict] = Field(None, description="Response format")
    tools: Optional[List[Dict]] = Field(None, description="Available tools")
    tool_choice: Optional[str] = Field(None, description="Tool selection")


class ChatResponse(BaseModel):
    """Chat completion response.
    
    Attributes:
        content: Generated text content
        model: Model that generated the response
        usage: Token usage statistics
        finish_reason: Reason for completion
        tool_calls: Any tool calls made by the model
    """
    
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: Optional[Dict] = Field(None, description="Token usage")
    finish_reason: Optional[str] = Field(None, description="Completion reason")
    tool_calls: Optional[List[Dict]] = Field(None, description="Tool calls")


class ServiceHealth(BaseModel):
    """Health status of the LLM service.
    
    Attributes:
        status: Overall health status
        providers: List of available providers
    """
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall status"
    )
    providers: List[str] = Field(..., description="Available providers")