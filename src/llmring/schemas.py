from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Any  # Can be str or structured content
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = (
        None  # For cache_control and other provider-specific metadata
    )


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Additional fields for unified interface
    cache: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    json_response: Optional[bool] = None
    stream: Optional[bool] = False  # Support for streaming responses
    extra_params: Dict[str, Any] = Field(
        default_factory=dict
    )  # Provider-specific parameters


class LLMResponse(BaseModel):
    """A response from an LLM provider."""

    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    parsed: Optional[Dict[str, Any]] = None  # Parsed JSON when response_format used

    @property
    def total_tokens(self) -> Optional[int]:
        """Get total tokens used."""
        if not self.usage:
            return None
        return self.usage.get("total_tokens") or (
            self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)
        )


class StreamChunk(BaseModel):
    """A chunk of a streaming response."""

    delta: str  # The text delta in this chunk
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # Only present in final chunk
    tool_calls: Optional[List[Dict[str, Any]]] = None
