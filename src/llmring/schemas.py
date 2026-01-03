"""Pydantic schemas for requests, responses, and file handling."""

from dataclasses import dataclass, field
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


class FileUploadResponse(BaseModel):
    """Response from file upload."""

    file_id: str
    provider: str
    filename: Optional[str] = None
    size_bytes: int
    created_at: datetime
    purpose: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FileMetadata(BaseModel):
    """File metadata."""

    file_id: str
    provider: str
    filename: Optional[str] = None
    size_bytes: int
    created_at: datetime
    purpose: str
    status: Literal["uploaded", "processing", "ready", "error", "expired"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    messages: List[Message]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Token budget for reasoning models' internal thinking. "
        "If not specified for reasoning models, defaults to min_recommended_reasoning_tokens from registry. "
        "For non-reasoning models, this parameter is ignored.",
    )
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Additional fields for unified interface
    cache: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    json_response: Optional[bool] = None
    files: Optional[List[str]] = None  # List of file_ids
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "Override request timeout in seconds. "
            "When omitted, inherits from the LLMRing default. "
            "Set explicitly to None to disable timeout for this request."
        ),
    )
    extra_params: Dict[str, Any] = Field(default_factory=dict)  # Provider-specific parameters


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

    @property
    def content(self) -> str:
        """Backward-compatible alias for `delta`."""
        return self.delta


# Internal data structures for file registration (not exported)


@dataclass
class ProviderFileUpload:
    """Provider-specific file upload information."""

    provider_file_id: str
    uploaded_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredFile:
    """Internal representation of a registered file."""

    id: str
    file_path: str
    content_hash: str
    uploads: Dict[str, ProviderFileUpload]
    registered_at: datetime
