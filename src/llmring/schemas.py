from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Any  # Can be str or structured content
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # New fields for unified interface and caching
    cache: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    json_response: Optional[bool] = None


class LLMResponse(BaseModel):
    """A response from an LLM provider."""

    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ===== DATABASE MODELS =====


class LLMModel(BaseModel):
    """Model information from the database."""

    id: Optional[int] = None
    provider: str = Field(
        ..., description="Model provider (anthropic, openai, google, ollama)"
    )
    model_name: str = Field(..., description="Model name")
    display_name: Optional[str] = None
    description: Optional[str] = None

    # Capabilities
    max_context: Optional[int] = Field(None, description="Maximum context length")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_parallel_tool_calls: bool = False
    tool_call_format: Optional[str] = None

    # Cost information (dollars per million tokens)
    dollars_per_million_tokens_input: Optional[Decimal] = Field(
        None, description="Cost in dollars per million input tokens"
    )
    dollars_per_million_tokens_output: Optional[Decimal] = Field(
        None, description="Cost in dollars per million output tokens"
    )

    # Status
    inactive_from: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        """Check if model is active based on inactive_from timestamp."""
        return self.inactive_from is None


class CallRecord(BaseModel):
    """API call record from the database."""

    id: UUID
    origin: str = Field(..., description="Name of the calling program/app")
    id_at_origin: str = Field(..., description="User identifier at the origin")

    # Model information
    model_id: Optional[int] = None
    provider: str
    model_name: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Cost information
    estimated_cost: Decimal
    dollars_per_million_tokens_input_used: Optional[Decimal] = None
    dollars_per_million_tokens_output_used: Optional[Decimal] = None

    # Performance
    response_time_ms: Optional[int] = None

    # Request details
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt_hash: Optional[str] = None
    tools_used: Optional[List[str]] = None

    # Status
    status: str = Field(
        default="success",
        description="Call status (success, error, timeout, rate_limited)",
    )
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Timestamp
    called_at: datetime


class UsageStats(BaseModel):
    """Usage statistics for an origin and user."""

    total_calls: int
    total_tokens: int
    total_cost: Decimal
    avg_cost_per_call: Decimal
    most_used_model: Optional[str] = None
    success_rate: Decimal = Field(
        ..., description="Success rate as decimal (0.95 = 95%)"
    )
    avg_response_time_ms: Optional[int] = None


class DailyAnalytics(BaseModel):
    """Daily aggregated analytics."""

    origin: str
    id_at_origin: str
    date: datetime
    provider: str
    model_name: str

    # Aggregated metrics
    total_calls: int
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: Decimal

    # Performance metrics
    avg_response_time_ms: Optional[int] = None
    success_rate: Decimal

    # Error counts
    error_count: int = 0
    timeout_count: int = 0
    rate_limit_count: int = 0

    created_at: Optional[datetime] = None
