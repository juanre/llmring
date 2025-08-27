"""
Pydantic schemas for MCP chat interface.
"""

import uuid
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# LLM schemas
class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any]

    def dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"id": self.id, "tool_name": self.tool_name, "arguments": self.arguments}


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_call_id: str
    result: Any
    timestamp: datetime | None = None

    def dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "result": self.result,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Message(BaseModel):
    """A message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Any  # Can be str or structured content
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    timestamp: datetime | None = None


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class LLMResponse(BaseModel):
    """A response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatSession(BaseModel):
    """A chat session with messages."""

    id: str
    user_id: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[Message] = []


class MCPServer(BaseModel):
    """Configuration for an MCP server."""

    id: str
    name: str
    base_url: str
    transport_type: str = "http"  # http, websocket, stdio
    auth_type: str = "none"  # bearer, api_key, oauth2, none
    auth_config: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LLMModel(BaseModel):
    """LLM model information."""

    provider: LLMProvider
    model_name: str
    display_name: str
    context_length: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_parallel_tool_calls: bool = False
    tool_call_format: Literal["openai", "anthropic", "google", "ollama"] | None = None
    cost_per_1k_input_tokens: float | None = None
    cost_per_1k_output_tokens: float | None = None
    is_active: bool = True


class ToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class OAuthToken(BaseModel):
    """OAuth token information."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    expires_at: datetime | None = None
    refresh_token: str | None = None
    scope: str | None = None

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) >= self.expires_at


class OAuthConfig(BaseModel):
    """OAuth configuration."""

    auth_server_url: str
    client_id: str
    client_secret: str | None = None
    redirect_uri: str = "http://localhost:8080/callback"
    scopes: list[str] = Field(
        default_factory=lambda: ["profile", "email", "mcp:read", "mcp:write"]
    )
    client_type: Literal["public", "confidential"] = "public"
