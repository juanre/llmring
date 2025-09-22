"""
Async Conversation Manager for MCP Client

Manages conversation persistence and retrieval using HTTP API calls to llmring-server.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from llmring.mcp.http_client import MCPHttpClient

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message model"""

    id: Optional[str]
    conversation_id: str
    role: str
    content: str
    timestamp: datetime
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """Complete conversation model"""

    id: str
    title: Optional[str]
    system_prompt: Optional[str]
    model: str
    temperature: float
    max_tokens: Optional[int]
    tool_config: Optional[Dict[str, Any]]
    created_by: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    total_tokens: int
    message_count: int


@dataclass
class ConversationSummary:
    """Conversation summary for list views"""

    id: str
    title: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    total_tokens: int
    last_message_preview: Optional[str]


class AsyncConversationManager:
    """Async conversation manager using HTTP API"""

    def __init__(
        self,
        llmring_server_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the async conversation manager.

        Args:
            llmring_server_url: URL of llmring-server
            api_key: Optional API key for authentication
        """
        self.http_client = MCPHttpClient(
            base_url=llmring_server_url,
            api_key=api_key,
        )

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = "mcp_agent",  # Use alias instead of hardcoded model
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new conversation via HTTP API"""
        conversation_id = await self.http_client.create_conversation(
            title=title or "New Conversation",
            system_prompt=system_prompt,
            model_alias=model,
            project_id=user_id,  # Using user_id as project_id for now
        )
        return str(conversation_id)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to a conversation via HTTP API"""
        message_data = await self.http_client.add_message(
            conversation_id=UUID(conversation_id),
            role=role,
            content=content,
            metadata=metadata,
        )
        return message_data.get("id", "")

    async def get_conversation(
        self,
        conversation_id: str,
        include_messages: bool = True,
    ) -> Optional[Conversation]:
        """Get a conversation with messages via HTTP API"""
        # Note: This would need a dedicated endpoint in llmring-server
        # For now, we'll construct from messages
        messages = await self.http_client.get_conversation_messages(
            conversation_id=UUID(conversation_id)
        )

        if not messages and not include_messages:
            return None

        # Construct conversation object from messages
        # This is a simplified version - real implementation would fetch full conversation data
        conversation = Conversation(
            id=conversation_id,
            title="Conversation",
            system_prompt=None,
            model="mcp_agent",  # Use alias instead of hardcoded model
            temperature=0.7,
            max_tokens=None,
            tool_config=None,
            created_by="user",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            messages=[
                Message(
                    id=msg.get("id"),
                    conversation_id=conversation_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    timestamp=datetime.fromisoformat(
                        msg.get("timestamp", datetime.now(UTC).isoformat())
                    ),
                    token_count=msg.get("token_count"),
                    metadata=msg.get("metadata"),
                )
                for msg in messages
            ],
            total_tokens=sum(
                msg.get("token_count", 0) for msg in messages if msg.get("token_count")
            ),
            message_count=len(messages),
        )

        return conversation

    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        """Get messages for a conversation via HTTP API"""
        messages = await self.http_client.get_conversation_messages(
            conversation_id=UUID(conversation_id),
            limit=limit,
        )

        return [
            Message(
                id=msg.get("id"),
                conversation_id=conversation_id,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                timestamp=datetime.fromisoformat(
                    msg.get("timestamp", datetime.now(UTC).isoformat())
                ),
                token_count=msg.get("token_count"),
                metadata=msg.get("metadata"),
            )
            for msg in messages[offset:]
        ]

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationSummary]:
        """List conversations for a user via HTTP API"""
        # Note: This would need a dedicated endpoint in llmring-server
        # For now, return empty list
        return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation via HTTP API"""
        # Note: This would need a dedicated endpoint in llmring-server
        # For now, return False
        return False

    async def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update conversation settings via HTTP API"""
        # Note: This would need a dedicated endpoint in llmring-server
        # For now, return False
        return False

    async def close(self):
        """Close the HTTP client"""
        await self.http_client.close()
