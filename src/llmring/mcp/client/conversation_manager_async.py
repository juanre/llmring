"""
Async Conversation Manager for MCP Client

Manages conversation persistence and retrieval using async database operations.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pgdbm import AsyncDatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message model"""

    id: str | None
    session_id: str
    role: str
    content: str
    timestamp: datetime
    token_count: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Conversation:
    """Complete conversation model"""

    id: str
    title: str | None
    system_prompt: str | None
    model: str
    temperature: float
    max_tokens: int | None
    tool_config: dict[str, Any] | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message]
    total_tokens: int
    message_count: int


@dataclass
class ConversationSummary:
    """Conversation summary for list views"""

    id: str
    title: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    total_tokens: int
    last_message_preview: str | None


class AsyncConversationManager:
    """Async conversation manager with database operations"""

    def __init__(self, db: AsyncDatabaseManager):
        self.db = db

    async def create_conversation(
        self,
        user_id: str,
        title: str | None = None,
        system_prompt: str | None = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new conversation using db-utils schema placeholders"""
        conversation_id = str(uuid.uuid4())

        query = """
        INSERT INTO {{tables.chat_sessions}} (
            id, title, system_prompt, model,
            temperature, max_tokens, tool_config, created_by
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """

        # Convert tool_config to JSON string if provided
        tool_config_json = json.dumps(tool_config) if tool_config else None

        result = await self.db.fetch_one(
            query,
            conversation_id,
            title,
            system_prompt,
            model,
            temperature,
            max_tokens,
            tool_config_json,
            user_id,
        )

        if not result:
            raise Exception("Failed to create conversation")

        return conversation_id

    async def get_conversation(
        self, conversation_id: str, user_id: str, include_messages: bool = True
    ) -> Conversation | None:
        """Get a conversation by ID using db-utils schema placeholders"""
        query = """
        SELECT cs.*
        FROM {{tables.chat_sessions}} cs
        WHERE cs.id = $1 AND cs.created_by = $2
        """

        conversation_data = await self.db.fetch_one(query, conversation_id, user_id)

        if not conversation_data:
            return None

        messages = []
        if include_messages:
            messages_query = """
            SELECT id, session_id, role, content, timestamp, token_count, metadata
            FROM {{tables.chat_messages}}
            WHERE session_id = $1
            ORDER BY timestamp ASC
            """
            message_rows = await self.db.fetch_all(messages_query, conversation_id)
            messages = [
                Message(
                    id=str(row["id"]),
                    session_id=str(row["session_id"]),
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    token_count=row["token_count"],
                    metadata=row["metadata"],
                )
                for row in message_rows
            ]

        # Calculate totals using schema placeholders
        stats_query = """
        SELECT
            COUNT(*) as message_count,
            COALESCE(SUM(token_count), 0) as total_tokens
        FROM {{tables.chat_messages}}
        WHERE session_id = $1
        """
        stats = await self.db.fetch_one(stats_query, conversation_id)

        return Conversation(
            id=str(conversation_data["id"]),
            title=conversation_data["title"],
            system_prompt=conversation_data["system_prompt"],
            model=conversation_data["model"],
            temperature=float(conversation_data["temperature"]),
            max_tokens=conversation_data["max_tokens"],
            tool_config=conversation_data["tool_config"],
            created_by=str(conversation_data["created_by"]),
            created_at=conversation_data["created_at"],
            updated_at=conversation_data["updated_at"],
            messages=messages,
            total_tokens=int(stats["total_tokens"]) if stats else 0,
            message_count=int(stats["message_count"]) if stats else 0,
        )

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSummary]:
        """List conversations for a user using db-utils schema placeholders"""
        query = """
        SELECT
            cs.id,
            cs.title,
            cs.created_by,
            cs.created_at,
            cs.updated_at,
            COUNT(cm.id) as message_count,
            COALESCE(SUM(cm.token_count), 0) as total_tokens,
            (
                SELECT content
                FROM {{tables.chat_messages}}
                WHERE session_id = cs.id
                ORDER BY timestamp DESC
                LIMIT 1
            ) as last_message_preview
        FROM {{tables.chat_sessions}} cs
        LEFT JOIN {{tables.chat_messages}} cm ON cs.id = cm.session_id
        WHERE cs.created_by = $1
        GROUP BY cs.id, cs.title, cs.created_by, cs.created_at, cs.updated_at
        ORDER BY cs.updated_at DESC
        LIMIT $2 OFFSET $3
        """

        rows = await self.db.fetch_all(query, user_id, limit, offset)

        return [
            ConversationSummary(
                id=str(row["id"]),
                title=row["title"],
                created_by=str(row["created_by"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=int(row["message_count"]),
                total_tokens=int(row["total_tokens"]),
                last_message_preview=row["last_message_preview"],
            )
            for row in rows
        ]

    async def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a message to a conversation"""
        # Verify user owns the conversation
        conversation = await self.get_conversation(
            conversation_id, user_id, include_messages=False
        )
        if not conversation:
            raise PermissionError(f"User {user_id} cannot access conversation {conversation_id}")

        message_id = str(uuid.uuid4())

        query = """
        INSERT INTO {{tables.chat_messages}} (
            id, session_id, role, content, token_count, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """

        # Convert metadata to JSON string if provided
        metadata_json = json.dumps(metadata) if metadata else None

        result = await self.db.fetch_one(
            query,
            message_id,
            conversation_id,
            role,
            content,
            token_count,
            metadata_json,
        )

        if not result:
            raise Exception("Failed to add message")

        # Update conversation timestamp using schema placeholders
        await self.db.execute(
            "UPDATE {{tables.chat_sessions}} SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            conversation_id,
        )

        return message_id

    async def update_conversation(self, conversation_id: str, user_id: str, **updates) -> None:
        """Update conversation metadata"""
        # Check access
        conversation = await self.get_conversation(
            conversation_id, user_id, include_messages=False
        )
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Build update query
        set_clauses = []
        params = []
        param_count = 1

        for field, value in updates.items():
            if field in [
                "title",
                "system_prompt",
                "model",
                "temperature",
                "max_tokens",
                "tool_config",
            ]:
                set_clauses.append(f"{field} = ${param_count}")
                params.append(value)
                param_count += 1

        if not set_clauses:
            return

        query = f"""
        UPDATE {{tables.chat_sessions}}
        SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
        WHERE id = ${param_count}
        """
        params.append(conversation_id)

        await self.db.execute(query, *params)

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation and all its messages"""
        # Check access
        conversation = await self.get_conversation(
            conversation_id, user_id, include_messages=False
        )
        if not conversation:
            return False

        # Delete messages first (due to foreign key constraint)
        await self.db.execute(
            "DELETE FROM {{tables.chat_messages}} WHERE session_id = $1",
            conversation_id,
        )

        # Delete conversation
        result = await self.db.execute(
            "DELETE FROM {{tables.chat_sessions}} WHERE id = $1", conversation_id
        )

        return result is not None

    async def get_or_create_default_conversation(self, user_id: str) -> str:
        """Get or create a default conversation for the user"""
        # Look for existing conversation
        conversations = await self.list_conversations(user_id=user_id, limit=1)

        if conversations:
            return conversations[0].id

        # Create new conversation
        return await self.create_conversation(
            user_id=user_id,
            title="Default Conversation",
            system_prompt="You are a helpful AI assistant.",
        )

    async def search_conversations(
        self, user_id: str, query: str, limit: int = 10
    ) -> list[ConversationSummary]:
        """Search conversations by title or message content"""
        search_query = """
        SELECT DISTINCT
            cs.id,
            cs.title,
            cs.created_by,
            cs.created_at,
            cs.updated_at,
            COUNT(cm.id) as message_count,
            COALESCE(SUM(cm.token_count), 0) as total_tokens,
            (
                SELECT content
                FROM {{tables.chat_messages}}
                WHERE session_id = cs.id
                ORDER BY timestamp DESC
                LIMIT 1
            ) as last_message_preview
        FROM {{tables.chat_sessions}} cs
        LEFT JOIN {{tables.chat_messages}} cm ON cs.id = cm.session_id
        WHERE cs.created_by = $1
        AND (
            cs.title ILIKE $2
            OR EXISTS (
                SELECT 1 FROM {{tables.chat_messages}}
                WHERE session_id = cs.id AND content ILIKE $2
            )
        )
        GROUP BY cs.id, cs.title, cs.created_by, cs.created_at, cs.updated_at
        ORDER BY cs.updated_at DESC
        LIMIT $3
        """

        params = [user_id, f"%{query}%", limit]
        rows = await self.db.fetch_all(search_query, *params)

        return [
            ConversationSummary(
                id=str(row["id"]),
                title=row["title"],
                created_by=str(row["created_by"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=int(row["message_count"]),
                total_tokens=int(row["total_tokens"]),
                last_message_preview=row["last_message_preview"],
            )
            for row in rows
        ]

    async def export_conversation(
        self, conversation_id: str, format: str, auth_context: dict[str, Any]
    ) -> str:
        """Export conversation in specified format"""
        user_id = auth_context["user_id"]
        conversation = await self.get_conversation(conversation_id, user_id)

        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        if format == "json":
            return json.dumps(
                {
                    "conversation": {
                        "id": conversation.id,
                        "title": conversation.title,
                        "created_at": conversation.created_at.isoformat(),
                        "model": conversation.model,
                    },
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                        }
                        for msg in conversation.messages
                    ],
                },
                indent=2,
            )
        elif format == "markdown":
            lines = [
                f"# {conversation.title or 'Conversation'}",
                "",
                f"**Created:** {conversation.created_at}",
                f"**Model:** {conversation.model}",
                f"**Messages:** {len(conversation.messages)}",
                "",
            ]

            for msg in conversation.messages:
                lines.append(f"## {msg.role.title()}")
                lines.append(f"{msg.content}")
                lines.append("")

            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
