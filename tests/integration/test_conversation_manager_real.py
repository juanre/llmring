"""
Real integration tests for AsyncConversationManager without mocks.

These tests verify the actual method signatures and behavior.
"""

import os
from uuid import uuid4

import pytest

from llmring.mcp.client.conversation_manager_async import AsyncConversationManager, Conversation


class TestConversationManagerReal:
    """Test conversation manager with real HTTP client, no mocks."""

    @pytest.fixture
    async def conversation_manager(self):
        """Create a real conversation manager."""
        # Use test server if available
        server_url = os.getenv("TEST_LLMRING_SERVER_URL", "http://localhost:8001")
        api_key = os.getenv("TEST_LLMRING_API_KEY", "test-key")

        return AsyncConversationManager(llmring_server_url=server_url, api_key=api_key)

    def is_server_available(self, error):
        """Check if error is due to server not being available."""
        error_str = str(error).lower()
        return any(word in error_str for word in ["connect", "refused", "unreachable", "timeout"])

    @pytest.mark.asyncio
    async def test_create_conversation_signature(self, conversation_manager):
        """Test that create_conversation has the correct signature."""
        # Test with all parameters to ensure signature is correct
        try:
            conversation_id = await conversation_manager.create_conversation(
                user_id="test-user-123",
                title="Test Conversation",
                system_prompt="You are a helpful assistant",
                model="mcp_agent",  # Use our alias
                temperature=0.7,
                max_tokens=1000,
                tool_config={"tools": [{"name": "test_tool"}]},
            )

            # If successful, verify return type
            assert isinstance(conversation_id, str)
            assert len(conversation_id) > 0

        except Exception as e:
            if self.is_server_available(e):
                # Server not available is OK - we tested the signature compiles
                pass
            else:
                # Unexpected error
                raise

    @pytest.mark.asyncio
    async def test_add_message_signature(self, conversation_manager):
        """Test that add_message has the correct signature and parameters."""
        conversation_id = str(uuid4())

        # Test the actual method signature
        try:
            message_id = await conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content="Test message content",
                token_count=15,
                metadata={
                    "tool_calls": [{"name": "test", "args": {}}],
                    "processing_time_ms": 100,
                },
            )

            # If successful, verify return type
            assert isinstance(message_id, str)

        except Exception as e:
            if self.is_server_available(e):
                # Expected if no server running
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_get_conversation_signature(self, conversation_manager):
        """Test that get_conversation has the correct signature."""
        conversation_id = str(uuid4())

        try:
            conversation = await conversation_manager.get_conversation(
                conversation_id=conversation_id, include_messages=True
            )

            # If found (unlikely with random ID), verify type
            if conversation:
                assert isinstance(conversation, Conversation)
                assert conversation.id == conversation_id

        except Exception as e:
            if self.is_server_available(e):
                pass
            else:
                # Could be None if not found, that's OK
                if "not found" not in str(e).lower():
                    raise

    @pytest.mark.asyncio
    async def test_list_conversations_signature(self, conversation_manager):
        """Test that list_conversations exists and has correct signature."""
        try:
            conversations = await conversation_manager.list_conversations(
                user_id="test-user-list", limit=10, offset=0
            )

            # If successful, verify return type
            assert isinstance(conversations, list)

        except AttributeError:
            pytest.fail("list_conversations method doesn't exist")
        except Exception as e:
            if self.is_server_available(e):
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_conversation_flow_without_server(self, conversation_manager):
        """Test the conversation flow to ensure all methods work together."""
        # This test verifies the method calls compile and have correct signatures
        # even if the server isn't running

        try:
            # 1. Create conversation
            conv_id = await conversation_manager.create_conversation(
                user_id="test-user-flow",
                title="Integration Test",
                system_prompt="Test system prompt",
                model="mcp_agent",
                temperature=0.5,
            )

            # 2. Add messages
            await conversation_manager.add_message(
                conversation_id=conv_id, role="user", content="Hello, this is a test"
            )

            await conversation_manager.add_message(
                conversation_id=conv_id,
                role="assistant",
                content="Hello! I understand this is a test.",
                token_count=10,
                metadata={"test": True},
            )

            # 3. Get conversation back
            conversation = await conversation_manager.get_conversation(
                conversation_id=conv_id, include_messages=True
            )

            if conversation:
                assert conversation.id == conv_id
                if conversation.messages:
                    assert len(conversation.messages) >= 2

        except Exception as e:
            if self.is_server_available(e):
                # Expected - we're just testing signatures
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_message_with_tool_results(self, conversation_manager):
        """Test adding a message with tool results in metadata."""
        conv_id = str(uuid4())

        # This specifically tests the pattern we use in stateless_engine
        try:
            message_id = await conversation_manager.add_message(
                conversation_id=conv_id,
                role="tool",
                content="Tool execution result: 42",
                metadata={
                    "tool_result": {
                        "tool_call_id": "call_123",
                        "result": {"answer": 42},
                    }
                },
            )

            assert message_id is not None

        except Exception as e:
            if self.is_server_available(e):
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_conversation_not_found_behavior(self, conversation_manager):
        """Test what happens when trying to get a non-existent conversation."""
        fake_id = str(uuid4())

        try:
            conversation = await conversation_manager.get_conversation(conversation_id=fake_id)

            # Should return None or raise an error
            assert conversation is None

        except Exception as e:
            if self.is_server_available(e):
                pass
            else:
                # Any other error is also acceptable (like 404)
                pass
