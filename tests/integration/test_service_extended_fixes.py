"""
Test service_extended fixes with real APIs.

Tests that the fixed response access (response.content instead of response.choices)
works correctly with actual API calls.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from llmring.service_extended import LLMRingExtended
from llmring.schemas import LLMRequest, Message


class TestServiceExtendedFixes:
    """Test the fixed service_extended code paths with real APIs."""

    @pytest.fixture
    def extended_service(self):
        """Create extended service with conversation support."""
        return LLMRingExtended(
            enable_conversations=True,
            message_logging_level="full"
        )

    @pytest.mark.asyncio
    async def test_chat_with_conversation_openai(self, extended_service):
        """Test that chat_with_conversation works with OpenAI (uses response.content)."""
        conversation_id = uuid4()

        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[
                Message(role="system", content="You are a helpful assistant. Respond with exactly: 'Test response'"),
                Message(role="user", content="Hello")
            ],
            max_tokens=10,
            temperature=0.1
        )

        # Mock the server client to avoid actual server calls
        with patch.object(extended_service, 'server_client', new=None):
            response = await extended_service.chat_with_conversation(
                request=request,
                conversation_id=conversation_id,
                store_messages=False  # Don't try to store to avoid server dependency
            )

        # Verify the response structure - this validates response.content access works
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert hasattr(response, 'model'), "Response should have model attribute"
        assert hasattr(response, 'usage'), "Response should have usage attribute"
        assert hasattr(response, 'finish_reason'), "Response should have finish_reason attribute"

        # Verify content is a string (not a choices array)
        assert isinstance(response.content, str), "response.content should be string, not choices array"
        assert len(response.content) > 0, "Response should have content"

        print(f"✓ OpenAI response.content access works: {response.content[:50]}...")

    @pytest.mark.asyncio
    async def test_chat_with_conversation_anthropic(self, extended_service):
        """Test that chat_with_conversation works with Anthropic (uses response.content)."""
        conversation_id = uuid4()

        request = LLMRequest(
            model="anthropic:claude-3-5-haiku",
            messages=[
                Message(role="system", content="You are a helpful assistant. Respond with exactly: 'Test response'"),
                Message(role="user", content="Hello")
            ],
            max_tokens=10,
            temperature=0.1
        )

        # Mock the server client to avoid actual server calls
        with patch.object(extended_service, 'server_client', new=None):
            response = await extended_service.chat_with_conversation(
                request=request,
                conversation_id=conversation_id,
                store_messages=False  # Don't try to store to avoid server dependency
            )

        # Verify the response structure - this validates response.content access works
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert hasattr(response, 'model'), "Response should have model attribute"
        assert hasattr(response, 'usage'), "Response should have usage attribute"
        assert hasattr(response, 'finish_reason'), "Response should have finish_reason attribute"

        # Verify content is a string (not a choices array)
        assert isinstance(response.content, str), "response.content should be string, not choices array"
        assert len(response.content) > 0, "Response should have content"

        print(f"✓ Anthropic response.content access works: {response.content[:50]}...")

    @pytest.mark.asyncio
    async def test_response_structure_validation(self, extended_service):
        """Test that response structure is correct (has content, not choices)."""
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Say hello")
            ],
            max_tokens=20,
            temperature=0.1
        )

        # Call chat method to verify response structure
        response = await extended_service.chat(request)

        # Verify response has correct structure (not the old choices format)
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert hasattr(response, 'model'), "Response should have model attribute"
        assert hasattr(response, 'usage'), "Response should have usage attribute"
        assert hasattr(response, 'finish_reason'), "Response should have finish_reason attribute"
        assert not hasattr(response, 'choices'), "Response should NOT have choices attribute"

        # Verify content is accessible and correct type
        assert isinstance(response.content, str), "Content should be string"
        assert len(response.content) > 0, "Content should not be empty"

        print(f"✓ Response structure is correct: {response.content[:50]}...")

    @pytest.mark.asyncio
    async def test_message_storage_structure(self, extended_service):
        """Test that message storage uses correct response structure."""
        conversation_id = uuid4()

        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[
                Message(role="user", content="Test message")
            ],
            max_tokens=10,
            temperature=0.1
        )

        # Mock server client to capture what would be sent
        mock_server_client = AsyncMock()
        extended_service.server_client = mock_server_client

        await extended_service.chat_with_conversation(
            request=request,
            conversation_id=conversation_id,
            store_messages=True
        )

        # Verify server client was called with correct structure
        mock_server_client.post.assert_called_once()
        call_args = mock_server_client.post.call_args

        # Check the payload structure
        json_payload = call_args[1]["json"]
        messages_to_store = json_payload["messages"]

        # Should have user message and assistant response
        assert len(messages_to_store) >= 2, "Should store user and assistant messages"

        # Check assistant message structure
        assistant_msg = None
        for msg in messages_to_store:
            if msg["role"] == "assistant":
                assistant_msg = msg
                break

        assert assistant_msg is not None, "Should have assistant message"
        assert "content" in assistant_msg, "Assistant message should have content"
        assert isinstance(assistant_msg["content"], str), "Content should be string"

        # Verify it's not trying to access choices (old broken code)
        assert "choices" not in str(call_args), "Should not reference choices anywhere"

        print("✓ Message storage structure is correct (uses response.content)")

    @pytest.mark.asyncio
    async def test_tool_calls_in_response_structure(self, extended_service):
        """Test that tool calls are properly handled in the fixed response structure."""
        request = LLMRequest(
            model="openai:gpt-4o",
            messages=[
                Message(role="system", content="You are a helpful assistant with access to tools."),
                Message(role="user", content="What's the weather like? Just say you'd need a weather tool.")
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }],
            max_tokens=50,
            temperature=0.1
        )

        response = await extended_service.chat(request)

        # Verify response structure includes tool_calls if present
        assert hasattr(response, 'tool_calls'), "Response should have tool_calls attribute"

        # Tool calls should be None or a list, never accessed as choices[0].message.tool_calls
        if response.tool_calls:
            assert isinstance(response.tool_calls, list), "tool_calls should be a list"

        print(f"✓ Tool calls structure correct: {response.tool_calls}")