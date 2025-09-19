"""
Test MCP client fixes with real APIs.

Tests that the fixed id_at_origin handling (now uses metadata instead of invalid kwarg)
works correctly with actual API calls.
"""

import pytest
from unittest.mock import patch, AsyncMock

from llmring.mcp.client.stateless_engine import StatelessChatEngine, ChatRequest


class TestMCPClientFixes:
    """Test the fixed MCP client code paths with real APIs."""

    @pytest.fixture
    def chat_engine(self):
        """Create chat engine for testing."""
        # Create LLMRing service with test lockfile
        import os
        from llmring.service import LLMRing
        test_lockfile = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'llmring.lock.json'
        )
        llm_service = LLMRing(origin="test", lockfile_path=test_lockfile)
        return StatelessChatEngine(llmring=llm_service)

    @pytest.mark.asyncio
    async def test_id_at_origin_in_metadata(self, chat_engine):
        """Test that id_at_origin is properly stored in metadata instead of passed as kwarg."""
        request = ChatRequest(
            message="Hello, what is 2+2?",
            model="openai_fast",
            auth_context={"user_id": "test-user-123"},
            save_to_db=False,  # Don't save to avoid DB dependency
        )

        # Patch the LLMRing.chat method to capture what's actually called
        original_chat = chat_engine.llmring.chat
        captured_request = None

        async def capture_chat(request):
            nonlocal captured_request
            captured_request = request
            # Call the real method
            return await original_chat(request)

        with patch.object(chat_engine.llmring, "chat", side_effect=capture_chat):
            response = await chat_engine.process_request(request)

        # Verify the request was called correctly
        assert captured_request is not None, "chat() should have been called"
        assert hasattr(captured_request, "metadata"), "Request should have metadata"
        assert captured_request.metadata is not None, "Metadata should not be None"
        assert "id_at_origin" in captured_request.metadata, (
            "id_at_origin should be in metadata"
        )
        assert captured_request.metadata["id_at_origin"] == "test-user-123", (
            "id_at_origin should match user_id"
        )

        # Verify response structure
        assert hasattr(response, "message"), "Response should have message attribute"
        assert hasattr(response.message, "content"), "Message should have content"
        assert isinstance(response.message.content, str), "Content should be string"

        print(
            f"✓ id_at_origin properly stored in metadata: {captured_request.metadata}"
        )

    @pytest.mark.asyncio
    async def test_streaming_with_metadata(self, chat_engine):
        """Test that streaming also properly handles metadata."""
        request = ChatRequest(
            message="Count to 3",
            model="openai_fast",
            auth_context={"user_id": "stream-user-456"},
            save_to_db=False,
        )

        # Patch the LLMRing.chat method to capture streaming request
        original_chat = chat_engine.llmring.chat
        captured_request = None

        async def capture_stream_chat(request):
            nonlocal captured_request
            captured_request = request
            # Call the real method
            return await original_chat(request)

        with patch.object(chat_engine.llmring, "chat", side_effect=capture_stream_chat):
            chunks = []
            async for chunk in chat_engine.process_request_stream(request):
                chunks.append(chunk)

        # Verify metadata was set correctly for streaming
        assert captured_request is not None, (
            "chat() should have been called for streaming"
        )
        assert hasattr(captured_request, "metadata"), "Request should have metadata"
        assert captured_request.metadata is not None, "Metadata should not be None"
        assert "id_at_origin" in captured_request.metadata, (
            "id_at_origin should be in metadata"
        )
        assert captured_request.metadata["id_at_origin"] == "stream-user-456", (
            "id_at_origin should match user_id"
        )

        # Verify streaming worked
        assert len(chunks) > 0, "Should have received streaming chunks"

        print(f"✓ Streaming with metadata works: {len(chunks)} chunks received")

    @pytest.mark.asyncio
    async def test_no_auth_context_handling(self, chat_engine):
        """Test that requests without auth_context work correctly."""
        request = ChatRequest(
            message="Simple test",
            model="openai_fast",
            auth_context=None,  # No auth context
            save_to_db=False,
        )

        # Patch to capture the request
        original_chat = chat_engine.llmring.chat
        captured_request = None

        async def capture_chat(request):
            nonlocal captured_request
            captured_request = request
            return await original_chat(request)

        with patch.object(chat_engine.llmring, "chat", side_effect=capture_chat):
            response = await chat_engine.process_request(request)

        # Should work without setting metadata
        assert captured_request is not None
        # Metadata might be None or empty - both should work
        if captured_request.metadata:
            assert (
                "id_at_origin" not in captured_request.metadata
                or captured_request.metadata["id_at_origin"] is None
            )

        print("✓ Requests without auth_context work correctly")

    @pytest.mark.asyncio
    async def test_conversation_manager_integration(self, chat_engine):
        """Test that the chat engine properly integrates with conversation manager."""
        # Mock conversation manager to avoid DB dependency
        mock_conversation_manager = AsyncMock()
        chat_engine.conversation_manager = mock_conversation_manager

        # Mock conversation creation
        mock_conversation_manager.create_conversation.return_value = "test-conv-id"
        mock_conversation_manager.add_message.return_value = None

        request = ChatRequest(
            message="Integration test",
            model="openai_fast",
            auth_context={"user_id": "integration-user"},
            save_to_db=True,  # This should trigger conversation manager calls
        )

        # Mock LLMRing.chat to avoid going to actual API for this integration test
        with patch.object(chat_engine.llmring, "chat") as mock_chat:
            # Mock LLMResponse
            mock_response = AsyncMock()
            mock_response.content = "Integration response"
            mock_response.model = "fast"
            mock_response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
            mock_response.finish_reason = "stop"
            mock_response.tool_calls = None
            mock_chat.return_value = mock_response

            response = await chat_engine.process_request(request)

        # Verify chat was called with proper metadata
        mock_chat.assert_called_once()
        called_request = mock_chat.call_args[0][0]
        assert called_request.metadata["id_at_origin"] == "integration-user"

        # Verify conversation manager was used
        mock_conversation_manager.add_message.assert_called()

        print("✓ Conversation manager integration works with metadata")
