"""
Real integration tests for stateless_engine without mocks.

These tests exercise the actual code paths to ensure all methods work correctly.
"""

import pytest
import os
from datetime import datetime, UTC
from uuid import UUID

from llmring.mcp.client.stateless_engine import (
    StatelessChatEngine,
    ChatRequest,
    ProcessingContext
)
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.http_client import MCPHttpClient
from llmring.schemas import Message, LLMResponse
from llmring.service import LLMRing


class TestStatelessEngineReal:
    """Test stateless engine with real components, no mocks."""

    @pytest.fixture
    def llmring_service(self):
        """Create a real LLMRing service."""
        # Use test lockfile if it exists, or create one
        test_lockfile = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'llmring.lock.json'
        )

        # If lockfile doesn't exist, use None to trigger default creation
        if not os.path.exists(test_lockfile):
            test_lockfile = None

        return LLMRing(origin="test", lockfile_path=test_lockfile)

    @pytest.fixture
    def http_client(self):
        """Create a real HTTP client pointing to test server."""
        # Use a test server URL or None to test without server
        return MCPHttpClient(
            base_url=os.getenv("TEST_LLMRING_SERVER_URL", "http://localhost:8001"),
            api_key=os.getenv("TEST_LLMRING_API_KEY", "test-key")
        )

    @pytest.fixture
    async def conversation_manager(self, http_client):
        """Create a real conversation manager."""
        return AsyncConversationManager(
            llmring_server_url=http_client.base_url,
            api_key=http_client.api_key
        )

    @pytest.fixture
    def chat_engine(self, llmring_service):
        """Create a real chat engine."""
        return StatelessChatEngine(
            llmring=llmring_service,
            default_model="mcp_agent"  # Use the alias we defined
        )

    @pytest.mark.asyncio
    async def test_prepare_messages_returns_correct_type(self, chat_engine):
        """Test that _prepare_messages returns list[Message] not list[dict]."""
        context = ProcessingContext(
            conversation_id="test-123",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!")
            ],
            system_prompt="You are a helpful assistant",
            model="mcp_agent",
            temperature=0.7,
            max_tokens=100,
            tools=None,
            auth_context={},
            mcp_client=None
        )

        messages = chat_engine._prepare_messages(context)

        # Verify it returns Message objects
        assert isinstance(messages, list)
        assert len(messages) == 3  # system + 2 messages
        assert all(isinstance(msg, Message) for msg in messages)
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    @pytest.mark.asyncio
    async def test_process_with_llm_type_assertion(self, chat_engine, llmring_service):
        """Test that _process_with_llm correctly handles LLMResponse type."""
        # Replace chat engine's llmring with our fixture
        chat_engine.llmring = llmring_service

        context = ProcessingContext(
            conversation_id="test-456",
            messages=[Message(role="user", content="What is 2+2?")],
            system_prompt=None,
            model="mcp_agent",
            temperature=0.5,
            max_tokens=50,
            tools=None,
            auth_context={"user_id": "test-user"},
            mcp_client=None
        )

        # This should call the real LLM and handle the response correctly
        try:
            response_message, tool_calls, tool_results, llm_response = \
                await chat_engine._process_with_llm(context)

            # Verify response types
            assert isinstance(response_message, Message)
            assert response_message.role == "assistant"
            assert isinstance(response_message.content, str)

            # Verify the LLMResponse was handled correctly
            assert llm_response is not None
            assert isinstance(llm_response, LLMResponse)

        except Exception as e:
            # If no API keys are configured, skip this test
            if "No provider found" in str(e) or "API key" in str(e):
                pytest.skip(f"Skipping due to missing API configuration: {e}")
            raise

    @pytest.mark.asyncio
    async def test_conversation_manager_add_message_signature(self, conversation_manager):
        """Test that add_message is called with correct parameters."""
        # Test the actual signature without mocks
        import uuid
        conversation_id = str(uuid.uuid4())  # Use proper UUID

        # This tests the actual method signature
        try:
            message_id = await conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content="Test message content",
                token_count=10,
                metadata={"test": "data"}
            )
            # If server is not running, this will fail with connection error
            # which is fine - we're testing the signature compiles
        except Exception as e:
            # Connection errors are expected if no server running
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                pass  # Signature is correct, just no server
            else:
                # Re-raise unexpected errors
                raise

    @pytest.mark.asyncio
    async def test_create_context_without_conversation(self, chat_engine):
        """Test _create_context when creating a new conversation."""
        request = ChatRequest(
            conversation_id=None,  # New conversation
            message="Hello world",
            model="mcp_agent",
            system_prompt="Be helpful",
            temperature=0.8,
            max_tokens=200,
            save_to_db=False  # Don't actually save
        )

        context = await chat_engine._create_context(request)

        # Verify context was created correctly
        assert context.conversation_id is not None
        assert isinstance(context.conversation_id, str)
        assert context.messages == []  # No messages yet
        assert context.system_prompt == "Be helpful"
        assert context.model == "mcp_agent"
        assert context.temperature == 0.8
        assert context.max_tokens == 200
        assert context.auth_context == {}

    @pytest.mark.asyncio
    async def test_create_context_with_existing_conversation(self, chat_engine, conversation_manager):
        """Test _create_context with an existing conversation ID."""
        # Replace the conversation manager with our real one
        chat_engine.conversation_manager = conversation_manager

        import uuid
        request = ChatRequest(
            conversation_id=str(uuid.uuid4()),  # Use proper UUID
            message="Continue conversation",
            save_to_db=False
        )

        # This should try to load the conversation
        try:
            context = await chat_engine._create_context(request)
            # If it doesn't exist, it should raise ValueError
        except ValueError as e:
            assert "not found" in str(e)
            # This is expected behavior
        except Exception as e:
            # Connection errors are OK if no server
            if "connect" not in str(e).lower():
                raise

    @pytest.mark.asyncio
    async def test_calculate_usage(self, chat_engine):
        """Test usage calculation."""
        messages = [
            Message(role="user", content="Hello, how are you today?"),
            Message(role="assistant", content="I'm doing well, thank you!")
        ]
        response = Message(
            role="assistant",
            content="This is a test response with several words."
        )

        usage = chat_engine._calculate_usage(messages, response)

        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_full_process_request_flow(self, chat_engine):
        """Test the full process_request flow without save_to_db."""
        request = ChatRequest(
            message="What is the capital of France?",
            model="mcp_agent",
            temperature=0.3,
            max_tokens=100,
            save_to_db=False,  # Don't save to avoid DB dependency
            auth_context={"user_id": "test-flow-user"}
        )

        try:
            response = await chat_engine.process_request(request)

            # Verify response structure
            assert response.conversation_id is not None
            assert isinstance(response.message, Message)
            assert response.message.role == "assistant"
            assert isinstance(response.message.content, str)
            assert len(response.message.content) > 0
            assert response.usage is not None
            assert response.model is not None
            assert isinstance(response.created_at, datetime)
            assert response.processing_time_ms > 0

        except Exception as e:
            # Skip if no API keys configured
            if "No provider found" in str(e) or "API key" in str(e):
                pytest.skip(f"Skipping due to missing API configuration: {e}")
            raise

    @pytest.mark.asyncio
    async def test_streaming_flow(self, chat_engine):
        """Test the streaming flow."""
        request = ChatRequest(
            message="Count from 1 to 5",
            model="mcp_agent",
            save_to_db=False,
            auth_context={"user_id": "stream-test"}
        )

        try:
            chunks = []
            async for chunk in chat_engine.process_request_stream(request):
                chunks.append(chunk)

            # Verify we got chunks
            assert len(chunks) > 0
            # Verify chunk structure (StreamChatChunk objects)
            for chunk in chunks[:-1]:  # All but last
                assert hasattr(chunk, "delta") or hasattr(chunk, "type")
                # Check it's a StreamChatChunk with expected attributes
                if hasattr(chunk, "delta"):
                    assert hasattr(chunk, "conversation_id")
                    assert hasattr(chunk, "finished")

        except Exception as e:
            # Skip if no API keys configured
            if "No provider found" in str(e) or "API key" in str(e):
                pytest.skip(f"Skipping due to missing API configuration: {e}")
            raise

    @pytest.mark.asyncio
    async def test_metadata_handling(self, chat_engine):
        """Test that metadata is properly set in LLM requests."""
        # Capture what gets sent to the LLM
        requests_sent = []
        original_chat = chat_engine.llmring.chat

        async def capture_chat(request):
            requests_sent.append(request)
            return await original_chat(request)

        # Temporarily replace chat method
        chat_engine.llmring.chat = capture_chat

        try:
            request = ChatRequest(
                message="Test metadata",
                model="mcp_agent",
                auth_context={"user_id": "metadata-test-user"},
                save_to_db=False
            )

            response = await chat_engine.process_request(request)

            # Check that metadata was set
            assert len(requests_sent) == 1
            sent_request = requests_sent[0]
            assert hasattr(sent_request, "metadata")
            assert sent_request.metadata is not None
            assert "id_at_origin" in sent_request.metadata
            assert sent_request.metadata["id_at_origin"] == "metadata-test-user"

        except Exception as e:
            if "No provider found" in str(e) or "API key" in str(e):
                pytest.skip(f"Skipping due to missing API configuration: {e}")
            raise
        finally:
            # Restore original method
            chat_engine.llmring.chat = original_chat