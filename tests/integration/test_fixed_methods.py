"""
Tests for the specific methods that were fixed in stateless_engine.py

These tests verify that:
1. Method names are correct (call_tool vs execute_tool, get_conversation vs load_conversation)
2. Method signatures match (add_message parameters)
3. Return types are correct
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from llmring.mcp.client.mcp_client import MCPClient
from llmring.mcp.client.models.schemas import ToolCall
from llmring.mcp.client.stateless_engine import ChatRequest, StatelessChatEngine
from llmring.schemas import LLMResponse, Message


class TestFixedMethods:
    """Test the methods that were fixed to ensure they use correct names and signatures."""

    @pytest.fixture
    def chat_engine(self):
        """Create a chat engine for testing."""
        from llmring.service import LLMRing

        llm_service = LLMRing(origin="test")
        return StatelessChatEngine(llmring=llm_service)

    @pytest.mark.asyncio
    async def test_mcp_client_call_tool_not_execute_tool(self):
        """Verify MCPClient has call_tool method, not execute_tool."""
        # Create a real MCPClient instance
        client = MCPClient(transport=MagicMock())

        # Verify it has call_tool method
        assert hasattr(client, "call_tool")
        assert callable(getattr(client, "call_tool"))

        # Verify it does NOT have execute_tool
        assert not hasattr(client, "execute_tool")

    @pytest.mark.asyncio
    async def test_conversation_manager_get_conversation_not_load(self, chat_engine):
        """Verify conversation manager has get_conversation, not load_conversation."""
        conv_manager = chat_engine.conversation_manager

        # Verify it has get_conversation
        assert hasattr(conv_manager, "get_conversation")
        assert callable(getattr(conv_manager, "get_conversation"))

        # Verify it does NOT have load_conversation
        assert not hasattr(conv_manager, "load_conversation")

    @pytest.mark.asyncio
    async def test_conversation_manager_add_message_signature(self, chat_engine):
        """Verify add_message takes individual parameters, not Message object."""
        conv_manager = chat_engine.conversation_manager
        import uuid

        # This should compile without errors
        # The actual call might fail if no server, but signature should be correct
        try:
            await conv_manager.add_message(
                conversation_id=str(uuid.uuid4()),
                role="user",
                content="Test content",
                token_count=10,
                metadata={"test": "data"},
            )
        except Exception as e:
            # Connection errors, 404, and method not allowed (405) are OK, we're testing the signature
            error_str = str(e).lower()
            if (
                "connect" not in error_str
                and "refused" not in error_str
                and "404" not in error_str
                and "not found" not in error_str
                and "405" not in error_str
                and "method not allowed" not in error_str
                and "401" not in error_str
                and "unauthorized" not in error_str
            ):
                # Re-raise if it's not a connection/404/405 error
                raise

    @pytest.mark.asyncio
    async def test_conversation_manager_no_add_tool_result(self, chat_engine):
        """Verify conversation manager does NOT have add_tool_result method."""
        conv_manager = chat_engine.conversation_manager

        # Verify it does NOT have add_tool_result
        assert not hasattr(conv_manager, "add_tool_result")

    @pytest.mark.asyncio
    async def test_execute_tool_uses_call_tool(self, chat_engine):
        """Test that execute_tool method properly calls call_tool on MCPClient."""
        import uuid
        from unittest.mock import AsyncMock

        # Use a proper UUID for conversation_id
        conv_id = str(uuid.uuid4())

        # Create a mock MCP client
        mock_mcp_client = MagicMock(spec=MCPClient)
        mock_mcp_client.call_tool = AsyncMock(return_value={"result": "test"})

        # Create a mock conversation with the MCP client
        mock_conversation = MagicMock()
        mock_conversation.id = conv_id
        mock_conversation.mcp_servers = {"test": mock_mcp_client}

        # Mock get_conversation to return our mock
        chat_engine.conversation_manager.get_conversation = AsyncMock(
            return_value=mock_conversation
        )

        # Mock add_message to avoid UUID conversion issues
        chat_engine.conversation_manager.add_message = AsyncMock(return_value="msg-id")

        # Mock _get_mcp_client to return our mock client
        chat_engine._get_mcp_client = AsyncMock(return_value=mock_mcp_client)

        # Create a tool call
        tool_call = ToolCall(id="call_123", tool_name="test_tool", arguments={"param": "value"})

        # Execute the tool
        await chat_engine.execute_tool(
            conversation_id=conv_id,
            tool_call=tool_call,
            auth_context={"user_id": "test"},
        )

        # Verify call_tool was called (not execute_tool)
        mock_mcp_client.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_prepare_messages_returns_message_objects(self, chat_engine):
        """Test that _prepare_messages returns Message objects, not dicts."""
        from llmring.mcp.client.stateless_engine import ProcessingContext

        context = ProcessingContext(
            conversation_id="test",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            system_prompt="Be helpful",
            model="mcp_agent",
            temperature=0.7,
            max_tokens=None,
            tools=None,
            auth_context={},
            mcp_client=None,
        )

        messages = chat_engine._prepare_messages(context)

        # All items should be Message objects
        assert all(isinstance(msg, Message) for msg in messages)
        assert len(messages) == 3  # system + 2 messages

        # Check the content
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    @pytest.mark.asyncio
    async def test_create_conversation_returns_id(self, chat_engine):
        """Test that create_conversation returns the conversation ID."""
        from unittest.mock import AsyncMock

        # Mock the conversation manager's create_conversation
        expected_id = str(uuid4())
        chat_engine.conversation_manager.create_conversation = AsyncMock(return_value=expected_id)

        request = ChatRequest(message="Test", save_to_db=True)

        # This should use the returned ID
        context = await chat_engine._create_context(request)

        # Verify the conversation_id matches what was returned
        assert context.conversation_id == expected_id

    @pytest.mark.asyncio
    async def test_tool_config_wrapped_in_dict(self, chat_engine):
        """Test that tool_config is wrapped in a dict when creating conversation."""
        # Mock create_conversation to capture what's passed
        captured_args = {}

        async def capture_create(**kwargs):
            captured_args.update(kwargs)
            return "test-id"

        chat_engine.conversation_manager.create_conversation = capture_create

        request = ChatRequest(
            message="Test",
            tools=[{"name": "test_tool", "description": "A test tool"}],
            save_to_db=True,
            auth_context={"user_id": "test-user-wrap"},
        )

        await chat_engine._create_context(request)

        # Verify tool_config was wrapped in a dict
        assert "tool_config" in captured_args
        assert captured_args["tool_config"] == {"tools": request.tools}
        # Verify user_id was passed
        assert "user_id" in captured_args
        assert captured_args["user_id"] == "test-user-wrap"

    @pytest.mark.asyncio
    async def test_llm_response_type_assertion(self, chat_engine):
        """Test that LLMResponse type assertion is in place."""
        from unittest.mock import AsyncMock

        from llmring.mcp.client.stateless_engine import ProcessingContext

        # Mock the llmring.chat to return a proper LLMResponse
        mock_response = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )

        chat_engine.llmring.chat = AsyncMock(return_value=mock_response)

        context = ProcessingContext(
            conversation_id="test",
            messages=[Message(role="user", content="Test")],
            system_prompt=None,
            model="mcp_agent",
            temperature=0.7,
            max_tokens=None,
            tools=None,
            auth_context={},
            mcp_client=None,
        )

        # This should not raise an assertion error
        response_message, tool_calls, tool_results, llm_response = (
            await chat_engine._process_with_llm(context)
        )

        # Verify the response was processed correctly
        assert isinstance(response_message, Message)
        assert response_message.content == "Test response"
        assert llm_response == mock_response
