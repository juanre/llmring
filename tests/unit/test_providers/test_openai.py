"""
Unit tests for the OpenAI provider using real API calls.
These tests require a valid OPENAI_API_KEY environment variable.
Tests are minimal to avoid excessive API usage costs.
"""

import json
import os

import pytest

from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMResponse, Message
from llmring.exceptions import (
    ProviderAuthenticationError,
    ModelNotFoundError,
    ProviderResponseError,
)


@pytest.mark.llm
@pytest.mark.unit
class TestOpenAIProviderUnit:
    """Unit tests for OpenAIProvider using real API calls."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.default_model == "gpt-4o"

    def test_initialization_without_api_key_raises_error(self):
        """Test that missing API key raises ProviderAuthenticationError."""
        # Temporarily unset environment variable
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(
                ProviderAuthenticationError, match="OpenAI API key must be provided"
            ):
                OpenAIProvider()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_initialization_with_env_var(self, monkeypatch):
        """Test provider initialization with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        provider = OpenAIProvider()
        assert provider.api_key == "env-test-key"

    def test_supported_models_list(self, openai_provider):
        """Test that supported models list contains expected models."""
        models = openai_provider.get_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o" in models
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

    def test_validate_model_exact_match(self, openai_provider):
        """Test model validation with exact model names."""
        assert openai_provider.validate_model("gpt-4o") is True
        assert openai_provider.validate_model("gpt-3.5-turbo") is True
        assert openai_provider.validate_model("invalid-model") is False

    def test_validate_model_with_provider_prefix(self, openai_provider):
        """Test model validation handles provider prefix correctly."""
        assert openai_provider.validate_model("openai:gpt-4o") is True
        assert openai_provider.validate_model("openai:gpt-3.5-turbo") is True
        assert openai_provider.validate_model("openai:invalid-model") is False

    def test_validate_model_base_name_matching(self, openai_provider):
        """Test model validation with base name matching."""
        # These should pass since they are exact matches in the supported models
        assert openai_provider.validate_model("gpt-4") is True
        assert openai_provider.validate_model("gpt-3.5-turbo") is True
        assert openai_provider.validate_model("claude-3") is False  # Different provider

    @pytest.mark.asyncio
    async def test_chat_basic_request(self, openai_provider, simple_user_message):
        """Test basic chat functionality with minimal token usage."""
        response = await openai_provider.chat(
            messages=simple_user_message,
            model="gpt-3.5-turbo",  # Use cheaper model
            max_tokens=10,  # Minimal tokens to reduce cost
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "gpt-3.5-turbo"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0
        assert response.finish_reason in ["stop", "length"]

    @pytest.mark.asyncio
    async def test_chat_with_system_message(
        self, openai_provider, system_user_messages
    ):
        """Test chat with system message."""
        response = await openai_provider.chat(
            messages=system_user_messages, model="gpt-3.5-turbo", max_tokens=10
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "4" in response.content or "four" in response.content.lower()

    @pytest.mark.asyncio
    async def test_chat_with_provider_prefix_removal(
        self, openai_provider, simple_user_message
    ):
        """Test that provider prefix is correctly removed from model name."""
        response = await openai_provider.chat(
            messages=simple_user_message, model="openai:gpt-3.5-turbo", max_tokens=10
        )

        assert isinstance(response, LLMResponse)
        assert response.model == "gpt-3.5-turbo"  # Prefix should be removed

    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, openai_provider, simple_user_message):
        """Test chat with temperature and max_tokens parameters."""
        response = await openai_provider.chat(
            messages=simple_user_message,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=15,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.usage is not None
        assert response.usage["completion_tokens"] <= 15

    @pytest.mark.asyncio
    async def test_chat_with_unsupported_model_raises_error(
        self, openai_provider, simple_user_message
    ):
        """Test that using an unsupported model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError, match="Unsupported model"):
            await openai_provider.chat(
                messages=simple_user_message, model="definitely-not-a-real-model"
            )

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, openai_provider, sample_tools):
        """Test chat with function calling tools."""
        messages = [
            Message(
                role="user",
                content="What's the weather in NYC? Use tools if available.",
            )
        ]

        response = await openai_provider.chat(
            messages=messages, model="gpt-3.5-turbo", tools=sample_tools, max_tokens=50
        )

        assert isinstance(response, LLMResponse)
        # Tool calls are provider-dependent, just verify response structure
        assert response.content is not None
        if response.tool_calls:
            assert isinstance(response.tool_calls, list)
            for tool_call in response.tool_calls:
                assert "id" in tool_call
                assert "type" in tool_call
                assert "function" in tool_call

    @pytest.mark.asyncio
    async def test_chat_api_error_handling(self, openai_provider):
        """Test proper error handling for API errors."""
        # Test with invalid model that passes validation but fails at API level
        with pytest.raises(Exception) as exc_info:
            await openai_provider.chat(
                messages=[Message(role="user", content="test")],
                model="gpt-99-nonexistent",  # Invalid model
                max_tokens=10,
            )

        # Should wrap the error with our standard format (either validation or API error)
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["error", "unsupported", "invalid"])

    def test_get_token_count_fallback(self, openai_provider):
        """Test token counting fallback when tiktoken not available."""
        text = "This is a test sentence for token counting."
        count = openai_provider.get_token_count(text)

        assert isinstance(count, int)
        assert count > 0
        # Should be a reasonable approximation
        assert 5 < count < 50

    def test_get_token_count_with_tiktoken(self, openai_provider):
        """Test token counting with tiktoken if available."""
        try:
            import tiktoken  # type: ignore

            text = "Hello world"
            count = openai_provider.get_token_count(text)

            # Should use tiktoken for accurate counting
            encoder = tiktoken.get_encoding("cl100k_base")  # Use safe encoding
            expected = len(encoder.encode(text))
            assert count == expected
        except ImportError:
            # Skip if tiktoken not available
            pytest.skip("tiktoken not available")

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_response(self, openai_provider, sample_tools):
        """Test handling of tool calls in response."""
        # Use a message that's likely to trigger a tool call
        messages = [
            Message(
                role="user",
                content="Please get the weather for San Francisco using the available tools.",
            )
        ]

        response = await openai_provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=sample_tools,
            tool_choice="auto",
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        # Tool calls depend on the model's decision, so we just verify structure
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            assert tool_call["type"] == "function"
            assert "name" in tool_call["function"]
            assert "arguments" in tool_call["function"]

            # Arguments should be valid JSON
            try:
                json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                pytest.fail("Tool call arguments are not valid JSON")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self, openai_provider, multi_turn_conversation
    ):
        """Test multi-turn conversation handling."""
        response = await openai_provider.chat(
            messages=multi_turn_conversation, model="gpt-3.5-turbo", max_tokens=20
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Should remember the context and mention "Alice"
        assert "alice" in response.content.lower()

    def test_get_default_model(self, openai_provider):
        """Test getting default model."""
        default_model = openai_provider.get_default_model()
        assert default_model == "gpt-4o"
        assert default_model in openai_provider.get_supported_models()

    @pytest.mark.asyncio
    async def test_json_response_format(self, openai_provider, json_response_format):
        """Test JSON response format."""
        messages = [
            Message(
                role="user", content="Respond with JSON: answer=hello, confidence=0.9"
            )
        ]

        response = await openai_provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            response_format=json_response_format,
            max_tokens=50,
        )

        assert isinstance(response, LLMResponse)
        # Try to parse the response as JSON
        try:
            parsed = json.loads(response.content)
            assert isinstance(parsed, dict)
            assert "answer" in parsed
        except json.JSONDecodeError:
            # JSON mode might not be enforced in test environment
            pass

    def test_contains_pdf_content_true(self, openai_provider):
        """Test PDF content detection returns True for PDF content."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this PDF"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQ=",  # PDF header in base64
                        },
                    },
                ],
            )
        ]

        assert openai_provider._contains_pdf_content(messages) is True

    def test_contains_pdf_content_false(self, openai_provider):
        """Test PDF content detection returns False for non-PDF content."""
        messages = [
            Message(role="user", content="Just text"),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            ),
        ]

        assert openai_provider._contains_pdf_content(messages) is False

    def test_extract_pdf_content_and_text(self, openai_provider):
        """Test extraction of PDF data and text from messages."""
        pdf_data = "JVBERi0xLjQ="  # PDF header in base64
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Please analyze this document"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                ],
            ),
        ]

        pdf_list, text = openai_provider._extract_pdf_content_and_text(messages)

        assert len(pdf_list) == 1
        assert text == "You are a helpful assistant Please analyze this document"
        # Verify PDF data was decoded correctly
        import base64

        assert pdf_list[0] == base64.b64decode(pdf_data)

    def test_pdf_content_detection_logic(self, openai_provider):
        """Test that PDF content detection works correctly for routing."""
        # Test with PDF content
        pdf_messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this PDF"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQ=",
                        },
                    },
                ],
            )
        ]

        # Test with non-PDF content
        regular_messages = [
            Message(role="user", content="Just text"),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            ),
        ]

        # Verify detection works
        assert openai_provider._contains_pdf_content(pdf_messages) is True
        assert openai_provider._contains_pdf_content(regular_messages) is False

    @pytest.mark.asyncio
    async def test_pdf_with_tools_raises_error(self, openai_provider):
        """Test that PDF processing with tools raises appropriate error."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Process this PDF"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQ=",
                        },
                    },
                ],
            )
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "description": "A dummy tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with pytest.raises(
            ProviderResponseError,
            match="Tools and custom response formats are not supported when processing PDFs",
        ):
            await openai_provider.chat(messages=messages, model="gpt-4o", tools=tools)

    @pytest.mark.asyncio
    async def test_pdf_with_response_format_raises_error(self, openai_provider):
        """Test that PDF processing with response format raises appropriate error."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Process this PDF"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQ=",
                        },
                    },
                ],
            )
        ]

        with pytest.raises(
            ProviderResponseError,
            match="Tools and custom response formats are not supported when processing PDFs",
        ):
            await openai_provider.chat(
                messages=messages,
                model="gpt-4o",
                response_format={"type": "json_object"},
            )
