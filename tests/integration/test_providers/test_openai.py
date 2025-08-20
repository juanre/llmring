"""
Integration tests for the OpenAI provider using real API calls.
These tests require a valid OPENAI_API_KEY environment variable.
"""

import asyncio
import base64
import json
import os

import pytest

from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMResponse, Message


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
class TestOpenAIProviderIntegration:
    """Integration tests for OpenAIProvider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create OpenAIProvider instance with real API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")

        return OpenAIProvider(api_key=api_key)

    @pytest.mark.asyncio
    async def test_simple_chat(self, provider):
        """Test simple chat with GPT."""
        messages = [Message(role="user", content="Say exactly 'Hello from GPT!'")]

        response = await provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",  # Use cheaper model for tests
            max_tokens=50,
        )

        assert isinstance(response, LLMResponse)
        assert "Hello from GPT" in response.content
        assert response.model == "gpt-3.5-turbo"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.finish_reason in ["stop", "length"]

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, provider):
        """Test chat with system message."""
        messages = [
            Message(
                role="system",
                content="You are a helpful math tutor. Always be encouraging.",
            ),
            Message(role="user", content="What is 2 + 2?"),
        ]

        response = await provider.chat(
            messages=messages, model="gpt-3.5-turbo", max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
        assert len(response.content) > 10  # Should be encouraging, not just "4"

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, provider):
        """Test chat with different temperature settings."""
        messages = [
            Message(role="user", content="Write a creative opening line for a story.")
        ]

        # Test with low temperature (more deterministic)
        response_low = await provider.chat(
            messages=messages, model="gpt-3.5-turbo", temperature=0.1, max_tokens=50
        )

        # Test with high temperature (more creative)
        response_high = await provider.chat(
            messages=messages, model="gpt-3.5-turbo", temperature=0.9, max_tokens=50
        )

        assert isinstance(response_low, LLMResponse)
        assert isinstance(response_high, LLMResponse)
        assert len(response_low.content) > 0
        assert len(response_high.content) > 0

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self, provider):
        """Test chat with max_tokens limit."""
        messages = [
            Message(
                role="user", content="Write a long essay about artificial intelligence."
            )
        ]

        response = await provider.chat(
            messages=messages, model="gpt-3.5-turbo", max_tokens=20  # Very small limit
        )

        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "length"
        assert response.usage["completion_tokens"] <= 20

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, provider):
        """Test multi-turn conversation."""
        messages = [
            Message(role="user", content="My name is Alice. What's your name?"),
            Message(role="assistant", content="Hello Alice! I'm an AI assistant."),
            Message(role="user", content="What was my name again?"),
        ]

        response = await provider.chat(
            messages=messages, model="gpt-3.5-turbo", max_tokens=50
        )

        assert isinstance(response, LLMResponse)
        assert "Alice" in response.content

    @pytest.mark.asyncio
    async def test_chat_with_json_mode(self, provider):
        """Test chat with JSON response format."""
        messages = [
            Message(
                role="user",
                content="List three colors as a JSON array with a 'colors' key.",
            )
        ]

        response = await provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"},
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)

        # Response should be valid JSON
        try:
            parsed = json.loads(response.content)
            assert isinstance(parsed, dict)
            assert "colors" in parsed
            assert isinstance(parsed["colors"], list)
            assert len(parsed["colors"]) == 3
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {response.content}")

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, provider):
        """Test chat with function calling."""
        messages = [
            Message(role="user", content="What's the weather like in San Francisco?")
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )

        assert isinstance(response, LLMResponse)

        # GPT should try to use the weather function
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            # Parse the arguments to ensure they're valid JSON
            args = json.loads(response.tool_calls[0]["function"]["arguments"])
            assert "location" in args
            assert "san francisco" in args["location"].lower()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, provider):
        """Test error handling with invalid model."""
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ValueError, match="Unsupported model"):
            await provider.chat(messages=messages, model="invalid-model-name")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider):
        """Test multiple concurrent requests."""

        async def make_request(i):
            messages = [Message(role="user", content=f"Count to {i}")]
            return await provider.chat(
                messages=messages, model="gpt-3.5-turbo", max_tokens=50
            )

        # Make 3 concurrent requests
        agents = [make_request(i) for i in range(1, 4)]
        responses = await asyncio.gather(*agents)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_model_validation(self, provider):
        """Test model validation methods."""
        # Test valid models
        assert provider.validate_model("gpt-4o") is True
        assert provider.validate_model("gpt-3.5-turbo") is True
        assert provider.validate_model("openai:gpt-4o") is True

        # Test invalid models
        assert provider.validate_model("claude-3-opus") is False
        assert provider.validate_model("invalid-model") is False

    def test_supported_models_list(self, provider):
        """Test that supported models list is comprehensive."""
        models = provider.get_supported_models()

        # Should include GPT-4 models
        assert "gpt-4o" in models
        assert "gpt-4-turbo" in models
        assert "gpt-4" in models

        # Should include GPT-3.5 models
        assert "gpt-3.5-turbo" in models

    def test_token_counting(self, provider):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        count = provider.get_token_count(text)

        # Should return a reasonable token count
        assert isinstance(count, int)
        assert count > 0
        assert count < 100  # Should be reasonable for this short text

    @pytest.mark.asyncio
    async def test_chat_with_image_content(self, provider):
        """Test chat with image content (vision)."""
        # Skip if not GPT-4o (which supports vision)
        if not provider.validate_model("gpt-4o"):
            pytest.skip("GPT-4o not available for vision testing")

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            )
        ]

        try:
            response = await provider.chat(
                messages=messages, model="gpt-4o", max_tokens=100
            )

            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            # Should mention something about the image (it's a tiny red pixel)

        except Exception as e:
            # Vision might not be available or might have restrictions
            if "billing" in str(e).lower() or "usage" in str(e).lower():
                pytest.skip(f"Vision API not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_tool_choice_none(self, provider):
        """Test tool choice set to 'none'."""
        messages = [Message(role="user", content="What's the weather like?")]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        response = await provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=tools,
            tool_choice="none",
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        # Should not have any tool calls since tool_choice is "none"
        assert response.tool_calls is None or len(response.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_pdf_processing_with_assistants_api(self, provider):
        """Test PDF processing using OpenAI Assistants API."""
        # Create a simple PDF content for testing
        # This is a minimal PDF with "Hello World" text
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000100 00000 n
0000000249 00000 n
0000000343 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
426
%%EOF"""

        # Create document content block
        pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")

        messages = [
            Message(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Extract the text from this PDF document.",
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                ],
            )
        ]

        try:
            response = await provider.chat(
                messages=messages,
                model="gpt-4o",  # Use a model that supports Assistants API
                max_tokens=200,
            )

            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            assert response.model == "gpt-4o"

            # The response should contain information about the PDF content
            # Since our test PDF contains "Hello World", check for that or related terms
            content_lower = response.content.lower()
            assert any(
                term in content_lower for term in ["hello", "world", "text", "document"]
            ), f"Response doesn't seem to reference PDF content: {response.content}"

            # Check that usage information is provided
            assert response.usage is not None
            assert response.usage.get("total_tokens", 0) > 0

        except Exception as e:
            # Handle potential API limitations
            error_str = str(e).lower()
            if any(
                term in error_str
                for term in ["billing", "quota", "usage limit", "rate limit"]
            ):
                pytest.skip(f"OpenAI API limit reached: {e}")
            elif "assistants" in error_str:
                pytest.skip(f"Assistants API not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_pdf_processing_with_tools_should_fail(self, provider):
        """Test that PDF processing with tools raises appropriate error."""
        pdf_content = b"dummy pdf content"
        pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Process this PDF."},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
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

        # Should raise error about tools not being supported with PDF processing
        with pytest.raises(
            ValueError,
            match="Tools and custom response formats are not supported when processing PDFs",
        ):
            await provider.chat(messages=messages, model="gpt-4o", tools=tools)
