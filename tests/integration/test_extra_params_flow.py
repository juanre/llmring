"""
Test extra_params flow with real APIs.

Tests that extra_params properly flow from LLMRequest through service to providers
and reach the actual API calls with correct provider-specific formats.
"""

import pytest
from unittest.mock import patch, MagicMock

from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message


class TestExtraParamsFlow:
    """Test that extra_params flow correctly from request to provider APIs."""

    @pytest.fixture
    def service(self):
        """Create LLMRing service."""
        return LLMRing()

    @pytest.mark.asyncio
    async def test_openai_extra_params_flow(self, service):
        """Test that OpenAI extra_params reach the API call."""
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Test")],
            max_tokens=10,
            temperature=0.1,
            extra_params={
                "logprobs": True,
                "top_logprobs": 3,
                "presence_penalty": 0.1
            }
        )

        # Patch the OpenAI client to capture request params
        with patch('llmring.providers.openai_api.AsyncOpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15

            mock_openai.return_value.chat.completions.create.return_value = mock_response

            response = await service.chat(request)

            # Verify the API was called with extra_params
            mock_openai.return_value.chat.completions.create.assert_called_once()
            call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]

            # Check that extra_params made it to the API call
            assert "logprobs" in call_kwargs, "logprobs should be passed to OpenAI API"
            assert call_kwargs["logprobs"] == True, "logprobs value should be preserved"
            assert "top_logprobs" in call_kwargs, "top_logprobs should be passed to OpenAI API"
            assert call_kwargs["top_logprobs"] == 3, "top_logprobs value should be preserved"
            assert "presence_penalty" in call_kwargs, "presence_penalty should be passed to OpenAI API"
            assert call_kwargs["presence_penalty"] == 0.1, "presence_penalty value should be preserved"

            print(f"✓ OpenAI extra_params flow works: {[k for k in call_kwargs.keys() if k in request.extra_params]}")

    @pytest.mark.asyncio
    async def test_anthropic_extra_params_flow(self, service):
        """Test that Anthropic extra_params reach the API call."""
        request = LLMRequest(
            model="anthropic:claude-3-5-haiku",
            messages=[Message(role="user", content="Test")],
            max_tokens=10,
            temperature=0.1,
            extra_params={
                "top_p": 0.9,
                "top_k": 40
            }
        )

        # Patch the Anthropic client to capture request params
        with patch('llmring.providers.anthropic_api.AsyncAnthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = "Test response"
            mock_response.model = "claude-3-5-haiku"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            mock_response.usage.cache_creation_input_tokens = 0
            mock_response.usage.cache_read_input_tokens = 0

            mock_anthropic.return_value.messages.create.return_value = mock_response

            response = await service.chat(request)

            # Verify the API was called with extra_params
            mock_anthropic.return_value.messages.create.assert_called_once()
            call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]

            # Check that extra_params made it to the API call
            assert "top_p" in call_kwargs, "top_p should be passed to Anthropic API"
            assert call_kwargs["top_p"] == 0.9, "top_p value should be preserved"
            assert "top_k" in call_kwargs, "top_k should be passed to Anthropic API"
            assert call_kwargs["top_k"] == 40, "top_k value should be preserved"

            print(f"✓ Anthropic extra_params flow works: {[k for k in call_kwargs.keys() if k in request.extra_params]}")

    @pytest.mark.asyncio
    async def test_extra_params_streaming_flow(self, service):
        """Test that extra_params flow correctly in streaming mode."""
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Count to 3")],
            max_tokens=20,
            temperature=0.1,
            stream=True,
            extra_params={
                "logprobs": True,
                "seed": 12345
            }
        )

        # Patch the OpenAI client to capture streaming request params
        with patch('llmring.providers.openai_api.AsyncOpenAI') as mock_openai:
            # Mock streaming response
            async def mock_stream():
                mock_chunk1 = MagicMock()
                mock_chunk1.choices = [MagicMock()]
                mock_chunk1.choices[0].delta.content = "1"
                mock_chunk1.choices[0].finish_reason = None
                yield mock_chunk1

                mock_chunk2 = MagicMock()
                mock_chunk2.choices = [MagicMock()]
                mock_chunk2.choices[0].delta.content = " 2 3"
                mock_chunk2.choices[0].finish_reason = "stop"
                yield mock_chunk2

            mock_openai.return_value.chat.completions.create.return_value = mock_stream()

            chunks = []
            async for chunk in await service.chat(request):
                chunks.append(chunk)

            # Verify the streaming API was called with extra_params
            mock_openai.return_value.chat.completions.create.assert_called_once()
            call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]

            # Check that extra_params made it to the streaming API call
            assert call_kwargs["stream"] == True, "stream should be True"
            assert "logprobs" in call_kwargs, "logprobs should be passed to streaming API"
            assert call_kwargs["logprobs"] == True, "logprobs value should be preserved"
            assert "seed" in call_kwargs, "seed should be passed to streaming API"
            assert call_kwargs["seed"] == 12345, "seed value should be preserved"

            print(f"✓ Streaming extra_params flow works: {len(chunks)} chunks, params: {[k for k in call_kwargs.keys() if k in request.extra_params]}")

    @pytest.mark.asyncio
    async def test_empty_extra_params_handling(self, service):
        """Test that empty extra_params are handled correctly."""
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Test")],
            max_tokens=10,
            # extra_params defaults to empty dict
        )

        # Patch the OpenAI client
        with patch('llmring.providers.openai_api.AsyncOpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15

            mock_openai.return_value.chat.completions.create.return_value = mock_response

            response = await service.chat(request)

            # Should work fine with empty extra_params
            assert response.content == "Test response"

            print("✓ Empty extra_params handling works correctly")

    def test_extra_params_schema_validation(self):
        """Test that extra_params schema validation works."""
        # Valid extra_params
        valid_request = LLMRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            extra_params={"valid": "param"}
        )
        assert valid_request.extra_params == {"valid": "param"}

        # Empty extra_params (default)
        empty_request = LLMRequest(
            model="test",
            messages=[Message(role="user", content="test")]
        )
        assert empty_request.extra_params == {}

        # Invalid extra_params (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            LLMRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                extra_params="not a dict"
            )

        print("✓ extra_params schema validation works correctly")