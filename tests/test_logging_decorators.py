"""
Tests for logging decorators (@log_llm_call and @log_llm_stream).

These tests verify that the decorators work with different provider SDKs
and properly log to llmring-server.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmring.logging import log_llm_call, log_llm_stream
from llmring.logging.normalizers import detect_provider, normalize_response

# -----------------------------------------------------------------------------
# Mock Provider Response Classes
# -----------------------------------------------------------------------------


class MockOpenAIUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt_tokens=10, completion_tokens=20, total_tokens=30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.prompt_tokens_details = None


class MockOpenAIChoice:
    """Mock OpenAI choice object."""

    def __init__(self, content="Hello from OpenAI!", finish_reason="stop"):
        self.message = MagicMock()
        self.message.content = content
        self.finish_reason = finish_reason


class MockOpenAIResponse:
    """Mock OpenAI ChatCompletion response."""

    __module__ = "openai.types.chat"

    def __init__(self, content="Hello from OpenAI!", model="gpt-4o"):
        self.choices = [MockOpenAIChoice(content=content)]
        self.model = model
        self.usage = MockOpenAIUsage()


class MockAnthropicUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class MockAnthropicContentBlock:
    """Mock Anthropic content block."""

    def __init__(self, text="Hello from Anthropic!"):
        self.text = text


class MockAnthropicResponse:
    """Mock Anthropic Message response."""

    __module__ = "anthropic.types"

    def __init__(self, content="Hello from Anthropic!", model="claude-3-opus-20240229"):
        self.content = [MockAnthropicContentBlock(content)]
        self.model = model
        self.stop_reason = "end_turn"
        self.usage = MockAnthropicUsage()


class MockGoogleUsageMetadata:
    """Mock Google usage metadata."""

    def __init__(self, prompt_tokens=10, candidates_tokens=20, total_tokens=30):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.total_token_count = total_tokens
        self.cached_content_token_count = 0


class MockGooglePart:
    """Mock Google content part."""

    def __init__(self, text="Hello from Google!"):
        self.text = text


class MockGoogleContent:
    """Mock Google content object."""

    def __init__(self, text="Hello from Google!"):
        self.parts = [MockGooglePart(text)]


class MockGoogleCandidate:
    """Mock Google candidate."""

    def __init__(self, text="Hello from Google!"):
        self.content = MockGoogleContent(text)
        self.finish_reason = "STOP"


class MockGoogleResponse:
    """Mock Google GenerateContentResponse."""

    __module__ = "google.generativeai"

    def __init__(self, text="Hello from Google!", model="gemini-pro"):
        self.text = text
        self.candidates = [MockGoogleCandidate(text)]
        self.model_version = model
        self.usage_metadata = MockGoogleUsageMetadata()


# -----------------------------------------------------------------------------
# Mock Streaming Response Classes
# -----------------------------------------------------------------------------


class MockOpenAIStreamDelta:
    """Mock OpenAI stream delta."""

    def __init__(self, content="chunk"):
        self.content = content


class MockOpenAIStreamChoice:
    """Mock OpenAI stream choice."""

    def __init__(self, content="chunk", finish_reason=None):
        self.delta = MockOpenAIStreamDelta(content)
        self.finish_reason = finish_reason


class MockOpenAIStreamChunk:
    """Mock OpenAI stream chunk."""

    def __init__(self, content="chunk", model="gpt-4o", finish_reason=None, usage=None):
        self.choices = [MockOpenAIStreamChoice(content, finish_reason)]
        self.model = model
        self.usage = usage


# -----------------------------------------------------------------------------
# Tests for Provider Detection
# -----------------------------------------------------------------------------


class TestProviderDetection:
    """Test auto-detection of provider from response objects."""

    def test_detect_openai_from_response(self):
        """Should detect OpenAI from response type."""
        response = MockOpenAIResponse()
        provider = detect_provider(response)
        assert provider == "openai"

    def test_detect_anthropic_from_response(self):
        """Should detect Anthropic from response type."""
        response = MockAnthropicResponse()
        provider = detect_provider(response)
        assert provider == "anthropic"

    def test_detect_google_from_response(self):
        """Should detect Google from response type."""
        response = MockGoogleResponse()
        provider = detect_provider(response)
        assert provider == "google"

    def test_detect_unknown_provider(self):
        """Should return None for unknown provider."""
        response = {"content": "hello"}
        provider = detect_provider(response)
        assert provider is None


# -----------------------------------------------------------------------------
# Tests for Response Normalization
# -----------------------------------------------------------------------------


class TestResponseNormalization:
    """Test normalization of provider-specific responses."""

    def test_normalize_openai_response(self):
        """Should normalize OpenAI response correctly."""
        response = MockOpenAIResponse(content="Test content", model="gpt-4o")
        content, model, usage, finish_reason = normalize_response(response, "openai")

        assert content == "Test content"
        assert model == "gpt-4o"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert finish_reason == "stop"

    def test_normalize_anthropic_response(self):
        """Should normalize Anthropic response correctly."""
        response = MockAnthropicResponse(content="Test content", model="claude-3-opus-20240229")
        content, model, usage, finish_reason = normalize_response(response, "anthropic")

        assert content == "Test content"
        assert model == "claude-3-opus-20240229"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert finish_reason == "end_turn"

    def test_normalize_google_response(self):
        """Should normalize Google response correctly."""
        response = MockGoogleResponse(text="Test content", model="gemini-pro")
        content, model, usage, finish_reason = normalize_response(response, "google")

        assert content == "Test content"
        assert model == "gemini-pro"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert finish_reason == "STOP"


# -----------------------------------------------------------------------------
# Tests for @log_llm_call Decorator
# -----------------------------------------------------------------------------


class TestLogLLMCallDecorator:
    """Test @log_llm_call decorator functionality."""

    @pytest.mark.asyncio
    async def test_decorator_returns_original_response(self):
        """Decorator should return the original response unchanged."""

        @log_llm_call(server_url="http://localhost:8000", provider="openai")
        async def mock_openai_call():
            return MockOpenAIResponse(content="Test", model="gpt-4o")

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_client.return_value.post = AsyncMock()

            response = await mock_openai_call()

            # Response should be unchanged
            assert isinstance(response, MockOpenAIResponse)
            assert response.choices[0].message.content == "Test"

    @pytest.mark.asyncio
    async def test_decorator_logs_metadata(self):
        """Decorator should log metadata when log_metadata=True."""

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="openai",
            log_metadata=True,
            log_conversations=False,
        )
        async def mock_openai_call():
            return MockOpenAIResponse(content="Test", model="gpt-4o")

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            await mock_openai_call()

            # Wait for background task to complete
            await asyncio.sleep(0.1)

            # Should have called POST /api/v1/log
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/log"
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "openai"
            assert log_data["model"] == "gpt-4o"
            assert log_data["input_tokens"] == 10
            assert log_data["output_tokens"] == 20

    @pytest.mark.asyncio
    async def test_decorator_logs_conversations(self):
        """Decorator should log full conversations when log_conversations=True."""

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="openai",
            log_conversations=True,
        )
        async def mock_openai_call(messages):
            return MockOpenAIResponse(content="Hello!", model="gpt-4o")

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            messages = [{"role": "user", "content": "Hi"}]
            await mock_openai_call(messages)

            # Wait for background task
            await asyncio.sleep(0.1)

            # Should have called POST /api/v1/conversations/log
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/conversations/log"
            conv_data = call_args[1]["json"]
            assert conv_data["messages"] == messages
            assert conv_data["response"]["content"] == "Hello!"
            assert conv_data["metadata"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_decorator_auto_detects_provider(self):
        """Decorator should auto-detect provider when provider='auto'."""

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="auto",
            log_metadata=True,
        )
        async def mock_anthropic_call():
            return MockAnthropicResponse(content="Test", model="claude-3-opus-20240229")

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            await mock_anthropic_call()
            await asyncio.sleep(0.1)

            # Should have detected anthropic
            call_args = mock_post.call_args
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_decorator_handles_logging_errors(self):
        """Decorator should handle logging errors gracefully."""

        @log_llm_call(server_url="http://localhost:8000", provider="openai")
        async def mock_openai_call():
            return MockOpenAIResponse()

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            # Make post raise an exception
            mock_client.return_value.post = AsyncMock(side_effect=Exception("Server error"))

            # Should not raise, just log warning
            response = await mock_openai_call()
            await asyncio.sleep(0.1)

            # Should still return response
            assert response is not None

    def test_decorator_rejects_non_async_functions(self):
        """Decorator should raise TypeError for non-async functions."""

        with pytest.raises(TypeError, match="can only decorate async functions"):

            @log_llm_call(server_url="http://localhost:8000")
            def sync_function():
                return "not async"


# -----------------------------------------------------------------------------
# Tests for @log_llm_stream Decorator
# -----------------------------------------------------------------------------


class TestLogLLMStreamDecorator:
    """Test @log_llm_stream decorator functionality."""

    @pytest.mark.asyncio
    async def test_decorator_yields_chunks_unchanged(self):
        """Decorator should yield all chunks unchanged."""

        @log_llm_stream(server_url="http://localhost:8000", provider="openai")
        async def mock_stream():
            for i in range(3):
                yield MockOpenAIStreamChunk(content=f"chunk{i}", model="gpt-4o")

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_client.return_value.post = AsyncMock()

            chunks = []
            async for chunk in mock_stream():
                chunks.append(chunk)

            # Should have received all chunks
            assert len(chunks) == 3
            assert chunks[0].choices[0].delta.content == "chunk0"
            assert chunks[1].choices[0].delta.content == "chunk1"
            assert chunks[2].choices[0].delta.content == "chunk2"

    @pytest.mark.asyncio
    async def test_decorator_logs_after_stream_completes(self):
        """Decorator should log after stream completes."""

        @log_llm_stream(
            server_url="http://localhost:8000",
            provider="openai",
            log_metadata=True,
        )
        async def mock_stream():
            yield MockOpenAIStreamChunk(content="chunk1", model="gpt-4o")
            yield MockOpenAIStreamChunk(content="chunk2", model="gpt-4o")
            # Final chunk with usage
            final_usage = MockOpenAIUsage(prompt_tokens=10, completion_tokens=20)
            yield MockOpenAIStreamChunk(
                content="", model="gpt-4o", finish_reason="stop", usage=final_usage
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            chunks = []
            async for chunk in mock_stream():
                chunks.append(chunk)

            # Wait for background logging task
            await asyncio.sleep(0.1)

            # Should have logged usage
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/log"
            log_data = call_args[1]["json"]
            assert log_data["input_tokens"] == 10
            assert log_data["output_tokens"] == 20

    def test_stream_decorator_rejects_non_generator(self):
        """Decorator should raise TypeError for non-generator functions."""

        with pytest.raises(TypeError, match="can only decorate async generator functions"):

            @log_llm_stream(server_url="http://localhost:8000")
            async def not_a_generator():
                return "not a generator"


# -----------------------------------------------------------------------------
# Integration Tests (if llmring-server available)
# -----------------------------------------------------------------------------


@pytest.mark.skipif("LLMRING_SERVER_AVAILABLE" not in dir(), reason="llmring-server not available")
class TestDecoratorWithRealServer:
    """Integration tests with real llmring-server."""

    @pytest.mark.asyncio
    async def test_decorator_with_real_server(self, llmring_server_client):
        """Test decorator with actual llmring-server instance."""

        @log_llm_call(
            server_url="http://testserver",
            provider="openai",
            log_metadata=True,
        )
        async def mock_openai_call():
            return MockOpenAIResponse(content="Test", model="gpt-4o")

        # Execute decorated function
        response = await mock_openai_call()

        # Wait for background logging
        await asyncio.sleep(0.2)

        # Verify response is unchanged
        assert response.choices[0].message.content == "Test"

        # TODO: Verify log was actually stored in database
        # This would require querying the test database through llmring_server_client
