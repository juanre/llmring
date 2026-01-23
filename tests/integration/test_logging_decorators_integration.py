"""
Integration tests for logging decorators using real provider API calls.

These tests verify that the @log_llm_call and @log_llm_stream decorators
work correctly with actual provider SDK responses.
"""

import asyncio
import os

import pytest

from llmring.logging import log_llm_call


class TestLogLLMCallWithRealProviders:
    """Test @log_llm_call decorator with real provider SDK responses."""

    @pytest.mark.asyncio
    async def test_decorator_with_real_openai_response(self):
        """Test decorator with actual OpenAI API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        from openai import AsyncOpenAI

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="openai",
            log_metadata=True,
            log_conversations=False,
        )
        async def call_openai():
            client = AsyncOpenAI()
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
                max_tokens=10,
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            response = await call_openai()

            # Wait for background logging
            await asyncio.sleep(0.2)

            # Verify response is real OpenAI response
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None

            # Verify decorator logged the request
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/log"
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "openai"
            assert "gpt-4o-mini" in log_data["model"]
            assert log_data["input_tokens"] > 0
            assert log_data["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_decorator_with_real_anthropic_response(self):
        """Test decorator with actual Anthropic API call."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        import anthropic

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="anthropic",
            log_metadata=True,
            log_conversations=False,
        )
        async def call_anthropic():
            client = anthropic.AsyncAnthropic()
            return await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            response = await call_anthropic()

            # Wait for background logging
            await asyncio.sleep(0.2)

            # Verify response is real Anthropic response
            assert hasattr(response, "content")
            assert len(response.content) > 0
            assert hasattr(response, "usage")

            # Verify decorator logged the request
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/log"
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "anthropic"
            assert "claude" in log_data["model"].lower()
            assert log_data["input_tokens"] > 0
            assert log_data["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_decorator_auto_detects_openai(self):
        """Test auto-detection with real OpenAI response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        from openai import AsyncOpenAI

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="auto",  # Auto-detect
            log_metadata=True,
        )
        async def call_openai():
            client = AsyncOpenAI()
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=10,
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            await call_openai()
            await asyncio.sleep(0.2)

            # Should have auto-detected openai
            call_args = mock_post.call_args
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_decorator_auto_detects_anthropic(self):
        """Test auto-detection with real Anthropic response."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        import anthropic

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="auto",  # Auto-detect
            log_metadata=True,
        )
        async def call_anthropic():
            client = anthropic.AsyncAnthropic()
            return await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test'"}],
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            await call_anthropic()
            await asyncio.sleep(0.2)

            # Should have auto-detected anthropic
            call_args = mock_post.call_args
            log_data = call_args[1]["json"]
            assert log_data["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_decorator_preserves_response_unchanged(self):
        """Verify decorator doesn't modify the response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        from openai import AsyncOpenAI

        # First call without decorator
        client = AsyncOpenAI()
        raw_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say exactly: UNCHANGED"}],
            max_tokens=20,
        )

        # Second call with decorator
        @log_llm_call(server_url="http://localhost:8000", provider="openai")
        async def decorated_call():
            client = AsyncOpenAI()
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say exactly: UNCHANGED"}],
                max_tokens=20,
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_client.return_value.post = AsyncMock()
            decorated_response = await decorated_call()

        # Both responses should have the same structure
        assert type(raw_response) == type(decorated_response)
        assert hasattr(decorated_response, "choices")
        assert hasattr(decorated_response, "usage")
        assert hasattr(decorated_response, "model")

    @pytest.mark.asyncio
    async def test_decorator_logs_conversations_with_real_response(self):
        """Test conversation logging with real API response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from unittest.mock import AsyncMock, patch

        from openai import AsyncOpenAI

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="openai",
            log_conversations=True,
        )
        async def call_openai(messages):
            client = AsyncOpenAI()
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=10,
            )

        with patch("llmring.logging.decorators.ServerClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.post = mock_post

            messages = [{"role": "user", "content": "Say hello"}]
            response = await call_openai(messages)
            await asyncio.sleep(0.2)

            # Verify conversation was logged
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/conversations/log"
            conv_data = call_args[1]["json"]
            assert conv_data["messages"] == messages
            assert "content" in conv_data["response"]
            assert len(conv_data["response"]["content"]) > 0
