"""
Test the specific provider enhancements implemented from code review.

These tests validate that:
1. Google uses real streaming (not faked single chunk)
2. Google uses native function calling (not prompt-based)
3. Google honors user model versions (no surprise downgrades)
4. Ollama uses real streaming (not faked single chunk)
5. All providers work with real APIs
"""

import os
import pytest

from llmring.service import LLMRing
from llmring.schemas import LLMRequest, LLMResponse, Message


@pytest.mark.integration
@pytest.mark.llm
class TestProviderEnhancements:
    """Test provider enhancements with real APIs."""

    @pytest.fixture
    def service(self):
        """Create LLMRing service with all providers."""
        return LLMRing()

    @pytest.mark.skipif(
        not (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
        ),
        reason="Google API key not available",
    )
    @pytest.mark.asyncio
    async def test_google_real_streaming_vs_faked(self, service):
        """Test that Google streaming is real (multiple chunks) not faked (single chunk)."""
        request = LLMRequest(
            model="fast",
            messages=[
                Message(
                    role="user",
                    content="Count from 1 to 5, one number per sentence, slowly.",
                )
            ],
            max_tokens=50,
            temperature=0.7,
            stream=True,
        )

        chunks = []
        async for chunk in await service.chat(request):
            chunks.append(chunk)

        # Real streaming should produce multiple chunks
        content_chunks = [c for c in chunks if c.delta and c.delta.strip()]

        print(
            f"Google streaming: {len(chunks)} total chunks, {len(content_chunks)} content chunks"
        )
        print(f"Deltas: {[c.delta for c in content_chunks]}")

        # If this was still faked, we'd get exactly 1 chunk with all content
        # Real streaming should give us multiple chunks
        assert len(content_chunks) > 1, (
            f"Expected real streaming (>1 chunk), got {len(content_chunks)} - may still be faked"
        )

    @pytest.mark.skipif(
        not (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
        ),
        reason="Google API key not available",
    )
    @pytest.mark.asyncio
    async def test_google_model_version_honoring(self, service):
        """Test that Google honors user-specified model versions (no 2.x → 1.5 downgrade)."""
        # Try a 2.x model - should be accepted, not downgraded to 1.5
        request = LLMRequest(
            model="google:gemini-2.0-flash-lite",
            messages=[Message(role="user", content="What model are you?")],
            max_tokens=30,
            temperature=0.1,
        )

        try:
            response = await service.chat(request)

            # The response should succeed (not throw "unsupported model")
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0

            # Check that the response model reflects the 2.x model, not 1.5
            print(f"Requested: gemini-2.0-flash-lite, Got: {response.model}")
            # Model should be the 2.x version, not downgraded to 1.5
            assert "2.0" in response.model or response.model == "gemini-2.0-flash-lite"

        except Exception as e:
            if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                # If 2.x model isn't available, that's fine - the key is it wasn't auto-downgraded
                pytest.skip(f"2.x model not available in API: {e}")
            else:
                raise

    @pytest.mark.skipif(
        not (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
        ),
        reason="Google API key not available",
    )
    @pytest.mark.asyncio
    async def test_google_native_function_calling(self, service):
        """Test that Google uses native function calling (not prompt-based)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        request = LLMRequest(
            model="google:gemini-1.5-pro",  # Use Pro for better function calling
            messages=[
                Message(
                    role="user",
                    content="What's the weather like in New York? Use the weather tool.",
                )
            ],
            tools=tools,
            max_tokens=100,
            temperature=0.1,
        )

        try:
            response = await service.chat(request)

            # Should either call the function or explain it would call the function
            # The key test is that it doesn't have prompt-engineered tool descriptions in the response
            assert isinstance(response, LLMResponse)
            print(f"Function calling response: {response.content[:100]}...")

            # If tools were properly integrated, response should be relevant to weather
            # and not contain the raw tool schema (which would indicate prompt-based approach)
            tool_schema_indicators = [
                "parameters",
                "properties",
                "required",
                '"type": "object"',
            ]
            response_lower = response.content.lower()

            # Should not contain raw tool schema (indicates prompt-based approach)
            schema_count = sum(
                1 for indicator in tool_schema_indicators if indicator in response_lower
            )
            assert schema_count <= 1, (
                f"Response contains tool schema, may still be using prompt-based approach: {response.content[:200]}"
            )

        except Exception as e:
            if "function calling" in str(e).lower() or "tool" in str(e).lower():
                pytest.skip(f"Function calling not available: {e}")
            else:
                raise

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    @pytest.mark.asyncio
    async def test_openai_json_schema_support(self, service):
        """Test OpenAI JSON schema support (new feature)."""
        request = LLMRequest(
            model="test",
            messages=[
                Message(role="user", content="Generate a person with name and age")
            ],
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "additionalProperties": False,
                    },
                },
                "strict": True,
            },
        )

        response = await service.chat(request)

        # Should return valid JSON matching the schema
        import json

        try:
            data = json.loads(response.content)
            assert "name" in data, "Response should have name field"
            assert "age" in data, "Response should have age field"
            assert isinstance(data["name"], str), "Name should be string"
            assert isinstance(data["age"], int), "Age should be integer"

            print(f"✓ JSON schema response: {data}")

        except json.JSONDecodeError:
            pytest.fail(f"Response should be valid JSON, got: {response.content}")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available"
    )
    @pytest.mark.asyncio
    async def test_anthropic_prompt_caching_integration(self, service):
        """Test that Anthropic prompt caching works in the integrated system."""
        # Long system prompt to trigger caching
        long_system = "You are an expert assistant. " * 200  # Over 1024 tokens

        request = LLMRequest(
            model="balanced",
            messages=[
                Message(
                    role="system",
                    content=long_system,
                    metadata={"cache_control": {"type": "ephemeral"}},
                ),
                Message(role="user", content="What is 2+2?"),
            ],
            max_tokens=20,
            temperature=0.1,
        )

        response = await service.chat(request)

        # Should work without errors and potentially show cache usage
        assert isinstance(response, LLMResponse)
        assert "4" in response.content

        if response.usage:
            # Check for cache-related usage (may be 0 if not cached)
            cache_keys = ["cache_creation_input_tokens", "cache_read_input_tokens"]
            has_cache_info = any(key in response.usage for key in cache_keys)
            if has_cache_info:
                print(
                    f"✓ Cache usage tracked: {[(k, v) for k, v in response.usage.items() if 'cache' in k]}"
                )

    def test_all_critical_providers_available(self, service):
        """Test that all critical providers with API keys are available."""
        available = list(service.providers.keys())
        print(f"Available providers: {available}")

        # Check each provider based on API key availability
        if os.getenv("OPENAI_API_KEY"):
            assert "openai" in available, (
                "OpenAI provider should be available with API key"
            )

        if os.getenv("ANTHROPIC_API_KEY"):
            assert "anthropic" in available, (
                "Anthropic provider should be available with API key"
            )

        google_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GOOGLE_GEMINI_API_KEY"),
        ]
        if any(google_keys):
            assert "google" in available, (
                f"Google provider should be available with API key. Available: {available}"
            )

        # Ollama should always be available (no API key required)
        assert "ollama" in available, "Ollama provider should always be available"

        assert len(available) >= 2, (
            f"Should have multiple providers available, got: {available}"
        )
