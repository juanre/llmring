"""
Test unified structured output across all providers.

Validates that the same response_format with JSON schema works consistently
across OpenAI (native), Anthropic (tool injection), Google (function calling),
and Ollama (best effort).
"""

import json
import pytest

from llmring.service import LLMRing
from llmring.schemas import LLMRequest, LLMResponse, Message


@pytest.mark.integration
@pytest.mark.llm
class TestUnifiedStructuredOutput:
    """Test unified structured output across all providers."""

    @pytest.fixture
    def service(self):
        """Create LLMRing service."""
        return LLMRing()

    @pytest.fixture
    def person_schema(self):
        """Simple person schema for testing."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

    @pytest.fixture
    def structured_request_template(self, person_schema):
        """Template for structured output requests."""
        return {
            "messages": [Message(role="user", content="Generate a person with name Alice, age 30")],
            "max_tokens": 100,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": person_schema
                },
                "strict": True
            }
        }

    @pytest.mark.skipif(not pytest.importorskip("openai", reason="OpenAI not available"), reason="OpenAI not available")
    @pytest.mark.asyncio
    async def test_openai_unified_structured_output(self, service, structured_request_template):
        """Test OpenAI native JSON schema support via unified interface."""
        request = LLMRequest(model="openai:gpt-4o-mini", **structured_request_template)
        response = await service.chat(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.parsed is not None

        # Verify JSON parsing
        assert isinstance(response.parsed, dict)
        assert "name" in response.parsed
        assert "age" in response.parsed
        assert isinstance(response.parsed["age"], int)

        print(f"✓ OpenAI structured output: {response.parsed}")

    @pytest.mark.skipif(not pytest.importorskip("anthropic", reason="Anthropic not available"), reason="Anthropic not available")
    @pytest.mark.asyncio
    async def test_anthropic_unified_structured_output(self, service, structured_request_template):
        """Test Anthropic tool injection approach via unified interface."""
        request = LLMRequest(model="anthropic:claude-3-5-haiku", **structured_request_template)
        response = await service.chat(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.parsed is not None
        assert response.tool_calls is not None  # Should have tool calls

        # Verify tool injection worked
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "respond_with_structure"

        # Verify JSON parsing
        assert isinstance(response.parsed, dict)
        assert "name" in response.parsed
        assert "age" in response.parsed

        print(f"✓ Anthropic structured output: {response.parsed}")

    @pytest.mark.skipif(not pytest.importorskip("google.genai", reason="Google GenAI not available"), reason="Google GenAI not available")
    @pytest.mark.asyncio
    async def test_google_unified_structured_output(self, service, structured_request_template):
        """Test Google function calling approach via unified interface."""
        request = LLMRequest(model="google:gemini-1.5-flash", **structured_request_template)
        response = await service.chat(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.parsed is not None
        assert response.tool_calls is not None  # Should have function calls

        # Verify function calling worked
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "respond_with_structure"

        # Verify JSON parsing
        assert isinstance(response.parsed, dict)
        assert "name" in response.parsed
        assert "age" in response.parsed

        print(f"✓ Google structured output: {response.parsed}")

    @pytest.mark.asyncio
    async def test_ollama_unified_structured_output(self, service, structured_request_template):
        """Test Ollama best-effort approach via unified interface."""
        request = LLMRequest(model="ollama:llama3.2:1b", **structured_request_template)

        try:
            response = await service.chat(request)

            # Verify response structure
            assert isinstance(response, LLMResponse)
            assert response.content is not None

            # Ollama best effort - parsing may or may not work
            if response.parsed:
                print(f"✓ Ollama structured output: {response.parsed}")
            else:
                print(f"✓ Ollama raw output: {response.content[:100]}...")

        except Exception as e:
            if "connect" in str(e).lower():
                pytest.skip("Ollama not running")
            else:
                raise

    @pytest.mark.asyncio
    async def test_cross_provider_consistency(self, service, person_schema):
        """Test that the same schema produces consistent results across providers."""
        request_template = {
            "messages": [Message(role="user", content="Generate: name=Bob, age=35")],
            "max_tokens": 50,
            "temperature": 0.0,  # Most deterministic
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": person_schema
                },
                "strict": True
            }
        }

        results = {}
        available_providers = list(service.providers.keys())

        for provider_name in ["openai", "anthropic", "google"]:
            if provider_name in available_providers:
                try:
                    if provider_name == "openai":
                        model = "openai:gpt-4o-mini"
                    elif provider_name == "anthropic":
                        model = "anthropic:claude-3-5-haiku"
                    elif provider_name == "google":
                        model = "google:gemini-1.5-flash"

                    request = LLMRequest(model=model, **request_template)
                    response = await service.chat(request)

                    if response.parsed:
                        results[provider_name] = response.parsed

                except Exception as e:
                    print(f"Provider {provider_name} failed: {e}")

        # Verify all providers that worked returned similar structure
        if len(results) >= 2:
            # Check that all results have required fields
            for provider_name, parsed_data in results.items():
                assert "name" in parsed_data, f"{provider_name} missing name field"
                assert "age" in parsed_data, f"{provider_name} missing age field"
                assert isinstance(parsed_data["age"], int), f"{provider_name} age not integer"

            print(f"✓ Cross-provider consistency verified across {len(results)} providers")
            print(f"Results: {results}")

    def test_schema_adaptation_logic(self, service):
        """Test that schema adaptation logic works correctly."""
        # Test OpenAI pass-through (should not be adapted)
        openai_request = LLMRequest(
            model="openai:gpt-4o",
            messages=[Message(role="user", content="test")],
            response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        )

        # Mock the adapter to test logic
        from unittest.mock import Mock
        mock_provider = Mock()

        # Should NOT adapt OpenAI
        adapted = asyncio.run(service._apply_structured_output_adapter(
            openai_request, "openai", mock_provider
        ))
        assert adapted.tools is None, "OpenAI should not have tools injected"

        # Should adapt Anthropic
        anthropic_request = LLMRequest(
            model="anthropic:claude-3-5-sonnet",
            messages=[Message(role="user", content="test")],
            response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        )

        adapted = asyncio.run(service._apply_structured_output_adapter(
            anthropic_request, "anthropic", mock_provider
        ))
        assert adapted.tools is not None, "Anthropic should have tools injected"
        assert len(adapted.tools) == 1
        assert adapted.tools[0]["function"]["name"] == "respond_with_structure"

        print("✓ Schema adaptation logic works correctly")