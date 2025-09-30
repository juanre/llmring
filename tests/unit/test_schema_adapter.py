"""Unit tests for SchemaAdapter service."""

import json
from unittest.mock import MagicMock

import pytest

from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.services.schema_adapter import SchemaAdapter


class TestSchemaAdapter:
    """Tests for SchemaAdapter service."""

    @pytest.fixture
    def adapter(self):
        """Create a SchemaAdapter instance."""
        return SchemaAdapter()

    @pytest.fixture
    def sample_schema(self):
        """Sample JSON schema for testing."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": ["string", "null"]},
            },
            "required": ["name"],
        }

    @pytest.mark.asyncio
    async def test_no_adaptation_for_openai(self, adapter, sample_schema):
        """OpenAI has native support, should not adapt."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="gpt-4",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": sample_schema},
            },
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "openai", provider)

        # Should not add tools for OpenAI
        assert result.tools is None
        assert result.metadata is not None
        assert result.metadata["_structured_output_adapted"] is True

    @pytest.mark.asyncio
    async def test_adapt_for_anthropic(self, adapter, sample_schema):
        """Should inject tool for Anthropic."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="claude-3",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": sample_schema},
            },
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "anthropic", provider)

        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "respond_with_structure"
        assert result.tool_choice == {"type": "any"}
        assert result.metadata["_structured_output_adapted"] is True

    @pytest.mark.asyncio
    async def test_adapt_for_google(self, adapter, sample_schema):
        """Should inject tool with normalized schema for Google."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="gemini-pro",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": sample_schema},
            },
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "google", provider)

        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "respond_with_structure"
        assert result.tool_choice == "any"
        # Should have normalization notes for the null type
        assert "_schema_normalization_notes" in result.metadata

    @pytest.mark.asyncio
    async def test_adapt_for_ollama(self, adapter, sample_schema):
        """Should add system message with schema for Ollama."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="llama2",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": sample_schema},
            },
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "ollama", provider)

        assert result.json_response is True
        assert len(result.messages) == 2  # System + user
        assert result.messages[0].role == "system"
        assert "IMPORTANT: Respond with valid JSON" in result.messages[0].content
        # Schema should be present (indented format)
        assert '"name"' in result.messages[0].content
        assert '"age"' in result.messages[0].content

    @pytest.mark.asyncio
    async def test_no_adaptation_when_tools_present(self, adapter, sample_schema):
        """Should not adapt if tools already present."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="claude-3",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": sample_schema},
            },
            tools=[{"type": "function", "function": {"name": "existing_tool"}}],
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "anthropic", provider)

        # Should not modify tools
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "existing_tool"
        assert "_structured_output_adapted" not in (result.metadata or {})

    @pytest.mark.asyncio
    async def test_no_adaptation_without_json_schema(self, adapter):
        """Should not adapt without json_schema format."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="claude-3",
        )

        provider = MagicMock()
        result = await adapter.apply_structured_output_adapter(request, "anthropic", provider)

        assert result.tools is None
        assert result.metadata is None or "_structured_output_adapted" not in result.metadata

    def test_normalize_google_schema_union_types(self, adapter):
        """Should normalize union types for Google."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "object",
            "properties": {
                "nullable_field": {"type": ["string", "null"]},
                "multi_type": {"type": ["string", "number"]},
            },
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        # Should remove null from union
        assert normalized["properties"]["nullable_field"]["type"] == "string"
        # Should fallback to string for multi-type union
        assert normalized["properties"]["multi_type"]["type"] == "string"
        # Should have notes about changes
        assert len(notes) == 2
        assert "removed 'null'" in notes[0]
        assert "multi-type union" in notes[1]

    def test_normalize_google_schema_unsupported_keywords(self, adapter):
        """Should remove unsupported keywords."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
            "anyOf": [{"type": "string"}],
            "pattern": "^[a-z]+$",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        # Should remove unsupported keywords
        assert "additionalProperties" not in normalized
        assert "anyOf" not in normalized
        assert "pattern" not in normalized
        # Should keep supported fields
        assert "properties" in normalized
        # Should have notes
        assert any("removed unsupported keywords" in note for note in notes)

    def test_normalize_google_schema_nested_objects(self, adapter):
        """Should recursively normalize nested objects."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"type": ["string", "null"]},
                    },
                },
            },
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        # Should normalize nested union type
        assert normalized["properties"]["user"]["properties"]["email"]["type"] == "string"
        assert any("properties.user.properties.email" in note for note in notes)

    def test_normalize_google_schema_arrays(self, adapter):
        """Should normalize array schemas."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "array",
            "items": {"type": ["string", "null"]},
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "array"
        assert normalized["items"]["type"] == "string"

    def test_normalize_google_schema_tuple_arrays(self, adapter):
        """Should normalize tuple-typed arrays to first schema."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "array",
            "items": [{"type": "string"}, {"type": "number"}],
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "array"
        assert normalized["items"]["type"] == "string"
        assert any("tuple-typed" in note for note in notes)

    def test_normalize_google_schema_preserves_supported_fields(self, adapter):
        """Should preserve supported JSON Schema fields."""
        from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer

        schema = {
            "type": "string",
            "title": "Name",
            "description": "User's name",
            "minLength": 1,
            "maxLength": 100,
            "default": "Anonymous",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["title"] == "Name"
        assert normalized["description"] == "User's name"
        assert normalized["minLength"] == 1
        assert normalized["maxLength"] == 100
        assert normalized["default"] == "Anonymous"

    @pytest.mark.asyncio
    async def test_post_process_openai_response(self, adapter, sample_schema):
        """Should parse JSON from OpenAI content."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="gpt-4",
            metadata={
                "_structured_output_adapted": True,
                "_original_schema": sample_schema,
            },
        )

        response = LLMResponse(
            content='{"name": "John", "age": 30}',
            model="gpt-4",
            provider="openai",
        )

        result = await adapter.post_process_structured_output(response, request, "openai")

        assert result.parsed is not None
        assert result.parsed["name"] == "John"
        assert result.parsed["age"] == 30

    @pytest.mark.asyncio
    async def test_post_process_anthropic_response(self, adapter, sample_schema):
        """Should extract JSON from Anthropic tool call."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="claude-3",
            metadata={
                "_structured_output_adapted": True,
                "_original_schema": sample_schema,
            },
        )

        response = LLMResponse(
            content="I'll provide the structured data",
            model="claude-3",
            provider="anthropic",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "respond_with_structure",
                        "arguments": {"name": "Jane", "age": 25},
                    },
                }
            ],
        )

        result = await adapter.post_process_structured_output(response, request, "anthropic")

        assert result.parsed is not None
        assert result.parsed["name"] == "Jane"
        assert result.parsed["age"] == 25
        # Content should be updated to JSON
        assert "Jane" in result.content

    @pytest.mark.asyncio
    async def test_post_process_skips_non_adapted(self, adapter):
        """Should skip post-processing if not adapted."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="gpt-4",
        )

        response = LLMResponse(
            content='{"name": "John"}',
            model="gpt-4",
            provider="openai",
        )

        result = await adapter.post_process_structured_output(response, request, "openai")

        # Should not modify response
        assert result.parsed is None

    @pytest.mark.asyncio
    async def test_post_process_handles_invalid_json(self, adapter, sample_schema):
        """Should handle invalid JSON gracefully."""
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="gpt-4",
            metadata={
                "_structured_output_adapted": True,
                "_original_schema": sample_schema,
            },
        )

        response = LLMResponse(
            content="This is not JSON",
            model="gpt-4",
            provider="openai",
        )

        result = await adapter.post_process_structured_output(response, request, "openai")

        # Should not crash, just not populate parsed
        assert result.parsed is None
