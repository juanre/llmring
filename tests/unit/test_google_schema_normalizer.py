import pytest

from llmring.services.schema_adapter import SchemaAdapter


class TestGoogleSchemaNormalizer:
    @pytest.fixture
    def adapter(self):
        return SchemaAdapter()

    def test_boolean_null_union_normalizes_to_boolean(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": ["boolean", "null"]},
            },
            "required": ["active"],
        }

        normalized, notes = adapter.normalize_google_schema(schema)

        assert normalized["type"] == "object"
        assert "properties" in normalized
        assert normalized["properties"]["active"]["type"] == "boolean"
        assert any("removed 'null'" in n for n in notes)

    def test_multi_type_union_falls_back_to_string(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"]},
            },
        }

        normalized, notes = adapter.normalize_google_schema(schema)

        assert normalized["properties"]["value"]["type"] == "string"
        assert any("multi-type union" in n for n in notes)

    def test_removes_unsupported_keywords(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": False,
                    "oneOf": [{"type": "object", "properties": {"a": {"type": "string"}}}],
                }
            },
        }

        normalized, notes = adapter.normalize_google_schema(schema)

        item_schema = normalized["properties"]["item"]
        assert item_schema["type"] == "object"
        assert "additionalProperties" not in item_schema
        assert "oneOf" not in item_schema
        # Notes should mention removed keywords
        joined = "\n".join(notes)
        assert "removed unsupported keywords" in joined
