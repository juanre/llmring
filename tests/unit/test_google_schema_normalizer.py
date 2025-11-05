"""Tests for GoogleSchemaNormalizer."""

from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer


class TestGoogleSchemaNormalizer:
    """Test Google schema normalization logic."""

    def test_simple_string_schema(self):
        """Test normalization of simple string schema."""
        schema = {"type": "string", "description": "A simple string"}

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert normalized["description"] == "A simple string"
        assert len(notes) == 0

    def test_nullable_type_union(self):
        """Test normalization of union type with null."""
        schema = {
            "type": ["string", "null"],
            "description": "Optional string",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert len(notes) == 1
        assert "removed 'null' from union type" in notes[0]

    def test_boolean_null_union_normalizes_to_boolean(self):
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": ["boolean", "null"]},
            },
            "required": ["active"],
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "object"
        assert "properties" in normalized
        assert normalized["properties"]["active"]["type"] == "boolean"
        assert any("removed 'null'" in n for n in notes)

    def test_multiple_non_null_types(self):
        """Test normalization of multiple non-null types (fallback to string)."""
        schema = {
            "type": ["string", "number"],
            "description": "String or number",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert len(notes) == 1
        assert "multi-type union" in notes[0]
        assert "normalized to 'string'" in notes[0]

    def test_multi_type_union_falls_back_to_string(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"]},
            },
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["properties"]["value"]["type"] == "string"
        assert any("multi-type union" in n for n in notes)

    def test_only_null_type(self):
        """Test normalization of null-only type (fallback to string)."""
        schema = {
            "type": ["null"],
            "description": "Only null",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert len(notes) == 1
        assert "normalized to 'string'" in notes[0]

    def test_remove_unsupported_keywords(self):
        """Test removal of unsupported keywords."""
        schema = {
            "type": "string",
            "pattern": "^[A-Z]+$",
            "format": "email",
            "additionalProperties": False,
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert "pattern" not in normalized
        assert "format" not in normalized
        assert "additionalProperties" not in normalized
        assert len(notes) == 1
        assert "removed unsupported keywords" in notes[0]

    def test_removes_unsupported_keywords(self):
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

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        item_schema = normalized["properties"]["item"]
        assert item_schema["type"] == "object"
        assert "additionalProperties" not in item_schema
        assert "oneOf" not in item_schema
        # Notes should mention removed keywords
        joined = "\n".join(notes)
        assert "removed unsupported keywords" in joined

    def test_supported_keywords_preserved(self):
        """Test that supported keywords are preserved."""
        schema = {
            "type": "string",
            "title": "Test String",
            "description": "A test string",
            "default": "hello",
            "enum": ["hello", "world"],
            "minLength": 1,
            "maxLength": 100,
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert normalized["title"] == "Test String"
        assert normalized["description"] == "A test string"
        assert normalized["default"] == "hello"
        assert normalized["enum"] == ["hello", "world"]
        assert normalized["minLength"] == 1
        assert normalized["maxLength"] == 100
        assert len(notes) == 0

    def test_object_with_properties(self):
        """Test normalization of object with properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "number", "minimum": 0},
            },
            "required": ["name"],
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "object"
        assert "properties" in normalized
        assert "name" in normalized["properties"]
        assert normalized["properties"]["name"]["type"] == "string"
        assert "age" in normalized["properties"]
        assert normalized["properties"]["age"]["type"] == "number"
        assert normalized["required"] == ["name"]
        assert len(notes) == 0

    def test_nested_object_normalization(self):
        """Test normalization of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": ["string", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "object"
        assert "user" in normalized["properties"]
        assert normalized["properties"]["user"]["type"] == "object"
        assert "email" in normalized["properties"]["user"]["properties"]
        assert normalized["properties"]["user"]["properties"]["email"]["type"] == "string"
        assert "additionalProperties" not in normalized["properties"]["user"]

        # Should have 2 notes: one for removing null from union, one for removing additionalProperties
        assert len(notes) == 2

    def test_array_with_dict_items(self):
        """Test normalization of array with dict items."""
        schema = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 10,
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "array"
        assert "items" in normalized
        assert normalized["items"]["type"] == "string"
        assert normalized["items"]["minLength"] == 1
        assert normalized["minItems"] == 1
        assert normalized["maxItems"] == 10
        assert len(notes) == 0

    def test_array_with_tuple_items(self):
        """Test normalization of array with tuple-typed items (uses first schema)."""
        schema = {
            "type": "array",
            "items": [
                {"type": "string"},
                {"type": "number"},
            ],
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "array"
        assert "items" in normalized
        assert normalized["items"]["type"] == "string"
        assert len(notes) == 1
        assert "tuple-typed 'items'" in notes[0]

    def test_remove_complex_composition(self):
        """Test removal of complex schema composition."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
            ],
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
            ],
            "allOf": [
                {"type": "object"},
                {"properties": {"name": {"type": "string"}}},
            ],
            "not": {"type": "null"},
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert "anyOf" not in normalized
        assert "oneOf" not in normalized
        assert "allOf" not in normalized
        assert "not" not in normalized
        assert len(notes) == 1
        assert "removed unsupported keywords" in notes[0]

    def test_empty_schema(self):
        """Test normalization of empty schema."""
        schema = {}

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized == {}
        assert len(notes) == 0

    def test_non_dict_schema(self):
        """Test that non-dict schemas are returned as-is."""
        schema = "not a dict"

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized == "not a dict"
        assert len(notes) == 0

    def test_complex_real_world_schema(self):
        """Test normalization of complex real-world schema."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "email": {"type": ["string", "null"], "format": "email"},
                "age": {"type": "number", "minimum": 0, "maximum": 150},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 10,
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["id", "name"],
            "additionalProperties": False,
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        # Check structure is preserved
        assert normalized["type"] == "object"
        assert "properties" in normalized
        assert normalized["required"] == ["id", "name"]

        # Check id field (format removed)
        assert normalized["properties"]["id"]["type"] == "string"
        assert "format" not in normalized["properties"]["id"]

        # Check email field (null removed, format removed)
        assert normalized["properties"]["email"]["type"] == "string"
        assert "format" not in normalized["properties"]["email"]

        # Check age field (preserved)
        assert normalized["properties"]["age"]["type"] == "number"
        assert normalized["properties"]["age"]["minimum"] == 0
        assert normalized["properties"]["age"]["maximum"] == 150

        # Check tags array (preserved)
        assert normalized["properties"]["tags"]["type"] == "array"
        assert normalized["properties"]["tags"]["items"]["type"] == "string"

        # Check metadata object (additionalProperties removed)
        assert normalized["properties"]["metadata"]["type"] == "object"
        assert "additionalProperties" not in normalized["properties"]["metadata"]

        # Check notes
        assert len(notes) > 0
        note_text = " ".join(notes)
        assert "removed 'null' from union type" in note_text
        assert "removed unsupported keywords" in note_text

    def test_number_constraints(self):
        """Test that number constraints are preserved."""
        schema = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "number"
        assert normalized["minimum"] == 0
        assert normalized["maximum"] == 100
        assert len(notes) == 0

    def test_enum_preserved(self):
        """Test that enum values are preserved."""
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"],
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert normalized["enum"] == ["red", "green", "blue"]
        assert len(notes) == 0

    def test_const_preserved(self):
        """Test that const values are preserved."""
        schema = {
            "type": "string",
            "const": "constant_value",
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        assert normalized["type"] == "string"
        assert normalized["const"] == "constant_value"
        assert len(notes) == 0

    def test_deeply_nested_structures(self):
        """Test normalization of deeply nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": {"type": ["string", "null"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        normalized, notes = GoogleSchemaNormalizer.normalize(schema)

        # Navigate to deeply nested field
        level3 = normalized["properties"]["level1"]["properties"]["level2"]["properties"]["level3"]
        assert level3["type"] == "array"
        assert level3["items"]["type"] == "object"
        assert level3["items"]["properties"]["value"]["type"] == "string"

        # Should have note about removing null
        assert len(notes) == 1
        assert "removed 'null' from union type" in notes[0]
