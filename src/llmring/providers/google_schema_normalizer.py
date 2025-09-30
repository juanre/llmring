"""
Google Gemini JSON Schema normalization utilities.

Google Gemini has specific requirements and limitations for function declaration schemas:
- No union types (except nullable)
- No complex composition (anyOf, oneOf, allOf, not)
- Limited keyword support

This module provides normalization to convert standard JSON Schemas to Google-compatible format.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class GoogleSchemaNormalizer:
    """
    Normalizes JSON Schemas for Google Gemini function declarations.

    Google Gemini function declarations require simplified schemas that don't use
    advanced JSON Schema features like union types, schema composition, or certain
    validation keywords.
    """

    # Supported basic keywords that can be copied as-is
    SUPPORTED_KEYWORDS = {
        "title",
        "description",
        "default",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }

    # Unsupported keywords that must be removed
    UNSUPPORTED_KEYWORDS = {
        "additionalProperties",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "patternProperties",
        "if",
        "then",
        "else",
        "pattern",
        "format",
        "dependencies",
    }

    @classmethod
    def normalize(cls, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Normalize a JSON Schema for Google Gemini function declarations.

        This method:
        - Converts union types to single types (removes null, falls back to string)
        - Removes unsupported keywords (additionalProperties, anyOf, oneOf, etc.)
        - Recursively normalizes nested properties and items
        - Records all modifications as notes

        Args:
            schema: Original JSON Schema

        Returns:
            Tuple of (normalized_schema, list_of_notes)
        """
        notes: List[str] = []

        def normalize_node(node: Any, path: str) -> Any:
            """Recursively normalize a schema node."""
            # Primitives or non-dict structures are returned as-is
            if not isinstance(node, dict):
                return node

            result: Dict[str, Any] = {}

            # Handle type normalization first
            node_type = node.get("type")
            if isinstance(node_type, list):
                # Remove null if present, pick a remaining type
                non_null_types = [t for t in node_type if t != "null"]
                if len(non_null_types) == 1:
                    result["type"] = non_null_types[0]
                    notes.append(f"{path or '<root>'}: removed 'null' from union type {node_type}")
                elif len(non_null_types) == 0:
                    # Only null provided; fallback to string
                    result["type"] = "string"
                    notes.append(
                        f"{path or '<root>'}: union type {node_type} normalized to 'string'"
                    )
                else:
                    # Multiple non-null types unsupported; fallback to string
                    result["type"] = "string"
                    notes.append(
                        f"{path or '<root>'}: multi-type union {node_type} normalized to 'string'"
                    )
            elif isinstance(node_type, str):
                result["type"] = node_type

            # Copy supported basic fields
            for key in cls.SUPPORTED_KEYWORDS:
                if key in node:
                    result[key] = node[key]

            # Remove/ignore unsupported or risky keywords
            removed_keywords = []
            for key in cls.UNSUPPORTED_KEYWORDS:
                if key in node:
                    removed_keywords.append(key)
            if removed_keywords:
                notes.append(f"{path or '<root>'}: removed unsupported keywords {removed_keywords}")

            # Object handling
            effective_type = result.get("type") or node.get("type")
            if effective_type == "object":
                # Normalize properties
                properties = node.get("properties", {})
                if isinstance(properties, dict):
                    norm_props: Dict[str, Any] = {}
                    for prop_name, prop_schema in properties.items():
                        norm_props[prop_name] = normalize_node(
                            prop_schema,
                            f"{path + '.' if path else ''}properties.{prop_name}",
                        )
                    result["properties"] = norm_props

                # Keep required list as-is
                if "required" in node and isinstance(node["required"], list):
                    result["required"] = [str(x) for x in node["required"]]

            # Array handling
            if effective_type == "array":
                items = node.get("items")
                if isinstance(items, list) and items:
                    # Tuple typing not supported; choose first
                    result["items"] = normalize_node(items[0], f"{path or '<root>'}.items[0]")
                    notes.append(
                        f"{path or '<root>'}: tuple-typed 'items' normalized to first schema"
                    )
                elif isinstance(items, dict):
                    result["items"] = normalize_node(items, f"{path or '<root>'}.items")

            return result

        normalized = normalize_node(schema, "")
        return normalized, notes
