"""
Schema adaptation service for LLMRing.

Handles provider-specific schema adaptations, including:
- Google Gemini JSON Schema normalization
- Structured output adaptation for providers without native support
- Schema validation
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from llmring.base import BaseLLMProvider
from llmring.providers.google_schema_normalizer import GoogleSchemaNormalizer
from llmring.schemas import LLMRequest, LLMResponse, Message

logger = logging.getLogger(__name__)


class SchemaAdapter:
    """
    Adapts schemas and structured outputs for provider-specific requirements.

    Different LLM providers have varying levels of support for structured outputs:
    - OpenAI: Native json_schema support
    - Anthropic: Uses tool-based approach
    - Google: Requires schema normalization and function declarations
    - Ollama: Best-effort with format hints
    """

    async def apply_structured_output_adapter(
        self, request: LLMRequest, provider_type: str, provider: BaseLLMProvider
    ) -> LLMRequest:
        """
        Apply structured output adapter for providers without native support.

        Converts json_schema requests to provider-specific approaches:
        - Anthropic: Tool injection with respond_with_structure
        - Google: Function declaration with normalized schema
        - Ollama: JSON mode with schema hints in system message

        Args:
            request: The original LLM request
            provider_type: Type of provider (anthropic, google, openai, ollama)
            provider: The provider instance

        Returns:
            Adapted request (or original if no adaptation needed)
        """
        # Only adapt if we have a json_schema request and no existing tools
        if (
            not request.response_format
            or request.response_format.get("type") != "json_schema"
            or request.tools
        ):
            return request

        schema = request.response_format.get("json_schema", {}).get("schema", {})
        if not schema:
            return request

        # Create a copy of the request to modify
        adapted_request = deepcopy(request)

        if provider_type == "anthropic":
            adapted_request = self._adapt_for_anthropic(adapted_request, schema)
        elif provider_type == "google":
            adapted_request = self._adapt_for_google(adapted_request, schema)
        elif provider_type == "ollama":
            adapted_request = self._adapt_for_ollama(adapted_request, schema)
        # OpenAI has native support, no adaptation needed

        # Mark request as adapted for post-processing
        adapted_request.metadata = adapted_request.metadata or {}
        adapted_request.metadata["_structured_output_adapted"] = True
        adapted_request.metadata["_original_schema"] = schema

        return adapted_request

    def _adapt_for_anthropic(self, request: LLMRequest, schema: Dict[str, Any]) -> LLMRequest:
        """Adapt structured output for Anthropic using tool injection."""
        respond_tool = {
            "type": "function",
            "function": {
                "name": "respond_with_structure",
                "description": "Respond with structured data matching the required schema",
                "parameters": schema,
            },
        }
        request.tools = [respond_tool]
        request.tool_choice = {"type": "any"}  # Force tool use
        return request

    def _adapt_for_google(self, request: LLMRequest, schema: Dict[str, Any]) -> LLMRequest:
        """Adapt structured output for Google using function declaration with normalized schema."""
        normalized_schema, notes = GoogleSchemaNormalizer.normalize(schema)

        if notes:
            try:
                logger.warning(
                    "Normalized JSON Schema for Google; potential downgrades: %s",
                    "; ".join(notes),
                )
            except Exception:
                # Avoid failing on logging issues
                pass

        respond_tool = {
            "type": "function",
            "function": {
                "name": "respond_with_structure",
                "description": "Respond with structured data matching the required schema",
                "parameters": normalized_schema,
            },
        }
        request.tools = [respond_tool]
        request.tool_choice = "any"  # Force function calling
        request.metadata = request.metadata or {}
        if notes:
            request.metadata["_schema_normalization_notes"] = notes

        return request

    def _adapt_for_ollama(self, request: LLMRequest, schema: Dict[str, Any]) -> LLMRequest:
        """Adapt structured output for Ollama using JSON mode and schema hints."""
        request.json_response = True

        # Add schema as system instruction
        schema_instruction = (
            f"\n\nIMPORTANT: Respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        # Add to system message or create one
        messages = list(request.messages)
        if messages and messages[0].role == "system":
            messages[0] = Message(
                role="system",
                content=messages[0].content + schema_instruction,
                metadata=messages[0].metadata,
            )
        else:
            messages.insert(
                0,
                Message(
                    role="system",
                    content=f"You are a helpful assistant.{schema_instruction}",
                ),
            )
        request.messages = messages

        return request

    async def post_process_structured_output(
        self, response: LLMResponse, request: LLMRequest, provider_type: str
    ) -> LLMResponse:
        """
        Post-process response from structured output adapter.

        Extracts JSON from tool calls (for Anthropic/Google) or parses from content,
        validates against schema if strict mode is enabled.

        Args:
            response: The LLM response
            request: The original request (with metadata)
            provider_type: Type of provider

        Returns:
            Response with parsed field populated
        """
        # Only process if request was adapted
        if not request.metadata or not request.metadata.get("_structured_output_adapted"):
            return response

        original_schema = request.metadata.get("_original_schema", {})

        try:
            if provider_type == "openai":
                # OpenAI native: Parse JSON from content
                try:
                    parsed_data = json.loads(response.content)
                    response.parsed = parsed_data

                    # Validate against schema if strict mode
                    if request.response_format and request.response_format.get("strict"):
                        self._validate_json_schema(parsed_data, original_schema)

                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from OpenAI response")

            elif provider_type in ["anthropic", "google"] and response.tool_calls:
                # Extract JSON from tool call arguments
                for tool_call in response.tool_calls:
                    if tool_call["function"]["name"] == "respond_with_structure":
                        # Parse the arguments as our structured response
                        tool_args = tool_call["function"]["arguments"]
                        if isinstance(tool_args, str):
                            parsed_data = json.loads(tool_args)
                        else:
                            parsed_data = tool_args

                        # Set content to JSON string and parsed to dict
                        response.content = json.dumps(parsed_data, indent=2)
                        response.parsed = parsed_data

                        # Validate against schema if strict mode
                        if request.response_format and request.response_format.get("strict"):
                            self._validate_json_schema(parsed_data, original_schema)

                        break

            elif provider_type == "ollama":
                # Try to parse JSON from content
                try:
                    parsed_data = json.loads(response.content)
                    response.parsed = parsed_data

                    # Validate against schema if strict mode
                    if request.response_format and request.response_format.get("strict"):
                        self._validate_json_schema(parsed_data, original_schema)

                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from Ollama response")

        except Exception as e:
            logger.warning(f"Failed to post-process structured output: {e}")

        return response

    @staticmethod
    def _validate_json_schema(data: Any, schema: Dict[str, Any]) -> None:
        """
        Validate data against JSON schema if jsonschema library is available.

        Args:
            data: The data to validate
            schema: The JSON schema

        Raises:
            ValueError: If validation fails
        """
        try:
            import jsonschema

            jsonschema.validate(instance=data, schema=schema)
        except ImportError:
            logger.warning(
                "jsonschema library not available - skipping validation. "
                "Install with: uv add llmring[validation]"
            )
        except Exception as e:
            raise ValueError(f"JSON Schema validation failed: {e}") from e
