"""LLM response formatter for structured outputs"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from jsonschema import Draft7Validator, ValidationError, validate

from ..models.schemas import LLMResponse

logger = logging.getLogger(__name__)


class LLMFormatter:
    """Handles formatting and validation of LLM responses"""

    def __init__(self):
        """Initialize the formatter"""
        self.validators = {}

    def validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against a JSON schema

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Returns:
            True if validation passes, False otherwise
        """
        try:
            validate(instance=data, schema=schema)
            return True
        except ValidationError as e:
            logger.error(f"JSON schema validation failed: {e}")
            return False

    def get_schema_errors(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Get detailed schema validation errors

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Returns:
            List of error messages
        """
        validator = Draft7Validator(schema)
        errors = []
        for error in validator.iter_errors(data):
            error_path = (
                " > ".join(str(x) for x in error.path) if error.path else "root"
            )
            errors.append(f"{error_path}: {error.message}")
        return errors

    def parse_json_response(
        self, content: str
    ) -> Union[Dict[str, Any], List[Any], None]:
        """Parse JSON from LLM response content

        Args:
            content: Response content from LLM

        Returns:
            Parsed JSON data or None if parsing fails
        """
        try:
            # First try direct JSON parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_blocks = self._extract_json_blocks(content)
            for block in json_blocks:
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

            # Try to extract JSON object from anywhere in text
            json_data = self._extract_json_object(content)
            if json_data:
                try:
                    return json.loads(json_data)
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON from response: {content[:200]}...")
            return None

    def format_json_response(
        self,
        llm_response: LLMResponse,
        schema: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Union[Dict[str, Any], List[Any], str, None]:
        """Format and optionally validate LLM response as JSON

        Args:
            llm_response: Response from LLM
            schema: Optional JSON schema to validate against
            strict: If True, return None on validation failure

        Returns:
            Formatted JSON data or original content if parsing fails
        """
        content = llm_response.content
        if not content:
            return None

        # Parse JSON from response
        json_data = self.parse_json_response(content)

        if json_data is None:
            if strict:
                return None
            return content

        # Validate against schema if provided
        if schema:
            if not self.validate_json_schema(json_data, schema):
                if strict:
                    errors = self.get_schema_errors(json_data, schema)
                    logger.error(f"Schema validation failed: {errors}")
                    return None
                logger.warning("Schema validation failed, returning data anyway")

        return json_data

    def create_schema_prompt(self, schema: Dict[str, Any]) -> str:
        """Create a prompt segment for JSON schema compliance

        Args:
            schema: JSON schema

        Returns:
            Prompt text to include in LLM request
        """
        return (
            "Your response MUST be valid JSON that conforms to this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            "Do not include any text outside the JSON response."
        )

    def _extract_json_blocks(self, content: str) -> List[str]:
        """Extract JSON from markdown code blocks

        Args:
            content: Text content containing code blocks

        Returns:
            List of JSON strings found in code blocks
        """
        json_blocks = []

        # Find all code blocks with optional language specifiers
        pattern = r"```(\w*)\s*\n(.*?)\n```"
        matches = re.findall(pattern, content, re.DOTALL)

        for lang, block_content in matches:
            if lang.lower() == "json":
                # Explicitly marked as JSON
                json_blocks.append(block_content)
            elif not lang:
                # No language specifier - check if it looks like JSON
                stripped = block_content.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    json_blocks.append(block_content)

        return json_blocks

    def _extract_json_object(self, content: str) -> Optional[str]:
        """Extract JSON object from text

        Args:
            content: Text content that may contain JSON

        Returns:
            JSON string if found, None otherwise
        """
        # Find the first occurrence of either '{' or '['
        first_obj_idx = content.find("{")
        first_arr_idx = content.find("[")

        candidates = []
        if first_obj_idx != -1:
            candidates.append((first_obj_idx, "{", "}"))
        if first_arr_idx != -1:
            candidates.append((first_arr_idx, "[", "]"))

        if not candidates:
            return None

        # Sort by position to find the first one
        candidates.sort(key=lambda x: x[0])
        first_idx, start, end = candidates[0]

        # Track nesting level
        level = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content[first_idx:], first_idx):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start:
                level += 1
            elif char == end:
                level -= 1
                if level == 0:
                    return content[first_idx : i + 1]

        return None

    def create_json_mode_request(
        self, messages: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create request parameters for JSON mode

        Args:
            messages: Chat messages
            schema: Optional JSON schema

        Returns:
            Request parameters with JSON mode enabled
        """
        request = {"messages": messages, "response_format": {"type": "json_object"}}

        # Add schema instructions to the last user message if schema provided
        if schema and messages:
            schema_prompt = self.create_schema_prompt(schema)
            last_message = messages[-1]
            if last_message["role"] == "user":
                messages[-1] = {
                    "role": "user",
                    "content": f"{last_message['content']}\n\n{schema_prompt}",
                }
            else:
                messages.append({"role": "user", "content": schema_prompt})

        return request
