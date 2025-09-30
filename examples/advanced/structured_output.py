"""
Structured output examples for LLMRing.

This example demonstrates:
- Using json_schema for structured output
- Provider-specific adaptation
- Validation and error handling
- Complex nested schemas
"""

import asyncio
from typing import List, Optional

from llmring import LLMRing
from llmring.schemas import LLMRequest, LLMResponse, Message


async def basic_structured_output():
    """Example 1: Basic structured output with json_schema."""
    llmring = LLMRing(origin="structured-output-example")

    # Define a JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "age": {"type": "number", "description": "Person's age"},
            "email": {"type": "string", "description": "Email address"},
        },
        "required": ["name", "age"],
    }

    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Extract structured data: John Doe, 30 years old, john@example.com",
            )
        ],
        model="openai:gpt-4",
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )

    response = await llmring.chat(request)

    # Parsed field contains structured data
    print(f"Parsed data: {response.parsed}")
    # {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}

    await llmring.close()


async def nested_schema():
    """Example 2: Complex nested schema."""
    llmring = LLMRing(origin="nested-schema-example")

    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["city", "country"],
                    },
                },
                "required": ["name", "address"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["user"],
    }

    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Extract: Alice Smith lives at 123 Main St, Paris, France. Tags: vip, premium",
            )
        ],
        model="anthropic:claude-3-sonnet-20240229",  # Works with Anthropic too
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )

    response = await llmring.chat(request)
    print(f"Structured output: {response.parsed}")

    await llmring.close()


async def validation_example():
    """Example 3: Schema validation with strict mode."""
    llmring = LLMRing(origin="validation-example")

    schema = {
        "type": "object",
        "properties": {
            "temperature": {"type": "number", "minimum": -273.15, "maximum": 1000},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]},
        },
        "required": ["temperature", "unit"],
    }

    request = LLMRequest(
        messages=[Message(role="user", content="The temperature is 25 degrees celsius")],
        model="openai:gpt-4",
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": schema},
            "strict": True,  # Enable validation
        },
    )

    try:
        response = await llmring.chat(request)
        print(f"Validated output: {response.parsed}")
    except ValueError as e:
        print(f"Validation failed: {e}")

    await llmring.close()


async def array_extraction():
    """Example 4: Extract arrays of structured data."""
    llmring = LLMRing(origin="array-example")

    schema = {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "in_stock": {"type": "boolean"},
                    },
                    "required": ["name", "price"],
                },
            }
        },
        "required": ["products"],
    }

    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Extract products: Laptop $999 in stock, Mouse $29 out of stock, Keyboard $79 in stock",
            )
        ],
        model="google:gemini-pro",  # Works with Google too
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )

    response = await llmring.chat(request)
    print(f"Products: {response.parsed}")

    await llmring.close()


async def error_handling():
    """Example 5: Handle structured output errors."""
    llmring = LLMRing(origin="error-handling-example")

    schema = {
        "type": "object",
        "properties": {"email": {"type": "string", "format": "email"}},
        "required": ["email"],
    }

    request = LLMRequest(
        messages=[Message(role="user", content="My email is not-an-email")],
        model="openai:gpt-4",
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )

    try:
        response = await llmring.chat(request)

        # Check if parsing succeeded
        if response.parsed:
            print(f"Parsed: {response.parsed}")
        else:
            print(f"Parsing failed, raw response: {response.content}")

    except Exception as e:
        print(f"Error: {e}")

    await llmring.close()


if __name__ == "__main__":
    print("Example 1: Basic structured output")
    asyncio.run(basic_structured_output())

    print("\nExample 2: Nested schema")
    asyncio.run(nested_schema())

    print("\nExample 3: Schema validation")
    asyncio.run(validation_example())

    print("\nExample 4: Array extraction")
    asyncio.run(array_extraction())

    print("\nExample 5: Error handling")
    asyncio.run(error_handling())
