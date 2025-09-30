# Unified Structured Output

LLMRing provides unified structured output across all providers using a single API that adapts to each provider's capabilities.

## Overview

The same `response_format` with JSON schema works across all providers:

- **OpenAI**: Native JSON schema support (pass-through)
- **Anthropic**: Automatic tool injection with schema validation
- **Google Gemini**: Function calling with schema enforcement, with JSON Schema normalization to Gemini-compatible subset
- **Ollama**: Best-effort prompt engineering with JSON parsing

## Basic Usage

```python
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

async with LLMRing() as service:
    # This works identically across ALL providers
    request = LLMRequest(
        model="balanced",  # or test, fast, local
        messages=[Message(role="user", content="Generate a person")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "age"]
                }
            },
            "strict": True
        }
    )

    response = await service.chat(request)

    # Access structured data
    print("JSON content:", response.content)        # JSON string
    print("Parsed object:", response.parsed)        # Python dict
```

## Response Fields

When using structured output, `LLMResponse` provides:

```python
response = await service.chat(request)

response.content    # str: Valid JSON string matching schema
response.parsed     # dict: Parsed Python object (None if parsing failed)
response.tool_calls # list: Tool calls used for adaptation (Anthropic/Google)
```

## Provider-Specific Behavior

### OpenAI (Native)

```python
# Uses OpenAI's native JSON schema support
request = LLMRequest(
    model="fast",
    messages=[Message(role="user", content="Generate user data")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False  # Supported by OpenAI
            }
        },
        "strict": True  # Enforced by OpenAI
    }
)

# Result: response.content = '{"name":"John","age":25}'
#         response.parsed = {"name":"John","age":25}
```

### Anthropic (Tool Injection)

```python
# Automatically converts to tool-based approach
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Generate user data")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        },
        "strict": True
    }
)

# Behind the scenes:
# 1. Injects a "respond_with_structure" tool with your schema
# 2. Forces tool use with tool_choice={"type": "any"}
# 3. Extracts JSON from tool call arguments
# 4. Sets response.content and response.parsed

# Result: response.content = '{\n  "name": "John",\n  "age": 25\n}'
#         response.parsed = {"name": "John", "age": 25}
#         response.tool_calls = [{"function": {"name": "respond_with_structure", ...}}]
```

### Google Gemini (Function Calling)

```python
# Automatically converts to function calling approach
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Generate user data")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
                # Note: additionalProperties not supported by Google
            }
        },
        "strict": True
    }
)

# Behind the scenes:
# 1. Creates FunctionDeclaration with your schema as parameters
# 2. Sets tool_config with mode="ANY" to force function calling
# 3. Extracts JSON from function call result
# 4. Sets response.content and response.parsed
# 5. Normalizes JSON Schema to Gemini-compatible subset when needed (see rules below)

# Result: response.content = '{\n  "age": 25,\n  "name": "John"\n}'
#         response.parsed = {"age": 25, "name": "John"}
#         response.tool_calls = [{"function": {"name": "respond_with_structure", ...}}]
```

### Ollama (Best Effort)

```python
# Uses prompt engineering with schema hints
request = LLMRequest(
    model="local",
    messages=[Message(role="user", content="Generate user data")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        },
        "strict": True  # Enables validation
    }
)

# Behind the scenes:
# 1. Adds schema to system prompt
# 2. Sets json_response=True for format="json"
# 3. Attempts to parse JSON from response
# 4. If strict=True and first parse fails, performs one automatic repair attempt
# 5. Validates against schema if strict=True

# Result: Best effort JSON parsing and validation
```

## Advanced Features

### Streaming with Structured Output

```python
# Streaming works with structured output
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Generate multiple users")],
    response_format={...}  # Same schema
)

async for chunk in service.chat_stream(request):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.tool_calls:
        print(f"\nStructured data: {chunk.tool_calls}")
```

### Error Handling

```python
from llmring.exceptions import ValidationError

try:
    response = await service.chat(request)

    if response.parsed:
        # Successfully parsed structured data
        user_data = response.parsed
        print(f"Name: {user_data['name']}, Age: {user_data['age']}")
    else:
        # Parsing failed but content available
        print("Raw response:", response.content)

except ValidationError as e:
    print(f"Schema validation failed: {e}")
```

### Schema Validation

Optional JSON schema validation (requires `pip install jsonschema`):

```python
# With jsonschema installed, strict=True enables validation
request = LLMRequest(
    model="any-provider",
    response_format={
        "type": "json_schema",
        "json_schema": {"schema": {...}},
        "strict": True  # Validates response against schema
    }
)

# Raises ValidationError if response doesn't match schema
```

## Best Practices

### Schema Design

```python
# Good: Simple, clear schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"}
    },
    "required": ["name", "age"]
}

# Avoid: Complex nested structures for better cross-provider compatibility
```

### Provider Selection

- **OpenAI**: Best for complex schemas with strict validation
- **Anthropic**: Excellent reliability, good for business logic
- **Google**: Good performance, handles most schemas well
- **Ollama**: Best effort, good for simple local use cases

### Performance Tips

- Use specific, descriptive property names
- Keep schemas reasonably simple for better reliability
- Test with your target providers to ensure compatibility
- Consider fallback handling for best-effort providers

## Examples

### E-commerce Product

```python
product_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "category": {"type": "string"},
        "in_stock": {"type": "boolean"}
    },
    "required": ["name", "price", "category"]
}

request = LLMRequest(
    model="fast",  # Works with any provider
    messages=[Message(role="user", content="Create a laptop product")],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "product", "schema": product_schema}
    }
)
```

### Data Extraction

```python
extraction_schema = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"}
                }
            }
        }
    },
    "required": ["companies"]
}

request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Extract company info from this text: ...")],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": extraction_schema}
    }
)
```

## Limitations

### Google Gemini
- JSON Schema is normalized automatically to Gemini's parameter schema:
  - Removes unsupported keywords: `additionalProperties`, `oneOf`, `anyOf`, `allOf`, `not`, `patternProperties`, `pattern`, `format`, `dependencies`, conditional `if/then/else`
  - Union types are reduced:
    - `['boolean','null']` → `boolean` (null removed)
    - Multi-type unions (e.g. `['string','number']`) → `string`
  - Array tuples (items as list) → first item only
  - Notes about downgrades are logged and attached to `request.metadata['_schema_normalization_notes']`
- Validation (when `strict=True`) still checks the original schema after extraction. If normalization changed semantics (e.g. nullability removed), ensure your schema avoids such constructs for cross-provider consistency.
- Complex nested schemas may still be unreliable; prefer simpler shapes.

### Ollama
- Best effort only - no guarantees
- Depends on model's JSON generation capability
- May include schema description in output
- With `strict=True`, one repair attempt is made on invalid JSON

### All Providers
- Very complex schemas may reduce reliability
- Streaming provides final structured data in last chunk
- Validation requires `jsonschema` package

This unified approach makes structured output a first-class feature across all LLM providers in LLMRing!

## Google Schema Normalization Rules (Details)

- **Types**
  - Union with `null`: remove `null` and keep the non-null type
  - Multi-type (non-null) unions: fallback to `string`
- **Objects**
  - `properties`: recursively normalized
  - `required`: preserved as-is
  - `additionalProperties`: removed
- **Arrays**
  - `items` as list (tuple typing): first schema is used
  - `items` as object: recursively normalized
- **Removed keywords**: `anyOf`, `oneOf`, `allOf`, `not`, `patternProperties`, `pattern`, `format`, `dependencies`, `if/then/else`
- **Diagnostics**
  - Normalization warnings are logged
  - Notes are added to `request.metadata['_schema_normalization_notes']`
