# Provider Layer Documentation

**Last Updated**: 2025-09-30

## Table of Contents

1. [Overview](#overview)
2. [Provider Protocol](#provider-protocol)
3. [AnthropicProvider](#anthropicprovider)
4. [OpenAIProvider](#openaiprovider)
5. [GoogleProvider](#googleprovider)
6. [OllamaProvider](#ollamaprovider)
7. [Provider Error Handling](#provider-error-handling)
8. [Adding a New Provider](#adding-a-new-provider)

---

## Overview

The provider layer implements the interface between LLMRing and external LLM APIs. Each provider translates the unified `LLMRequest` format to provider-specific API calls and converts responses back to the unified `LLMResponse` format.

### Design Principles

1. **Protocol-Based**: Providers implement `BaseLLMProvider` protocol, no inheritance required
2. **Provider-Specific**: Each provider handles quirks of its API
3. **Error Translation**: Provider errors mapped to common exception types
4. **Streaming Support**: All providers support both streaming and non-streaming
5. **Type Safety**: Full type annotations, no `Any` in public interfaces

### Provider Location

All providers are in `src/llmring/providers/`:

```
src/llmring/providers/
├── __init__.py
├── anthropic_api.py          # Anthropic Claude
├── openai_api.py             # OpenAI GPT
├── google_api.py             # Google Gemini
├── ollama_api.py             # Ollama (local models)
├── error_handler.py          # Unified error handling
└── google_schema_normalizer.py  # Google schema normalization
```

---

## Provider Protocol

**File**: `src/llmring/base.py`

All providers implement the `BaseLLMProvider` protocol:

```python
from typing import Protocol, AsyncIterator

class BaseLLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        Execute a non-streaming chat request.

        Args:
            request: The LLM request

        Returns:
            The LLM response

        Raises:
            ProviderError: If the API call fails
        """
        ...

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """
        Execute a streaming chat request.

        Args:
            request: The LLM request

        Yields:
            Stream chunks

        Raises:
            ProviderError: If the API call fails
        """
        ...
```

### Why Protocol?

Using a protocol instead of an abstract base class provides:
- **Duck typing**: No need to inherit from a base class
- **Flexibility**: Providers can have additional methods
- **Type checking**: MyPy/PyRight can verify implementation
- **Testing**: Easy to create test doubles

---

## AnthropicProvider

**File**: `src/llmring/providers/anthropic_api.py`

### API Details

- **SDK**: `anthropic` (official Python SDK)
- **Models**: Claude 3 family (Opus, Sonnet, Haiku)
- **Special Features**:
  - Tool calling (beta)
  - PDF support (beta)
  - Prompt caching (beta)

### Message Format

Anthropic uses a specific message format:

```python
# Input: LLMRing format
Message(role="user", content="Hello")

# Converted to: Anthropic format
{
    "role": "user",
    "content": [{"type": "text", "text": "Hello"}]
}
```

### System Messages

Anthropic requires system messages to be separate from the message list:

```python
# If first message is system
if messages[0]["role"] == "system":
    system_content = messages[0]["content"]
    messages = messages[1:]

# Pass to API
response = await client.messages.create(
    system=system_content,
    messages=messages,
    ...
)
```

### Tool Calling

Anthropic tool calling is in beta and requires a header:

```python
client = anthropic.AsyncAnthropic(
    api_key=api_key,
    default_headers={"anthropic-beta": "tools-2024-04-04"}
)
```

### PDF Support

Anthropic supports PDF documents as base64-encoded content:

```python
{
    "type": "document",
    "source": {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "<base64-encoded PDF>"
    }
}
```

**Size Limit**: Validated by `InputValidator` (50MB encoded, 100MB decoded)

### Streaming

Anthropic uses server-sent events (SSE) for streaming:

```python
async with client.messages.stream(...) as stream:
    async for chunk in stream:
        if chunk.type == "content_block_delta":
            yield StreamChunk(
                content=chunk.delta.text,
                role="assistant",
                ...
            )
```

### Error Handling

Common Anthropic errors:
- `anthropic.RateLimitError` → `ProviderRateLimitError`
- `anthropic.AuthenticationError` → `ProviderAuthenticationError`
- `anthropic.APITimeoutError` → `ProviderTimeoutError`
- `anthropic.APIError` → `ProviderResponseError`

---

## OpenAIProvider

**File**: `src/llmring/providers/openai_api.py`

### API Details

- **SDK**: `openai` (official Python SDK)
- **Models**: GPT-4, GPT-3.5, o1 series
- **Special Features**:
  - Native `json_schema` support
  - PDF file upload
  - Vision (GPT-4V)
  - Function calling

### Message Format

OpenAI uses a straightforward message format:

```python
{
    "role": "user",
    "content": "Hello"
}
```

For multimodal content:

```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
    ]
}
```

### o1 Model Restrictions

The o1 series has restrictions:
- No system messages
- No streaming support
- No temperature control
- No top_p control

These are validated before the API call:

```python
if model.startswith("o1-"):
    if kwargs.get("temperature") or kwargs.get("top_p"):
        raise ProviderResponseError(
            "o1 models don't support temperature/top_p"
        )
```

### PDF Document Handling

OpenAI requires PDF files to be uploaded first, then referenced:

```python
# 1. Upload PDF
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(pdf_data)
    tmp.flush()
    tmp_path = tmp.name

try:
    with open(tmp_path, "rb") as f:
        file_obj = await self.client.files.create(
            file=f,
            purpose="assistants"
        )
finally:
    os.unlink(tmp_path)

# 2. Reference in message
{
    "type": "file",
    "file_id": file_obj.id
}
```

**Why the temp file dance?** OpenAI SDK expects a file handle, not raw bytes. We close the NamedTemporaryFile and reopen it in 'rb' mode to avoid race conditions.

### Structured Output

OpenAI has native `json_schema` support:

```python
response = await client.chat.completions.create(
    messages=messages,
    model=model,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {...}
        }
    }
)
```

No adaptation needed (unlike Anthropic/Google).

### Streaming

OpenAI streaming yields deltas:

```python
async for chunk in await client.chat.completions.create(
    messages=messages,
    model=model,
    stream=True
):
    delta = chunk.choices[0].delta
    if delta.content:
        yield StreamChunk(
            content=delta.content,
            role="assistant",
            ...
        )
```

### Error Handling

Common OpenAI errors:
- `openai.RateLimitError` → `ProviderRateLimitError`
- `openai.AuthenticationError` → `ProviderAuthenticationError`
- `openai.Timeout` → `ProviderTimeoutError`
- `openai.APIError` → `ProviderResponseError`

---

## GoogleProvider

**File**: `src/llmring/providers/google_api.py`

### API Details

- **SDK**: `google-generativeai` (official Python SDK)
- **Models**: Gemini family (Pro, Flash)
- **Special Features**:
  - Vision support
  - Function calling
  - Safety settings

### Message Format

Google uses a different role structure:

```python
# LLMRing "user" → Google "user"
# LLMRing "assistant" → Google "model"
# LLMRing "system" → Converted to user message with prefix
```

Google doesn't have a native system role, so we convert:

```python
if role == "system":
    content = f"[System instruction: {content}]"
    role = "user"
```

### Content Conversion

Google requires inline data for images:

```python
{
    "mime_type": "image/jpeg",
    "data": "<base64-data>"
}
```

For documents:

```python
{
    "mime_type": "application/pdf",
    "data": "<base64-data>"
}
```

### Schema Normalization

Google has strict schema requirements. We use `GoogleSchemaNormalizer` to:

- Remove union types: `["string", "null"]` → `"string"`
- Remove unsupported keywords: `additionalProperties`, `anyOf`, `oneOf`
- Flatten tuple arrays: Take first schema only

See `src/llmring/providers/google_schema_normalizer.py` for details.

### Function Calling

Google uses "function declarations":

```python
tools = [
    genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            )
        ]
    )
]

response = model.generate_content(
    contents=messages,
    tools=tools,
    tool_config={"function_calling_config": {"mode": "any"}}
)
```

### Streaming Quirk

Google's SDK uses thread-based streaming (not native async):

```python
# Run blocking streaming in thread
response_stream = await asyncio.to_thread(
    model.generate_content,
    contents=messages,
    stream=True,
    generation_config=generation_config
)

# Iterate in thread
for chunk in response_stream:
    yield StreamChunk(...)
```

**Why?** The `google-generativeai` SDK doesn't have native async support for streaming.

### Safety Settings

Google has built-in safety filters that can be configured:

```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    # ...
}
```

### Error Handling

Common Google errors:
- Rate limits → `ProviderRateLimitError`
- Invalid API key → `ProviderAuthenticationError`
- Model not found → `ModelNotFoundError`
- Other API errors → `ProviderResponseError`

Google errors are extracted from various exception types:

```python
def _extract_error_message(self, error: Exception) -> str:
    """Extract meaningful error message from Google exception."""
    if hasattr(error, "message"):
        return error.message
    if hasattr(error, "details"):
        return error.details
    return str(error)
```

---

## OllamaProvider

**File**: `src/llmring/providers/ollama_api.py`

### API Details

- **Connection**: HTTP API (not official SDK)
- **Models**: Any local Ollama model
- **Special Features**:
  - Local execution
  - Custom models
  - Vision support (llava)
  - Function calling (experimental)

### Message Format

Ollama follows OpenAI-style messages:

```python
{
    "role": "user",
    "content": "Hello"
}
```

For images:

```python
{
    "role": "user",
    "content": "What's in this image?",
    "images": ["<base64-data>"]
}
```

### API Endpoint

Ollama runs locally (default: `http://localhost:11434`):

```python
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

### JSON Mode

Ollama supports JSON mode:

```python
response = requests.post(
    f"{base_url}/api/chat",
    json={
        "model": model,
        "messages": messages,
        "format": "json"  # Force JSON output
    }
)
```

### Streaming

Ollama uses newline-delimited JSON (NDJSON) for streaming:

```python
response = requests.post(
    f"{base_url}/api/chat",
    json={...},
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk_data = json.loads(line)
        if "message" in chunk_data:
            yield StreamChunk(
                content=chunk_data["message"]["content"],
                ...
            )
```

### Tool Calling (Experimental)

Ollama has experimental tool calling support:

```python
response = requests.post(
    f"{base_url}/api/chat",
    json={
        "model": model,
        "messages": messages,
        "tools": tools  # OpenAI-style tools
    }
)
```

**Note**: Quality varies by model. Best with `mistral` and `llama3.1`.

### Error Handling

Common Ollama errors:
- Connection refused → `ProviderResponseError` ("Ollama server not running?")
- Model not found → `ModelNotFoundError`
- Timeout → `ProviderTimeoutError`

---

## Provider Error Handling

**File**: `src/llmring/providers/error_handler.py`

### Purpose

Unified error handling across all providers. Maps provider-specific errors to common exception types.

### Error Types

```python
from llmring.exceptions import (
    ProviderError,                 # Base class
    ProviderTimeoutError,          # Timeout during API call
    ProviderRateLimitError,        # Rate limit exceeded
    ProviderAuthenticationError,   # Invalid API key
    ModelNotFoundError,            # Model not available
    ProviderResponseError,         # Other API errors
)
```

### Usage

All providers use `ProviderErrorHandler`:

```python
class AnthropicProvider:
    def __init__(self, ...):
        self._error_handler = ProviderErrorHandler(
            provider_name="anthropic",
            circuit_breaker=self._circuit_breaker
        )

    async def chat(self, request):
        try:
            # API call
            response = await self.client.messages.create(...)
        except Exception as e:
            # Unified error handling
            await self._error_handler.handle_error(e, model)
```

### Error Detection

The error handler detects error types from:

1. **Exception type**:
   - `TimeoutError` → `ProviderTimeoutError`
   - `anthropic.RateLimitError` → `ProviderRateLimitError`

2. **Error message patterns**:
   - "rate limit" in message → `ProviderRateLimitError`
   - "authentication" in message → `ProviderAuthenticationError`
   - "not found" in message → `ModelNotFoundError`

3. **HTTP status codes** (for Ollama):
   - 429 → `ProviderRateLimitError`
   - 401/403 → `ProviderAuthenticationError`
   - 404 → `ModelNotFoundError`

### RetryError Unwrapping

The `tenacity` library wraps exceptions in `RetryError`. The error handler unwraps them:

```python
if isinstance(error, tenacity.RetryError):
    # Extract the original exception
    if error.last_attempt and error.last_attempt.exception():
        original_error = error.last_attempt.exception()
        # Detect error type from original
```

### Circuit Breaker Integration

The error handler works with the circuit breaker pattern:

```python
async def handle_error(self, error, model):
    # Detect error type
    error_type = self._detect_error_type(error)

    # Raise appropriate exception
    if error_type == "timeout":
        raise ProviderTimeoutError(f"Timeout calling {model}: {error}")

    # Circuit breaker will track failures
```

---

## Adding a New Provider

### Step 1: Implement the Protocol

Create a new file in `src/llmring/providers/`:

```python
# src/llmring/providers/my_provider_api.py

from typing import AsyncIterator
from llmring.schemas import LLMRequest, LLMResponse, StreamChunk
from llmring.providers.error_handler import ProviderErrorHandler

class MyProvider:
    """Provider for MyLLM API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._error_handler = ProviderErrorHandler(
            provider_name="myprovider",
            circuit_breaker=None  # Optional
        )

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming chat request."""
        try:
            # 1. Prepare messages
            messages = self._prepare_messages(request.messages)

            # 2. Call API
            response = await self._api_client.chat(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # 3. Parse response
            return LLMResponse(
                content=response.content,
                model=response.model,
                usage=response.usage,
                finish_reason=response.finish_reason,
            )

        except Exception as e:
            await self._error_handler.handle_error(e, request.model)

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat request."""
        try:
            # 1. Prepare messages
            messages = self._prepare_messages(request.messages)

            # 2. Stream API call
            async for chunk in self._api_client.chat_stream(
                model=request.model,
                messages=messages,
            ):
                # 3. Yield chunks
                yield StreamChunk(
                    content=chunk.content,
                    role="assistant",
                    finish_reason=chunk.finish_reason,
                )

        except Exception as e:
            await self._error_handler.handle_error(e, request.model)

    def _prepare_messages(self, messages):
        """Convert LLMRing messages to provider format."""
        # Implement message conversion
        pass
```

### Step 2: Register the Provider

Add to `LLMRing` in `src/llmring/service.py`:

```python
PROVIDER_CLASSES = {
    "anthropic": "llmring.providers.anthropic_api:AnthropicProvider",
    "openai": "llmring.providers.openai_api:OpenAIProvider",
    "google": "llmring.providers.google_api:GoogleProvider",
    "ollama": "llmring.providers.ollama_api:OllamaProvider",
    "myprovider": "llmring.providers.my_provider_api:MyProvider",  # Add here
}
```

### Step 3: Add Error Detection

Update `ProviderErrorHandler._detect_error_type()`:

```python
def _detect_error_type(self, error, provider_name):
    # Add provider-specific error detection
    if provider_name == "myprovider":
        if isinstance(error, MyProviderRateLimitError):
            return "rate_limit"
        if isinstance(error, MyProviderAuthError):
            return "authentication"
        # ...
```

### Step 4: Add Tests

Create `tests/integration/providers/test_myprovider.py`:

```python
import os
import pytest
from llmring import LLMRing
from llmring.schemas import LLMRequest, Message

@pytest.mark.skipif(
    not os.getenv("MYPROVIDER_API_KEY"),
    reason="MYPROVIDER_API_KEY not set"
)
class TestMyProviderIntegration:
    @pytest.fixture
    def llmring(self):
        return LLMRing(origin="test")

    @pytest.mark.asyncio
    async def test_basic_chat(self, llmring):
        request = LLMRequest(
            messages=[Message(role="user", content="Say hello")],
            model="myprovider:model-name",
        )

        response = await llmring.chat(request)

        assert response.content
        assert response.usage
        assert response.model
```

### Step 5: Document

Add documentation:
- Update this file with provider-specific details
- Add example to `README.md`
- Document any quirks or limitations

---

## Provider Comparison

| Feature | Anthropic | OpenAI | Google | Ollama |
|---------|-----------|--------|--------|--------|
| **Streaming** | SSE | SSE | Thread-based | NDJSON |
| **System Messages** | Separate param | In messages | Convert to user | In messages |
| **Vision** | Yes (beta) | Yes (GPT-4V) | Yes | Yes (llava) |
| **PDF Support** | Base64 inline | File upload | Base64 inline | Not supported |
| **Tool Calling** | Yes (beta) | Yes | Yes | Experimental |
| **JSON Schema** | Tool injection | Native | Tool injection | Schema in prompt |
| **Max Tokens** | Required | Optional | Optional | Optional |
| **SDK** | Async | Async | Sync (threading) | HTTP |

---

## Best Practices

### 1. Always Use Error Handler

Don't handle errors directly - use `ProviderErrorHandler`:

```python
# Bad
try:
    response = await api_call()
except SomeError as e:
    raise ProviderResponseError(str(e))

# Good
try:
    response = await api_call()
except Exception as e:
    await self._error_handler.handle_error(e, model)
```

### 2. Validate Provider-Specific Restrictions

Check for provider limitations before calling the API:

```python
# OpenAI o1 models don't support temperature
if model.startswith("o1-") and request.temperature:
    raise ProviderResponseError(
        "o1 models don't support temperature parameter"
    )
```

### 3. Strip Provider Prefix

Providers should not know about `provider:model` format. Strip the prefix:

```python
# In provider implementation
if model.startswith("openai:"):
    model = model.split(":", 1)[1]
```

### 4. Handle Missing Features Gracefully

Not all providers support all features:

```python
# Ollama doesn't support max_tokens
if hasattr(request, "max_tokens") and request.max_tokens:
    logger.warning("Ollama doesn't support max_tokens, ignoring")
```

### 5. Document Quirks

Use docstrings and comments to explain provider-specific behavior:

```python
def _prepare_messages(self, messages):
    """
    Convert messages to Anthropic format.

    Anthropic requires:
    - System messages separate from message list
    - Content as list of content blocks
    - Tool use results as separate messages
    """
```

---

## Related Documentation

- [Architecture Overview](./overview.md)
- [Service Layer Documentation](./services.md)
- [Architecture Decision Records](../decisions/)
