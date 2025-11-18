# LLMRing API Reference

## Core Classes

### LLMRing

Main service class for LLM interactions.

```python
from llmring.service import LLMRing

# Using context manager (recommended for automatic cleanup)
async with LLMRing(
    origin="myapp",                    # Optional: Origin identifier
    registry_url=None,                 # Optional: Custom registry URL
    lockfile_path="/path/to/custom.lock",  # Optional: Explicit lockfile path
    timeout=120.0                      # Optional: Default request timeout
) as service:
    # Use service here
    response = await service.chat(request)

# If you omit lockfile_path, resolution order is:
# 1. LLMRING_LOCKFILE_PATH environment variable (file must exist)
# 2. ./llmring.lock in the current directory (if present)
# 3. Bundled fallback at src/llmring/llmring.lock

# Or manual resource management
service = LLMRing()
try:
    response = await service.chat(request)
finally:
    await service.close()  # Ensure resources are cleaned up
```

The `timeout` parameter controls how long LLMRing waits for provider responses. It defaults to 60 seconds, honors the `LLMRING_PROVIDER_TIMEOUT_S` environment value when unspecified, and can be set to `None` to disable the timer entirely.

**Methods:**
- `async chat(request: LLMRequest, profile: str | None = None) -> LLMResponse`
- `async chat_stream(request: LLMRequest, profile: str | None = None) -> AsyncIterator[StreamChunk]`
- `get_provider(name: str) -> BaseLLMProvider`
- `resolve_alias(alias_or_model: str, profile: str | None = None) -> str`
- `calculate_cost(response: LLMResponse) -> Dict[str, float]`
- `async close() -> None` - Clean up resources
- `async __aenter__() -> LLMRing` - Context manager entry
- `async __aexit__(...) -> None` - Context manager exit with automatic cleanup

### LLMRequest

Request schema for LLM interactions.

```python
from llmring.schemas import LLMRequest, Message

request = LLMRequest(
    model="fast",                    # Required: Model identifier or alias
    messages=[Message(...)],                 # Required: Conversation messages
    temperature=0.7,                         # Optional: 0.0-1.0
    max_tokens=1000,                         # Optional: Response length limit
    response_format={"type": "json_object"}, # Optional: Response format
    tools=[...],                             # Optional: Available functions
    tool_choice="auto",                      # Optional: Tool usage mode
    timeout=None,                            # Optional: Override or disable timeout
    extra_params={}                          # Optional: Provider-specific params
)
```
> Note: streaming is handled via `LLMRing.chat_stream(...)`, not by passing a `stream` flag on the request.

`timeout` inherits the service default when omitted and can be set to `None` to wait indefinitely for a single request.

### LLMResponse

Response schema from LLM providers.

```python
response = await service.chat(request)

# Available fields:
response.content          # str: Generated text content
response.model           # str: Model that generated response
response.usage           # Dict: Token usage and costs
response.finish_reason   # str: Why generation stopped
response.tool_calls      # List[Dict]: Function calls if any
response.parsed          # Any: Parsed structured output (if response_format used)
```

### File Registration

LLMRing provides a provider-agnostic file registration system. Files are registered once and uploaded lazily on first use with each provider.

**Methods:**

```python
# Register a file (no upload yet)
file_id = service.register_file("data.csv")

# List all registered files
files = service.list_registered_files()

# Deregister file (removes registration and all provider uploads)
await service.deregister_file(file_id)
```

**Key Concepts:**
- **Provider-agnostic**: Register once, use with any provider
- **Lazy uploads**: Upload happens on first use per provider, not at registration
- **Staleness detection**: Files are re-hashed before each use to detect changes
- **Cross-provider tracking**: Each provider maintains its own upload

See [File Registration Documentation](file-uploads.md) for complete details.

## Provider-Specific Features

### OpenAI
- JSON schema with strict mode
- o1 models support (temperature filtering)
- PDF processing with file upload
- Advanced parameters via extra_params

### Anthropic
- Prompt caching (90% cost savings)
- Large context windows (200K+ tokens)
- Streaming with tool calls
- Cache usage tracking

### Google Gemini
- Real streaming via native SDK
- Native function calling (improved tool support)
- Multimodal content (text, images)
- 2M+ token context windows

### Ollama
- Local model execution
- Real streaming
- Custom model options
- No API key required

## Error Handling

Comprehensive typed exceptions for better error handling:

```python
from llmring.exceptions import (
    ProviderAuthenticationError,    # Invalid API key
    ProviderRateLimitError,        # Rate limit exceeded
    ModelNotFoundError,            # Invalid model
    ProviderResponseError,         # API error
    ProviderTimeoutError,          # Request timeout
    CircuitBreakerError           # Service unavailable
)
```

## Extended Classes

### LLMRingExtended

Extended LLM service with conversation tracking and server integration support.

```python
from llmring import LLMRingExtended

service = LLMRingExtended(
    origin="myapp",
    registry_url=None,
    lockfile_path=None,
    server_url="https://api.llmring.ai",  # Optional: llmring-server URL
    api_key="your-api-key",                # Optional: API key for server
    enable_conversations=True,             # Enable conversation tracking
    message_logging_level="full"           # Options: none, metadata, full
)
```

**Additional Methods:**
- `async create_conversation(title, system_prompt, model_alias, temperature, max_tokens) -> UUID`
  - Create a new conversation with server tracking
- `async chat_with_conversation(request, conversation_id, store_messages, profile) -> LLMResponse`
  - Chat with automatic message storage in conversation
- `async get_conversation_history(conversation_id, limit) -> Dict`
  - Retrieve conversation history with messages
- `async list_conversations(limit, offset) -> List[Dict]`
  - List all conversations for authenticated user

**Use Cases:**
- Applications requiring conversation persistence
- Multi-user chat systems
- Analytics and monitoring
- Audit logging and compliance

**Example:**

```python
from llmring import LLMRingExtended, LLMRequest, Message

async with LLMRingExtended(
    server_url="https://api.llmring.ai",
    api_key="your-key",
    enable_conversations=True
) as service:
    # Create a conversation
    conv_id = await service.create_conversation(
        title="Customer Support Chat",
        system_prompt="You are a helpful support assistant",
        model_alias="balanced"
    )

    # Chat with conversation tracking
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Hello!")]
    )
    response = await service.chat_with_conversation(
        request=request,
        conversation_id=conv_id
    )

    # Retrieve history
    history = await service.get_conversation_history(conv_id)
    print(f"Messages in conversation: {len(history['messages'])}")
```

---

### ConversationManager

Helper class for managing conversations with simplified API.

```python
from llmring import LLMRingExtended, ConversationManager

service = LLMRingExtended(
    server_url="https://api.llmring.ai",
    api_key="your-key",
    enable_conversations=True
)

manager = ConversationManager(service)
```

**Methods:**
- `async start_conversation(title, system_prompt, model_alias) -> UUID`
  - Start a new conversation and set it as current
- `async send_message(content, model, temperature, max_tokens) -> LLMResponse`
  - Send a message in the current conversation
- `async load_conversation(conversation_id) -> bool`
  - Load an existing conversation and restore history
- `clear_history() -> None`
  - Clear local message history (keeps conversation ID)
- `get_history() -> List[Dict]`
  - Get current message history

**Use Cases:**
- Simplifying multi-turn conversations
- Building chat interfaces
- Managing conversation state
- Session persistence

**Example:**

```python
from llmring import LLMRingExtended, ConversationManager

async def chat_session():
    service = LLMRingExtended(
        server_url="https://api.llmring.ai",
        api_key="your-key",
        enable_conversations=True
    )

    manager = ConversationManager(service)

    # Start new conversation
    conv_id = await manager.start_conversation(
        title="Code Review",
        system_prompt="You are an expert code reviewer",
        model_alias="balanced"
    )

    # Send messages
    response = await manager.send_message("Review this code...")
    print(response.content)

    response = await manager.send_message("What about performance?")
    print(response.content)

    # Get full history
    history = manager.get_history()
    print(f"Conversation has {len(history)} messages")
```

---

## Alias Resolution Cache

LLMRing includes an in-memory cache for alias resolution to improve performance.

**Configuration:**

```python
from llmring import LLMRing

service = LLMRing(
    # Cache is automatically configured
    # Default: 1000 entries, 1 hour TTL
)
```

**Cache Behavior:**
- Aliases are resolved once and cached
- Cache expires after 1 hour (configurable in code)
- Maximum 1000 cached entries
- Thread-safe for concurrent access
- Automatically cleared on lockfile updates

**When Cache is Useful:**
- High-frequency API calls with same aliases
- Reducing lockfile I/O operations
- Improving response time
- Multi-threaded applications

**Cache Statistics:**

The cache uses an LRU (Least Recently Used) eviction policy when the maximum size is reached.

---

## File Utilities

LLMRing exports comprehensive file handling utilities for vision and multimodal capabilities.

See [File Utilities Documentation](file-utilities.md) for complete details on:
- `encode_file_to_base64` - Encode files to base64
- `get_file_mime_type` - Detect MIME types
- `create_data_url` - Create data URLs
- `validate_image_file` - Validate image formats
- `create_image_content` - Create image content for messages
- `create_multi_image_content` - Handle multiple images
- `create_base64_image_content` - Explicit base64 images
- `analyze_image` - Convenience for image analysis
- `extract_text_from_image` - OCR convenience function
- `compare_images` - Image comparison convenience function

**Quick Example:**

```python
from llmring import LLMRing, LLMRequest, Message, analyze_image

async with LLMRing() as service:
    content = analyze_image("screenshot.png", "What's in this image?")
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
    print(response.content)
```

---

## Best Practices

- Use aliases with fallback models for resilience
- Use streaming for long responses
- Implement proper error handling
- Use profiles for environment-specific configurations
- Leverage provider-specific features via extra_params
- Access raw clients for advanced features when needed
- Use `LLMRingExtended` for conversation persistence
- Use `ConversationManager` for simplified multi-turn chats
- Leverage file utilities for vision/multimodal tasks
