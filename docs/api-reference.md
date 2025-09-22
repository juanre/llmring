# LLMRing API Reference

## Core Classes

### LLMRing

Main service class for LLM interactions.

```python
from llmring.service import LLMRing

service = LLMRing(
    origin="myapp",                    # Optional: Origin identifier
    registry_url=None,                 # Optional: Custom registry URL
    lockfile_path="./llmring.lock"     # Optional: Custom lockfile path
)
```

**Methods:**
- `async chat(request: LLMRequest) -> LLMResponse | AsyncIterator[StreamChunk]`
- `get_provider(name: str) -> BaseLLMProvider`
- `resolve_alias(alias: str, profile: str = None) -> str`
- `calculate_cost(response: LLMResponse) -> Dict[str, float]`

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
    stream=False,                            # Optional: Streaming mode
    extra_params={}                          # Optional: Provider-specific params
)
```

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
```

## Provider-Specific Features

### OpenAI
- JSON schema with strict mode
- o1 models via Responses API
- PDF processing with file upload
- Advanced parameters via extra_params

### Anthropic
- Prompt caching (90% cost savings)
- Large context windows (200K+ tokens)
- Streaming with tool calls
- Cache usage tracking

### Google Gemini
- Real streaming via native SDK
- Native function calling
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
    CircuitBreakerError           # Service unavailable
)
```

## Best Practices

- Use streaming for long responses
- Implement proper error handling
- Leverage provider-specific features
- Use model aliases for flexibility
- Access raw clients for advanced features
