# LLMRing

A comprehensive Python library for LLM integration with unified interface, advanced features, and MCP support. Supports OpenAI, Anthropic, Google Gemini, and Ollama with consistent APIs.

## ✨ Key Features

- **🔄 Unified Interface**: Single API for all major LLM providers
- **⚡ Streaming Support**: Real streaming for all providers (not simulated)
- **🛠️ Native Tool Calling**: Provider-native function calling with consistent interface
- **📋 Unified Structured Output**: JSON schema works across all providers with automatic adaptation
- **📋 Alias Management**: Semantic model aliases via lockfile (`deep`, `fast`, `balanced`)
- **💰 Cost Tracking**: Automatic cost calculation and receipt generation
- **🎯 Registry Integration**: Centralized model capabilities and pricing
- **🔧 Advanced Features**:
  - OpenAI: JSON schema, o1 models, PDF processing
  - Anthropic: Prompt caching (90% cost savings)
  - Google: Native function calling, multimodal, 2M+ context
  - Ollama: Local models, streaming, custom options
- **🔒 Type Safety**: Comprehensive typed exceptions and error handling
- **🌐 MCP Integration**: Model Context Protocol support for tool ecosystems

## 🚀 Quick Start

### Installation

```bash
# With uv (recommended)
uv add llmring

# With pip
pip install llmring
```

### Basic Usage

```python
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

# Initialize service (auto-detects API keys)
service = LLMRing()

# Simple chat
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!")
    ]
)

response = await service.chat(request)
print(response.content)
```

### Streaming

```python
# Real streaming for all providers
request = LLMRequest(
    model="anthropic:claude-3-5-sonnet",
    messages=[Message(role="user", content="Count to 10")],
    stream=True
)

async for chunk in await service.chat(request):
    print(chunk.delta, end="", flush=True)
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

request = LLMRequest(
    model="google:gemini-1.5-pro",
    messages=[Message(role="user", content="What's the weather in NYC?")],
    tools=tools
)

response = await service.chat(request)
if response.tool_calls:
    print("Function called:", response.tool_calls[0]["function"]["name"])
```

## 🔧 Advanced Features

### Provider-Specific Parameters

```python
# OpenAI: Structured output with JSON schema
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Generate a person")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        },
        "strict": True
    }
)

# Anthropic: Prompt caching for 90% cost savings
request = LLMRequest(
    model="anthropic:claude-3-5-sonnet",
    messages=[
        Message(
            role="system",
            content="Very long system prompt...",  # 1024+ tokens
            metadata={"cache_control": {"type": "ephemeral"}}
        ),
        Message(role="user", content="Hello")
    ]
)

# Extra parameters for provider-specific features
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "logprobs": True,
        "top_logprobs": 5,
        "presence_penalty": 0.1,
        "seed": 12345
    }
)
```

### Model Aliases

```bash
# Initialize lockfile with smart defaults
llmring lock init

# Use semantic aliases instead of specific models
```

```python
request = LLMRequest(
    model="deep",      # → claude-3-opus (powerful reasoning)
    model="fast",      # → gpt-4o-mini (quick responses)
    model="balanced",  # → claude-3-5-sonnet (best overall)
    messages=[Message(role="user", content="Hello")]
)
```

### Raw SDK Access

```python
# Full SDK power when needed
openai_client = service.get_provider("openai").client
anthropic_client = service.get_provider("anthropic").client
google_client = service.get_provider("google").client
ollama_client = service.get_provider("ollama").client

# Use any SDK feature directly
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    # Any OpenAI parameter
)
```

## 🌐 Provider Support

| Provider | Models | Streaming | Tools | Special Features |
|----------|--------|-----------|-------|------------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, o1 | ✅ Real | ✅ Native | JSON schema, PDF processing |
| **Anthropic** | Claude 3.5 Sonnet/Haiku | ✅ Real | ✅ Native | Prompt caching, large context |
| **Google** | Gemini 1.5/2.0 Pro/Flash | ✅ Real | ✅ Native | Multimodal, 2M+ context |
| **Ollama** | Llama, Mistral, etc. | ✅ Real | 🔧 Prompt | Local models, custom options |

## 📦 Setup

### Environment Variables

```bash
# Add to your .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GEMINI_API_KEY=AIza...

# Optional
OLLAMA_BASE_URL=http://localhost:11434  # Default
```

### Dependencies

```python
# Required for specific providers
pip install openai>=1.0     # OpenAI
pip install anthropic>=0.67  # Anthropic
pip install google-genai    # Google Gemini
pip install ollama>=0.4     # Ollama
```

## 🔗 MCP Integration

```python
from llmring.mcp.client.enhanced_llm import create_enhanced_llm

# Create MCP-enabled LLM with tool ecosystem
llm = await create_enhanced_llm(
    model="openai:gpt-4o",
    mcp_server_path="path/to/mcp/server"
)

# Now has access to MCP tools
response = await llm.chat([
    Message(role="user", content="Use available tools to help me")
])
```

## 📚 Documentation

- **[Provider Usage Guide](docs/provider-usage.md)** - Provider-specific features and examples
- **[API Reference](docs/api-reference.md)** - Detailed API documentation
- **[Structured Output](docs/structured-output.md)** - Unified JSON schema across all providers
- **[MCP Integration](docs/mcp-integration.md)** - Model Context Protocol guide
- **[Examples](examples/)** - Working code examples

## 🧪 Development

```bash
# Install for development
uv sync --group dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## 🛠️ Error Handling

LLMRing uses typed exceptions for better error handling:

```python
from llmring.exceptions import (
    ProviderAuthenticationError,
    ModelNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError
)

try:
    response = await service.chat(request)
except ProviderAuthenticationError:
    print("Invalid API key")
except ModelNotFoundError:
    print("Model not supported")
except ProviderRateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
```

## 🎯 Key Benefits

- **🔄 Unified Interface**: Switch providers without code changes
- **⚡ Performance**: Real streaming, prompt caching, optimized requests
- **🛡️ Reliability**: Circuit breakers, retries, typed error handling
- **📊 Observability**: Cost tracking, usage analytics, receipt generation
- **🔧 Flexibility**: Provider-specific features + raw SDK access
- **📏 Standards**: Type-safe, well-tested, production-ready

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## 🌟 Examples

See the `examples/` directory for complete working examples:
- Basic chat and streaming
- Tool calling and function execution
- Provider-specific features
- MCP integration
- Cost tracking and receipts

---

**LLMRing: The comprehensive LLM library for Python developers** 🚀
