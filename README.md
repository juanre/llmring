# LLMRing

A comprehensive Python library for LLM integration with unified interface, advanced features, and MCP support. Supports OpenAI, Anthropic, Google Gemini, and Ollama with consistent APIs.

## ✨ Key Features

- **🔄 Unified Interface**: Single API for all major LLM providers
- **⚡ Streaming Support**: Real streaming for all providers (not simulated)
- **🛠️ Native Tool Calling**: Provider-native function calling with consistent interface
- **📋 Unified Structured Output**: JSON schema works across all providers with automatic adaptation
- **🧠 Intelligent Configuration**: AI-powered lockfile creation with registry analysis
- **📋 Smart Aliases**: Always-current semantic aliases (`deep`, `fast`, `balanced`) via intelligent recommendations
- **💰 Cost Tracking**: Automatic cost calculation and receipt generation
- **🎯 Registry Integration**: Centralized model capabilities and pricing
- **🔧 Advanced Features**:
  - OpenAI: JSON schema, o1 models, PDF processing
  - Anthropic: Prompt caching (90% cost savings)
  - Google: Native function calling, multimodal, 2M+ context
  - Ollama: Local models, streaming, custom options
- **🔒 Type Safety**: Comprehensive typed exceptions and error handling
- **🌐 MCP Integration**: Model Context Protocol support for tool ecosystems
- **💬 MCP Chat Client**: Generic chat interface with persistent history for any MCP server

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

# Initialize service with context manager (auto-closes resources)
async with LLMRing() as service:
    # Simple chat
    request = LLMRequest(
        model="fast",
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
async with LLMRing() as service:
    # Real streaming for all providers
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Count to 10")],
        stream=True
    )

    async for chunk in await service.chat(request):
        print(chunk.delta, end="", flush=True)
```

### Tool Calling

```python
async with LLMRing() as service:
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
        model="balanced",
        messages=[Message(role="user", content="What's the weather in NYC?")],
        tools=tools
    )

    response = await service.chat(request)
    if response.tool_calls:
        print("Function called:", response.tool_calls[0]["function"]["name"])
```

## 📚 Resource Management

### Context Manager (Recommended)

```python
# Automatic resource cleanup with context manager
async with LLMRing() as service:
    response = await service.chat(request)
    # Resources are automatically cleaned up when exiting the context
```

### Manual Cleanup

```python
# Manual resource management
service = LLMRing()
try:
    response = await service.chat(request)
finally:
    await service.close()  # Ensure resources are cleaned up
```

## 🔧 Advanced Features

### 🎯 Unified Structured Output (All Providers)

```python
# Same JSON schema API works across ALL providers!
request = LLMRequest(
    model="balanced",  # Works with any provider
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
        "strict": True  # Validates across all providers
    }
)

response = await service.chat(request)
print("JSON:", response.content)   # Valid JSON string
print("Data:", response.parsed)    # Python dict ready to use
```

### Provider-Specific Parameters

```python

# Anthropic: Prompt caching for 90% cost savings
request = LLMRequest(
    model="balanced",
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
    model="fast",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "logprobs": True,
        "top_logprobs": 5,
        "presence_penalty": 0.1,
        "seed": 12345
    }
)
```

### 🧠 Intelligent Model Aliases

LLMRing features intelligent lockfile creation that analyzes the current registry and recommends optimal aliases:

```bash
# Interactive mode - answers prompts about your needs
llmring lock init --interactive

# With requirements from a file
llmring lock init --interactive --requirements-file requirements.txt

# With requirements directly in command
llmring lock init --interactive --requirements "I need cost-effective models for coding"

# Analyze your configuration
llmring lock analyze

# Optimize existing lockfile
llmring lock optimize
```

**Interactive Mode** prompts you for:
- **Use cases**: What you'll primarily use LLMs for (coding, writing, analysis, etc.)
- **Budget preference**: Low cost, balanced, or maximum performance
- **Capabilities**: Vision, function calling, or auto-detect from registry
- **Usage volume**: Expected monthly request volume
- **Custom aliases**: Specific aliases you want (fast, deep, coder, writer, vision, etc.)

```python
# Use semantic aliases (always current, registry-based)
request = LLMRequest(
    model="deep",      # → most capable reasoning model
    model="fast",      # → cost-effective quick responses
    model="balanced",  # → optimal all-around model
    model="advisor",   # → Claude Opus 4.1 - powers intelligent lockfile creation
    messages=[Message(role="user", content="Hello")]
)
```

**Key Benefits:**
- **Always current**: Aliases point to latest registry models, not outdated hardcoded ones
- **Intelligent selection**: AI advisor analyzes registry and recommends optimal configuration
- **Cost-aware**: Transparent cost analysis and recommendations
- **Self-hosted**: Uses LLMRing's own API to power intelligent lockfile creation

### 🚪 Advanced: Direct Model Access

While aliases are recommended, you can still use direct provider:model format when needed:

```python
# Direct model specification (escape hatch)
request = LLMRequest(
    model="anthropic:claude-3-5-sonnet",  # Direct provider:model format
    messages=[Message(role="user", content="Hello")]
)

# Or mix aliases with direct models
request = LLMRequest(
    model="openai:gpt-4o",  # Specific model when needed
    messages=[Message(role="user", content="Hello")]
)
```

**Recommendation**: Use aliases for maintainability and cost optimization. Use direct model strings only when you need a specific model version or provider-specific features.

### 🚪 Raw SDK Access (Escape Hatch)

When you need the full power of the underlying SDKs:

```python
# Access any provider's raw client for maximum SDK features
openai_client = service.get_provider("openai").client      # openai.AsyncOpenAI
anthropic_client = service.get_provider("anthropic").client # anthropic.AsyncAnthropic
google_client = service.get_provider("google").client       # google.genai.Client
ollama_client = service.get_provider("ollama").client       # ollama.AsyncClient

# Use any SDK feature not exposed by LLMRing
response = await openai_client.chat.completions.create(
    model="fast",  # Use alias or provider:model format when needed
    messages=[{"role": "user", "content": "Hello"}],
    logprobs=True,
    top_logprobs=10,
    parallel_tool_calls=False,
    # Any OpenAI parameter
)

# Anthropic with all SDK features
response = await anthropic_client.messages.create(
    model="balanced",  # Use alias or provider:model format when needed
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    top_p=0.9,
    top_k=40,
    system=[{
        "type": "text",
        "text": "You are helpful",
        "cache_control": {"type": "ephemeral"}
    }]
)

# Google with native SDK features
response = google_client.models.generate_content(
    model="balanced",  # Use alias or provider:model format when needed
    contents="Hello",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 3
    },
    safety_settings=[{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }]
)
```

**When to use raw clients:**
- Advanced SDK features not in LLMRing
- Provider-specific optimizations
- Complex configurations
- Performance-critical applications

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

### Intelligent Setup

```bash
# Create optimized configuration with AI advisor
llmring lock init --interactive

# The advisor analyzes the current registry and your API keys
# to recommend optimal model aliases for your workflow
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
    model="fast",
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
- **[MCP Chat Client](docs/mcp-chat-client.md)** - Generic MCP chat client with persistent history
- **[Conversational Lockfile](examples/conversational_lockfile.md)** - Natural language lockfile management
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
