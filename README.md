# LLMRing

A comprehensive Python library for LLM integration with unified interface, advanced features, and MCP support. Supports OpenAI, Anthropic, Google Gemini, and Ollama with consistent APIs.

## ✨ Key Features

- **🔄 Unified Interface**: Single API for all major LLM providers
- **⚡ Streaming Support**: Real streaming for all providers (not simulated)
- **🛠️ Native Tool Calling**: Provider-native function calling with consistent interface
- **📋 Unified Structured Output**: JSON schema works across all providers with automatic adaptation
- **🧠 Conversational Configuration**: MCP chat interface for natural language lockfile setup
- **📋 Smart Aliases**: Always-current semantic aliases (`deep`, `fast`, `balanced`) via intelligent recommendations
- **💰 Cost Tracking**: Automatic cost calculation and receipt generation
- **🎯 Registry Integration**: Centralized model capabilities and pricing
- **🔄 Fallback Models**: Automatic failover to alternative models for resilience
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

**Including Lockfiles in Your Package:**

To ship your `llmring.lock` with your package (like llmring does), add to your `pyproject.toml`:

```toml
[tool.hatch.build]
include = [
    "src/yourpackage/**/*.py",
    "src/yourpackage/**/*.lock",  # Include lockfiles
]
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

    accumulated_usage = None
    async for chunk in await service.chat(request):
        print(chunk.delta, end="", flush=True)
        # Capture final usage stats
        if chunk.usage:
            accumulated_usage = chunk.usage

    print()  # Newline after streaming
    if accumulated_usage:
        print(f"Tokens used: {accumulated_usage.get('total_tokens', 0)}")
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
from llmring import LLMRing, LLMRequest, Message

# Automatic resource cleanup with context manager
async with LLMRing() as service:
    request = LLMRequest(
        model="fast",
        messages=[Message(role="user", content="Hello!")]
    )
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

### 🧠 Model Aliases and Lockfiles

LLMRing uses lockfiles to map semantic aliases to models, with support for fallback models and environment-specific profiles:

```bash
# Initialize lockfile (explicit creation at current directory)
llmring lock init

# Conversational configuration with AI advisor (recommended)
llmring lock chat  # Natural language interface for lockfile management

# Analyze your configuration
llmring lock analyze

# View current aliases
llmring aliases
```

**Lockfile Resolution Order:**
1. Explicit path via `lockfile_path` parameter (file must exist)
2. `LLMRING_LOCKFILE_PATH` environment variable (file must exist)
3. `./llmring.lock` in current directory (if exists)
4. Bundled lockfile at `src/llmring/llmring.lock` (minimal fallback with advisor alias)

**Packaging Your Own Lockfile:**
Libraries using LLMRing can ship with their own lockfiles. See [Lockfile Documentation](docs/lockfile.md) for details on:
- Including lockfiles in your package distribution
- Lockfile resolution order and precedence
- Creating lockfiles with fallback models
- Environment-specific profiles and configuration

**Conversational Configuration** via `llmring lock chat`:
- Describe your requirements in natural language
- Get AI-powered recommendations based on registry analysis
- Configure aliases with multiple fallback models
- Understand cost implications and tradeoffs
- Set up environment-specific profiles

```python
# Use semantic aliases (always current, with fallbacks)
request = LLMRequest(
    model="deep",      # → most capable reasoning model
    messages=[Message(role="user", content="Hello")]
)
# Or use other aliases:
# model="fast"      → cost-effective quick responses
# model="balanced"  → optimal all-around model
# model="advisor"   → Claude Opus 4.1 - powers conversational config
```

**Key Benefits:**
- **Always current**: Registry-based recommendations, no hardcoded models
- **Resilient**: Fallback models provide automatic failover
- **Cost-aware**: Transparent cost analysis and recommendations
- **Environment-specific**: Different configurations for dev/staging/prod

### 🎭 Profiles: Environment-Specific Configurations

LLMRing supports **profiles** to manage different model configurations for different environments (dev, staging, prod, etc.):

```python
# Use different models based on environment
# Development: Use cheaper/faster models
# Production: Use higher-quality models

# Set profile via environment variable
export LLMRING_PROFILE=dev  # or prod, staging, etc.

# Or specify profile in code
async with LLMRing() as service:
    # Uses 'dev' profile bindings
    response = await service.chat(request, profile="dev")
```

**Profile Configuration in Lockfiles:**

```toml
# llmring.lock - Different models per environment
[profiles.default]
[[profiles.default.bindings]]
alias = "assistant"
models = ["anthropic:claude-3-5-sonnet"]  # Production quality

[profiles.dev]
[[profiles.dev.bindings]]
alias = "assistant"
models = ["openai:gpt-4o-mini"]  # Cheaper for development

[profiles.test]
[[profiles.test.bindings]]
alias = "assistant"
models = ["ollama:llama3"]  # Local model for testing
```

**Using Profiles with CLI:**

```bash
# Bind aliases to specific profiles
llmring bind assistant "openai:gpt-4o-mini" --profile dev
llmring bind assistant "anthropic:claude-3-5-sonnet" --profile prod

# List aliases in a profile
llmring aliases --profile dev

# Use profile for chat
llmring chat "Hello" --profile dev

# Set default profile via environment
export LLMRING_PROFILE=dev
llmring chat "Hello"  # Now uses dev profile
```

**Profile Selection Priority:**
1. Explicit parameter: `profile="dev"` or `--profile dev` (highest priority)
2. Environment variable: `LLMRING_PROFILE=dev`
3. Default: `default` profile (if not specified)

**Common Use Cases:**
- **Development**: Use cheaper models to reduce costs during development
- **Testing**: Use local models (Ollama) or mock responses
- **Staging**: Use production models but with different rate limits
- **Production**: Use highest quality models for best user experience
- **A/B Testing**: Test different models for the same alias

### 🏯 Fallback Models

Aliases can specify multiple models for automatic failover:

```toml
# In llmring.lock
[[bindings]]
alias = "assistant"
models = [
    "anthropic:claude-3-5-sonnet",  # Primary
    "openai:gpt-4o",                 # First fallback
    "google:gemini-1.5-pro"          # Second fallback
]
```

If the primary model fails (rate limit, availability, etc.), LLMRing automatically tries the fallbacks.

### 🚪 Advanced: Direct Model References

While aliases are recommended, you can still use direct `provider:model` references when needed:

```python
# Direct model reference (escape hatch)
request = LLMRequest(
    model="anthropic:claude-3-5-sonnet",  # Direct provider:model reference
    messages=[Message(role="user", content="Hello")]
)

# Or specify exact model versions
request = LLMRequest(
    model="openai:gpt-4o",  # Specific model version when needed
    messages=[Message(role="user", content="Hello")]
)
```

**Terminology:**
- **Alias**: Semantic name like `fast`, `balanced`, `deep` (recommended)
- **Model Reference**: Full `provider:model` format like `openai:gpt-4o` (escape hatch)
- **Raw SDK Access**: Bypassing LLMRing entirely using provider clients directly (see [Provider Guide](docs/providers.md))

**Recommendation**: Use aliases for maintainability and cost optimization. Use direct model references only when you need a specific model version or provider-specific features.

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

### Conversational Setup

```bash
# Create optimized configuration with AI advisor
llmring lock chat

# This opens an interactive chat where you can describe your needs
# and get personalized recommendations based on the registry
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
from llmring.mcp.client import create_enhanced_llm

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

- **[Lockfile Documentation](docs/lockfile.md)** - Complete guide to lockfiles, aliases, and profiles
- **[MCP Integration](docs/mcp.md)** - Model Context Protocol and chat client
- **[API Reference](docs/api-reference.md)** - Core API documentation
- **[Provider Guide](docs/providers.md)** - Provider-specific features
- **[Structured Output](docs/structured-output.md)** - Unified JSON schema support
- **[File Utilities](docs/file-utilities.md)** - Vision and multimodal file handling
- **[CLI Reference](docs/cli-reference.md)** - Command-line interface guide
- **[Receipts & Cost Tracking](docs/receipts.md)** - Cost tracking and receipt system
- **[Examples](examples/)** - Working code examples:
  - [Quick Start](examples/quick_start.py) - Basic usage patterns
  - [MCP Chat](examples/mcp_chat_example.py) - MCP integration
  - [Streaming](examples/mcp_streaming_example.py) - Streaming with tools
  - [Conversational Lockfile](examples/conversational_lockfile.md) - Lockfile setup guide

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
