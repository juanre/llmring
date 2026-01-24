# LLMRing

A Python library for LLM integration with unified interface and MCP support. Supports OpenAI, Anthropic, Google Gemini, and Ollama with consistent APIs.

## Features

- Unified Interface: Single API for all major LLM providers
- Streaming Support: Streaming for all providers
- Native Tool Calling: Provider-native function calling with consistent interface
- Unified Structured Output: JSON schema works across all providers with automatic adaptation
- Conversational Configuration: MCP chat interface for natural language lockfile setup
- Aliases: Semantic aliases (`deep`, `fast`, `balanced`) with registry-based recommendations
- Lockfile Composability: Extend lockfiles from other packages with namespaced aliases
- Cost Tracking: Token usage and cost calculation
- Registry Integration: Centralized model capabilities and pricing
- Fallback Models: Automatic failover to alternative models
- Type Safety: Typed exceptions and error handling
- MCP Integration: Model Context Protocol support for tool ecosystems
- MCP Chat Client: Chat interface with persistent history for any MCP server

## Getting Started

### 1. Install

```bash
# With uv (recommended)
uv add llmring

# With pip
pip install llmring
```

Requires Python 3.11 or higher.

### 2. Set up API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# Add whichever providers you want to use
```

### 3. Create a lockfile

LLMRing uses lockfiles to map semantic aliases (like `fast`, `deep`, `balanced`) to actual models. Create one interactively:

```bash
llmring lock chat
```

This starts a conversation where you describe your needs and get AI-powered recommendations. Or create a minimal lockfile manually:

```bash
llmring lock init
```

View your configured aliases:

```bash
llmring aliases
```

### 4. Use it

```python
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="fast",  # Uses your lockfile alias
        messages=[
            Message(role="user", content="Hello!")
        ]
    )
    response = await service.chat(request)
    print(response.content)
```

The alias `fast` resolves to whatever model you configured in your lockfile. Change models by editing the lockfile, not your code.

## Model Aliases and Lockfiles

Lockfiles let you:
- Use semantic aliases (`fast`, `deep`, `balanced`) instead of model IDs
- Configure fallback models for automatic failover
- Set up environment-specific profiles (dev/staging/prod)
- Share configurations across projects

**Lockfile resolution order:**
1. Explicit `lockfile_path` parameter
2. `LLMRING_LOCKFILE_PATH` environment variable
3. `./llmring.lock` in current directory
4. Bundled fallback lockfile

**Lockfile composability**: Libraries can ship their own lockfiles, and users can extend them with `[extends]` to use namespaced aliases (`my-library:summarizer`) while keeping control of model selection.

See [Lockfile Documentation](docs/lockfile.md) for complete details.

## Overview

### Streaming

```python
async with LLMRing() as service:
    # Streaming for all providers
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Count to 10")]
    )

    accumulated_usage = None
    async for chunk in service.chat_stream(request):
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

## Resource Management

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

## Advanced Features

### Unified Structured Output

```python
# JSON schema API works across all providers
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

## File Registration

Register files once and use them with any provider. Files are uploaded lazily on first use:

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # Register file once (no upload yet)
    file_id = service.register_file("data.csv")

    # Use with Anthropic (lazy upload happens here)
    request = LLMRequest(
        model="anthropic:claude-3-5-haiku-20241022",
        messages=[Message(role="user", content="Analyze this data")],
        files=[file_id],
        tools=[{"type": "code_execution"}]
    )

    response = await service.chat(request)

    # Use same file with Google (separate upload happens automatically)
    request = LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(role="user", content="Summarize this data")],
        files=[file_id]
    )

    response = await service.chat(request)

    # Manage registered files
    files = service.list_registered_files()
    await service.deregister_file(file_id)
```

**Key Features:**

- **Provider-agnostic**: Register once, use with any provider (Anthropic, OpenAI, Google)
- **Lazy uploads**: Files upload only when first used, not at registration
- **Automatic staleness detection**: Files are re-hashed before each use to detect changes
- **Cross-provider caching**: Upload tracking per provider prevents redundant uploads

**Provider Support:**

| Provider      | Lazy Upload     | Use in Chat        | Notes                                |
|---------------|-----------------|--------------------|--------------------------------------|
| **Anthropic** | ✅ On first use | ✅ Document blocks | 500MB limit, code execution          |
| **OpenAI**    | ✅ On first use | ⚠️ Assistants only  | 512MB limit, not in Chat Completions |
| **Google**    | ✅ On first use | ✅ Cached content  | Text-only, TTL-based                 |

See [File Registration Documentation](docs/file-uploads.md) for complete guide.

## Current Limitations & Workarounds

LLMRing provides a unified interface for core LLM functionality. Some advanced provider-specific features require workarounds or direct SDK access.

### ✅ Fully Supported (Unified API)

- **Chat completions** - Single and multi-turn conversations
- **Streaming** - Server-sent events (SSE) for incremental responses
- **Tool calling** - Function calling with native provider support
- **Structured output** - JSON schema across all providers
- **Vision & multimodal** - Images, documents, PDFs
- **File registration** - Provider-agnostic registration with lazy uploads (v1.5.0+)
- **Provider fallback** - Automatic failover between models
- **Cost tracking** - Token usage and cost calculation

### ⚠️ Requires Workarounds

#### 1. Provider-Specific Parameters

**Status:** ✅ Works via `extra_params` (needs better documentation)

Access provider-specific features using the `extra_params` field:

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # OpenAI: logprobs, seed, frequency_penalty, etc.
    request = LLMRequest(
        model="openai:gpt-4o",
        messages=[Message(role="user", content="Hello")],
        extra_params={
            "logprobs": True,
            "top_logprobs": 5,
            "seed": 12345,
            "frequency_penalty": 0.5
        }
    )

    # Google: safety settings, top_k, candidate_count
    request = LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(role="user", content="Hello")],
        extra_params={
            "safety_settings": [{
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }],
            "top_k": 40,
            "candidate_count": 3
        }
    )

    # Anthropic: thinking budget, top_k
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5-20250929",
        messages=[Message(role="user", content="Hello")],
        extra_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 5000
            },
            "top_k": 40
        }
    )

    response = await service.chat(request)
```

**Common Parameters:**

| Parameter | OpenAI | Anthropic | Google | Description |
|-----------|--------|-----------|--------|-------------|
| `logprobs` | ✅ | ❌ | ❌ | Log probabilities for tokens |
| `top_logprobs` | ✅ | ❌ | ❌ | Number of top logprobs to return |
| `seed` | ✅ | ❌ | ❌ | Deterministic sampling |
| `frequency_penalty` | ✅ | ❌ | ❌ | Penalize repeated tokens |
| `presence_penalty` | ✅ | ❌ | ❌ | Penalize token presence |
| `thinking` | ❌ | ✅ | ❌ | Extended thinking budget |
| `top_k` | ❌ | ✅ | ✅ | Top-k sampling |
| `safety_settings` | ❌ | ❌ | ✅ | Content filtering |
| `candidate_count` | ❌ | ❌ | ✅ | Number of response candidates |

See provider documentation for complete parameter lists:
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages)
- [Google Gemini API Reference](https://ai.google.dev/api/generate-content)

#### 2. Direct SDK Access

**Status:** ✅ Works via `get_provider().client`

For features not exposed by llmring, access the raw provider SDK:

```python
from llmring import LLMRing

async with LLMRing() as service:
    # Get underlying SDK clients
    anthropic_client = service.get_provider("anthropic").client  # anthropic.AsyncAnthropic
    openai_client = service.get_provider("openai").client        # openai.AsyncOpenAI
    google_client = service.get_provider("google").client         # google.genai.Client
    ollama_client = service.get_provider("ollama").client         # ollama.AsyncClient

    # Use any SDK feature directly
    # Example: Anthropic with all SDK parameters
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello"}],
        system=[{
            "type": "text",
            "text": "You are helpful",
            "cache_control": {"type": "ephemeral"}
        }],
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        }
    )

    # Example: OpenAI with logprobs and seed
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        logprobs=True,
        top_logprobs=10,
        seed=42,
        parallel_tool_calls=False
    )
```

**When to use direct SDK access:**
- Experimental/beta features
- Complex provider-specific workflows
- Features requiring different API endpoints
- Provider-specific frameworks (Agents SDK, etc.)

### ❌ Not Currently Supported

#### Real-time Audio/Video Streaming

**Status:** Not supported (WebSocket transport not implemented)

**Providers:**
- ✅ OpenAI Realtime API - WebSocket-based voice/video streaming
- ✅ Google Live API - Multimodal real-time streaming
- ❌ Anthropic - No real-time API yet

**Workaround:** Use provider SDK directly

```python
# OpenAI Realtime API example
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your-key")

async with client.beta.realtime.connect(model="gpt-4o-realtime") as connection:
    await connection.session.update(session={'modalities': ['text', 'audio']})

    await connection.send_audio(audio_bytes)

    async for event in connection:
        if event.type == "response.audio.delta":
            # Handle streaming audio
            play_audio(event.delta)
```

**Future plans:** Real-time streaming is planned for llmring v2.0 (Q2 2026)

#### OpenAI Agents SDK

**Status:** Out of scope (framework-level abstraction)

The OpenAI Agents SDK is a separate framework (like LangChain) built on top of the OpenAI API. Use it directly with llmring's OpenAI client:

```python
from openai_agents import Agent
from llmring import LLMRing

service = LLMRing()
openai_client = service.get_provider("openai").client

# Use Agents SDK with llmring's client
agent = Agent(
    name="assistant",
    client=openai_client,
    instructions="You are a helpful assistant",
    tools=[...]
)

result = await agent.run("Hello")
```

**Why not unified?** Agent frameworks (OpenAI Agents SDK, LangChain, CrewAI, etc.) are high-level abstractions for orchestration, state management, and multi-agent workflows. llmring focuses on the lower-level API layer.

---

## When to Use What?

### Use llmring's Unified API for:
- ✅ Multi-provider applications
- ✅ Provider fallback strategies
- ✅ Cost optimization across providers
- ✅ Standard chat/tool calling workflows
- ✅ Rapid prototyping and MVPs

### Use `extra_params` for:
- ⚠️ Provider-specific parameter tuning
- ⚠️ Well-documented provider features
- ⚠️ One-off provider requirements

### Use Direct SDK Access for:
- ❌ Experimental/beta features
- ❌ Real-time audio/video
- ❌ Complex provider-specific workflows
- ❌ Provider-specific frameworks

---

## Using LLMRing in Libraries

If you're building a library that uses LLMRing, follow this pattern to ship with defaults while allowing users to override your model choices:

### Library Pattern

**Libraries should:**
1. Ship with a bundled `llmring.lock` in their package
2. Accept optional `lockfile_path` parameter
3. Validate required aliases on initialization
4. Document which aliases they require

**This allows:**
- Library works out of the box with defaults
- Users can override with their own lockfile
- Clear errors if user's lockfile is incomplete

### Simple Library Example

```python
# my-library/src/my_library/__init__.py
from pathlib import Path
from llmring import LLMRing

# Library's bundled lockfile (shipped with package)
DEFAULT_LOCKFILE = Path(__file__).parent / "llmring.lock"
REQUIRED_ALIASES = ["summarizer"]

class MyLibrary:
    """Example library using llmring with configurable lockfile."""

    def __init__(self, lockfile_path=None):
        """Initialize library with optional custom lockfile.

        Args:
            lockfile_path: Path to lockfile. If None, uses library's bundled lockfile.
                          Users can override to control model choices.

        Raises:
            ValueError: If lockfile missing required aliases
        """
        # Use provided lockfile or library's default
        lockfile = lockfile_path or DEFAULT_LOCKFILE

        # Initialize LLMRing with explicit lockfile
        self.ring = LLMRing(lockfile_path=lockfile)

        # Validate required aliases exist (fail fast with clear error)
        self.ring.require_aliases(REQUIRED_ALIASES, context="my-library")

    def summarize(self, text: str) -> str:
        """Summarize text using 'summarizer' alias."""
        response = self.ring.chat("summarizer", messages=[
            {"role": "user", "content": f"Summarize: {text}"}
        ])
        return response.content
```

**Library's lockfile** (`my-library/src/my_library/llmring.lock`):

```toml
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
models = [
    "anthropic:claude-3-5-haiku-20241022",
    "openai:gpt-4o-mini",
    "google:gemini-2.5-flash"
]
```

### User Override Pattern

**Users can use library defaults:**

```python
from my_library import MyLibrary

# Uses library's bundled lockfile automatically
lib = MyLibrary()
result = lib.summarize("Some text")
```

**Or override with their own lockfile:**

```python
# Create custom lockfile: ./my-llmring.lock
# [profiles.default]
# [[profiles.default.bindings]]
# alias = "summarizer"
# models = ["anthropic:claude-sonnet-4-5-20250929", "openai:gpt-4o"]

# Use custom lockfile
lib = MyLibrary(lockfile_path="./my-llmring.lock")
result = lib.summarize("Some text")
```

### Library Composition

When Library B uses Library A, pass the same lockfile to both:

```python
# library-b/src/library_b/__init__.py
from pathlib import Path
from llmring import LLMRing
from library_a import LibraryA

DEFAULT_LOCKFILE = Path(__file__).parent / "llmring.lock"
REQUIRED_ALIASES = ["analyzer"]

class LibraryB:
    def __init__(self, lockfile_path=None):
        """Initialize Library B (which uses Library A).

        Args:
            lockfile_path: Lockfile controlling models for both libraries.
                          Must include aliases required by both Library A and Library B.
        """
        lockfile = lockfile_path or DEFAULT_LOCKFILE

        # Pass lockfile to Library A (controls Library A's model choices)
        self.lib_a = LibraryA(lockfile_path=lockfile)

        # Initialize our own LLMRing with same lockfile
        self.ring = LLMRing(lockfile_path=lockfile)
        self.ring.require_aliases(REQUIRED_ALIASES, context="library-b")

    def analyze(self, text: str):
        # Use Library A (which uses our lockfile)
        summary = self.lib_a.summarize(text)

        # Do our own analysis
        analysis = self.ring.chat("analyzer", messages=[...])

        return {"summary": summary, "analysis": analysis}
```

**Library B's lockfile must include aliases for both libraries:**

```toml
# library-b/src/library_b/llmring.lock
[profiles.default]
name = "default"

# Library A's requirement (we choose which model)
[[profiles.default.bindings]]
alias = "summarizer"
models = [
    "anthropic:claude-sonnet-4-5-20250929",
    "openai:gpt-4o"
]

# Library B's requirement
[[profiles.default.bindings]]
alias = "analyzer"
models = [
    "openai:gpt-4o",
    "google:gemini-2.5-pro"
]
```

**Users can override the entire chain:**

```python
# User's lockfile with their preferred models for BOTH libraries
lib_b = LibraryB(lockfile_path="./user-models.lock")
# This lockfile controls both Library A and Library B
```

### Validation Helpers

LLMRing provides validation helpers for library authors:

```python
from llmring import LLMRing

ring = LLMRing(lockfile_path="./my.lock")

# Check if alias exists (returns bool, never raises)
if ring.has_alias("summarizer"):
    # Safe to use
    response = ring.chat("summarizer", messages=[...])

# Validate required aliases (raises ValueError with helpful message if missing)
ring.require_aliases(
    ["summarizer", "analyzer"],
    context="my-library"  # Included in error message
)
# Raises: "Lockfile missing required aliases for my-library: analyzer.
#          Lockfile path: /path/to/lockfile.lock
#          Please ensure your lockfile defines these aliases."
```

### Packaging Lockfiles

Include lockfiles in your package distribution:

**pyproject.toml:**

```toml
[tool.hatch.build]
include = [
    "src/my_library/**/*.py",
    "src/my_library/**/*.lock",  # Include lockfiles
]
```

**Or with setuptools in MANIFEST.in:**

```
include src/my_library/*.lock
```

### Library Best Practices

1. **Ship with bundled lockfile** - Include your defaults in the package
2. **Accept `lockfile_path` parameter** - Let users override everything
3. **Validate early** - Use `require_aliases()` in `__init__`
4. **Document requirements** - List required aliases in README
5. **Use semantic names** - Aliases like "summarizer" are clearer than model IDs
6. **Pass lockfile down** - When using other libraries, pass your lockfile to them

### Lockfile Composability

Users can extend library lockfiles using the `[extends]` section:

```toml
# User's llmring.lock
[extends]
packages = ["my-library", "another-lib"]  # Extend these libraries

# Override a library's alias if needed
[[profiles.default.bindings]]
alias = "my-library:summarizer"
models = ["openai:gpt-4o"]  # Use different model
```

This allows:
- Libraries define their required aliases
- Users control which models are used via namespaced aliases (`my-library:summarizer`)
- Users can override specific aliases while keeping library defaults

**Important notes:**
- Extends are **not recursive**: If `libA` extends `libCore`, extending `libA` does not include `libCore`'s aliases
- Only extend packages you trust, as package discovery may execute import hooks

Validate the setup with:

```bash
llmring lock check
```

See the [Lockfile Composability Guide](skills/lockfile/SKILL.md#lockfile-composability) for full documentation.

### Profiles: Environment-Specific Configurations

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
# llmring.lock (truncated for brevity)
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"
[[profiles.default.bindings]]
alias = "assistant"
models = ["anthropic:claude-sonnet-4-5-20250929"]

[profiles.dev]
name = "dev"
[[profiles.dev.bindings]]
alias = "assistant"
models = ["openai:gpt-4o-mini"]  # Cheaper for development

[profiles.test]
name = "test"
[[profiles.test.bindings]]
alias = "assistant"
models = ["ollama:llama3"]  # Local model for testing
```

**Using Profiles with CLI:**

```bash
# Bind aliases to specific profiles
llmring bind assistant "openai:gpt-4o-mini" --profile dev
llmring bind assistant "anthropic:claude-sonnet-4-5-20250929" --profile prod

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

### Fallback Models

Aliases can specify multiple models for automatic failover:

```toml
# In llmring.lock
[profiles.default]
name = "default"
[[profiles.default.bindings]]
alias = "assistant"
models = [
    "anthropic:claude-sonnet-4-5-20250929",  # Primary
    "openai:gpt-4o",                         # First fallback
    "google:gemini-2.5-pro"                  # Second fallback
]
```

If the primary model fails (rate limit, availability, etc.), LLMRing automatically tries the fallbacks.

### Advanced: Direct Model References

While aliases are recommended, you can still use direct `provider:model` references when needed:

```python
# Direct model reference (escape hatch)
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5-20250929",  # Direct provider:model reference
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

Recommendation: Use aliases for maintainability and cost optimization. Use direct model references only when you need a specific model version or provider-specific features.

### Raw SDK Access

When you need direct access to the underlying SDKs:

```python
# Access provider SDK clients directly
openai_client = service.get_provider("openai").client      # openai.AsyncOpenAI
anthropic_client = service.get_provider("anthropic").client # anthropic.AsyncAnthropic
google_client = service.get_provider("google").client       # google.genai.Client
ollama_client = service.get_provider("ollama").client       # ollama.AsyncClient

# Use SDK features not exposed by LLMRing
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

When to use raw clients:
- SDK features not exposed by LLMRing
- Provider-specific optimizations
- Complex configurations
- Performance-critical applications

## Provider Support

| Provider | Models | Streaming | Tools | Special Features |
|----------|--------|-----------|-------|------------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, o1 | Yes | Native | JSON schema, PDF processing |
| **Anthropic** | Claude 3.5 Sonnet/Haiku | Yes | Native | Prompt caching, large context |
| **Google** | Gemini 1.5/2.0 Pro/Flash | Yes | Native | Multimodal, 2M+ context |
| **Ollama** | Llama, Mistral, etc. | Yes | Prompt-based | Local models, custom options |

## Setup

### Environment Variables

```bash
# Add to your .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GEMINI_API_KEY=AIza...

# Optional
OLLAMA_BASE_URL=http://localhost:11434  # Default
```

### Self-Hosted Quickstart

```bash
# 1. Start the local llmring-server stack (production-like, Docker)
cd ../llmring-server
make docker

# 2. Bootstrap local credentials from your application repo
cd ../llmring
llmring server init --server http://localhost:9100 --env-file .env.llmring

# 3. Load the environment variables in your shell
source .env.llmring

# 4. Verify everything works
llmring stats --json
```

Command hints:

- `llmring server status` – Inspect current configuration and test connectivity
- `llmring server key rotate` – Generate a new API key and refresh `.env.llmring`
- `llmring server key list` – Show active key information from the environment and env file
- `llmring lock init` – Create a default `llmring.lock` so aliases like `fast` are available

### Inspect Usage Data

```bash
# Summaries (JSON with --json)
llmring server stats --json

# Raw logs (table, JSON or CSV)
llmring server logs --limit 20 --output json

# Conversation archive
llmring server conversations
llmring server conversations --conversation-id <uuid> --messages
```

All commands honour `LLMRING_SERVER_URL` and `LLMRING_API_KEY`; run
`llmring server init` first so they target your self-hosted container.

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

## MCP Integration

```python
from llmring.mcp.client import create_enhanced_llm

# Create MCP-enabled LLM with tools
llm = await create_enhanced_llm(
    model="fast",
    mcp_server_path="path/to/mcp/server"
)

# Now has access to MCP tools
response = await llm.chat([
    Message(role="user", content="Use available tools to help me")
])
```

## Documentation

- **[Lockfile Documentation](docs/lockfile.md)** - Complete guide to lockfiles, aliases, and profiles
- **[Conversational Lockfile](docs/conversational-lockfile.md)** - Natural language lockfile management
- **[MCP Integration](docs/mcp.md)** - Model Context Protocol and chat client
- **[API Reference](docs/api-reference.md)** - Core API documentation
- **[Provider Guide](docs/providers.md)** - Provider-specific features
- **[Structured Output](docs/structured-output.md)** - Unified JSON schema support
- **[File Utilities](docs/file-utilities.md)** - Vision and multimodal file handling
- **[CLI Reference](docs/cli-reference.md)** - Command-line interface guide
- **[Examples](examples/)** - Working code examples:
  - [Quick Start](examples/quick_start.py) - Basic usage patterns
  - [MCP Chat](examples/mcp_chat_example.py) - MCP integration
  - [Streaming](examples/mcp_streaming_example.py) - Streaming with tools

## Claude Code Skills

LLMRing provides skills for Claude Code that activate automatically when you ask about streaming, lockfiles, tool calling, etc. Install with:

```bash
/plugin marketplace add juanre/ai-tools
/plugin install llmring@juanre-ai-tools
```

## Development

```bash
# Install for development
uv sync --group dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## Error Handling

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

## Key Features Summary

- Unified Interface: Switch providers without code changes
- Performance: Streaming, prompt caching, optimized requests
- Reliability: Circuit breakers, retries, typed error handling
- Observability: Cost tracking and usage monitoring
- Flexibility: Provider-specific features and raw SDK access
- Standards: Type-safe, well-tested

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## Examples

See the `examples/` directory for complete working examples:
- Basic chat and streaming
- Tool calling and function execution
- Provider-specific features
- MCP integration
- Cost tracking and usage monitoring
