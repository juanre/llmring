# Gap Documentation Drafts

These are draft sections to add to README.md and llmring.ai website to clearly document current limitations and workarounds.

---

## 1. README.md Addition

**Location:** Add after "Advanced Features" section, before "Provider Support"

```markdown
## Current Limitations & Workarounds

LLMRing provides a unified interface for core LLM functionality. Some advanced provider-specific features require workarounds or direct SDK access.

### ✅ Fully Supported (Unified API)

- **Chat completions** - Single and multi-turn conversations
- **Streaming** - Server-sent events (SSE) for incremental responses
- **Tool calling** - Function calling with native provider support
- **Structured output** - JSON schema across all providers
- **Vision & multimodal** - Images, documents, PDFs
- **File uploads** - Upload-once-reference-many pattern (v1.4.0+)
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
        model="anthropic:claude-sonnet-4-5",
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

## Feedback & Contributions

Have a feature request or found a workaround we should document?
- Open an issue: [github.com/juanre/llmring/issues](https://github.com/juanre/llmring/issues)
- Contribute: [github.com/juanre/llmring/blob/main/CONTRIBUTING.md](https://github.com/juanre/llmring/blob/main/CONTRIBUTING.md)
```

---

## 2. Website Documentation Page

**File:** `llmring.ai/content/docs/limitations-and-workarounds.md`

```markdown
+++
title = "Limitations & Workarounds"
description = "Understanding llmring's scope and how to access advanced features"
weight = 85
+++

# Limitations & Workarounds

llmring provides a **unified interface for core LLM functionality** across OpenAI, Anthropic, Google, and Ollama. This page clarifies what's supported, what requires workarounds, and what's out of scope.

## Design Philosophy

### What llmring Unifies ✅

We unify features that:

1. **Work across all providers** - Chat, streaming, tools, structured output
2. **Have similar semantics** - File uploads, multimodal input, cost tracking
3. **Add clear value** - Simplified API, multi-provider fallback, consistent error handling

Our goal: **Make common tasks easy and consistent while providing escape hatches for advanced features.**

### What llmring Doesn't Unify ❌

We intentionally don't unify:

1. **Provider-unique features** - Use direct SDK access
2. **Experimental APIs** - Too unstable to abstract
3. **Framework-level concerns** - Agent orchestration, chains, memory
4. **Real-time protocols** - WebSocket streaming (planned for v2.0)

Our philosophy: **Stay focused on API-level unification, not framework-level abstraction.**

---

## Feature Support Matrix

| Feature | Support | Access Method | Notes |
|---------|---------|---------------|-------|
| **Core Features** |
| Chat completions | ✅ First-class | `service.chat()` | All providers |
| Streaming (SSE) | ✅ First-class | `service.chat_stream()` | All providers |
| Tool calling | ✅ First-class | `tools` parameter | Native support |
| Structured output | ✅ First-class | `response_format` | JSON schema |
| Multimodal | ✅ First-class | `create_file_content()` | Images, PDFs, documents |
| File uploads | ✅ First-class | `service.upload_file()` | v1.4.0+ |
| Cost tracking | ✅ First-class | `response.usage` | Built-in |
| Provider fallback | ✅ First-class | Lockfile bindings | Automatic |
| **Provider-Specific** |
| Logprobs | ⚠️ extra_params | `extra_params` | OpenAI only |
| Safety settings | ⚠️ extra_params | `extra_params` | Google only |
| Thinking budget | ⚠️ extra_params | `extra_params` | Anthropic only |
| Prompt caching | ⚠️ metadata | `message.metadata` | Anthropic only |
| **Advanced** |
| Raw SDK features | ⚠️ Direct access | `get_provider().client` | All providers |
| Real-time audio | ❌ Not supported | Use provider SDK | OpenAI, Google |
| Agent frameworks | ❌ Out of scope | Use framework directly | Integration supported |

---

## Detailed Usage Guides

### ✅ First-Class Features

These work seamlessly across all providers through llmring's unified API.

#### Chat Completions

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="balanced",  # Alias resolves to best model
        messages=[
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello!")
        ]
    )
    response = await service.chat(request)
```

#### Streaming

```python
async with LLMRing() as service:
    request = LLMRequest(
        model="fast",
        messages=[Message(role="user", content="Count to 10")]
    )

    async for chunk in service.chat_stream(request):
        print(chunk.delta, end="", flush=True)
```

#### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Weather in NYC?")],
    tools=tools
)

response = await service.chat(request)
if response.tool_calls:
    print(response.tool_calls[0]["function"]["name"])
```

---

### ⚠️ Provider-Specific Features

These require `extra_params` or special handling but work reliably.

#### OpenAI: Logprobs & Deterministic Sampling

```python
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "logprobs": True,
        "top_logprobs": 5,
        "seed": 42,  # Deterministic output
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3
    }
)
```

**Use cases:**
- Analyzing model confidence (logprobs)
- Reproducible outputs (seed)
- Reducing repetition (penalties)

#### Google: Safety Settings & Sampling

```python
request = LLMRequest(
    model="google:gemini-2.5-flash",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "safety_settings": [{
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }],
        "top_k": 40,
        "candidate_count": 3  # Multiple completions
    }
)
```

**Use cases:**
- Content moderation
- Fine-grained sampling control
- Generating multiple candidates

#### Anthropic: Extended Thinking & Caching

```python
# Extended thinking budget
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[Message(role="user", content="Complex problem...")],
    extra_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 10000
        }
    }
)

# Prompt caching (via metadata)
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[
        Message(
            role="system",
            content="Very long system prompt (1024+ tokens)...",
            metadata={"cache_control": {"type": "ephemeral"}}
        ),
        Message(role="user", content="Question")
    ]
)
```

**Use cases:**
- Complex reasoning tasks (thinking)
- Cost optimization for repeated prompts (caching)

---

### ⚠️ Direct SDK Access

For features not exposed by llmring, use the raw provider SDK.

#### Pattern: Raw SDK Integration

```python
from llmring import LLMRing

async with LLMRing() as service:
    # Get provider's official SDK client
    anthropic_client = service.get_provider("anthropic").client
    openai_client = service.get_provider("openai").client
    google_client = service.get_provider("google").client

    # Use any SDK feature
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello"}],
        # Any Anthropic SDK parameter
        system=[{
            "type": "text",
            "text": "System prompt",
            "cache_control": {"type": "ephemeral"}
        }]
    )
```

**When to use:**
- Experimental features
- Complex provider-specific workflows
- Beta APIs

**Benefits:**
- Full provider SDK access
- Still uses llmring's provider management
- Credentials automatically configured

---

### ❌ Not Currently Supported

#### Real-time Audio/Video

**Providers:** OpenAI Realtime API, Google Live API

**Why not unified?**
- Different protocol (WebSocket vs HTTP)
- Different interaction model (bidirectional vs request-response)
- Only 2/3 providers support (Anthropic missing)

**Workaround:** Use provider SDK directly

```python
# OpenAI Realtime API
from openai import AsyncOpenAI

client = AsyncOpenAI()
async with client.beta.realtime.connect(model="gpt-4o-realtime") as conn:
    await conn.send_audio(audio_bytes)
    async for event in conn:
        if event.type == "response.audio.delta":
            handle_audio(event.delta)
```

**Timeline:** Planned for llmring v2.0 (Q2 2026)

#### Agent Frameworks (OpenAI Agents SDK, LangChain, etc.)

**Why not unified?**
- Framework-level abstraction (not API-level)
- Each framework has different philosophy
- Out of llmring's scope

**Integration:** Use frameworks with llmring's clients

```python
from openai_agents import Agent
from llmring import LLMRing

service = LLMRing()
openai_client = service.get_provider("openai").client

agent = Agent(
    name="assistant",
    client=openai_client,
    instructions="You are helpful"
)
```

**Works with:**
- OpenAI Agents SDK
- LangChain
- CrewAI
- AutoGen
- Any framework using OpenAI/Anthropic/Google SDKs

---

## Decision Framework

### Should I use llmring?

**YES, if you want:**
- Multi-provider support
- Provider fallback/redundancy
- Cost optimization across providers
- Clean, simple API
- Type-safe requests/responses

**MAYBE, if you need:**
- Provider-specific features (use extra_params)
- Experimental APIs (use raw SDK)
- Simple single-provider app (still valuable for abstractions)

**NO, if you primarily need:**
- Real-time audio/video streaming
- Heavy agent framework usage
- Single provider with deep integration

---

## Roadmap

### v1.4.0 (Current) - File Uploads
- ✅ Upload-once-reference-many pattern
- ✅ Anthropic Files API
- ✅ OpenAI Files API
- ✅ Google Context Caching

### v1.5.0 (Q1 2026) - Enhanced Provider Features
- 📋 Documented extra_params reference
- 📋 Type hints for common parameters
- 📋 Validation for provider-specific params

### v2.0.0 (Q2 2026) - Real-time Streaming
- 📋 WebSocket transport layer
- 📋 OpenAI Realtime API support
- 📋 Google Live API support
- 📋 Unified real-time interface

---

## Get Help

**Have questions?**
- 📖 [Documentation](https://llmring.ai/docs)
- 💬 [Discord Community](https://discord.gg/llmring)
- 🐛 [GitHub Issues](https://github.com/juanre/llmring/issues)

**Found a workaround?**
Share it! We'd love to document community solutions.

**Want a feature?**
Open a feature request with your use case. We prioritize based on:
1. Multi-provider applicability
2. User demand
3. API stability
```

---

## 3. Provider-Specific Features Quick Reference

**File:** `llmring.ai/content/docs/provider-specific-features.md`

```markdown
+++
title = "Provider-Specific Features"
description = "Quick reference for accessing unique provider capabilities"
weight = 86
+++

# Provider-Specific Features Quick Reference

Complete reference for accessing provider-specific features through `extra_params`.

## OpenAI Parameters

### Sampling & Generation

```python
extra_params={
    "temperature": 0.7,           # Randomness (0-2, default 1)
    "top_p": 0.9,                 # Nucleus sampling
    "frequency_penalty": 0.5,     # Penalize token frequency (0-2)
    "presence_penalty": 0.3,      # Penalize token presence (0-2)
    "seed": 42,                   # Deterministic sampling
    "max_completion_tokens": 1000 # Alias for max_tokens
}
```

### Logprobs & Analysis

```python
extra_params={
    "logprobs": True,        # Enable log probabilities
    "top_logprobs": 5        # Number of top logprobs (1-20)
}
```

### Tools & Behavior

```python
extra_params={
    "parallel_tool_calls": False,  # Disable parallel tool execution
    "response_format": {           # Structured output
        "type": "json_schema",
        "json_schema": {...}
    }
}
```

### Reasoning Models (o1, o3-mini)

```python
extra_params={
    "reasoning_effort": "high",    # low, medium, high
    "max_completion_tokens": 5000  # Required for o1
}
```

---

## Anthropic Parameters

### Thinking & Reasoning

```python
extra_params={
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000    # Thinking token budget
    }
}
```

### Sampling

```python
extra_params={
    "top_k": 40,                  # Top-k sampling
    "top_p": 0.9                  # Nucleus sampling
}
```

### System Prompts (with caching)

Use `message.metadata` instead of `extra_params`:

```python
messages=[
    Message(
        role="system",
        content="Long system prompt (1024+ tokens)...",
        metadata={
            "cache_control": {"type": "ephemeral"}
        }
    ),
    Message(role="user", content="Question")
]
```

---

## Google Parameters

### Safety Settings

```python
extra_params={
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
}
```

**Thresholds:**
- `BLOCK_NONE`
- `BLOCK_ONLY_HIGH`
- `BLOCK_MEDIUM_AND_ABOVE`
- `BLOCK_LOW_AND_ABOVE`

### Sampling & Generation

```python
extra_params={
    "top_k": 40,                  # Top-k sampling
    "top_p": 0.9,                 # Nucleus sampling
    "candidate_count": 3,         # Generate multiple candidates
    "stop_sequences": ["END"]     # Custom stop sequences
}
```

---

## Comparison Table

| Feature | OpenAI | Anthropic | Google | Access |
|---------|--------|-----------|--------|--------|
| **Sampling** |
| Temperature | ✅ | ✅ | ✅ | Standard param |
| Top-p | ✅ | ✅ | ✅ | extra_params |
| Top-k | ❌ | ✅ | ✅ | extra_params |
| Frequency penalty | ✅ | ❌ | ❌ | extra_params |
| Presence penalty | ✅ | ❌ | ❌ | extra_params |
| **Analysis** |
| Logprobs | ✅ | ❌ | ❌ | extra_params |
| Top logprobs | ✅ | ❌ | ❌ | extra_params |
| **Reasoning** |
| Thinking budget | ❌ | ✅ | ❌ | extra_params |
| Reasoning effort | ✅ | ❌ | ❌ | extra_params |
| **Safety** |
| Safety settings | ❌ | ❌ | ✅ | extra_params |
| Content filtering | ✅ | ✅ | ✅ | Moderation API |
| **Optimization** |
| Deterministic (seed) | ✅ | ❌ | ❌ | extra_params |
| Prompt caching | ❌ | ✅ | ✅ | message.metadata |
| **Output** |
| Multiple candidates | ❌ | ❌ | ✅ | extra_params |
| Parallel tools | ✅ | ❌ | ❌ | extra_params |

---

## Complete Examples

### OpenAI: Maximum Control

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="openai:gpt-4o",
        messages=[Message(role="user", content="Explain quantum computing")],
        temperature=0.7,
        max_tokens=1000,
        extra_params={
            # Sampling
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "seed": 42,

            # Analysis
            "logprobs": True,
            "top_logprobs": 5,

            # Behavior
            "parallel_tool_calls": False
        }
    )
    response = await service.chat(request)

    # Access logprobs
    if hasattr(response, 'logprobs'):
        print(response.logprobs)
```

### Anthropic: Deep Thinking

```python
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[
        Message(
            role="system",
            content="You are an expert mathematician. " * 200,  # 1024+ tokens
            metadata={"cache_control": {"type": "ephemeral"}}
        ),
        Message(role="user", content="Prove Fermat's Last Theorem")
    ],
    temperature=0.7,
    max_tokens=4000,
    extra_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 15000
        },
        "top_k": 40
    }
)
response = await service.chat(request)
```

### Google: Safe & Varied

```python
request = LLMRequest(
    model="google:gemini-2.5-pro",
    messages=[Message(role="user", content="Write a story about...")],
    temperature=0.9,
    max_tokens=2000,
    extra_params={
        # Safety
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ],

        # Sampling
        "top_k": 40,
        "top_p": 0.95,
        "candidate_count": 3,  # Generate 3 variations

        # Control
        "stop_sequences": ["THE END"]
    }
)
response = await service.chat(request)
```

---

## Getting Raw Provider Responses

For maximum control, access the raw SDK:

```python
from llmring import LLMRing

async with LLMRing() as service:
    # Access raw clients
    openai_client = service.get_provider("openai").client
    anthropic_client = service.get_provider("anthropic").client
    google_client = service.get_provider("google").client

    # Full SDK access with all parameters
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        # Any OpenAI SDK parameter
        logprobs=True,
        top_logprobs=10,
        seed=42,
        stream_options={"include_usage": True}
    )
```

---

## See Also

- [Limitations & Workarounds](./limitations-and-workarounds.md)
- [API Reference](./api-reference.md)
- [Advanced Patterns](./advanced-patterns.md)
```

---

## Summary

These drafts provide:

1. **README.md section:** Comprehensive overview of limitations, workarounds, and when to use what
2. **Website documentation page:** Detailed guide for understanding scope and accessing features
3. **Provider-specific features:** Quick reference for all provider parameters

All three work together to give users a clear understanding of:
- What llmring does well (unified API)
- What requires workarounds (extra_params, raw SDK)
- What's out of scope (real-time, frameworks)
- How to access any feature they need

Ready for review and integration into the actual documentation sites.
