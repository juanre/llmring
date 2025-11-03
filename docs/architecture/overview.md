# LLMRing Architecture Overview

**Last Updated**: 2025-09-30

## Table of Contents

1. [Introduction](#introduction)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Diagram](#component-diagram)
4. [Request Flow](#request-flow)
5. [Key Design Principles](#key-design-principles)
6. [Layer Responsibilities](#layer-responsibilities)

---

## Introduction

LLMRing is a lightweight, provider-agnostic LLM orchestration library. It provides a unified interface for interacting with multiple LLM providers (Anthropic, OpenAI, Google, Ollama) while managing costs, aliases, and provider-specific quirks.

### Core Goals

- **Provider Abstraction**: Single API for multiple LLM providers
- **Cost Tracking**: Automatic cost calculation and usage monitoring
- **Alias Management**: Model aliases via lockfiles for reproducible deployments
- **Type Safety**: Full type annotations and protocol-based design
- **Security**: Input validation and secure API handling

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Application                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                         LLMRing                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Service Orchestration Layer               │  │
│  │  - Alias Resolution                                    │  │
│  │  - Schema Adaptation                                   │  │
│  │  - Cost Calculation                                    │  │
│  │  - Usage Logging                                       │  │
│  │  - Validation                                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Provider Abstraction Layer                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │Anthropic │ │  OpenAI  │ │  Google  │ │  Ollama  │ │  │
│  │  │ Provider │ │ Provider │ │ Provider │ │ Provider │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Anthropic  │ │    OpenAI    │ │   Google AI  │
│     API      │ │     API      │ │     API      │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## Component Diagram

### Core Components

```
LLMRing (service.py)
├── Services/
│   ├── AliasResolver - Resolves model aliases from lockfiles
│   ├── SchemaAdapter - Adapts schemas for provider-specific APIs
│   ├── CostCalculator - Calculates token costs from registry
│   ├── LoggingService - Sends usage metadata to llmring-server / SaaS
│   └── ValidationService - Validates requests and context limits
│
├── Providers/
│   ├── AnthropicProvider - Anthropic Claude API
│   ├── OpenAIProvider - OpenAI GPT API
│   ├── GoogleProvider - Google Gemini API
│   ├── OllamaProvider - Ollama local models
│   ├── ProviderErrorHandler - Unified error handling
│   └── GoogleSchemaNormalizer - Google-specific schema normalization
│
├── Infrastructure/
│   ├── ModelRegistryClient - Fetches model metadata from registry
│   ├── TokenCounter - Counts tokens for cost calculation
│   ├── Lockfile - Manages model aliases and profiles
│   └── InputValidator - Security validation for user inputs
│
└── Schemas/
    ├── LLMRequest - Unified request format
    ├── LLMResponse - Unified response format
    ├── StreamChunk - Streaming response chunks
    └── Message - Chat message format
```

---

## Request Flow

### Non-Streaming Chat Request

```
1. User calls: await llmring.chat(request)
                    │
                    ▼
2. Service Layer: LLMRing.chat()
   ├─► Resolve alias (AliasResolver)
   │   └─► "gpt4" → "openai:gpt-4-turbo"
   │
   ├─► Validate input (InputValidator)
   │   └─► Check message sizes, base64 data
   │
   ├─► Validate context (ValidationService)
   │   └─► Check token limits, capabilities
   │
   ├─► Adapt schema (SchemaAdapter)
   │   └─► Convert structured output to provider format
   │
   ├─► Get provider (openai)
   │
   └─► Call provider.chat(request)
                    │
                    ▼
3. Provider Layer: OpenAIProvider._chat_non_streaming()
   ├─► Prepare messages (convert format)
   ├─► Build API parameters
   ├─► Execute API call with retries
   ├─► Parse response
   └─► Return LLMResponse
                    │
                    ▼
4. Service Layer: Post-processing
   ├─► Post-process structured output (SchemaAdapter)
   ├─► Calculate cost (CostCalculator)
   ├─► Log usage / conversations (LoggingService + ServerClient)
   └─► Return response to user
```

### Streaming Chat Request

```
1. User calls: async for chunk in llmring.chat_stream(request)
                    │
                    ▼
2. Service Layer: LLMRing.chat_stream()
   ├─► Resolve alias
   ├─► Validate input
   ├─► Validate context
   ├─► Adapt schema
   └─► Get provider
                    │
                    ▼
3. Provider Layer: OpenAIProvider._stream_chat()
   ├─► Prepare messages
   ├─► Build API parameters
   ├─► Start streaming API call
   └─► Yield StreamChunk for each delta
                    │
                    ▼
4. User processes chunks in async loop
   └─► Each chunk contains: content, role, usage (if final)
5. Optional logging when stream finishes (LoggingService)
```

### Error Handling Flow

```
Provider API Error
        │
        ▼
Provider catches exception
        │
        ▼
ProviderErrorHandler.handle_error()
   ├─► Unwrap RetryError if present
   ├─► Detect error type:
   │   ├─► TimeoutError → ProviderTimeoutError
   │   ├─► Rate limit → ProviderRateLimitError
   │   ├─► Auth error → ProviderAuthenticationError
   │   ├─► Model not found → ModelNotFoundError
   │   └─► Other → ProviderResponseError
   │
   └─► Raise typed exception with context
        │
        ▼
Service Layer catches typed exception
   └─► Optional: Try fallback provider
        │
        ▼
User receives exception with clear error message
```

### Alias Resolution Flow

```
User specifies: "gpt4"
        │
        ▼
AliasResolver.resolve("gpt4", profile="production")
   │
   ├─► Check cache (TTL: 1 hour)
   │   └─► Cache hit? Return cached result
   │
   ├─► Load lockfile
   │
   ├─► Check profile-specific aliases
   │   └─► profiles.production.aliases["gpt4"]
   │
   ├─► Check global aliases
   │   └─► aliases["gpt4"]
   │
   ├─► If not found, assume it's "provider:model" format
   │   └─► Parse "gpt4" → Error (no colon)
   │
   └─► Return "openai:gpt-4-turbo"
        │
        ▼
Parse result: provider="openai", model="gpt-4-turbo"
```

---

## Key Design Principles

### 1. Single Responsibility

Each service has one clear responsibility:
- **AliasResolver**: Only resolves aliases
- **SchemaAdapter**: Only adapts schemas
- **CostCalculator**: Only calculates costs

### 2. Dependency Inversion

Services depend on abstractions (protocols), not concrete implementations:

```python
class BaseLLMProvider(Protocol):
    async def chat(self, request: LLMRequest) -> LLMResponse: ...
    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: ...
```

Providers implement the protocol without inheriting from a base class.

### 3. Separation of Concerns

Clear boundaries between layers:
- **User API Layer** (`LLMRing`): Public interface, orchestration
- **Service Layer** (`services/`): Business logic, no provider details
- **Provider Layer** (`providers/`): API-specific implementation
- **Infrastructure Layer**: Registry, lockfiles, validation

### 4. Type Safety

Full type annotations throughout:
- No `Union[LLMResponse, AsyncIterator[StreamChunk]]`
- Separate methods: `chat()` vs `chat_stream()`
- Explicit return types, no `Any` in public APIs

### 5. Fail-Fast Validation

Validate inputs early:
1. Input validation (message sizes, base64 data)
2. Context validation (token limits, capabilities)
3. Provider validation (API key, model availability)

### 6. Provider Agnostic

Services don't know about provider details:
- SchemaAdapter handles provider-specific schema conversion
- ProviderErrorHandler maps provider errors to common exceptions
- Each provider implements the same protocol

---

## Layer Responsibilities

### User API Layer (`service.py`)

**Responsibilities:**
- Public API surface (`chat()`, `chat_stream()`, `resolve_alias()`)
- Service orchestration
- Provider registration and management
- Backward compatibility

**Does NOT:**
- Know about provider-specific APIs
- Implement business logic directly
- Handle provider-specific errors

### Service Layer (`services/`)

**Responsibilities:**
- Business logic implementation
- Provider-agnostic operations
- Caching and optimization
- State management

**Examples:**
- `AliasResolver`: Lockfile parsing, cache management
- `CostCalculator`: Registry fetching, cost computation
- `ValidationService`: Token estimation, capability checking

### Provider Layer (`providers/`)

**Responsibilities:**
- API-specific implementation
- Message format conversion
- Streaming implementation
- Error translation

**Examples:**
- `AnthropicProvider`: Anthropic API calls, beta headers
- `OpenAIProvider`: OpenAI API calls, PDF document handling
- `GoogleProvider`: Gemini API calls, threading for streaming

### Infrastructure Layer

**Responsibilities:**
- External integrations (registry, lockfiles)
- Token counting
- Security validation

**Examples:**
- `ModelRegistryClient`: GitHub Pages registry fetching
- `Lockfile`: TOML/JSON parsing, alias resolution
- `InputValidator`: Base64 size validation, URL validation

---

## Cross-Cutting Concerns

### Error Handling

All layers use typed exceptions from `exceptions.py`:
- `ProviderError` - Base class for provider errors
- `ProviderTimeoutError` - Timeout during API call
- `ProviderRateLimitError` - Rate limit exceeded
- `ProviderAuthenticationError` - Invalid API key
- `ModelNotFoundError` - Model not available
- `ProviderResponseError` - Other API errors

### Logging

Structured logging throughout:
```python
logger.debug(f"Resolved alias '{alias}' to '{resolved}'")
logger.info(f"Chat request completed in {duration:.2f}s")
logger.warning(f"Cost information unavailable for {model}")
logger.error(f"Provider error: {error}", exc_info=True)
```

### Caching

Consistent caching strategy using `cachetools.TTLCache`:
- Alias resolution: 1 hour TTL
- Registry models: 24 hours TTL
- Token encodings: 1 hour TTL

### Testing

No mocks per `CLAUDE.md` - all tests use real functionality:
- Unit tests for services (mock external APIs)
- Integration tests for providers (use real APIs with skip flags)
- E2E tests for full request flow

---

## Extension Points

### Adding a New Provider

1. Implement `BaseLLMProvider` protocol
2. Add provider to `PROVIDER_CLASSES` in `service.py`
3. Add error detection rules to `ProviderErrorHandler`
4. Add integration tests with `@pytest.mark.skipif`

### Adding a New Service

1. Create service class in `src/llmring/services/`
2. Initialize in `LLMRing.__init__()`
3. Add delegation methods in `LLMRing`
4. Add comprehensive unit tests

### Adding a New Capability

1. Add field to `LLMRequest` schema
2. Add validation in `ValidationService`
3. Update provider implementations
4. Add registry support for capability checking

---

## Performance Considerations

### Caching Strategy

- Alias resolution cached for 1 hour (lockfile rarely changes)
- Registry models cached for 24 hours (pricing changes slowly)
- Token encodings cached for 1 hour (limited model types)

### Async/Await

- All I/O operations are async (API calls, file reads)
- No blocking calls in hot path
- Proper async context manager usage

### Token Estimation

Two-stage approach:
1. Quick character-based check (4 chars/token estimate)
2. Full tokenization only if likely to fit

### Streaming

- Zero-copy streaming where possible
- Immediate chunk yielding (no buffering)
- Proper cleanup on cancellation

---

## Security Considerations

### Input Validation

- Base64 data size limits (50MB encoded, 100MB decoded)
- Message content size limits (1MB per message)
- URL validation for registry (HTTPS or file:// only)

### Error Messages

- No sensitive data in error messages
- Stack traces only in debug mode
- Sanitized API errors

---

## Related Documentation

- [Service Layer Documentation](./services.md)
- [Provider Layer Documentation](./providers.md)
- [Architecture Decision Records](../decisions/)
