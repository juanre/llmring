# Changelog

All notable changes to LLMRing will be documented in this file.

## [1.0.0] - 2025-09-13

**🎉 First Stable Release!**

This major version release includes significant breaking changes and new features.
The extensive breaking changes (provider interface, exception handling, service layer)
warrant a major version bump to 1.0.0, signaling production stability.

### 🚀 Major Features

#### Unified Structured Output
- **NEW**: JSON schema support across all providers with single API
- **OpenAI**: Native JSON schema with strict mode (enhanced)
- **Anthropic**: Automatic tool injection for schema enforcement
- **Google Gemini**: Function calling approach with schema validation
- **Ollama**: Best-effort with optional retry and validation
- **API**: Added `response.parsed` field for direct Python object access
- **Validation**: Optional strict mode with jsonschema validation

#### Provider Enhancements
- **Google Gemini**: Real streaming via `generate_content_stream` (no more simulation)
- **Google Gemini**: Native function calling via `types.Tool` and `FunctionDeclaration`
- **Google Gemini**: Honor user model versions (stop 2.x→1.5 downgrades)
- **Google Gemini**: Tool choice support with proper mode mapping
- **Google Gemini**: Non-blocking streaming with thread executor
- **Ollama**: Real streaming via SDK `stream=True` (no more simulation)
- **Ollama**: Enhanced options passthrough for advanced parameters
- **OpenAI**: JSON schema strict mode support
- **Anthropic**: Prompt caching with 90% cost savings (enhanced)

### 🔧 API Improvements

#### Unified Interface
- **BREAKING**: Base class signature unified across all providers
- **NEW**: `extra_params` passthrough for provider-specific features
- **Enhanced**: All providers support identical method signatures
- **Fixed**: LSP compliance - base class matches implementations

#### Error Handling
- **NEW**: Comprehensive typed exception hierarchy
- **Enhanced**: `ProviderAuthenticationError`, `ModelNotFoundError`, `ProviderRateLimitError`, etc.
- **Fixed**: All providers use typed exceptions instead of generic `ValueError`
- **Enhanced**: Exception context includes provider and model information

#### Service Layer
- **Fixed**: `service_extended.py` uses `response.content` (not `response.choices`)
- **Fixed**: MCP client uses metadata for correlation IDs (not invalid kwargs)
- **Enhanced**: All request parameters properly passed to providers

### 📊 Quality & Testing

#### Test Coverage
- **NEW**: 284+ tests with 100% pass rate
- **Enhanced**: Real API testing (no mocks) validates actual functionality
- **Fixed**: All exception expectations match typed exception behavior
- **Enhanced**: Provider-specific test coverage unlocked

#### Documentation
- **NEW**: Comprehensive README.md with current features
- **NEW**: `docs/structured-output.md` - Complete structured output guide
- **NEW**: `docs/api-reference.md` - Full API documentation
- **NEW**: `docs/mcp-integration.md` - MCP usage guide
- **NEW**: `docs/provider-usage.md` - Provider-specific examples
- **Cleaned**: Removed process artifacts and outdated documentation

### 🛠️ Developer Experience

#### Features
- **Enhanced**: Provider auto-detection with proper API key handling
- **Enhanced**: Real streaming for all providers
- **Enhanced**: Native tool calling where supported
- **Enhanced**: Cost tracking and receipt generation
- **Enhanced**: Circuit breaker and retry logic

#### Configuration
- **Fixed**: Environment variable detection (`GOOGLE_GEMINI_API_KEY`, etc.)
- **Enhanced**: Conditional test skipping based on service availability
- **Enhanced**: Proper lockfile integration

### 📦 Dependencies

#### Added
- `jsonschema>=4.0.0` - For structured output validation

#### Updated
- `anthropic>=0.67.0` - Latest features and prompt caching
- `openai>=1.99.0` - JSON schema support
- All provider SDKs updated to latest versions

### 🐛 Bug Fixes

#### Critical Fixes
- **Fixed**: Ollama provider returning `None` (missing return statement)
- **Fixed**: Error detection case sensitivity (`"API key"` vs `"api key"`)
- **Fixed**: Exception double-wrapping (LLMRingError passthrough)
- **Fixed**: Google tool parameter handling (tools in config, not direct)
- **Fixed**: Service layer exception handling (typed exceptions)

#### Provider Fixes
- **Fixed**: Google API key detection and provider initialization
- **Fixed**: Anthropic error pattern matching (`authentication_error`, `x-api-key`)
- **Fixed**: OpenAI streaming tool call accumulation
- **Fixed**: PDF comment accuracy (no vector stores implemented)

### ⚠️ Breaking Changes

#### Provider Interface (Major Breaking Change)
- **BREAKING**: `BaseLLMProvider.chat()` signature changed from `chat(request: LLMRequest)` to `chat(messages, model, temperature, ...)` with explicit parameters
- **Impact**: Custom provider implementations must update their `chat()` method signature
- **Migration**: Update provider classes to match new signature with `extra_params` parameter

#### Exception Handling (Breaking Change)
- **BREAKING**: All providers now raise typed exceptions instead of generic `ValueError`/`Exception`
- **Changed**:
  - Invalid API key: `ValueError` → `ProviderAuthenticationError`
  - Unsupported model: `ValueError` → `ModelNotFoundError`
  - Rate limiting: `Exception` → `ProviderRateLimitError`
  - Timeouts: `Exception` → `ProviderTimeoutError`
  - API errors: `Exception` → `ProviderResponseError`
- **Impact**: Exception handling code must be updated to catch specific exception types
- **Migration**: Replace generic `except ValueError` with specific exception types from `llmring.exceptions`

#### Service Layer Changes
- **BREAKING**: `service_extended.py` response access pattern changed
- **Changed**: Code accessing `response.choices[0].message.content` will fail
- **Fixed**: Now uses `response.content` directly
- **Impact**: Custom code using service_extended must update response access

#### MCP Client Changes
- **BREAKING**: `LLMRing.chat()` no longer accepts `id_at_origin` parameter
- **Changed**: Correlation IDs now passed via `LLMRequest.metadata["id_at_origin"]`
- **Impact**: MCP integration code must update parameter passing

#### Response Schema Enhancement
- **NEW**: Added `LLMResponse.parsed` field for structured output
- **Non-breaking**: Existing code continues to work, new field optional

### 🎯 Migration Guide

#### From 0.3.x to 1.0.0

**Exception Handling:**
```python
# Old (0.3.x)
try:
    response = await service.chat(request)
except ValueError as e:
    print("Error:", e)

# New (1.0.0)
from llmring.exceptions import ProviderAuthenticationError, ModelNotFoundError

try:
    response = await service.chat(request)
except ProviderAuthenticationError:
    print("Invalid API key")
except ModelNotFoundError:
    print("Model not supported")
```

**Structured Output:**
```python
# New in 1.0.0 - Works across all providers
request = LLMRequest(
    model="any-provider",
    response_format={
        "type": "json_schema",
        "json_schema": {"schema": {...}},
        "strict": True
    }
)

response = await service.chat(request)
data = response.parsed  # Python dict ready to use
```

---

## [0.3.0] and Earlier

See git history for previous versions.