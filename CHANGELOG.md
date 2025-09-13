# Changelog

All notable changes to LLMRing will be documented in this file.

## [0.4.0] - 2025-09-13

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

### 🔄 Breaking Changes

- **Provider interface**: All providers now use explicit parameters instead of single `LLMRequest`
- **Exception types**: Generic `ValueError`/`Exception` replaced with typed exceptions
- **Response format**: Enhanced with `parsed` field for structured output

### 🎯 Migration Guide

#### From 0.3.x to 0.4.0

**Exception Handling:**
```python
# Old (0.3.x)
try:
    response = await service.chat(request)
except ValueError as e:
    print("Error:", e)

# New (0.4.0)
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
# New in 0.4.0 - Works across all providers
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