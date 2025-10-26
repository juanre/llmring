# Changelog

All notable changes to LLMRing will be documented in this file.

## [1.2.0] - 2025-10-28

### 🚀 Features

- **Detailed cost breakdowns**: Prompt caching (reads/writes), long-context tiers, and thinking tokens are now itemised in every `LLMResponse.usage.cost_breakdown` entry and propagated to receipts/logging APIs.
- **Cache-aware pricing**: Prompt cache reads/writes (`cache_read_input_tokens`, `cache_creation_*`) bill against provider-specific rates automatically.
- **Long-context support**: When requests exceed a model's `long_context_threshold_tokens`, LLMRing switches to the higher-tier pricing without custom code.

### 🔧 Improvements

- **Registry schema**: Added caching, long-context, and thinking-token pricing fields (`dollars_per_million_tokens_cached_input`, `dollars_per_million_tokens_cache_write_{5m,1h}`, `supports_long_context_pricing`, `supports_thinking`, etc.).
- **Cost calculator**: Refactored to handle cache reads/writes, reasoning tokens, and long-context overflow while preventing double billing.
- **Receipts & logs**: Include per-feature cost breakdown and extended token counters for downstream analytics.

### 🧪 Testing

- **Cost calculator**: Added unit coverage for cache read/write pricing, reasoning-token separation, and long-context thresholds.

---

## [1.1.1] - 2025-10-14

### 🚀 Features

#### Reasoning Model Support
- **NEW**: Support for OpenAI reasoning models (o1, o3, gpt-5 series)
- **NEW**: `reasoning_tokens` parameter for controlling internal reasoning budget
- **NEW**: Registry-based detection of reasoning models via `is_reasoning_model` flag
- **NEW**: `max_completion_tokens` used instead of `max_tokens` for reasoning models
- **NEW**: `min_recommended_reasoning_tokens` field in registry for optimal settings

#### MCP Enhancements
- **IMPROVED**: Prepopulate lockfile chat with registry models for better recommendations

### 🔧 Improvements

#### Provider Updates
- **ADDED**: `reasoning_tokens` parameter to Anthropic provider signature (ignored, for API consistency)
- **FIXED**: OpenAI reasoning model configuration with proper defaults
- **IMPROVED**: Registry model schema validation for reasoning capabilities

#### Testing
- **FIXED**: Integration tests to handle 404 errors gracefully
- **FIXED**: Receipt test expectations for automatic generation
- **UPDATED**: Test registry data with correct reasoning token recommendations
- **ADDED**: OpenAI validation tests for reasoning models

#### Dependencies
- **FIXED**: Use PyPI version of llmring-server (>=0.1.2) instead of local path
- **REMOVED**: Local development path override for llmring-server

#### Documentation
- **IMPROVED**: Removed marketing language and emojis from README for professional tone

### 🐛 Bug Fixes
- **FIXED**: Reasoning model detection using registry instead of hardcoded list
- **FIXED**: Token parameter selection based on model capabilities
- **FIXED**: Test failures related to receipt generation behavior

### 📦 Technical Details
- Reasoning models use `max_completion_tokens` parameter (OpenAI API requirement)
- Non-reasoning models continue using standard `max_tokens` parameter
- Registry provides single source of truth for model capabilities
- Automatic fallback to hardcoded defaults if registry unavailable

---

## [1.1.0] - 2025-09-29

### 📚 Documentation

#### Comprehensive Documentation Overhaul (2025-09-29)
- **NEW**: `docs/file-utilities.md` - Complete guide to vision/multimodal file handling (10 functions)
- **NEW**: `docs/cli-reference.md` - Comprehensive CLI command reference (15+ commands)
- **NEW**: `docs/receipts.md` - Cost tracking and receipt system documentation
- **NEW**: `docs/lockfile.md` - Complete lockfile configuration guide with JSON format
- **NEW**: `docs/mcp.md` - MCP integration guide (replaces mcp-integration.md and mcp-chat-client.md)
- **RENAMED**: `docs/provider-usage.md` → `docs/providers.md` with use-case explanations
- **ENHANCED**: README.md - Fixed MCP import paths, improved examples, standardized terminology
- **ENHANCED**: `docs/api-reference.md` - Added LLMRingExtended, ConversationManager, alias cache docs
- **FIXED**: Import path: `from llmring.mcp.client import create_enhanced_llm` (was .enhanced_llm)
- **IMPROVED**: All code examples verified against source code for accuracy
- **IMPROVED**: Standardized terminology: Alias, Model Reference, Raw SDK Access
- **IMPROVED**: Added cross-references throughout all documentation

### 🚀 Features

#### Fallback Models (2025-09-20)
- **NEW**: Aliases can specify multiple models for automatic failover
- **Example**: `models = ["anthropic:claude-3-5-sonnet", "openai:gpt-4o", "google:gemini-1.5-pro"]`
- **Behavior**: If primary model fails (rate limit, availability), automatically tries fallbacks
- **Breaking**: Removed backward compatibility with old single-model alias format

#### MCP Conversational Lockfile Management (2025-09-18)
- **NEW**: `llmring lock chat` - Natural language lockfile configuration
- **NEW**: AI-powered recommendations based on live registry analysis
- **NEW**: Persistent chat history in `~/.llmring/mcp_chat/`
- **NEW**: Complete tool execution loop with native function calling
- **NEW**: Session management with save/load capabilities
- **Enhanced**: MCP tools explain fallback models and provide better guidance

#### Lockfile Packaging (2025-09-17)
- **NEW**: Package lockfiles with your library distribution
- **NEW**: Bundled fallback lockfile with `advisor` alias
- **NEW**: Project root discovery for optimal lockfile placement
- **IMPROVED**: Lockfile resolution order documented and consistent
- **FIXED**: Lockfile placement uses package directory for distribution

### 🔧 Improvements

#### Provider Enhancements
- **FIXED**: Temperature parameter filtering for models that don't support it (2025-09-29)
- **IMPROVED**: Google tool-calling loop and error handling (2025-09-21)
- **FIXED**: Google FunctionResponse formatting for tool errors (2025-09-21)
- **IMPROVED**: Code quality and performance across all providers (2025-09-27)

#### CLI & Tools
- **REMOVED**: Hardcoded model IDs and obsolete CLI commands (2025-09-19)
- **IMPROVED**: MCP tool parameter naming for better clarity
- **IMPROVED**: MCP advisor understanding of configuration concepts

#### Testing
- **REPLACED**: All mocks with real integration tests
- **FIXED**: Test failures after fallback models implementation
- **IMPROVED**: Test fixtures and alias usage

### 🐛 Bug Fixes
- **FIXED**: Google provider parts parsing (use parts not response.text)
- **FIXED**: Function call/response parts in conversation history
- **FIXED**: MCP client initialization and tool discovery
- **FIXED**: Lockfile manager class structure
- **FIXED**: Conversational lockfile tests

### 📦 Maintenance
- **MOVED**: `demo_mcp_lockfile.py` to examples/ directory (2025-09-29)
- **REMOVED**: Obsolete `replicate_google_loop.py` test file
- **IMPROVED**: Pre-commit hooks and code formatting
- **REFACTORED**: Eliminated technical debt across codebase

### ⚠️ Breaking Changes
- **Fallback Models**: Old single-model alias format no longer supported
- **Lockfile Format**: Now requires `models` array instead of single `model` string
- **Migration**: Update lockfiles to use `models = ["provider:model"]` array format

---

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

#### Intelligent Configuration Management
- **NEW**: AI-powered lockfile creation with `llmring lock init --interactive`
- **Registry-driven**: All recommendations based on live registry analysis, not hardcoded defaults
- **Self-hosted**: Uses LLMRing's own API with advisor model to power intelligent recommendations
- **Smart aliases**: Always-current semantic aliases that adapt as registry evolves
- **Cost-aware**: Transparent cost analysis and optimization suggestions
- **CLI tools**: `analyze`, `optimize`, `validate` commands for lockfile management

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
