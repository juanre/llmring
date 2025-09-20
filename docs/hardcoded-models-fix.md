# LLMRing: Removing Hardcoded Model IDs - Implementation Plan

## Overview

This document outlines the plan to remove all hardcoded model IDs from the LLMRing codebase, ensuring full compliance with the source-of-truth principle that requires all model information to come from the registry.

## Principles

Per source-of-truth v4.1:
- **No user-facing hardcoded model IDs** anywhere in the codebase
- **Registry is the single source of truth** for model capabilities and pricing
- **Registry-backed provider validation** for all model references
- **Intelligent lockfile creation** uses registry data for recommendations

## Issues to Fix

### 1. Hardcoded Model IDs in Lockfile Defaults

**Location**: `llmring/src/llmring/lockfile.py:_suggest_defaults()`

**Current State**:
```python
defaults["long_context"] = "openai:gpt-4-turbo-preview"
defaults["deep"] = "anthropic:claude-3-opus-20240229"
defaults["vision"] = "google:gemini-1.5-pro"
```

**Solution**:
- Query registry for available models based on API keys present
- Select models based on capabilities (e.g., highest max_input_tokens for "long_context")
- Use pricing data to select appropriate "low_cost" options
- Make the method async to support registry queries

### 2. Model Name Inference Logic

**Location**: `llmring/src/llmring/service.py:_parse_model_string()`

**Current State**:
```python
if model.startswith("gpt"):
    return "openai", model
elif model.startswith("claude"):
    return "anthropic", model
elif model.startswith("gemini"):
    return "google", model
```

**Solution**:
- Remove all inference logic
- Require explicit provider:model format
- Return error for ambiguous model names without provider prefix
- Update documentation to reflect this requirement

### 3. Provider Fallback Models

**Locations**:
- `google_api.py`: `"gemini-1.5-flash"`
- `anthropic_api.py`: Similar patterns
- `openai_api.py`: Similar patterns

**Current State**:
```python
self.default_model = "gemini-1.5-flash"
```

**Solution**:
- Query registry for active models from the provider
- Select fallback based on criteria (e.g., lowest cost, most capable)
- Cache the fallback selection
- Handle case where no models are available

## Implementation Steps

### Phase 1: Immediate Actions

#### 1.1 Fix Lockfile Defaults
- Make `_suggest_defaults()` async
- Add `RegistryClient` parameter
- Query registry for each provider based on available API keys
- Select models based on capabilities:
  - `long_context`: Highest `max_input_tokens`
  - `low_cost`: Lowest `dollars_per_million_tokens_input`
  - `deep`: Balance of capability and cost
  - `vision`: `supports_vision=true`
  - `json_mode`: `supports_json_mode=true`

#### 1.2 Remove Model Inference
- Simplify `_parse_model_string()` to only handle explicit provider:model format
- Remove all prefix-based inference
- Add clear error messages for invalid formats

#### 1.3 Fix Provider Fallbacks
- Add registry client to provider initialization
- Query registry in `_get_default_model()` methods
- Implement fallback selection logic based on capabilities
- Add caching to avoid repeated registry queries

### Phase 2: Near-term Improvements

#### 2.1 Registry Version Pinning
- Update `LLMRing.validate_model()` to use pinned registry versions from lockfile
- Add version parameter to registry queries
- Implement version mismatch warnings
- Support `llmring lock bump-registry` command properly

### Phase 3: Architecture Enhancements

#### 3.1 Alias Resolution Caching
- Add LRU cache for resolved aliases in `LLMRing` service
- Cache key: (alias, profile) tuple
- Cache value: resolved provider:model string
- Invalidate on lockfile reload
- Configurable cache size and TTL

## Testing Strategy

1. **Unit Tests**:
   - Mock registry responses for lockfile default generation
   - Test model parsing without inference
   - Verify provider fallback selection logic

2. **Integration Tests**:
   - Test against live registry (when available)
   - Verify lockfile creation with real API keys
   - Test provider initialization with registry

3. **CI Checks**:
   - Grep for hardcoded model patterns
   - Fail build if user-facing hardcoded models detected
   - Allow exceptions only for test fixtures

## Migration Path

1. **Backwards Compatibility**:
   - Existing lockfiles continue to work
   - Model inference removed but clear error messages guide users
   - Registry fallback for missing models

2. **User Communication**:
   - Update CLI to show registry source for recommendations
   - Add `--verbose` flag to show registry queries
   - Clear error messages for model format requirements

## Success Criteria

- [ ] No hardcoded model IDs in src/ directory (except test fixtures)
- [ ] All model selection uses registry data
- [ ] CI gates prevent regression
- [ ] Existing lockfiles continue to function
- [ ] Clear migration path for users

## Timeline

- **Week 1**: Implement Phase 1 (Immediate Actions)
- **Week 2**: Implement Phase 2 (Registry Version Pinning)
- **Week 3**: Implement Phase 3 (Caching) and testing
- **Week 4**: Documentation and user communication

## Risk Mitigation

1. **Registry Unavailability**:
   - Use cached registry data when available
   - Fail gracefully with clear error messages
   - Consider embedded minimal registry for bootstrap

2. **Performance Impact**:
   - Cache registry queries aggressively
   - Async operations where possible
   - Batch registry requests

3. **Breaking Changes**:
   - Deprecation warnings before removal
   - Clear migration documentation
   - Support period for old format