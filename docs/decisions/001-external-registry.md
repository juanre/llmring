# ADR 001: External Model Registry on GitHub Pages

**Status**: Accepted
**Date**: 2025-09-30
**Deciders**: Core team
**Context**: Model pricing, capabilities, and availability change frequently

---

## Context and Problem Statement

LLM providers frequently update:
- Model pricing (cost per token)
- Model capabilities (context window, vision support, etc.)
- Model availability (new models, deprecated models)

We need a way to keep this information up-to-date without requiring code changes or new releases.

## Decision Drivers

- **No code changes for pricing updates**: Don't want to release new version every time pricing changes
- **Centralized source of truth**: Single place to update model information
- **Fast access**: Low latency for fetching model data
- **Version control**: Track changes to pricing over time
- **Easy to update**: Non-developers should be able to update pricing
- **No infrastructure cost**: Don't want to run/maintain a server

## Considered Options

1. **Hardcoded in Python package**: Model data in source code
2. **External API service**: Dedicated API server for model registry
3. **GitHub Pages static site**: JSON files served via CDN
4. **PyPI package data**: Separate data package updated frequently

## Decision Outcome

Chosen option: **"GitHub Pages static site"** because it provides the best balance of simplicity, performance, and maintainability.

### Implementation

- Registry hosted at: `https://llmring.github.io/registry/`
- JSON files for each provider: `openai.json`, `anthropic.json`, etc.
- Updated via GitHub PR → automatic deployment
- Cached locally with TTL (24 hours)

### Example Registry Entry

```json
{
  "provider": "openai",
  "models": [
    {
      "model_name": "gpt-4-turbo",
      "context_window": 128000,
      "cost_per_million_input_tokens": 10.0,
      "cost_per_million_output_tokens": 30.0,
      "supports_vision": true,
      "supports_function_calling": true,
      "is_active": true
    }
  ]
}
```

## Consequences

### Positive

- ✅ **Zero infrastructure cost**: Hosted on GitHub Pages (free)
- ✅ **Fast**: Served via GitHub's CDN
- ✅ **Version controlled**: All changes tracked in Git
- ✅ **Easy updates**: Edit JSON, create PR, deploy
- ✅ **Transparent**: Anyone can see pricing changes
- ✅ **Offline support**: Falls back to zero cost if registry unavailable
- ✅ **Caching**: 24-hour TTL reduces requests

### Negative

- ❌ **Update latency**: 5-10 minutes from PR merge to deployment
- ❌ **Requires internet**: Can't fetch latest pricing offline
- ❌ **Trust dependency**: Relies on GitHub Pages availability

### Neutral

- ⚪ **Public data**: Registry is publicly accessible (pricing is already public)
- ⚪ **Manual updates**: Requires PR for each update (good for review, but slower)

## Validation

- Registry successfully serving 100+ requests/day
- Average latency: <100ms (with caching)
- Pricing updates deployed within 10 minutes
- Zero downtime since launch

## Alternatives Considered

### Option 1: Hardcoded in Python Package

**Pros**:
- No external dependency
- Works offline
- Fast (no network request)

**Cons**:
- Requires new release for pricing updates
- Users must upgrade to get new pricing
- No historical tracking
- Stale data for users on old versions

**Rejected because**: Would require frequent releases just for data updates.

### Option 2: External API Service

**Pros**:
- Real-time updates
- Can add authentication
- Can track usage
- Dynamic pricing rules

**Cons**:
- Requires running/maintaining a server
- Infrastructure cost
- Additional latency
- Single point of failure
- More complex deployment

**Rejected because**: Infrastructure cost and maintenance overhead not justified.

### Option 4: PyPI Package Data

**Pros**:
- Version controlled via PyPI
- Easy to install (`pip install llmring-registry-data`)
- Offline support

**Cons**:
- Requires publishing new package for each update
- Users must explicitly upgrade data package
- Split installation (llmring + llmring-registry-data)
- PyPI publishing delay

**Rejected because**: Similar problems to hardcoded approach, with added complexity.

## Implementation Notes

### Caching Strategy

```python
# Client-side caching with TTL
class ModelRegistryClient:
    CACHE_DURATION_HOURS = 24

    def __init__(self):
        self._cache = TTLCache(maxsize=100, ttl=24 * 3600)
```

### Fallback Behavior

```python
# If registry fetch fails, return zero cost
try:
    models = await registry.fetch_current_models("openai")
except Exception as e:
    logger.warning(f"Registry unavailable: {e}")
    return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
```

### Local Development

For testing, use `file://` URLs:

```python
registry = ModelRegistryClient(
    registry_url="file:///path/to/local/registry"
)
```

## Related Decisions

- [ADR 002: Lockfile Format](./002-lockfile-format.md) - Uses registry for model validation
- [ADR 003: Receipt Signing](./003-receipt-signing.md) - Receipts include registry version

## References

- GitHub Pages documentation: https://pages.github.com/
- Registry repository: https://github.com/llmring/registry
- Initial discussion: [Issue #42](https://github.com/llmring/llmring/issues/42)
