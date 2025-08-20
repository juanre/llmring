# Technical Specification

## Architecture

### System Components

```
[Lockfile] → [Service] → [Provider APIs]
     ↓           ↓            ↓
  [Config]   [Registry]   [LLM Response]
```

1. **Lockfile**: Configuration file mapping aliases to models with profiles
2. **Registry**: Model metadata served from GitHub Pages (capabilities, pricing, limits)
3. **Service**: Stateless routing layer that resolves aliases and forwards requests
4. **Providers**: Direct API integration with OpenAI, Anthropic, Google, Ollama

### Data Flow

1. User references alias (e.g., "summarizer")
2. Service resolves alias via lockfile
3. Registry provides model metadata (cached 24h)
4. Request forwarded to provider API
5. Response returned with usage data
6. Cost calculated from registry pricing

### Caching Strategy

Three-tier caching minimizes network requests:

- **Memory**: Process lifetime, < 1ms access
- **Disk**: `~/.cache/llmring/registry/`, 24h TTL, 1-5ms access  
- **Service**: Instance lifetime cache

## Lockfile Specification

### Format

TOML (default) or JSON supported:

```toml
version = "1.0"
default_profile = "default"

[[profiles.default.bindings]]
alias = "summarizer"
provider = "openai"
model = "gpt-4o-mini"
constraints = { temperature = 0.3, max_tokens = 500 }

[profiles.default.registry_versions]
openai = 3
anthropic = 2
```

### Schema

- **version**: Lockfile format version (currently "1.0")
- **default_profile**: Active profile name
- **profiles**: Map of profile configurations
  - **bindings**: List of alias→model mappings
    - **alias**: Semantic task name
    - **provider**: LLM provider name
    - **model**: Model identifier
    - **constraints**: Optional parameters (temperature, max_tokens, etc.)
  - **registry_versions**: Pinned registry versions for drift detection

## Registry Specification

### Structure

```
registry/
├── openai/models.json
├── anthropic/models.json
└── google/models.json
```

### Model Schema

```json
{
  "provider": "openai",
  "version": 1,
  "updated_at": "2025-08-20T12:00:00Z",
  "models": {
    "openai:gpt-4": {
      "model_name": "gpt-4",
      "display_name": "GPT-4",
      "max_input_tokens": 8192,
      "max_output_tokens": 4096,
      "dollars_per_million_tokens_input": 30.00,
      "dollars_per_million_tokens_output": 60.00,
      "supports_vision": false,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "is_active": true
    }
  }
}
```

### Fields

- **model_name**: Provider's model identifier
- **display_name**: Human-readable name
- **max_input_tokens**: Context window limit
- **max_output_tokens**: Generation limit
- **dollars_per_million_tokens_***: Pricing
- **supports_***: Capability flags
- **is_active**: Availability status

## API Specification

### Core Methods

```python
class LLMRing:
    async def chat(request: LLMRequest) -> LLMResponse
    async def chat_with_alias(alias: str, messages: List, **kwargs) -> LLMResponse
    async def calculate_cost(response: LLMResponse) -> Dict[str, float]
    async def validate_context_limit(request: LLMRequest) -> Optional[str]
    
    def bind_alias(alias: str, model: str, profile: str = None) -> None
    def resolve_alias(alias: str, profile: str = None) -> str
    def list_aliases(profile: str = None) -> Dict[str, str]
```

### Request/Response

```python
class LLMRequest:
    messages: List[Message]
    model: str  # Alias or provider:model
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    json_response: Optional[bool]

class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    total_tokens: int  # Computed property
```

## Security Model

### API Key Management

- Keys only from environment variables
- Never stored in files or lockfile
- No transmission to LLMRing infrastructure

### Data Privacy

- No telemetry or usage tracking
- No request/response logging
- Direct provider API calls only
- Local cache contains only public metadata

### Network Security

- HTTPS required for production registry
- File:// URLs only for testing
- Proxy support via environment
- Timeouts on all network operations

## Performance Characteristics

### Latency

- Alias resolution: < 1ms
- Registry cache hit: 1-5ms
- Registry network fetch: 50-500ms
- Provider API call: 100-2000ms

### Memory Usage

- Registry cache: ~200KB total
- Service instance: ~10MB base
- Per request: ~1KB + message size

### Scalability

- Stateless service (horizontal scaling)
- No database dependencies
- Registry CDN distributed
- Provider rate limits apply

## Deployment

### Environment Variables

```bash
# Required (at least one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional
LLMRING_PROFILE=prod
LLMRING_REGISTRY_URL=https://custom.registry.com
HTTPS_PROXY=https://proxy:8080
```

### Production Checklist

1. Use secrets management for API keys
2. Pin registry versions in lockfile
3. Set appropriate profile (prod)
4. Monitor API usage and costs
5. Implement retry logic for failures
6. Cache registry data for resilience

## Testing

### Mock Registry

```python
# Use local test files
ring = LLMRing(registry_url="file:///tests/registry")
```

### Test Structure

```
tests/resources/registry/
├── openai/models.json
├── anthropic/models.json
└── google/models.json
```

### Integration Tests

```python
# Test with live registry
TEST_LIVE_REGISTRY=1 pytest tests/integration/

# Test with mock data
pytest tests/unit/
```