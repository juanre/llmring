# Service Layer Documentation

**Last Updated**: 2025-09-30

## Table of Contents

1. [Overview](#overview)
2. [AliasResolver](#aliasresolver)
3. [SchemaAdapter](#schemaadapter)
4. [CostCalculator](#costcalculator)
5. [ReceiptManager](#receiptmanager)
6. [ValidationService](#validationservice)
7. [Service Interaction Patterns](#service-interaction-patterns)

---

## Overview

The service layer contains business logic extracted from the main `LLMRing` class. Each service has a single, well-defined responsibility and operates independently of provider-specific details.

### Design Principles

1. **Single Responsibility**: Each service does one thing well
2. **Provider Agnostic**: No knowledge of provider-specific APIs
3. **Testable**: Easy to unit test in isolation
4. **Stateless (mostly)**: Minimal state, clear lifecycle
5. **Composable**: Services can be used together or independently

### Service Location

All services are in `src/llmring/services/`:

```
src/llmring/services/
├── __init__.py
├── alias_resolver.py      # Alias resolution from lockfiles
├── schema_adapter.py      # Schema adaptation for providers
├── cost_calculator.py     # Cost calculation from registry
├── receipt_manager.py     # Receipt generation and management
└── validation_service.py  # Request validation
```

---

## AliasResolver

**File**: `src/llmring/services/alias_resolver.py`

### Purpose

Resolves model aliases from lockfiles to concrete `provider:model` strings. Supports profile-specific aliases and caching for performance.

### Responsibilities

- Load and parse lockfiles (TOML/JSON)
- Resolve aliases with profile support
- Cache resolved aliases with TTL
- Validate lockfile format
- Handle missing aliases gracefully

### Key Methods

#### `resolve(model: str, profile: Optional[str] = None) -> str`

Resolves a model string or alias to a concrete `provider:model` format.

**Algorithm**:
1. Check cache (TTL: 1 hour)
2. If contains `:`, parse as `provider:model` directly
3. Check profile-specific aliases (`profiles.{profile}.aliases`)
4. Check global aliases (`aliases`)
5. Raise error if not found

**Example**:
```python
resolver = AliasResolver(lockfile)

# Resolve alias
model = resolver.resolve("gpt4", profile="production")
# Returns: "openai:gpt-4-turbo"

# Direct provider:model (no lockfile lookup)
model = resolver.resolve("openai:gpt-4")
# Returns: "openai:gpt-4"
```

#### `update_available_providers(available_providers: List[str])`

Updates the set of available providers for validation.

**Use Case**: When a new provider is registered at runtime, sync the resolver.

```python
llmring.register_provider("custom", CustomProvider())
resolver.update_available_providers(["openai", "anthropic", "custom"])
```

#### `clear_cache()`

Clears the alias resolution cache.

**Use Case**: After lockfile changes, force re-resolution.

```python
resolver.clear_cache()
```

### Caching Strategy

Uses `cachetools.TTLCache`:
- **Max Size**: 100 entries
- **TTL**: 1 hour (lockfiles rarely change during runtime)
- **Key**: `(model, profile)` tuple
- **Value**: Resolved `provider:model` string

### Error Handling

- `ValueError`: If alias not found in lockfile
- `FileNotFoundError`: If lockfile path invalid
- `json.JSONDecodeError` / `tomli.TOMLDecodeError`: If lockfile malformed

### Testing

See `tests/unit/test_alias_resolver.py` (15 tests):
- Alias resolution with profiles
- Cache behavior
- Error handling
- Edge cases (empty lockfile, missing profiles)

---

## SchemaAdapter

**File**: `src/llmring/services/schema_adapter.py`

### Purpose

Adapts structured output schemas for providers that don't have native `json_schema` support. Converts OpenAI-style `json_schema` requests to provider-specific formats.

### Responsibilities

- Detect `json_schema` requests
- Convert to provider-specific format:
  - **Anthropic**: Tool injection (`respond_with_structure`)
  - **Google**: Function declaration with normalized schema
  - **Ollama**: JSON mode + schema in system message
  - **OpenAI**: No adaptation (native support)
- Normalize Google schemas (remove unsupported features)
- Post-process responses (extract JSON from tool calls)
- Validate against schema if strict mode enabled

### Key Methods

#### `apply_structured_output_adapter(request, provider_type, provider) -> LLMRequest`

Main entry point for schema adaptation.

**Algorithm**:
1. Check if request has `response_format.type == "json_schema"`
2. Check if request already has tools (skip if so)
3. Adapt based on provider type
4. Mark request as adapted in metadata

**Example**:
```python
adapter = SchemaAdapter()

request = LLMRequest(
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name"]
            }
        }
    }
)

# For Anthropic: Converts to tool injection
adapted = await adapter.apply_structured_output_adapter(
    request, "anthropic", provider
)
# adapted.tools = [{"type": "function", "function": {...}}]
# adapted.tool_choice = {"type": "any"}

# For Google: Converts to function declaration with normalized schema
adapted = await adapter.apply_structured_output_adapter(
    request, "google", provider
)
# adapted.tools = [{"type": "function", "function": {...}}]
# adapted.tool_choice = "any"
# Schema normalized (removes unsupported keywords)
```

#### `post_process_structured_output(response, request, provider_type) -> LLMResponse`

Extracts and validates structured output from provider responses.

**Algorithm**:
1. Check if request was adapted (metadata flag)
2. Based on provider:
   - **OpenAI**: Parse JSON from `response.content`
   - **Anthropic/Google**: Extract from `tool_calls[0].function.arguments`
   - **Ollama**: Parse JSON from `response.content`
3. Validate against original schema if strict mode
4. Set `response.parsed` field

**Example**:
```python
# After API call
response = await provider.chat(adapted_request)

# Extract structured output
response = await adapter.post_process_structured_output(
    response, adapted_request, "anthropic"
)

# Now response.parsed contains the structured data
print(response.parsed)
# {'name': 'Alice', 'age': 30}
```

### Google Schema Normalization

Google Gemini has strict schema requirements. The adapter uses `GoogleSchemaNormalizer` to:

- Remove `null` from union types: `["string", "null"]` → `"string"`
- Collapse multi-type unions to string: `["string", "number"]` → `"string"`
- Remove unsupported keywords: `additionalProperties`, `anyOf`, `oneOf`, `allOf`, `pattern`, `format`
- Handle tuple-typed arrays: Take first schema only
- Preserve supported keywords: `type`, `description`, `enum`, `minimum`, `maximum`, etc.

See `src/llmring/providers/google_schema_normalizer.py` for details.

### Testing

See `tests/unit/test_schema_adapter.py` (16 tests):
- Adaptation for each provider
- Google schema normalization
- Post-processing extraction
- Validation in strict mode

---

## CostCalculator

**File**: `src/llmring/services/cost_calculator.py`

### Purpose

Calculates the cost of LLM requests based on token usage and pricing from the model registry.

### Responsibilities

- Fetch pricing from model registry
- Calculate input/output token costs
- Handle missing pricing gracefully (return $0.00)
- Add cost information to responses
- Cache registry lookups

### Key Methods

#### `calculate_cost(response, registry_model) -> Optional[Dict[str, float]]`

Calculates cost based on usage and pricing.

**Algorithm**:
1. Check if response has usage information
2. Get pricing from registry model (input/output cost per million tokens)
3. Calculate input cost: `(input_tokens / 1_000_000) * input_cost_per_million`
4. Calculate output cost: `(output_tokens / 1_000_000) * output_cost_per_million`
5. Calculate total cost
6. Return cost dictionary

**Returns**:
```python
{
    "input_cost": 0.0030,    # $0.003 for input tokens
    "output_cost": 0.0045,   # $0.0045 for output tokens
    "total_cost": 0.0075     # $0.0075 total
}
```

**Example**:
```python
calculator = CostCalculator(registry)

# After API call
response = await provider.chat(request)

# Get registry model for pricing
registry_model = await calculator._get_registry_model("openai", "gpt-4-turbo")

# Calculate cost
cost_info = await calculator.calculate_cost(response, registry_model)

# Add to response
calculator.add_cost_to_response(response, cost_info)

# response now has response.cost field
print(response.cost)
# {'input_cost': 0.003, 'output_cost': 0.0045, 'total_cost': 0.0075}
```

#### `add_cost_to_response(response, cost_info)`

Adds cost information to the response object.

**Side Effect**: Mutates `response.cost` field.

#### `get_zero_cost_info() -> Dict[str, float]`

Returns zero-cost fallback when pricing unavailable.

**Returns**: `{"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}`

### Registry Integration

The cost calculator depends on the model registry for pricing:

```python
registry_model = await registry.fetch_current_models("openai")
model = next(m for m in registry_model if m.model_name == "gpt-4-turbo")

# model.cost_per_million_input_tokens
# model.cost_per_million_output_tokens
```

### Caching

Registry calls are cached (24 hour TTL) to minimize external requests.

### Error Handling

- If no usage in response → Return `None`
- If pricing unavailable → Return zero cost
- If registry fetch fails → Log warning, return zero cost

### Testing

See `tests/unit/test_cost_calculator.py` (13 tests):
- Cost calculation with various token counts
- Registry fetching and caching
- Error handling (no usage, no pricing)
- Edge cases (zero tokens, missing data)

---

## ReceiptManager

**File**: `src/llmring/services/receipt_manager.py`

### Purpose

Generates cryptographically signed receipts for LLM requests. Receipts provide tamper-proof records of usage, costs, and model versions.

### Responsibilities

- Generate receipts for completed requests
- Sign receipts with Ed25519 private key
- Store receipts in memory
- Resolve profile names (explicit > env > lockfile > default)
- Handle streaming vs non-streaming receipts

### Key Methods

#### `generate_receipt(response, original_alias, provider, model, cost_info, profile) -> Optional[Receipt]`

Generates a signed receipt for a non-streaming request.

**Algorithm**:
1. Extract usage from response
2. Resolve profile name (explicit > `LLMRING_PROFILE` > lockfile default > "default")
3. Create Receipt object with metadata
4. Sign receipt with Ed25519 key
5. Store receipt
6. Return receipt

**Example**:
```python
manager = ReceiptManager(lockfile)

response = await provider.chat(request)
cost_info = await calculator.calculate_cost(response, registry_model)

receipt = await manager.generate_receipt(
    response=response,
    original_alias="gpt4",
    provider="openai",
    model="gpt-4-turbo",
    cost_info=cost_info,
    profile="production"
)

print(receipt.signature)  # Ed25519 signature
print(receipt.total_cost)  # 0.0075
print(receipt.profile)     # "production"
```

#### `generate_streaming_receipt(usage, original_alias, provider, model, cost_info, profile) -> Optional[Receipt]`

Generates a receipt for streaming requests (usage accumulated during stream).

**Similar to `generate_receipt()` but takes `usage` dict instead of `response`.**

#### `get_receipts() -> List[Receipt]`

Returns copies of all stored receipts.

**Note**: Returns copies to prevent external mutation.

#### `clear_receipts()`

Clears all stored receipts.

**Use Case**: Testing or memory management.

### Profile Resolution Priority

1. **Explicit profile parameter**: `generate_receipt(..., profile="prod")`
2. **Environment variable**: `LLMRING_PROFILE=staging`
3. **Lockfile default**: `default_profile = "production"`
4. **Fallback**: `"default"`

### Receipt Structure

See `src/llmring/receipt.py`:

```python
@dataclass
class Receipt:
    id: str                    # UUID
    timestamp: datetime        # ISO 8601
    model: str                 # "gpt-4-turbo"
    profile: str               # "production"
    original_alias: str        # "gpt4"
    input_tokens: int          # 100
    output_tokens: int         # 150
    total_tokens: int          # 250
    input_cost: float          # 0.003
    output_cost: float         # 0.0045
    total_cost: float          # 0.0075
    signature: str             # Ed25519 hex signature
    public_key: str            # Ed25519 public key
```

### Cryptographic Signing

Uses **Ed25519** for signatures:
- Private key from `LLMRING_SIGNING_KEY` environment variable
- If not set, receipts generated without signatures
- Public key included in receipt for verification

**Verification**:
```python
from llmring.receipt import verify_receipt

is_valid = verify_receipt(receipt)
# True if signature matches, False otherwise
```

### Testing

See `tests/unit/test_receipt_manager.py` (19 tests):
- Receipt generation for various scenarios
- Profile resolution priority
- Signature generation
- Error handling

---

## ValidationService

**File**: `src/llmring/services/validation_service.py`

### Purpose

Validates LLM requests before sending to providers. Checks context limits, token counts, and model capabilities.

### Responsibilities

- Validate context limits (input + output tokens)
- Estimate token counts efficiently
- Check model capabilities (vision, tools, JSON mode)
- Fetch model metadata from registry
- Fail fast with clear error messages

### Key Methods

#### `validate_context_limit(request, registry_model) -> Optional[str]`

Main validation entry point. Returns error message if validation fails, `None` if valid.

**Algorithm**:
1. Get context limit from registry model
2. Estimate input tokens
3. Check if `input_tokens + max_tokens > context_limit`
4. Return error message or `None`

**Example**:
```python
validator = ValidationService(registry, token_counter)

request = LLMRequest(
    messages=[Message(role="user", content="Hello" * 10000)],
    model="openai:gpt-4",
    max_tokens=4000
)

registry_model = await registry.fetch_current_models("openai")
model = next(m for m in registry_model if m.model_name == "gpt-4")

error = await validator.validate_context_limit(request, model)

if error:
    raise ValueError(error)
    # "Request exceeds context limit: 50000 input tokens + 4000 max tokens > 8192 limit"
```

#### `validate_model_capabilities(request, registry_model) -> Optional[str]`

Checks if model supports requested capabilities.

**Checks**:
- **Vision**: If request has image content, model must support vision
- **Tool Calling**: If request has tools, model must support function calling
- **JSON Mode**: If request has `json_response=True`, model must support JSON mode

**Returns**: Error message if capability missing, `None` if valid.

**Example**:
```python
request = LLMRequest(
    messages=[Message(role="user", content=[
        {"type": "image_url", "image_url": {"url": "..."}}
    ])],
    model="openai:gpt-3.5-turbo"  # No vision support
)

error = await validator.validate_model_capabilities(request, model)
# "Model gpt-3.5-turbo does not support vision"
```

### Token Estimation Strategy

Two-stage approach for efficiency:

1. **Quick Character Check**:
   - Estimate: `characters / 4` (rough approximation)
   - If obviously too large, fail fast without tokenization

2. **Accurate Tokenization**:
   - Use `tiktoken` to count actual tokens
   - Only if quick check passes

**Why**: Tokenization is expensive. Quick check avoids unnecessary work for obviously oversized inputs.

### Registry Integration

Fetches model metadata for validation:

```python
registry_model = await validator._get_registry_model("openai", "gpt-4")

# registry_model.context_window
# registry_model.supports_vision
# registry_model.supports_function_calling
# registry_model.supports_json_response
```

### Error Messages

Clear, actionable error messages:

- `"Request exceeds context limit: 10000 input + 4096 max > 8192 limit"`
- `"Model gpt-3.5-turbo does not support vision"`
- `"Model gpt-4-turbo does not support JSON mode"`

### Testing

See `tests/unit/test_validation_service.py` (19 tests):
- Context limit validation
- Token estimation strategies
- Capability validation (vision, tools, JSON)
- Error handling

---

## Service Interaction Patterns

### Pattern 1: Sequential Service Calls

Most common pattern - services called in sequence:

```python
# In LLMRing.chat()

# 1. Resolve alias
resolved = self._alias_resolver.resolve(model, profile)

# 2. Validate input
InputValidator.validate_message_content(messages)

# 3. Validate context
error = await self._validation_service.validate_context_limit(request, model)

# 4. Adapt schema
request = await self._schema_adapter.apply_structured_output_adapter(request, ...)

# 5. Call provider
response = await provider.chat(request)

# 6. Post-process
response = await self._schema_adapter.post_process_structured_output(response, ...)

# 7. Calculate cost
cost = await self._cost_calculator.calculate_cost(response, model)

# 8. Generate receipt
receipt = await self._receipt_manager.generate_receipt(...)
```

### Pattern 2: Conditional Service Usage

Some services only called when needed:

```python
# Schema adapter only if structured output requested
if request.response_format and request.response_format.get("type") == "json_schema":
    request = await self._schema_adapter.apply_structured_output_adapter(...)
```

### Pattern 3: Service with Side Effects

Some services mutate state:

```python
# CostCalculator adds cost to response
calculator.add_cost_to_response(response, cost_info)
# response.cost now set

# ReceiptManager stores receipts
receipt = await manager.generate_receipt(...)
# receipt stored in manager._receipts list
```

### Pattern 4: Service Chaining

Services depend on results from other services:

```python
# CostCalculator needs registry model
registry_model = await calculator._get_registry_model(provider, model)
cost_info = await calculator.calculate_cost(response, registry_model)

# ReceiptManager needs cost info
receipt = await manager.generate_receipt(..., cost_info=cost_info)
```

### Pattern 5: Service Initialization

All services initialized in `LLMRing.__init__()`:

```python
class LLMRing:
    def __init__(self):
        self._alias_resolver = AliasResolver(lockfile)
        self._schema_adapter = SchemaAdapter()
        self._cost_calculator = CostCalculator(registry)
        self._receipt_manager = ReceiptManager(lockfile)
        self._validation_service = ValidationService(registry, token_counter)
```

### Pattern 6: Service Synchronization

When state changes, sync services:

```python
def register_provider(self, name: str, provider: BaseLLMProvider):
    self.providers[name] = provider

    # Sync alias resolver
    self._alias_resolver.update_available_providers(
        list(self.providers.keys())
    )
```

---

## Service Dependencies

```
AliasResolver
├── Depends on: Lockfile
└── No service dependencies

SchemaAdapter
├── Depends on: GoogleSchemaNormalizer
└── No service dependencies

CostCalculator
├── Depends on: ModelRegistryClient
└── No service dependencies

ReceiptManager
├── Depends on: Lockfile
└── No service dependencies

ValidationService
├── Depends on: ModelRegistryClient
├── Depends on: TokenCounter
└── No service dependencies
```

**Key Insight**: Services are independent and don't depend on each other. This makes testing easier and reduces coupling.

---

## Best Practices

### 1. Keep Services Focused

Each service should have one clear responsibility. If a service grows too large, consider splitting it.

### 2. Avoid Service-to-Service Calls

Services should not call other services directly. Orchestration happens in `LLMRing`.

**Bad**:
```python
class CostCalculator:
    async def calculate_cost(self, response, model):
        # Don't do this!
        receipt = self.receipt_manager.generate_receipt(...)
```

**Good**:
```python
class LLMRing:
    async def chat(self, request):
        # Orchestrate service calls
        cost = await self._cost_calculator.calculate_cost(response, model)
        receipt = await self._receipt_manager.generate_receipt(..., cost_info=cost)
```

### 3. Immutable by Default

Services should not mutate inputs unless explicitly documented:

```python
# CostCalculator.add_cost_to_response() mutates response
# This is documented in docstring as a side effect
```

### 4. Clear Error Messages

Services should raise exceptions with clear, actionable error messages:

```python
raise ValueError(
    f"Request exceeds context limit: {input_tokens} input + {max_tokens} max > {limit} limit. "
    f"Reduce message length or max_tokens."
)
```

### 5. Testability

Services should be easy to unit test without complex setup:

```python
# Easy to test - no complex dependencies
def test_alias_resolver():
    lockfile = Lockfile(content={"aliases": {"gpt4": "openai:gpt-4"}})
    resolver = AliasResolver(lockfile)
    assert resolver.resolve("gpt4") == "openai:gpt-4"
```

---

## Related Documentation

- [Architecture Overview](./overview.md)
- [Provider Layer Documentation](./providers.md)
- [Architecture Decision Records](../decisions/)
