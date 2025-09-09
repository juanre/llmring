# LLMRing API Design Analysis

## Current API Overview

### Core Interface
```python
# User-facing API (schemas.py)
class LLMRequest:
    messages: List[Message]
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    response_format: Optional[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    cache: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    json_response: Optional[bool]

class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: Any  # Can be str or structured content
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_call_id: Optional[str]
    timestamp: Optional[datetime]

class LLMResponse:
    content: str
    model: str
    usage: Optional[Dict[str, Any]]
    finish_reason: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
```

## Analysis: Strengths ✅

### 1. **Provider Abstraction Works Well**
- Users can switch between providers by changing model names
- Unified interface hides provider complexity
- Alias system allows semantic naming

### 2. **Basic Features Are Universal**
- Temperature, max_tokens work across all providers
- System/user/assistant roles are standard
- Tool calling interface is consistent

### 3. **Multimodal Support Exists**
- `content: Any` allows structured content
- Providers handle image_url and document types
- Format conversion happens transparently

## Analysis: Critical Gaps 🔴

### 1. **No Streaming Support**
```python
# Missing in current API:
async def stream_chat(request) -> AsyncIterator[StreamChunk]:
    pass
```
**Impact**: All providers support streaming, but LLMRing doesn't expose it. This is a major limitation for user experience.

### 2. **Missing Provider-Specific Parameters**
```python
# Not exposed in LLMRequest:
- top_p              # All providers
- frequency_penalty  # OpenAI, Anthropic
- presence_penalty   # OpenAI, Anthropic  
- seed              # OpenAI (reproducibility)
- stop_sequences    # All providers
- logprobs          # OpenAI
- top_logprobs      # OpenAI
- logit_bias        # OpenAI
- user              # OpenAI (tracking)
- n                 # OpenAI (multiple completions)
```

### 3. **No Provider Capabilities Discovery**
```python
# Users can't query what a provider supports:
- Max context window size
- Vision support
- Tool calling support
- Available models
- Cost per token
```

### 4. **Limited Error Information**
```python
# Current: generic exceptions
# Missing: provider-specific error codes, rate limit info, retry-after headers
```

### 5. **No Async Context Management**
```python
# Missing conversation state management:
- No session/conversation ID
- No automatic context trimming
- No token counting utilities
```

## Provider Feature Comparison

| Feature | OpenAI | Anthropic | Google | Ollama | LLMRing |
|---------|--------|-----------|--------|--------|---------|
| **Basic Chat** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Vision** | ✅ | ✅ | ✅ | ❌ | ✅ (partial) |
| **Documents** | ✅ (Assistants) | ✅ | ✅ | ❌ | ✅ (partial) |
| **Tools/Functions** | ✅ | ✅ | ✅ (limited) | ❌ | ✅ |
| **JSON Mode** | ✅ | ✅ (via prompting) | ✅ | ❌ | ✅ |
| **Multiple Responses (n)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Logprobs** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Fine-tuning Info** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Token Counting** | ✅ (tiktoken) | ✅ | ✅ | ✅ | ❌ |
| **Cost Tracking** | Via usage | Via usage | Via usage | N/A | ✅ |
| **Caching** | ❌ | ✅ (beta) | ❌ | ❌ | ❌ |

## Design Issues

### 1. **Incomplete Abstraction**
The current API is neither:
- **Lowest common denominator**: Missing features like streaming that all support
- **Union of all features**: Missing provider-specific capabilities

### 2. **Type Safety Issues**
```python
content: Any  # Too permissive
response_format: Optional[Dict[str, Any]]  # Not typed
tools: Optional[List[Dict[str, Any]]]  # Not typed
```

### 3. **Provider Switching Not Seamless**
```python
# User code that works with OpenAI:
request = LLMRequest(
    model="gpt-4",
    messages=[...],
    response_format={"type": "json_object"}  # OpenAI specific
)

# Fails with Anthropic (different JSON mode approach)
request.model = "claude-3-opus"  # 💥 Different response_format handling
```

### 4. **No Feature Negotiation**
```python
# Users can't check capabilities before using:
if provider.supports_streaming():
    async for chunk in provider.stream(...):
        ...
```

## Recommendations

### 1. **Add Streaming Support (Priority 1)**
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def stream_chat(
        self, messages: List[Message], model: str, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        pass

class StreamChunk(BaseModel):
    content: Optional[str]
    role: Optional[str]
    finish_reason: Optional[str]
    usage: Optional[Dict]  # Final chunk only
```

### 2. **Expose Common Advanced Parameters**
```python
class LLMRequest(BaseModel):
    # ... existing fields ...
    
    # Common parameters (all providers)
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    
    # Provider hints (auto-mapped)
    frequency_penalty: Optional[float] = None  # OpenAI/Anthropic
    presence_penalty: Optional[float] = None   # OpenAI/Anthropic
    seed: Optional[int] = None                 # OpenAI
    
    # Catch-all for provider-specific
    provider_params: Optional[Dict[str, Any]] = None
```

### 3. **Add Provider Capabilities API**
```python
class ProviderCapabilities(BaseModel):
    supports_streaming: bool
    supports_vision: bool
    supports_tools: bool
    supports_json_mode: bool
    max_context_tokens: int
    max_output_tokens: int
    cost_per_1k_input_tokens: Optional[float]
    cost_per_1k_output_tokens: Optional[float]

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_capabilities(self, model: str) -> ProviderCapabilities:
        pass
```

### 4. **Improve Type Safety**
```python
from typing import Union, Literal

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: Dict[str, str]

class DocumentContent(BaseModel):
    type: Literal["document"]
    source: Dict[str, Any]

MessageContent = Union[str, List[Union[TextContent, ImageContent, DocumentContent]]]

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    # ...
```

### 5. **Add Token Counting**
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def count_tokens(self, messages: List[Message], model: str) -> int:
        pass
```

### 6. **Provider-Aware Parameter Mapping**
```python
class ParameterMapper:
    """Maps unified parameters to provider-specific ones."""
    
    def to_openai(self, request: LLMRequest) -> Dict:
        params = self.base_params(request)
        if request.frequency_penalty:
            params["frequency_penalty"] = request.frequency_penalty
        return params
    
    def to_anthropic(self, request: LLMRequest) -> Dict:
        params = self.base_params(request)
        if request.frequency_penalty:
            # Map to Anthropic's equivalent
            params["repetition_penalty"] = request.frequency_penalty
        return params
```

## Migration Path

### Phase 1: Non-Breaking Additions (v0.4.0)
- Add streaming support via new method
- Add capabilities API
- Add provider_params field
- Improve type hints

### Phase 2: Enhanced Parameters (v0.5.0)
- Add common parameters to LLMRequest
- Add token counting
- Add parameter mapping layer

### Phase 3: Breaking Changes (v1.0.0)
- Make content strongly typed
- Require capabilities check for advanced features
- Deprecate untyped Dict[str, Any] fields

## Impact on Users

### Current State
```python
# Users must know provider differences
if "openai" in model:
    request.response_format = {"type": "json_object"}
elif "anthropic" in model:
    # Handle differently
    messages[0].content += "\nRespond in JSON format"
```

### Improved State
```python
# Unified handling
request = LLMRequest(
    model="gpt-4",  # or "claude-3-opus"
    messages=messages,
    json_response=True  # Works across providers
)

# Streaming (new)
async for chunk in llmring.stream_chat(request):
    print(chunk.content, end="")

# Capabilities check (new)
caps = llmring.get_capabilities(model)
if caps.supports_vision:
    # Include images
```

## Conclusion

LLMRing's current API provides good basic abstraction but lacks critical features that all major providers support (especially streaming). The design should evolve toward:

1. **Complete common features** (streaming, token counting)
2. **Expose provider capabilities** (for feature discovery)
3. **Better type safety** (structured content types)
4. **Smart parameter mapping** (handle provider differences internally)

This would make LLMRing a true superset API where users can:
- Switch providers with minimal code changes
- Access provider-specific features when needed
- Get consistent behavior for common operations
- Discover capabilities programmatically