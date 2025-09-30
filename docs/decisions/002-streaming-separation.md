# ADR 002: Separate Streaming and Non-Streaming Methods

**Status**: Accepted
**Date**: 2025-09-30
**Deciders**: Core team
**Context**: Type safety and API clarity

---

## Context and Problem Statement

How should we handle streaming vs non-streaming LLM responses in our API?

The original design used a `stream` parameter that returned `Union[LLMResponse, AsyncIterator[StreamChunk]]`, but this caused several issues:
- Type checkers couldn't infer return type
- Users had to check response type at runtime
- Provider implementations were complex
- Testing was harder (mocking unions is awkward)

## Decision Drivers

- **Type safety**: Clear return types for type checkers
- **User experience**: Simple, intuitive API
- **No technical debt**: Clean code from the start
- **Consistency**: Same pattern across all providers
- **Testability**: Easy to test both modes

## Considered Options

1. **Single method with Union return type** (original)
2. **Separate methods** (`chat()` and `chat_stream()`)
3. **Wrapper class** that handles both modes
4. **Protocol-based dispatch** with separate types

## Decision Outcome

Chosen option: **"Separate methods"** because it provides the clearest API and best type safety.

### Implementation

```python
class LLMRing:
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Non-streaming chat completion."""
        ...

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion."""
        ...
```

### Provider Protocol

```python
class BaseLLMProvider(Protocol):
    async def chat(self, request: LLMRequest) -> LLMResponse: ...
    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: ...
```

## Consequences

### Positive

- ✅ **Type safety**: No `Union` types, type checkers work perfectly
- ✅ **Clear intent**: Method name shows what you get
- ✅ **Simple implementation**: Each method handles one case
- ✅ **Easy testing**: Test streaming and non-streaming separately
- ✅ **No runtime checks**: Don't need `isinstance()` checks
- ✅ **Better docs**: Each method has focused documentation

### Negative

- ❌ **Two methods**: Slightly larger API surface
- ❌ **Some duplication**: Request validation logic shared

### Neutral

- ⚪ **Migration**: No breaking change (we hadn't shipped yet)

## Usage Examples

### Non-Streaming

```python
response = await llmring.chat(request)
print(response.content)  # Type: str
```

### Streaming

```python
async for chunk in llmring.chat_stream(request):
    print(chunk.content, end="", flush=True)  # Type: str
```

## Validation

After implementing separate methods:
- ✅ All 401 tests pass
- ✅ MyPy/PyRight pass with no errors
- ✅ Code is simpler (removed ~50 lines of Union handling)
- ✅ User feedback positive (clearer API)

## Alternatives Considered

### Option 1: Single Method with Union Return

**Original Implementation**:
```python
async def chat(
    self,
    request: LLMRequest,
    stream: bool = False
) -> Union[LLMResponse, AsyncIterator[StreamChunk]]:
    if stream:
        return self._chat_stream(request)
    return self._chat_non_streaming(request)
```

**Pros**:
- Single method (smaller API)
- Familiar pattern (matches OpenAI SDK)

**Cons**:
- Type checkers can't infer return type
- Users must check type at runtime
- Complex implementation
- Harder to test
- Technical debt

**Rejected because**: Type safety is more important than API size.

### Option 3: Wrapper Class

**Proposed Implementation**:
```python
@dataclass
class StreamingLLMResponse:
    stream: AsyncIterator[StreamChunk]
    request: LLMRequest

class NonStreamingLLMResponse:
    response: LLMResponse
    request: LLMRequest
```

**Pros**:
- Single method
- Type safe
- Can add metadata

**Cons**:
- Extra wrapping/unwrapping
- More complex
- Users must unwrap result

**Rejected because**: Added complexity without clear benefit.

### Option 4: Protocol-Based Dispatch

**Proposed Implementation**:
```python
class StreamingResponse(Protocol):
    chunks: AsyncIterator[StreamChunk]

class NonStreamingResponse(Protocol):
    response: LLMResponse

async def chat(self, request: LLMRequest) -> StreamingResponse | NonStreamingResponse:
    ...
```

**Pros**:
- Flexible
- Type safe
- Single method

**Cons**:
- Complex dispatch logic
- Still have Union type
- Hard to understand

**Rejected because**: Too clever, not worth the complexity.

## Implementation Notes

### Shared Logic

Validation and pre-processing shared via internal methods:

```python
async def chat(self, request: LLMRequest) -> LLMResponse:
    # Shared validation
    await self._validate_request(request)

    # Non-streaming specific
    return await provider.chat(request)

async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
    # Same validation
    await self._validate_request(request)

    # Streaming specific
    async for chunk in provider.chat_stream(request):
        yield chunk
```

### Provider Implementation

Each provider implements both methods:

```python
class AnthropicProvider:
    async def chat(self, request: LLMRequest) -> LLMResponse:
        return await self._chat_non_streaming(request)

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream_chat(request):
            yield chunk
```

## Design Principles Applied

1. **Explicit is better than implicit**: Method name shows streaming vs non-streaming
2. **Simple is better than complex**: No Union types or runtime checks
3. **Flat is better than nested**: No wrapper classes
4. **Type safety**: Static typing catches errors early

## Related Decisions

- [ADR 003: Provider Protocol](./003-provider-protocol.md) - Providers implement both methods
- [ADR 004: Service Extraction](./004-service-extraction.md) - Services handle both modes

## References

- Python typing documentation: https://docs.python.org/3/library/typing.html
- AsyncIterator protocol: https://docs.python.org/3/library/typing.html#typing.AsyncIterator
- Original issue: [Issue #89](https://github.com/llmring/llmring/issues/89)
