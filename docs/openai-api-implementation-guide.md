# OpenAI API Implementation Guide

## Overview

OpenAI provides two main APIs for text generation:
1. **Responses API** (`client.responses.create`) - The modern, recommended API
2. **Chat Completions API** (`client.chat.completions.create`) - The traditional API, still supported

## Responses API vs Chat Completions API

### Responses API
- **Status**: Recommended for all new projects
- **Performance**: 3% better on benchmarks with reasoning models
- **Cost**: 40-80% lower due to improved cache utilization
- **Features**: Built-in tools (web search, file search, code interpreter, MCP)
- **Stateful**: Can maintain context with `store: true`
- **Limitation**: 256K character input limit

### Chat Completions API
- **Status**: Supported indefinitely
- **Flexibility**: Maximum control for complex scenarios
- **Input Size**: Only limited by model's context window
- **Features**: Well-established, extensive documentation
- **Use Cases**: Function calling, complex multi-turn conversations

## Model Categories

### Reasoning Models
Models optimized for complex reasoning tasks:
- **O-series**: o1, o1-mini, o3, o3-mini, o3-pro, o4-mini
- **GPT-5 series**: gpt-5, gpt-5-mini, gpt-5-nano
- **Recommendation**: Use Responses API for better performance

### Standard Models
General-purpose conversational models:
- **GPT-4 series**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
- **GPT-3.5**: gpt-3.5-turbo
- **Recommendation**: Can use either API based on needs

## Implementation Strategy

### Decision Tree
```
1. Is it a reasoning model (o-series, gpt-5)?
   YES → Check input size
      ├─ Under 256K chars → Use Responses API
      └─ Over 256K chars → Use Chat Completions API
   NO → Continue to step 2

2. Does the request need tools/functions?
   YES → Check if built-in tools suffice
      ├─ Built-in tools work → Use Responses API
      └─ Need custom functions → Use Chat Completions API
   NO → Continue to step 3

3. Is input under 256K characters?
   YES → Use Responses API (recommended)
   NO → Use Chat Completions API
```

### Code Implementation Pattern

```python
class OpenAIProvider:
    def _should_use_responses_api(
        self, 
        model: str, 
        messages: List[Message],
        tools: Optional[List] = None,
        response_format: Optional[Dict] = None
    ) -> bool:
        """
        Determine whether to use Responses API or Chat Completions API.
        """
        # Calculate input size
        input_size = self._calculate_input_size(messages)
        
        # If over 256K chars, must use Chat Completions
        if input_size > 256000:
            return False
        
        # Reasoning models benefit from Responses API
        reasoning_models = [
            'o1', 'o1-mini', 'o3', 'o3-mini', 'o3-pro', 'o4-mini',
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
        ]
        
        is_reasoning_model = any(
            model.startswith(prefix) for prefix in reasoning_models
        )
        
        # If reasoning model, prefer Responses API
        if is_reasoning_model:
            # But check for unsupported features
            if tools and not self._are_tools_supported_in_responses(tools):
                return False
            return True
        
        # For standard models, use Responses API if no complex features needed
        if not tools and not response_format:
            return True
        
        # Check if tools are built-in (supported by Responses)
        if tools and self._are_tools_supported_in_responses(tools):
            return True
        
        # Default to Chat Completions for maximum compatibility
        return False
```

## API Method Signatures

### Responses API
```python
response = await client.responses.create(
    model="gpt-5",
    input="Your prompt here",  # or input=[{"role": "user", "content": "..."}]
    max_output_tokens=4096,     # Note: not max_tokens
    temperature=0.7,
    store=True,                 # Optional: maintain state
)
# Access: response.output_text or response.output
```

### Chat Completions API
```python
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Your prompt here"}],
    max_tokens=4096,            # Note: different parameter name
    temperature=0.7,
    tools=[...],                # Custom tools support
    response_format={...},      # JSON mode, etc.
)
# Access: response.choices[0].message.content
```

## Testing Strategy

### Unit Tests
1. **Model Routing Tests**
   - Test reasoning models route to Responses API
   - Test large inputs (>256K) route to Chat Completions
   - Test tool requirements affect routing

2. **API Compatibility Tests**
   - Verify correct parameter mapping (max_tokens vs max_output_tokens)
   - Test response extraction from different formats
   - Ensure error handling for API-specific failures

3. **Feature Coverage Tests**
   - Test built-in tools with Responses API
   - Test custom functions with Chat Completions
   - Test multimodal inputs with both APIs

### Integration Tests
```python
@pytest.mark.parametrize("model,expected_api", [
    ("o1-mini", "responses"),
    ("gpt-5", "responses"),
    ("gpt-4o", "chat_or_responses"),  # Depends on features
    ("gpt-3.5-turbo", "chat_or_responses"),
])
async def test_api_routing(model, expected_api):
    """Test that models route to the correct API."""
    pass

@pytest.mark.parametrize("input_size,expected_api", [
    (100_000, "responses"),    # Under limit
    (256_000, "responses"),    # At limit
    (300_000, "chat"),         # Over limit
])
async def test_input_size_routing(input_size, expected_api):
    """Test that input size affects API selection."""
    pass
```

## Maintaining API Compatibility

### Unified Interface
Our LLMRing API should remain consistent regardless of which OpenAI API is used internally:

```python
# User always calls the same interface
response = await llmring.chat(
    messages=[...],
    model="gpt-5",
    max_tokens=1000,
    temperature=0.7
)

# Internally, we handle:
# 1. API selection (Responses vs Chat Completions)
# 2. Parameter mapping (max_tokens → max_output_tokens)
# 3. Response normalization (output_text → content)
```

### Parameter Mapping
```python
def _map_to_responses_params(self, **kwargs):
    """Map unified params to Responses API."""
    params = {}
    if 'max_tokens' in kwargs:
        params['max_output_tokens'] = kwargs['max_tokens']
    if 'messages' in kwargs:
        params['input'] = self._convert_messages_to_input(kwargs['messages'])
    # ... other mappings
    return params

def _map_to_chat_params(self, **kwargs):
    """Map unified params to Chat Completions API."""
    # Direct mapping for most parameters
    return kwargs
```

### Response Normalization
```python
def _normalize_response(self, api_response, api_type: str) -> LLMResponse:
    """Convert API-specific response to unified format."""
    if api_type == "responses":
        return LLMResponse(
            content=api_response.output_text or api_response.output,
            model=api_response.model,
            usage=self._extract_usage_from_responses(api_response),
            finish_reason="stop"
        )
    else:  # chat_completions
        choice = api_response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=api_response.model,
            usage=api_response.usage.model_dump() if api_response.usage else None,
            finish_reason=choice.finish_reason
        )
```

## Error Handling

### API-Specific Errors
```python
try:
    if use_responses_api:
        response = await client.responses.create(...)
except Exception as e:
    error_msg = str(e)
    if "256000" in error_msg:
        # Retry with Chat Completions API
        response = await client.chat.completions.create(...)
    elif "not supported" in error_msg.lower():
        # Feature not supported in Responses, fallback
        response = await client.chat.completions.create(...)
    else:
        raise
```

## Migration Path

### Phase 1: Dual Support
- Implement both API paths
- Default to Responses for reasoning models
- Fallback to Chat Completions when needed

### Phase 2: Optimization
- Monitor performance metrics
- Adjust routing logic based on real usage
- Optimize for cost and performance

### Phase 3: Full Migration
- Move all compatible requests to Responses API
- Keep Chat Completions only for edge cases
- Document any limitations clearly

## Best Practices

1. **Always check input size** before choosing API
2. **Cache API selection** for similar requests
3. **Log API usage** for monitoring and optimization
4. **Handle fallbacks gracefully** when Responses API limitations are hit
5. **Test both paths** thoroughly with real-world data
6. **Document model-specific behaviors** as they're discovered

## Future Considerations

- Monitor OpenAI announcements for Responses API improvements
- The 256K limit may be increased in the future
- New models may have different optimal API choices
- Built-in tools list will likely expand over time

## References

- [OpenAI Responses API Documentation](https://platform.openai.com/docs/guides/responses)
- [Migrate to Responses Guide](https://platform.openai.com/docs/guides/migrate-to-responses)
- [Model Comparison](https://platform.openai.com/docs/models)
- [API Reference](https://platform.openai.com/docs/api-reference)