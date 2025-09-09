# OpenAI Provider Implementation Plan

## Goal
Maximize the benefits of OpenAI's dual API system (Responses API and Chat Completions API) while maintaining complete backward compatibility with the existing LLMRing unified API.

## Design Principles

1. **Transparent to Users**: The LLMRing API remains unchanged
2. **Intelligent Routing**: Automatically choose the best API for each request
3. **Graceful Fallback**: Handle limitations transparently
4. **Performance First**: Use Responses API when beneficial
5. **Feature Complete**: Support all OpenAI capabilities

## Implementation Architecture

```python
class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider with intelligent API routing.
    
    Automatically routes requests to either:
    - Responses API (modern, recommended)
    - Chat Completions API (traditional, flexible)
    """
    
    def __init__(self, api_key: Optional[str] = None, ...):
        # Initialize both API clients
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Define model categories for routing
        self.reasoning_models = {
            'o1', 'o1-mini', 'o3', 'o3-mini', 'o3-pro', 'o4-mini',
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
        }
        
        # Track metrics for optimization
        self.metrics = OpenAIMetrics()
    
    async def chat(
        self,
        messages: List[Message],
        model: str,
        **kwargs
    ) -> LLMResponse:
        """
        Unified chat interface that intelligently routes to the best API.
        """
        # Determine which API to use
        api_choice = self._determine_api(model, messages, **kwargs)
        
        # Route to appropriate implementation
        if api_choice == "responses":
            return await self._chat_via_responses(messages, model, **kwargs)
        else:
            return await self._chat_via_completions(messages, model, **kwargs)
```

## Routing Logic Implementation

### Phase 1: Basic Routing (Initial Implementation)

```python
def _determine_api(
    self,
    model: str,
    messages: List[Message],
    tools: Optional[List] = None,
    response_format: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Determine which API to use based on model and request characteristics.
    
    Returns: "responses" or "chat_completions"
    """
    # Step 1: Check input size constraint
    input_size = self._calculate_input_size(messages)
    if input_size > 256000:  # Responses API limit
        self.metrics.log_routing_reason("input_size_limit")
        return "chat_completions"
    
    # Step 2: Check if model is a reasoning model
    model_base = model.split('-')[0]
    is_reasoning = any(model.startswith(m) for m in self.reasoning_models)
    
    if is_reasoning:
        # Reasoning models benefit from Responses API
        # But check for unsupported features
        if tools and not self._are_tools_builtin(tools):
            self.metrics.log_routing_reason("custom_tools_needed")
            return "chat_completions"
        
        self.metrics.log_routing_reason("reasoning_model")
        return "responses"
    
    # Step 3: For standard models, check feature requirements
    if tools:
        if self._are_tools_builtin(tools):
            self.metrics.log_routing_reason("builtin_tools")
            return "responses"
        else:
            self.metrics.log_routing_reason("custom_tools")
            return "chat_completions"
    
    # Step 4: Default to Responses API for new projects
    self.metrics.log_routing_reason("default_responses")
    return "responses"
```

### Phase 2: Advanced Routing (After Initial Testing)

```python
def _determine_api_advanced(
    self,
    model: str,
    messages: List[Message],
    **kwargs
) -> str:
    """
    Advanced routing with learned optimizations.
    """
    # Use cached routing decisions for similar requests
    cache_key = self._generate_routing_cache_key(model, messages, **kwargs)
    if cache_key in self.routing_cache:
        return self.routing_cache[cache_key]
    
    # Apply learned heuristics from metrics
    if self.metrics.should_prefer_responses(model, len(messages)):
        api_choice = "responses"
    else:
        api_choice = self._determine_api(model, messages, **kwargs)
    
    # Cache the decision
    self.routing_cache[cache_key] = api_choice
    return api_choice
```

## Parameter Mapping

### Input Parameter Mapping

```python
def _map_to_responses_params(
    self,
    messages: List[Message],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[List] = None,
    **kwargs
) -> Dict:
    """
    Map LLMRing parameters to Responses API format.
    """
    params = {
        "model": model,
        "input": self._messages_to_input(messages),
    }
    
    # Map token limit
    if max_tokens:
        params["max_output_tokens"] = max_tokens
    
    # Map temperature
    if temperature is not None:
        params["temperature"] = temperature
    
    # Map tools (only built-in tools)
    if tools:
        params["tools"] = self._map_builtin_tools(tools)
    
    # Handle stateful context (new Responses feature)
    if kwargs.get("maintain_state", False):
        params["store"] = True
    
    # Add reasoning effort for reasoning models
    if self._is_reasoning_model(model):
        params["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")
    
    return params

def _map_to_chat_params(
    self,
    messages: List[Message],
    model: str,
    **kwargs
) -> Dict:
    """
    Map LLMRing parameters to Chat Completions API format.
    """
    # Most parameters map directly
    params = {
        "model": model,
        "messages": self._messages_to_openai_format(messages),
    }
    
    # Direct mappings
    for key in ["max_tokens", "temperature", "tools", "tool_choice", 
                "response_format", "seed", "user"]:
        if key in kwargs and kwargs[key] is not None:
            params[key] = kwargs[key]
    
    return params
```

### Message Format Conversion

```python
def _messages_to_input(self, messages: List[Message]) -> Union[str, List[Dict]]:
    """
    Convert messages to Responses API input format.
    """
    if len(messages) == 1 and messages[0].role == "user":
        # Simple format for single user message
        if isinstance(messages[0].content, str):
            return messages[0].content
    
    # Multi-turn format
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg.role,
            "content": self._format_content(msg.content)
        })
    return formatted

def _messages_to_openai_format(self, messages: List[Message]) -> List[Dict]:
    """
    Convert messages to Chat Completions format.
    """
    openai_messages = []
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            # Handle tool calls
            message_dict = {
                "role": msg.role,
                "content": msg.content or "",
                "tool_calls": msg.tool_calls
            }
        elif hasattr(msg, "tool_call_id"):
            # Handle tool responses
            message_dict = {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            }
        else:
            # Regular messages
            message_dict = {
                "role": msg.role,
                "content": self._format_content(msg.content)
            }
        openai_messages.append(message_dict)
    return openai_messages
```

## Response Normalization

```python
def _normalize_response(
    self,
    api_response: Any,
    api_type: str,
    original_request: Dict
) -> LLMResponse:
    """
    Normalize response from either API to unified LLMResponse format.
    """
    if api_type == "responses":
        return self._normalize_responses_api(api_response, original_request)
    else:
        return self._normalize_chat_completions(api_response, original_request)

def _normalize_responses_api(self, response: Any, request: Dict) -> LLMResponse:
    """
    Normalize Responses API response.
    """
    # Extract content
    content = response.output_text if hasattr(response, 'output_text') else response.output
    
    # Extract usage information
    usage = None
    if hasattr(response, 'usage'):
        usage = {
            "prompt_tokens": response.usage.get('prompt_tokens', 0),
            "completion_tokens": response.usage.get('completion_tokens', 0),
            "total_tokens": response.usage.get('total_tokens', 0),
        }
        # Add reasoning tokens if present (for reasoning models)
        if 'reasoning_tokens' in response.usage:
            usage['reasoning_tokens'] = response.usage['reasoning_tokens']
    
    # Build unified response
    return LLMResponse(
        content=content,
        model=response.model,
        usage=usage,
        finish_reason="stop",
        raw_response=response  # Keep original for debugging
    )

def _normalize_chat_completions(self, response: Any, request: Dict) -> LLMResponse:
    """
    Normalize Chat Completions API response.
    """
    choice = response.choices[0]
    
    # Build unified response
    llm_response = LLMResponse(
        content=choice.message.content or "",
        model=response.model,
        usage=response.usage.model_dump() if response.usage else None,
        finish_reason=choice.finish_reason,
        raw_response=response
    )
    
    # Add tool calls if present
    if choice.message.tool_calls:
        llm_response.tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in choice.message.tool_calls
        ]
    
    return llm_response
```

## Error Handling and Fallback

```python
async def _chat_with_fallback(
    self,
    messages: List[Message],
    model: str,
    **kwargs
) -> LLMResponse:
    """
    Try Responses API first, fallback to Chat Completions if needed.
    """
    try:
        # Try Responses API first
        return await self._chat_via_responses(messages, model, **kwargs)
    except Exception as e:
        error_msg = str(e)
        
        # Check if error is due to API limitations
        if any(limit in error_msg.lower() for limit in [
            "256000", "string too long", "not supported", "invalid parameter"
        ]):
            # Log the fallback
            self.metrics.log_fallback(model, error_msg)
            
            # Try Chat Completions API
            return await self._chat_via_completions(messages, model, **kwargs)
        else:
            # Re-raise other errors
            raise

async def _chat_via_responses(
    self,
    messages: List[Message],
    model: str,
    **kwargs
) -> LLMResponse:
    """
    Chat using Responses API.
    """
    params = self._map_to_responses_params(messages, model, **kwargs)
    
    try:
        response = await retry_async(
            lambda: self.client.responses.create(**params)
        )
        self.metrics.log_success("responses", model)
        return self._normalize_response(response, "responses", params)
    except Exception as e:
        self.metrics.log_error("responses", model, str(e))
        raise

async def _chat_via_completions(
    self,
    messages: List[Message],
    model: str,
    **kwargs
) -> LLMResponse:
    """
    Chat using Chat Completions API.
    """
    params = self._map_to_chat_params(messages, model, **kwargs)
    
    try:
        response = await retry_async(
            lambda: self.client.chat.completions.create(**params)
        )
        self.metrics.log_success("chat_completions", model)
        return self._normalize_response(response, "chat_completions", params)
    except Exception as e:
        self.metrics.log_error("chat_completions", model, str(e))
        raise
```

## Feature Support Matrix

| Feature | Responses API | Chat Completions | LLMRing Handling |
|---------|--------------|------------------|------------------|
| Basic Chat | ✅ | ✅ | Use Responses by default |
| Large Input (>256K) | ❌ | ✅ | Auto-route to Chat Completions |
| Custom Functions | ❌ | ✅ | Detect and route to Chat Completions |
| Built-in Tools | ✅ | ❌ | Use Responses for web_search, etc. |
| Reasoning Models | ✅ (Better) | ✅ | Prefer Responses |
| JSON Mode | ✅ | ✅ | Both supported |
| Streaming | ✅ | ✅ | Both supported |
| Vision | ✅ | ✅ | Both supported |
| Stateful Context | ✅ | ❌ | Enable with Responses |

## Migration Timeline

### Week 1: Core Implementation
- Implement routing logic
- Add parameter mapping
- Create response normalization

### Week 2: Testing
- Unit tests for all paths
- Integration tests with real API
- Performance benchmarks

### Week 3: Optimization
- Analyze metrics
- Tune routing heuristics
- Add caching layer

### Week 4: Documentation & Release
- Update user documentation
- Create migration guide
- Release with feature flag

## Rollout Strategy

### Phase 1: Opt-in (Week 1-2)
```python
# Enable new routing with environment variable
LLMRING_OPENAI_USE_RESPONSES=true
```

### Phase 2: Gradual Rollout (Week 3-4)
```python
# Roll out to percentage of requests
if random.random() < 0.1:  # 10% of requests
    use_new_routing = True
```

### Phase 3: Default (Week 5+)
```python
# Make new routing the default
# Keep flag to disable if needed
LLMRING_OPENAI_USE_LEGACY=true  # To disable new routing
```

## Success Metrics

1. **Performance**: 3% improvement for reasoning models
2. **Cost**: 40% reduction in API costs
3. **Reliability**: <0.1% fallback rate
4. **Compatibility**: 100% backward compatible
5. **User Experience**: No breaking changes

## Risk Mitigation

1. **Feature Gaps**: Maintain compatibility matrix, auto-fallback
2. **API Changes**: Abstract behind interface, version lock SDK
3. **Performance Issues**: Monitor metrics, quick rollback capability
4. **User Confusion**: Clear documentation, no visible changes

## Code Organization

```
src/llmring/providers/
├── openai_api.py           # Main provider implementation
├── openai_routing.py        # Routing logic (new)
├── openai_responses.py      # Responses API handling (new)
├── openai_completions.py    # Chat Completions handling (refactored)
├── openai_utils.py          # Shared utilities (new)
└── openai_metrics.py        # Metrics and monitoring (new)
```

## Next Steps

1. Review this plan with the team
2. Create feature branch for implementation
3. Implement core routing logic
4. Add comprehensive tests
5. Deploy behind feature flag
6. Monitor metrics and iterate