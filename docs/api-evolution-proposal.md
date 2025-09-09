# LLMRing API Evolution Proposal

## Executive Summary

This proposal outlines a comprehensive evolution of the LLMRing API to become a true superset interface for LLM providers. The goal is to enable seamless provider switching for common use cases while exposing advanced capabilities when needed.

**Key Principles:**
1. **Progressive disclosure** - Simple things simple, complex things possible
2. **Zero breaking changes** in minor versions - All additions are optional
3. **Provider intelligence** - Automatically handle provider differences
4. **Feature discovery** - Let users query capabilities before using them
5. **Escape hatches** - Always provide access to native features

---

## Proposed API Architecture

### Core Types (Enhanced)

```python
from typing import Union, Literal, AsyncIterator, Optional, Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field

# ============================================================
# Content Types (Strongly Typed)
# ============================================================

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image: Union[str, bytes]  # URL or base64
    detail: Optional[Literal["low", "high", "auto"]] = "auto"

class DocumentContent(BaseModel):
    type: Literal["document"] = "document"
    document: Union[str, bytes]  # Path, URL, or bytes
    mime_type: Optional[str] = None

class AudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    audio: Union[str, bytes]  # Path, URL, or bytes
    mime_type: Optional[str] = "audio/wav"

ContentType = Union[str, TextContent, ImageContent, DocumentContent, AudioContent]
MessageContent = Union[str, List[ContentType]]

# ============================================================
# Message Types
# ============================================================

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: MessageContent
    name: Optional[str] = None  # For multi-user conversations
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None  # Anthropic caching
    metadata: Optional[Dict[str, Any]] = None

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: "FunctionCall"

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string

# ============================================================
# Tool Definitions
# ============================================================

class FunctionParameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    required: Optional[bool] = False

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, FunctionParameter]

class Tool(BaseModel):
    type: Literal["function", "web_search", "code_interpreter", "file_search"]
    function: Optional[FunctionDefinition] = None

# ============================================================
# Request Types (Comprehensive)
# ============================================================

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[Dict[str, Any]] = None

class LLMRequest(BaseModel):
    """Comprehensive request supporting all provider capabilities."""
    
    # Required
    messages: List[Message]
    
    # Model selection
    model: Optional[str] = None  # Can be alias or provider:model
    
    # Common parameters (all providers)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    
    # Common advanced (most providers)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    
    # Response control
    response_format: Optional[ResponseFormat] = None
    json_mode: Optional[bool] = None  # Simplified alternative to response_format
    
    # Tools and functions
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], Dict]] = None
    
    # Provider-specific common
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)  # OpenAI/Anthropic
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)   # OpenAI/Anthropic
    repetition_penalty: Optional[float] = Field(None, ge=0.0, le=2.0)  # Anthropic/Ollama
    seed: Optional[int] = None  # OpenAI (reproducibility)
    
    # Advanced provider-specific
    n: Optional[int] = Field(None, description="Number of completions (OpenAI)")
    logprobs: Optional[bool] = None  # OpenAI
    top_logprobs: Optional[int] = None  # OpenAI
    logit_bias: Optional[Dict[str, float]] = None  # OpenAI
    user: Optional[str] = None  # OpenAI (tracking)
    
    # Reasoning models
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None  # OpenAI o-series
    
    # State management
    store: Optional[bool] = None  # OpenAI Responses API
    conversation_id: Optional[str] = None  # For maintaining context
    
    # Provider escape hatch
    provider_params: Optional[Dict[str, Any]] = Field(
        None, 
        description="Provider-specific parameters not covered above"
    )
    
    # LLMRing features
    cache: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    profile: Optional[str] = None  # For alias resolution

# ============================================================
# Response Types
# ============================================================

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None  # For reasoning models
    cache_read_tokens: Optional[int] = None  # Anthropic
    cache_write_tokens: Optional[int] = None  # Anthropic
    
    # Cost tracking
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None
    total_cost: Optional[float] = None

class LLMResponse(BaseModel):
    """Unified response from any provider."""
    
    # Content
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    
    # Metadata
    model: str
    provider: str  # Added: explicit provider identification
    usage: Optional[Usage] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    
    # Additional responses (OpenAI n>1)
    choices: Optional[List["LLMResponse"]] = None
    
    # Advanced
    logprobs: Optional[Dict[str, Any]] = None  # OpenAI
    
    # LLMRing additions
    receipt_id: Optional[str] = None
    cached: Optional[bool] = False
    latency_ms: Optional[float] = None

# ============================================================
# Streaming Types
# ============================================================

class StreamChoice(BaseModel):
    delta: Dict[str, Any]
    index: int
    finish_reason: Optional[str] = None

class StreamChunk(BaseModel):
    """Streaming response chunk."""
    
    # Content delta
    content: Optional[str] = None  # Incremental content
    role: Optional[str] = None  # First chunk only
    
    # Tool calls
    tool_calls: Optional[List[Dict]] = None  # Incremental
    
    # Metadata
    model: Optional[str] = None  # First chunk
    provider: Optional[str] = None  # First chunk
    
    # Final chunk
    usage: Optional[Usage] = None  # Last chunk only
    finish_reason: Optional[str] = None  # Last chunk only
    
    # Raw for debugging
    raw: Optional[Dict[str, Any]] = None

# ============================================================
# Provider Capabilities
# ============================================================

class ModelCapabilities(BaseModel):
    """Capabilities of a specific model."""
    
    # Basic info
    model_id: str
    provider: str
    display_name: str
    
    # Context limits
    max_input_tokens: int
    max_output_tokens: int
    total_context_tokens: int
    
    # Feature support
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_audio: bool = False
    supports_documents: bool = False
    supports_tools: bool = False
    supports_parallel_tools: bool = False
    supports_json_mode: bool = False
    supports_json_schema: bool = False
    supports_logprobs: bool = False
    supports_n_completions: bool = False
    supports_caching: bool = False
    supports_fine_tuning: bool = False
    
    # Advanced features
    supports_reasoning: bool = False  # o-series, GPT-5
    supports_web_search: bool = False  # Responses API
    supports_code_interpreter: bool = False  # Responses API
    
    # Costs (per million tokens)
    cost_per_million_input_tokens: Optional[float] = None
    cost_per_million_output_tokens: Optional[float] = None
    
    # Performance hints
    is_fast: bool = False  # For real-time use cases
    is_smart: bool = False  # For complex reasoning
    
    # Availability
    requires_tier: Optional[int] = None  # OpenAI tier requirement
    is_deprecated: bool = False
    deprecation_date: Optional[str] = None
```

### Provider Interface (Enhanced)

```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Enhanced base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """
        Send a chat request to the provider.
        Handles all parameter mapping internally.
        """
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        request: LLMRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from the provider.
        """
        pass
    
    @abstractmethod
    def get_capabilities(
        self,
        model: str
    ) -> ModelCapabilities:
        """
        Get capabilities for a specific model.
        """
        pass
    
    @abstractmethod
    def count_tokens(
        self,
        messages: List[Message],
        model: str
    ) -> int:
        """
        Count tokens for the given messages.
        """
        pass
    
    @abstractmethod
    def estimate_cost(
        self,
        usage: Usage,
        model: str
    ) -> Dict[str, float]:
        """
        Estimate cost for the given usage.
        """
        pass
    
    # Optional: Override for provider-specific optimizations
    
    def validate_request(
        self,
        request: LLMRequest,
        model: str
    ) -> Optional[str]:
        """
        Validate request against model capabilities.
        Returns error message if invalid, None if valid.
        """
        caps = self.get_capabilities(model)
        
        # Check token limits
        token_count = self.count_tokens(request.messages, model)
        if token_count > caps.max_input_tokens:
            return f"Input exceeds limit: {token_count} > {caps.max_input_tokens}"
        
        # Check feature support
        if request.tools and not caps.supports_tools:
            return f"Model {model} does not support tools"
        
        if request.stream and not caps.supports_streaming:
            return f"Model {model} does not support streaming"
        
        return None
    
    def map_parameters(
        self,
        request: LLMRequest,
        model: str
    ) -> Dict[str, Any]:
        """
        Map unified parameters to provider-specific format.
        Override in provider implementations.
        """
        return {}
```

### Service Interface (Enhanced)

```python
class LLMRing:
    """Enhanced LLMRing service with full feature support."""
    
    # ============================================================
    # Core Methods
    # ============================================================
    
    async def chat(
        self,
        request: Union[LLMRequest, Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat request to the appropriate provider.
        
        Can accept either LLMRequest or kwargs for convenience:
        - chat(request=LLMRequest(...))
        - chat(messages=[...], model="gpt-4", temperature=0.7)
        """
        if isinstance(request, dict) or kwargs:
            request = LLMRequest(**(request if isinstance(request, dict) else kwargs))
        
        # Resolve alias
        model = self.resolve_alias(request.model)
        provider = self.get_provider_for_model(model)
        
        # Validate request
        error = provider.validate_request(request, model)
        if error:
            raise ValueError(error)
        
        # Execute with telemetry
        start_time = time.time()
        response = await provider.chat(request)
        response.latency_ms = (time.time() - start_time) * 1000
        
        # Generate receipt if configured
        if self.receipt_generator:
            response.receipt_id = await self.generate_receipt(request, response)
        
        return response
    
    async def stream_chat(
        self,
        request: Union[LLMRequest, Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from the appropriate provider.
        """
        if isinstance(request, dict) or kwargs:
            request = LLMRequest(**(request if isinstance(request, dict) else kwargs))
        
        request.stream = True  # Ensure streaming is enabled
        
        model = self.resolve_alias(request.model)
        provider = self.get_provider_for_model(model)
        
        # Validate
        error = provider.validate_request(request, model)
        if error:
            raise ValueError(error)
        
        # Stream with telemetry
        start_time = time.time()
        usage_aggregator = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        
        async for chunk in provider.stream_chat(request):
            # Aggregate usage for final receipt
            if chunk.usage:
                usage_aggregator = chunk.usage
            
            yield chunk
        
        # Generate receipt after streaming completes
        if self.receipt_generator and usage_aggregator.total_tokens > 0:
            # Create synthetic response for receipt
            response = LLMResponse(
                content="[streamed]",
                model=model,
                provider=provider.name,
                usage=usage_aggregator,
                latency_ms=(time.time() - start_time) * 1000
            )
            await self.generate_receipt(request, response)
    
    # ============================================================
    # Convenience Methods
    # ============================================================
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple completion interface for single prompts.
        """
        request = LLMRequest(
            messages=[Message(role="user", content=prompt)],
            model=model,
            **kwargs
        )
        response = await self.chat(request)
        return response.content or ""
    
    async def stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream completion for single prompts.
        """
        request = LLMRequest(
            messages=[Message(role="user", content=prompt)],
            model=model,
            stream=True,
            **kwargs
        )
        async for chunk in self.stream_chat(request):
            if chunk.content:
                yield chunk.content
    
    # ============================================================
    # Discovery Methods
    # ============================================================
    
    def get_capabilities(
        self,
        model: str
    ) -> ModelCapabilities:
        """
        Get capabilities for a specific model.
        """
        model = self.resolve_alias(model)
        provider = self.get_provider_for_model(model)
        return provider.get_capabilities(model)
    
    def list_models(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[ModelCapabilities]:
        """
        List available models, optionally filtered.
        
        Args:
            provider: Filter by provider (e.g., "openai")
            capability: Filter by capability (e.g., "supports_vision")
        """
        models = []
        
        for provider_name, provider_instance in self.providers.items():
            if provider and provider_name != provider:
                continue
            
            for model in provider_instance.list_models():
                caps = provider_instance.get_capabilities(model)
                
                if capability:
                    if not getattr(caps, capability, False):
                        continue
                
                models.append(caps)
        
        return models
    
    def find_model(
        self,
        requirements: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find a model that meets the requirements.
        
        Example:
            model = ring.find_model({
                "supports_vision": True,
                "max_input_tokens": 100000,
                "is_fast": True
            })
        """
        for caps in self.list_models():
            matches = True
            for key, value in requirements.items():
                if not hasattr(caps, key):
                    matches = False
                    break
                
                attr_value = getattr(caps, key)
                if isinstance(value, bool):
                    if attr_value != value:
                        matches = False
                        break
                elif isinstance(value, (int, float)):
                    if attr_value < value:
                        matches = False
                        break
            
            if matches:
                return f"{caps.provider}:{caps.model_id}"
        
        return None
    
    # ============================================================
    # Utility Methods
    # ============================================================
    
    def count_tokens(
        self,
        messages: List[Message],
        model: str
    ) -> int:
        """
        Count tokens for the given messages.
        """
        model = self.resolve_alias(model)
        provider = self.get_provider_for_model(model)
        return provider.count_tokens(messages, model)
    
    def estimate_cost(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate cost for a request.
        """
        model = self.resolve_alias(model)
        provider = self.get_provider_for_model(model)
        
        input_tokens = provider.count_tokens(messages, model)
        
        # Estimate output tokens if not provided
        if max_tokens is None:
            max_tokens = min(1000, provider.get_capabilities(model).max_output_tokens)
        
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=max_tokens,
            total_tokens=input_tokens + max_tokens
        )
        
        return provider.estimate_cost(usage, model)
    
    def validate_request(
        self,
        request: LLMRequest
    ) -> Optional[str]:
        """
        Validate a request before sending.
        Returns error message if invalid.
        """
        model = self.resolve_alias(request.model)
        provider = self.get_provider_for_model(model)
        return provider.validate_request(request, model)
```

---

## Provider Implementation Strategy

### OpenAI Provider with Intelligent Routing

```python
class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with Responses/Chat Completions routing."""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.reasoning_models = {
            'o1', 'o1-mini', 'o3', 'o3-mini', 
            'gpt-5', 'gpt-5-mini'
        }
    
    async def chat(self, request: LLMRequest) -> LLMResponse:
        # Determine API based on model and request
        use_responses = self._should_use_responses(request)
        
        if use_responses:
            return await self._chat_via_responses(request)
        else:
            return await self._chat_via_completions(request)
    
    def _should_use_responses(self, request: LLMRequest) -> bool:
        """Intelligent routing logic."""
        
        # Check input size limit (256K for Responses API)
        input_size = sum(len(str(m.content)) for m in request.messages)
        if input_size > 256000:
            return False
        
        # Reasoning models benefit from Responses
        model_base = request.model.split('-')[0]
        if any(request.model.startswith(m) for m in self.reasoning_models):
            # But check for unsupported features
            if request.tools and not self._are_builtin_tools(request.tools):
                return False
            return True
        
        # New projects should use Responses when possible
        if not request.tools or self._are_builtin_tools(request.tools):
            return True
        
        return False
    
    async def _chat_via_responses(self, request: LLMRequest) -> LLMResponse:
        """Use Responses API."""
        params = {
            "model": request.model,
            "input": self._messages_to_input(request.messages),
        }
        
        # Map parameters
        if request.max_tokens:
            params["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.store:
            params["store"] = True
        
        # Add reasoning effort for reasoning models
        if any(request.model.startswith(m) for m in self.reasoning_models):
            params["reasoning_effort"] = request.reasoning_effort or "medium"
        
        response = await self.client.responses.create(**params)
        return self._normalize_responses_response(response)
    
    async def _chat_via_completions(self, request: LLMRequest) -> LLMResponse:
        """Use Chat Completions API."""
        params = {
            "model": request.model,
            "messages": self._messages_to_openai_format(request.messages),
        }
        
        # Direct parameter mapping
        for key in ["max_tokens", "temperature", "top_p", "n", 
                    "frequency_penalty", "presence_penalty", "seed",
                    "stop", "tools", "tool_choice", "logprobs", 
                    "top_logprobs", "user"]:
            value = getattr(request, key, None)
            if value is not None:
                params[key] = value
        
        # Handle response format
        if request.json_mode or request.response_format:
            params["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**params)
        return self._normalize_completions_response(response)
    
    async def stream_chat(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream using either API."""
        use_responses = self._should_use_responses(request)
        
        if use_responses:
            async for chunk in self._stream_via_responses(request):
                yield chunk
        else:
            async for chunk in self._stream_via_completions(request):
                yield chunk
```

### Anthropic Provider with Smart Mapping

```python
class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider with parameter mapping."""
    
    async def chat(self, request: LLMRequest) -> LLMResponse:
        params = {
            "model": request.model,
            "messages": self._messages_to_anthropic_format(request.messages),
            "max_tokens": request.max_tokens or 4096,
        }
        
        # Map common parameters
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop:
            params["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
        
        # Map OpenAI-style to Anthropic
        if request.frequency_penalty is not None:
            # Anthropic doesn't have exact equivalent, could map to top_k
            pass
        
        # Handle JSON mode via prompting
        if request.json_mode or request.response_format:
            if "system" in params:
                params["system"] += "\n\nYou must respond with valid JSON only."
            else:
                params["system"] = "You must respond with valid JSON only."
        
        # Handle caching
        if any(m.cache_control for m in request.messages):
            # Apply cache control to messages
            params["messages"] = self._apply_cache_control(params["messages"])
        
        response = await self.client.messages.create(**params)
        return self._normalize_response(response)
```

---

## Migration Path

### Phase 1: Foundation (v0.4.0) - Non-Breaking
**Timeline: 2 weeks**

1. **Add streaming support**
   - Add `stream_chat()` method to base class
   - Implement for all providers
   - Add `StreamChunk` type

2. **Add capabilities API**
   - Add `get_capabilities()` method
   - Implement `ModelCapabilities` type
   - Populate from registry + hardcoded knowledge

3. **Add common parameters**
   - Add `top_p`, `stop` to `LLMRequest`
   - Add `provider_params` escape hatch
   - Map in provider implementations

**Usage after Phase 1:**
```python
# New streaming support
async for chunk in llmring.stream_chat(messages=messages, model="gpt-4"):
    print(chunk.content, end="")

# New capabilities check
caps = llmring.get_capabilities("claude-3-opus")
if caps.supports_vision:
    # Include images
```

### Phase 2: Intelligence (v0.5.0) - Non-Breaking
**Timeline: 3 weeks**

1. **Implement OpenAI routing**
   - Add Responses API support
   - Implement intelligent routing
   - Add fallback mechanisms

2. **Add token counting**
   - Implement for each provider
   - Add `count_tokens()` method
   - Add cost estimation

3. **Add discovery methods**
   - `list_models()` with filtering
   - `find_model()` with requirements
   - Model recommendation system

**Usage after Phase 2:**
```python
# Token counting
tokens = llmring.count_tokens(messages, "gpt-4")
cost = llmring.estimate_cost(messages, "gpt-4", max_tokens=1000)

# Model discovery
model = llmring.find_model({
    "supports_vision": True,
    "max_input_tokens": 100000,
    "is_fast": True
})
```

### Phase 3: Type Safety (v0.6.0) - Minor Breaking
**Timeline: 2 weeks**

1. **Strongly typed content**
   - Implement content type classes
   - Add validators
   - Deprecation warnings for `Any` types

2. **Enhanced request validation**
   - Validate against capabilities
   - Better error messages
   - Suggest alternatives

3. **Response enhancements**
   - Add provider identification
   - Add latency tracking
   - Improve usage reporting

**Usage after Phase 3:**
```python
# Strongly typed content
message = Message(
    role="user",
    content=[
        TextContent(text="Analyze this image:"),
        ImageContent(image="http://example.com/image.jpg", detail="high")
    ]
)

# Better validation
error = llmring.validate_request(request)
if error:
    print(f"Invalid request: {error}")
```

### Phase 4: Advanced Features (v0.7.0) - Non-Breaking
**Timeline: 3 weeks**

1. **Conversation management**
   - Add conversation_id support
   - Automatic context trimming
   - State persistence

2. **Advanced provider features**
   - Multiple completions (n parameter)
   - Logprobs support
   - Fine-tuning indicators

3. **Monitoring and observability**
   - Request/response logging
   - Performance metrics
   - Cost tracking dashboard

**Usage after Phase 4:**
```python
# Conversation management
response = await llmring.chat(
    messages=messages,
    model="gpt-4",
    conversation_id="user-123-session-456",
    store=True  # Maintain state
)

# Multiple completions
response = await llmring.chat(
    messages=messages,
    model="gpt-4",
    n=3  # Get 3 different responses
)
for choice in response.choices:
    print(choice.content)
```

### Phase 5: Production Ready (v1.0.0) - Major Release
**Timeline: 4 weeks**

1. **Performance optimizations**
   - Connection pooling
   - Request batching
   - Cache layer

2. **Production features**
   - Rate limiting
   - Retry strategies
   - Circuit breakers

3. **Documentation and tooling**
   - Comprehensive docs
   - Migration guides
   - CLI tools

---

## Breaking Changes Strategy

### Deprecation Process

```python
# v0.4.0 - Add deprecation warning
class Message(BaseModel):
    content: Any  # Deprecated: Will be strongly typed in v1.0
    
    def __init__(self, **data):
        if isinstance(data.get("content"), dict):
            warnings.warn(
                "Dict content is deprecated. Use typed content classes.",
                DeprecationWarning,
                stacklevel=2
            )
        super().__init__(**data)

# v0.6.0 - Provide migration helper
def migrate_content(content: Any) -> MessageContent:
    """Helper to migrate old content format to new."""
    if isinstance(content, str):
        return content
    # ... migration logic
    return MessageContent(...)

# v1.0.0 - Remove deprecated code
class Message(BaseModel):
    content: MessageContent  # Now required to be typed
```

### Backward Compatibility Layer

```python
class LLMRingCompat:
    """Compatibility wrapper for old API."""
    
    def __init__(self, ring: LLMRing):
        self.ring = ring
    
    async def chat_old_style(self, **kwargs) -> Dict:
        """Old-style API for backward compatibility."""
        # Convert old format to new
        request = self._convert_old_request(kwargs)
        response = await self.ring.chat(request)
        # Convert new format to old
        return self._convert_new_response(response)
```

---

## Success Metrics

### Technical Metrics
- **API Coverage**: 95% of provider features accessible
- **Provider Switching**: <5 lines of code change
- **Performance**: <5% overhead vs direct SDK usage
- **Reliability**: 99.9% uptime with fallbacks

### User Metrics
- **Adoption**: 80% of users on latest version within 6 months
- **Satisfaction**: >4.5/5 developer satisfaction score
- **Breaking Changes**: <5% of users affected by breaking changes

### Business Metrics
- **Cost Optimization**: 20% average cost reduction through intelligent routing
- **Time to Integration**: <30 minutes for new users
- **Provider Coverage**: Support for 95% of use cases

---

## Risk Mitigation

### Technical Risks

1. **Provider API Changes**
   - Mitigation: Version lock SDKs, abstract behind interfaces
   - Fallback: Provider-specific escape hatches

2. **Performance Degradation**
   - Mitigation: Comprehensive benchmarking, caching
   - Fallback: Direct SDK access option

3. **Feature Gaps**
   - Mitigation: provider_params escape hatch
   - Fallback: Direct provider access

### User Experience Risks

1. **Migration Difficulty**
   - Mitigation: Compatibility layer, migration tools
   - Fallback: Support old API indefinitely

2. **Learning Curve**
   - Mitigation: Progressive disclosure, excellent docs
   - Fallback: Simple API subset

3. **Breaking Changes**
   - Mitigation: Semantic versioning, long deprecation periods
   - Fallback: LTS versions

---

## Implementation Priority

### Must Have (P0)
1. Streaming support
2. Provider capabilities API
3. Common parameters (top_p, stop)
4. OpenAI Responses API routing

### Should Have (P1)
1. Token counting
2. Model discovery
3. Strong typing for content
4. Cost estimation

### Nice to Have (P2)
1. Conversation management
2. Multiple completions
3. Logprobs support
4. Advanced caching

---

## Conclusion

This evolution plan transforms LLMRing from a basic abstraction layer into a comprehensive, production-ready LLM interface that:

1. **Preserves simplicity** for basic use cases
2. **Enables advanced features** when needed
3. **Handles provider differences** intelligently
4. **Maintains backward compatibility** through careful migration
5. **Provides escape hatches** for edge cases

The phased approach ensures we can deliver value incrementally while maintaining stability for existing users.