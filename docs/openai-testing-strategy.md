# OpenAI Provider Testing Strategy

## Testing Goals

1. **Ensure correct API routing** based on model, input size, and features
2. **Maintain backwards compatibility** with existing LLMRing API
3. **Verify parameter mapping** between APIs
4. **Test fallback mechanisms** when limits are hit
5. **Validate response normalization** across both APIs

## Test Categories

### 1. Model Routing Tests

Test that different models route to the appropriate API:

```python
# tests/providers/test_openai_routing.py

class TestOpenAIRouting:
    """Test API routing logic for OpenAI provider."""
    
    @pytest.mark.parametrize("model,expected_api", [
        # Reasoning models should use Responses API
        ("o1", "responses"),
        ("o1-mini", "responses"),
        ("o3", "responses"),
        ("o3-mini", "responses"),
        ("o3-pro", "responses"),
        ("o4-mini", "responses"),
        ("gpt-5", "responses"),
        ("gpt-5-mini", "responses"),
        ("gpt-5-nano", "responses"),
        
        # Standard models can use either (depends on features)
        ("gpt-4o", "auto"),
        ("gpt-4o-mini", "auto"),
        ("gpt-4-turbo", "auto"),
        ("gpt-3.5-turbo", "auto"),
    ])
    async def test_model_routing(self, model, expected_api):
        """Test that models route to the correct API."""
        provider = OpenAIProvider()
        api_choice = provider._determine_api(model, [], None, None)
        
        if expected_api == "auto":
            assert api_choice in ["responses", "chat_completions"]
        else:
            assert api_choice == expected_api
```

### 2. Input Size Tests

Test routing based on input size limits:

```python
class TestInputSizeRouting:
    """Test API selection based on input size."""
    
    @pytest.mark.parametrize("char_count,expected_api", [
        (100_000, "responses"),      # Well under limit
        (250_000, "responses"),      # Near limit
        (256_000, "responses"),      # At limit
        (256_001, "chat_completions"), # Over limit
        (500_000, "chat_completions"), # Well over limit
    ])
    async def test_input_size_routing(self, char_count, expected_api):
        """Test that input size affects API selection."""
        provider = OpenAIProvider()
        
        # Create message with specific character count
        message = Message(role="user", content="x" * char_count)
        
        api_choice = provider._determine_api("gpt-5", [message], None, None)
        assert api_choice == expected_api
    
    async def test_automatic_fallback_on_size_limit(self):
        """Test automatic fallback when Responses API hits size limit."""
        provider = OpenAIProvider()
        
        # Create large message
        large_message = Message(role="user", content="x" * 300_000)
        
        with patch.object(provider, '_chat_via_responses') as mock_responses:
            mock_responses.side_effect = Exception("Invalid 'input': string too long. Expected a string with maximum length 256000")
            
            with patch.object(provider, '_chat_via_completions') as mock_completions:
                mock_completions.return_value = LLMResponse(content="success")
                
                response = await provider.chat([large_message], "gpt-5")
                
                # Should not call Responses API
                mock_responses.assert_not_called()
                # Should call Chat Completions API
                mock_completions.assert_called_once()
                assert response.content == "success"
```

### 3. Feature Compatibility Tests

Test that features route to appropriate APIs:

```python
class TestFeatureRouting:
    """Test API selection based on requested features."""
    
    async def test_builtin_tools_use_responses(self):
        """Test that built-in tools work with Responses API."""
        provider = OpenAIProvider()
        
        builtin_tools = [
            {"type": "web_search"},
            {"type": "file_search"},
            {"type": "code_interpreter"}
        ]
        
        api_choice = provider._determine_api("gpt-5", [], builtin_tools, None)
        assert api_choice == "responses"
    
    async def test_custom_functions_use_chat_completions(self):
        """Test that custom functions require Chat Completions API."""
        provider = OpenAIProvider()
        
        custom_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        api_choice = provider._determine_api("gpt-5", [], custom_tools, None)
        assert api_choice == "chat_completions"
    
    async def test_json_mode_compatibility(self):
        """Test JSON response format with both APIs."""
        provider = OpenAIProvider()
        
        response_format = {"type": "json_object"}
        
        # Both APIs should support JSON mode
        for model in ["gpt-4o", "gpt-5"]:
            response = await provider.chat(
                [Message(role="user", content="Return a JSON object")],
                model=model,
                response_format=response_format
            )
            # Should return valid JSON
            import json
            json.loads(response.content)  # Should not raise
```

### 4. Parameter Mapping Tests

Test correct parameter translation between APIs:

```python
class TestParameterMapping:
    """Test parameter mapping between unified API and provider APIs."""
    
    async def test_responses_parameter_mapping(self):
        """Test parameter mapping to Responses API."""
        provider = OpenAIProvider()
        
        unified_params = {
            "messages": [Message(role="user", content="test")],
            "model": "gpt-5",
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        
        responses_params = provider._map_to_responses_params(**unified_params)
        
        assert "max_output_tokens" in responses_params
        assert responses_params["max_output_tokens"] == 1000
        assert "max_tokens" not in responses_params
        assert "input" in responses_params
        assert "messages" not in responses_params
    
    async def test_chat_completions_parameter_mapping(self):
        """Test parameter mapping to Chat Completions API."""
        provider = OpenAIProvider()
        
        unified_params = {
            "messages": [Message(role="user", content="test")],
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        
        chat_params = provider._map_to_chat_params(**unified_params)
        
        assert "max_tokens" in chat_params
        assert chat_params["max_tokens"] == 1000
        assert "max_output_tokens" not in chat_params
        assert "messages" in chat_params
```

### 5. Response Normalization Tests

Test that responses are normalized correctly:

```python
class TestResponseNormalization:
    """Test response normalization from different APIs."""
    
    async def test_responses_api_normalization(self):
        """Test normalizing Responses API response."""
        provider = OpenAIProvider()
        
        # Mock Responses API response
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_response.model = "gpt-5"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        
        normalized = provider._normalize_response(mock_response, "responses")
        
        assert normalized.content == "Test response"
        assert normalized.model == "gpt-5"
        assert normalized.usage["prompt_tokens"] == 10
    
    async def test_chat_completions_normalization(self):
        """Test normalizing Chat Completions API response."""
        provider = OpenAIProvider()
        
        # Mock Chat Completions response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "gpt-4o"
        mock_response.usage = Mock()
        mock_response.usage.model_dump = lambda: {
            "prompt_tokens": 10,
            "completion_tokens": 5
        }
        
        normalized = provider._normalize_response(mock_response, "chat_completions")
        
        assert normalized.content == "Test response"
        assert normalized.model == "gpt-4o"
        assert normalized.usage["prompt_tokens"] == 10
```

### 6. Error Handling Tests

Test error handling and fallback mechanisms:

```python
class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    async def test_fallback_on_responses_error(self):
        """Test fallback to Chat Completions on Responses API error."""
        provider = OpenAIProvider()
        
        with patch.object(provider.client.responses, 'create') as mock_create:
            mock_create.side_effect = Exception("Feature not supported")
            
            with patch.object(provider.client.chat.completions, 'create') as mock_chat:
                mock_chat.return_value = Mock(
                    choices=[Mock(message=Mock(content="fallback"))],
                    usage=None
                )
                
                response = await provider.chat(
                    [Message(role="user", content="test")],
                    "gpt-5"
                )
                
                assert response.content == "fallback"
    
    async def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        provider = OpenAIProvider()
        
        with patch.object(provider.client.responses, 'create') as mock_create:
            mock_create.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(ValueError, match="rate limit"):
                await provider.chat(
                    [Message(role="user", content="test")],
                    "gpt-5"
                )
```

### 7. Integration Tests

End-to-end tests with real API calls (using test API keys):

```python
@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests with real OpenAI API."""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
    async def test_reasoning_model_with_responses(self):
        """Test reasoning model uses Responses API correctly."""
        provider = OpenAIProvider()
        
        response = await provider.chat(
            [Message(role="user", content="What is 2+2?")],
            "gpt-5-mini"  # or another available model
        )
        
        assert response.content
        assert "4" in response.content
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
    async def test_large_input_fallback(self):
        """Test large input automatically falls back to Chat Completions."""
        provider = OpenAIProvider()
        
        # Create input larger than 256K chars
        large_input = "Please summarize: " + ("word " * 60000)
        
        response = await provider.chat(
            [Message(role="user", content=large_input)],
            "gpt-4o"
        )
        
        assert response.content
        # Should have used Chat Completions (verify via logs or metrics)
```

## Test Data

### Mock Responses
Create realistic mock responses for both APIs:

```python
# tests/fixtures/openai_responses.py

MOCK_RESPONSES_API_RESPONSE = {
    "id": "resp_123",
    "object": "response",
    "created": 1234567890,
    "model": "gpt-5",
    "output_text": "This is a test response",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }
}

MOCK_CHAT_COMPLETIONS_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "This is a test response"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }
}
```

## Performance Testing

### Benchmark Tests
Compare performance between APIs:

```python
@pytest.mark.benchmark
class TestPerformance:
    """Performance comparison tests."""
    
    async def test_api_latency_comparison(self):
        """Compare latency between Responses and Chat Completions."""
        provider = OpenAIProvider()
        
        messages = [Message(role="user", content="Hello")]
        
        # Time Responses API
        start = time.time()
        await provider._chat_via_responses(messages, "gpt-5")
        responses_time = time.time() - start
        
        # Time Chat Completions API
        start = time.time()
        await provider._chat_via_completions(messages, "gpt-4o")
        chat_time = time.time() - start
        
        # Log for analysis
        print(f"Responses API: {responses_time:.3f}s")
        print(f"Chat Completions: {chat_time:.3f}s")
```

## Continuous Testing

### CI/CD Integration
```yaml
# .github/workflows/test-openai.yml
name: OpenAI Provider Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      - name: Run unit tests
        run: |
          pytest tests/providers/test_openai_*.py -v
      - name: Run integration tests (if API key present)
        if: ${{ secrets.OPENAI_API_KEY }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/providers/test_openai_*.py -v -m integration
```

## Monitoring & Metrics

Track API usage in production:

```python
class OpenAIMetrics:
    """Track API usage metrics."""
    
    def __init__(self):
        self.api_calls = {
            "responses": 0,
            "chat_completions": 0,
            "fallbacks": 0,
            "errors": 0
        }
    
    def log_api_call(self, api_type: str, success: bool, fallback: bool = False):
        """Log an API call for metrics."""
        self.api_calls[api_type] += 1
        if fallback:
            self.api_calls["fallbacks"] += 1
        if not success:
            self.api_calls["errors"] += 1
```

## Test Coverage Goals

- **Unit test coverage**: >90% for routing logic
- **Integration test coverage**: Key paths with real API
- **Error scenarios**: All known error types handled
- **Performance regression**: Detect >10% latency increase
- **Compatibility**: 100% backward compatibility with existing code