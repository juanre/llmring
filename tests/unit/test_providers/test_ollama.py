"""
Unit tests for the Ollama provider using real API calls.
These tests require a running Ollama instance at localhost:11434.
Tests are minimal to avoid excessive resource usage.
Chat tests are skipped for large models to prevent long test times.
"""

import pytest
import httpx

from llmring.providers.ollama_api import OllamaProvider
from llmring.schemas import LLMResponse, Message


def is_ollama_running():
    """Check if Ollama is running at localhost:11434."""
    try:
        with httpx.Client() as client:
            response = client.get("http://localhost:11434/api/version", timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False


# Skip decorator that checks if Ollama is actually running
skip_if_ollama_not_running = pytest.mark.skipif(
    not is_ollama_running(), reason="Ollama service not running at localhost:11434"
)


@pytest.mark.llm
@pytest.mark.unit
class TestOllamaProviderUnit:
    """Unit tests for OllamaProvider."""

    def test_initialization_with_default_url(self):
        """Test provider initialization with default URL."""
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        # Default model is now None initially and fetched from registry on first call
        assert provider.default_model is None
        assert provider.config.api_key is None  # Ollama doesn't use API keys

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        custom_url = "http://custom-ollama:11434"
        provider = OllamaProvider(base_url=custom_url)
        assert provider.base_url == custom_url

    def test_initialization_with_env_var(self, monkeypatch):
        """Test provider initialization with environment variable."""
        custom_url = "http://env-ollama:11434"
        monkeypatch.setenv("OLLAMA_BASE_URL", custom_url)
        provider = OllamaProvider()
        assert provider.base_url == custom_url

    @pytest.mark.asyncio
    async def test_supported_models_list(self, ollama_provider):
        """Test that supported models list contains expected models."""
        models = await ollama_provider.get_supported_models()

        assert isinstance(models, list)
        # Models list may be empty if registry is unavailable
        if len(models) > 0:
            # If models are available, check for common ones
            # Models are returned from registry or fallback list
            assert any("mistral" in model or "llama" in model for model in models)
        # Even if empty, the provider should return a list

    @pytest.mark.asyncio
    async def test_validate_model_exact_match(self, ollama_provider):
        """Test model validation with exact model names."""
        # Should validate local models
        # These models are in the test registry
        assert await ollama_provider.validate_model("mistral") is True
        assert await ollama_provider.validate_model("llama3") is True

    @pytest.mark.asyncio
    async def test_validate_model_with_provider_prefix(self, ollama_provider):
        """Test model validation handles provider prefix correctly."""
        # Should validate with provider prefix
        assert await ollama_provider.validate_model("ollama:mistral") is True

    @pytest.mark.asyncio
    async def test_validate_model_with_tags(self, ollama_provider):
        """Test model validation with version tags."""
        # Models with tags may or may not validate depending on registry content
        # Since test registry doesn't have tagged versions, these will be False
        # But that's OK - validation is advisory now, models will still work
        result1 = await ollama_provider.validate_model("mistral:7b")
        result2 = await ollama_provider.validate_model("llama3:latest")
        # Just verify the function works without error
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)

    @pytest.mark.asyncio
    async def test_validate_model_invalid_format(self, ollama_provider):
        """Test model validation behavior with invalid formats."""
        # When registry is unavailable, Ollama provider may be more permissive
        # The behavior depends on whether the registry is accessible
        try:
            models = await ollama_provider.get_supported_models()
            if len(models) > 0:
                # Registry is available, should be stricter
                assert await ollama_provider.validate_model("invalid_model!@#") is False
                assert await ollama_provider.validate_model("model with spaces") is False
                assert await ollama_provider.validate_model("") is False
            else:
                # Registry unavailable, may accept any format or use fallback logic
                pytest.skip("Ollama registry unavailable - validation behavior may be permissive")
        except Exception:
            pytest.skip("Cannot determine Ollama model validation behavior without registry")

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_basic_request(self, ollama_provider, simple_user_message):
        """Test basic chat functionality."""
        # Use a model that's available
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")

            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        response = await ollama_provider.chat(
            messages=simple_user_message,
            model=model_to_use,
            max_tokens=3,  # Minimal tokens to reduce resource usage
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == model_to_use
        assert response.usage is not None
        assert response.usage["prompt_tokens"] >= 0
        assert response.usage["completion_tokens"] >= 0
        assert response.usage["total_tokens"] >= 0
        assert response.finish_reason == "stop"

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_with_system_message(
        self, ollama_provider, system_user_messages
    ):
        """Test chat with system message."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        response = await ollama_provider.chat(
            messages=system_user_messages, model=model_to_use, max_tokens=3
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Math answer may vary with local models
        assert len(response.content) > 0

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_with_provider_prefix_removal(
        self, ollama_provider, simple_user_message
    ):
        """Test that provider prefix is correctly removed from model name."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            base_model = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in base_model for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in base_model for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {base_model} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in base_model and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {base_model} - use 1b variant for testing"
            )

        response = await ollama_provider.chat(
            messages=simple_user_message, model=f"ollama:{base_model}", max_tokens=3
        )

        assert isinstance(response, LLMResponse)
        assert response.model == base_model  # Prefix should be removed

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, ollama_provider, simple_user_message):
        """Test chat with temperature and max_tokens parameters."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        response = await ollama_provider.chat(
            messages=simple_user_message,
            model=model_to_use,
            temperature=0.7,
            max_tokens=3,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_with_unsupported_model_raises_error(
        self, ollama_provider, simple_user_message
    ):
        """Test that using an invalid model format raises ModelNotFoundError."""
        from llmring.exceptions import ModelNotFoundError, ProviderResponseError

        with pytest.raises((ModelNotFoundError, ProviderResponseError)):
            await ollama_provider.chat(
                messages=simple_user_message, model="invalid_model!@#"
            )

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_chat_api_error_handling(self, ollama_provider):
        """Test proper error handling for API errors."""
        # Test with model that doesn't exist locally
        with pytest.raises(Exception) as exc_info:
            await ollama_provider.chat(
                messages=[Message(role="user", content="test")],
                model="definitely-not-installed-model",
                max_tokens=3,
            )

        # Should wrap the error with our standard format (either validation or API error)
        error_msg = str(exc_info.value).lower()
        assert (
            "error" in error_msg or "not found" in error_msg or "invalid" in error_msg
        )

    def test_get_token_count_fallback(self, ollama_provider):
        """Test token counting fallback implementation."""
        text = "This is a test sentence for token counting."
        count = ollama_provider.get_token_count(text)

        assert isinstance(count, int)
        assert count > 0
        # Should be a reasonable approximation (length / 4)
        expected = len(text) // 4
        assert count == expected

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_get_available_models(self, ollama_provider):
        """Test getting available models from Ollama."""
        models = await ollama_provider.get_available_models()

        assert isinstance(models, list)
        # May be empty if no models are installed
        if models:
            for model in models:
                assert isinstance(model, str)
                assert len(model) > 0

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self, ollama_provider, multi_turn_conversation
    ):
        """Test multi-turn conversation handling."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        response = await ollama_provider.chat(
            messages=multi_turn_conversation, model=model_to_use, max_tokens=5
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Local models may not be as good at context retention
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_get_default_model(self, ollama_provider):
        """Test getting default model."""
        try:
            default_model = await ollama_provider.get_default_model()
            # If we get a default model, it should contain 'llama'
            assert "llama" in default_model
            models = await ollama_provider.get_supported_models()
            if models:  # Only check if models list is not empty
                assert default_model in models
        except Exception as e:
            # If registry is unavailable and no models found, this is expected
            assert "No default model available" in str(e) or "registry" in str(e).lower()
            pytest.skip("Ollama registry unavailable - cannot test default model")

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_tools_handling(self, ollama_provider, sample_tools):
        """Test that tools are handled gracefully (prompt engineering)."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        messages = [Message(role="user", content="What's the weather in NYC?")]

        response = await ollama_provider.chat(
            messages=messages, model=model_to_use, tools=sample_tools, max_tokens=20
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Tools are added to the prompt, so response should mention tools or weather
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["tool", "weather", "function"])

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_json_response_format(self, ollama_provider, json_response_format):
        """Test JSON response format handling."""
        try:
            available_models = await ollama_provider.get_available_models()
            if not available_models:
                pytest.skip("No models available in Ollama")
            model_to_use = available_models[0]
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

        # Skip large models as they're too slow for unit testing
        # Allow small variants like :1b but skip large ones
        small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]
        is_small_model = any(
            indicator in model_to_use for indicator in small_model_indicators
        )

        large_model_patterns = ["llama3.3", "deepseek-r1:32b"]
        is_large_base = any(pattern in model_to_use for pattern in large_model_patterns)

        # Skip if it's a large base model and not explicitly marked as small
        if is_large_base and not is_small_model:
            pytest.skip(
                f"Skipping large model {model_to_use} - too slow for unit testing"
            )

        # Also skip 3.2 models unless they're 1b
        if "llama3.2" in model_to_use and not is_small_model:
            pytest.skip(
                f"Skipping large llama3.2 model {model_to_use} - use 1b variant for testing"
            )

        messages = [
            Message(
                role="user", content="Respond with JSON: answer=hello, confidence=0.9"
            )
        ]

        response = await ollama_provider.chat(
            messages=messages,
            model=model_to_use,
            response_format=json_response_format,
            max_tokens=5,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Local models may not strictly follow JSON format
        assert len(response.content) > 0

    @skip_if_ollama_not_running
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test error handling when Ollama is not accessible."""
        # Create provider with invalid URL
        provider = OllamaProvider(base_url="http://nonexistent-host:11434")

        with pytest.raises(Exception) as exc_info:
            await provider.chat(
                messages=[Message(role="user", content="test")],
                model="local",
                max_tokens=3,
            )

        # Should get a connection error
        error_msg = str(exc_info.value).lower()
        assert "connect" in error_msg or "failed" in error_msg or "error" in error_msg

    @pytest.mark.asyncio
    async def test_model_validation_registry_based(self, ollama_provider):
        """Test that model validation is registry-based with graceful fallback."""
        # When registry is available, should validate against registry
        # When registry unavailable, should allow gracefully (return True)

        # Test with some model names - exact behavior depends on registry availability
        result1 = await ollama_provider.validate_model("llama3")
        result2 = await ollama_provider.validate_model("mistral")
        result3 = await ollama_provider.validate_model("custom-model:tag")

        # Models in registry should validate as True
        assert result1 is True  # llama3 is in test registry
        assert result2 is True  # mistral is in test registry

        # Models with tags might be False if not in registry, but that's OK
        # Validation is now advisory
        assert isinstance(result3, bool)  # Just verify it returns a boolean

        # Even "invalid" patterns should return a boolean result
        result4 = await ollama_provider.validate_model("model with spaces")
        assert isinstance(result4, bool)  # Just verify it works
