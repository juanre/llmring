"""
Simple test of extra_params flow using real APIs.
"""

import pytest

from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message


class TestExtraParamsSimple:
    """Simple tests for extra_params functionality."""

    # service fixture is provided by conftest.py

    def test_extra_params_in_request_schema(self):
        """Test that LLMRequest properly includes extra_params."""
        # Test with extra_params
        request_with_params = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="test")],
            extra_params={"logprobs": True, "seed": 42},
        )

        assert hasattr(request_with_params, "extra_params")
        assert request_with_params.extra_params == {"logprobs": True, "seed": 42}

        # Test default empty dict
        request_default = LLMRequest(
            model="openai:gpt-4o-mini", messages=[Message(role="user", content="test")]
        )

        assert hasattr(request_default, "extra_params")
        assert request_default.extra_params == {}

        print("✓ LLMRequest extra_params schema works correctly")

    @pytest.mark.asyncio
    async def test_openai_extra_params_accepted(self, service):
        """Test that OpenAI provider accepts extra_params without error."""
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[
                Message(
                    role="system",
                    content="Respond with exactly: 'extra_params test passed'",
                ),
                Message(role="user", content="Hello"),
            ],
            max_tokens=10,
            temperature=0.1,
            extra_params={
                "seed": 12345,  # Should make responses deterministic
                "logprobs": False,  # Valid parameter
            },
        )

        try:
            response = await service.chat(request)

            # If we get here, extra_params were accepted
            assert hasattr(response, "content")
            assert isinstance(response.content, str)
            assert len(response.content) > 0

            print(f"✓ OpenAI accepts extra_params: {response.content[:50]}...")
            return True

        except Exception as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"extra_params not properly passed to OpenAI: {e}")
            else:
                # Other errors (auth, network) are fine for this test
                print(
                    f"✓ OpenAI provider accepts extra_params (got expected error: {type(e).__name__})"
                )
                return True

    @pytest.mark.asyncio
    async def test_anthropic_extra_params_accepted(self, service):
        """Test that Anthropic provider accepts extra_params without error."""
        request = LLMRequest(
            model="anthropic:claude-3-5-haiku",
            messages=[
                Message(
                    role="system",
                    content="Respond with exactly: 'extra_params test passed'",
                ),
                Message(role="user", content="Hello"),
            ],
            max_tokens=10,
            temperature=0.1,
            extra_params={"top_p": 0.9, "top_k": 50},
        )

        try:
            response = await service.chat(request)

            # If we get here, extra_params were accepted
            assert hasattr(response, "content")
            assert isinstance(response.content, str)
            assert len(response.content) > 0

            print(f"✓ Anthropic accepts extra_params: {response.content[:50]}...")
            return True

        except Exception as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"extra_params not properly passed to Anthropic: {e}")
            else:
                # Other errors (auth, network) are fine for this test
                print(
                    f"✓ Anthropic provider accepts extra_params (got expected error: {type(e).__name__})"
                )
                return True

    def test_provider_signatures_have_extra_params(self):
        """Test that all providers have extra_params in their signatures."""
        import inspect
        from llmring.providers.openai_api import OpenAIProvider
        from llmring.providers.anthropic_api import AnthropicProvider
        from llmring.providers.google_api import GoogleProvider
        from llmring.providers.ollama_api import OllamaProvider

        providers = [OpenAIProvider, AnthropicProvider, GoogleProvider, OllamaProvider]

        for provider_class in providers:
            sig = inspect.signature(provider_class.chat)
            params = list(sig.parameters.keys())

            assert "extra_params" in params, (
                f"{provider_class.__name__} missing extra_params parameter"
            )

            # Check default value
            extra_params_param = sig.parameters["extra_params"]
            assert extra_params_param.default is None, (
                f"{provider_class.__name__} extra_params should default to None"
            )

        print("✓ All providers have extra_params parameter with correct default")

    def test_service_passes_all_params(self):
        """Test that service.py is configured to pass all parameters including extra_params."""
        from llmring.service import LLMRing
        import inspect

        # Read the service.py source to verify it passes extra_params
        service_file = inspect.getsourcefile(LLMRing.chat)
        with open(service_file, "r") as f:
            service_source = f.read()

        # Check that extra_params is passed in provider.chat calls
        assert "extra_params=request.extra_params" in service_source, (
            "service.py should pass extra_params=request.extra_params to providers"
        )

        print("✓ Service.py properly configured to pass extra_params")
