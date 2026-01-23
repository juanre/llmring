"""
Tests for logging decorator validation logic.

For integration tests with real provider SDK responses, see:
- tests/integration/test_logging_normalizers.py
- tests/integration/test_logging_decorators_integration.py
"""

import pytest

from llmring.logging import log_llm_call, log_llm_stream
from llmring.logging.normalizers import detect_provider


class TestProviderDetection:
    """Test provider detection edge cases."""

    def test_detect_unknown_provider(self):
        """Should return None for unknown provider."""
        response = {"content": "hello"}
        provider = detect_provider(response)
        assert provider is None


class TestDecoratorValidation:
    """Test decorator input validation."""

    def test_log_llm_call_rejects_non_async_functions(self):
        """Decorator should raise TypeError for non-async functions."""

        with pytest.raises(TypeError, match="can only decorate async functions"):

            @log_llm_call(server_url="http://localhost:8000")
            def sync_function():
                return "not async"

    def test_log_llm_stream_rejects_non_generator(self):
        """Decorator should raise TypeError for non-generator functions."""

        with pytest.raises(TypeError, match="can only decorate async generator functions"):

            @log_llm_stream(server_url="http://localhost:8000")
            async def not_a_generator():
                return "not a generator"
