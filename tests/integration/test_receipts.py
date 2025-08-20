"""Integration tests for receipt generation."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmring import LLMRequest, LLMRing, Message
from llmring.receipts import Receipt


@pytest.mark.asyncio
class TestReceiptGeneration:
    """Test receipt generation during chat operations."""

    async def test_receipt_generated_on_chat(self):
        """Test that receipts are generated when chat completes with usage info."""
        import tempfile
        from pathlib import Path

        from llmring.lockfile import Lockfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a lockfile with test bindings
            lockfile_path = Path(tmpdir) / "llmring.lock"
            lockfile = Lockfile.create_default()
            lockfile.set_binding("low_cost", "openai:gpt-3.5-turbo", profile="test")
            lockfile.save(lockfile_path)

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-test",
                    "LLMRING_PROFILE": "test",
                },
                clear=False,
            ):
                ring = LLMRing(lockfile_path=str(lockfile_path))

                # Mock the provider's chat method
                mock_response = MagicMock()
                mock_response.content = "Test response"
                mock_response.model = "openai:gpt-3.5-turbo"
                mock_response.usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
                mock_response.finish_reason = "stop"

                with patch.object(
                    ring.providers["openai"],
                    "chat",
                    new=AsyncMock(return_value=mock_response),
                ):
                    # Create a request
                    request = LLMRequest(
                        model="low_cost",  # This should resolve to an alias
                        messages=[Message(role="user", content="Test message")],
                    )

                    # Send the chat request
                    response = await ring.chat(request, profile="test")

                # Verify response
                assert response.content == "Test response"
                assert response.usage["total_tokens"] == 30

                # Check that a receipt was generated if lockfile exists
                if ring.lockfile:
                    assert len(ring.receipts) == 1
                    receipt = ring.receipts[0]

                    assert isinstance(receipt, Receipt)
                    assert receipt.alias == "low_cost"
                    assert receipt.profile == "test"
                    assert receipt.provider == "openai"
                    assert receipt.model == "gpt-3.5-turbo"
                    assert receipt.prompt_tokens == 10
                    assert receipt.completion_tokens == 20
                    assert receipt.total_tokens == 30
                    assert receipt.lock_digest  # Should have a digest

    async def test_receipt_not_generated_without_usage(self):
        """Test that receipts are not generated when no usage info is provided."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test"},
            clear=False,
        ):
            ring = LLMRing()

            # Mock the provider's chat method without usage
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.model = "openai:gpt-3.5-turbo"
            mock_response.usage = None  # No usage info
            mock_response.finish_reason = "stop"

            with patch.object(
                ring.providers["openai"],
                "chat",
                new=AsyncMock(return_value=mock_response),
            ):
                request = LLMRequest(
                    model="openai:gpt-3.5-turbo",
                    messages=[Message(role="user", content="Test message")],
                )

                response = await ring.chat(request)

                # No receipt should be generated
                assert len(ring.receipts) == 0

    async def test_receipt_digest_calculation(self):
        """Test that lockfile digest is calculated correctly."""
        from llmring.lockfile import Lockfile

        # Create a lockfile
        lockfile = Lockfile.create_default()

        # Calculate digest
        digest1 = lockfile.calculate_digest()
        assert digest1
        assert len(digest1) == 64  # SHA256 hex digest length

        # Digest should be consistent
        digest2 = lockfile.calculate_digest()
        assert digest1 == digest2

        # Changing lockfile should change digest
        lockfile.set_binding("test_alias", "openai:gpt-4")
        digest3 = lockfile.calculate_digest()
        assert digest3 != digest1
