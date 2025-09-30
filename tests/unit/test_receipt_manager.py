"""Unit tests for ReceiptManager service."""

from unittest.mock import MagicMock, patch

import pytest

from llmring.lockfile_core import Lockfile
from llmring.receipts import Receipt
from llmring.schemas import LLMResponse
from llmring.services.receipt_manager import ReceiptManager


class TestReceiptManager:
    """Tests for ReceiptManager service."""

    @pytest.fixture
    def lockfile(self):
        """Create a mock lockfile."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.calculate_digest.return_value = "abc123"
        lockfile.default_profile = "default"
        return lockfile

    @pytest.fixture
    def manager(self, lockfile):
        """Create a ReceiptManager instance."""
        return ReceiptManager(lockfile)

    @pytest.fixture
    def sample_response(self):
        """Sample LLM response with usage."""
        return LLMResponse(
            content="Hello, world!",
            model="openai:gpt-4",
            provider="openai",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    @pytest.fixture
    def sample_cost_info(self):
        """Sample cost information."""
        return {
            "input_cost": 0.003,
            "output_cost": 0.003,
            "total_cost": 0.006,
        }

    def test_generate_receipt_success(self, manager, sample_response, sample_cost_info):
        """Should generate receipt successfully."""
        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
            profile="production",
        )

        assert receipt is not None
        assert receipt.alias == "fast"
        assert receipt.profile == "production"
        assert receipt.provider == "openai"
        assert receipt.model == "gpt-4"
        assert len(manager.receipts) == 1

    def test_generate_receipt_no_usage(self, manager, sample_cost_info):
        """Should return None when no usage info."""
        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
        )

        receipt = manager.generate_receipt(
            response=response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )

        assert receipt is None
        assert len(manager.receipts) == 0

    def test_generate_receipt_no_lockfile(self, sample_response, sample_cost_info):
        """Should return None when no lockfile."""
        manager = ReceiptManager(lockfile=None)

        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )

        assert receipt is None
        assert len(manager.receipts) == 0

    def test_generate_receipt_direct_model(self, manager, sample_response, sample_cost_info):
        """Should mark as direct_model when using provider:model format."""
        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="openai:gpt-4",  # Direct model, not alias
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )

        assert receipt is not None
        assert receipt.alias == "direct_model"

    def test_generate_receipt_no_cost_info(self, manager, sample_response):
        """Should use zero cost when no cost info provided."""
        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=None,
        )

        assert receipt is not None
        # Receipt should still be generated with zero costs

    def test_generate_receipt_uses_profile_priority(self, manager, sample_response):
        """Should use profile with correct priority."""
        # Explicit profile argument takes precedence
        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            profile="explicit",
        )
        assert receipt.profile == "explicit"

    @patch.dict("os.environ", {"LLMRING_PROFILE": "from_env"})
    def test_generate_receipt_uses_env_profile(self, manager, sample_response):
        """Should use LLMRING_PROFILE env var when no explicit profile."""
        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
        )
        assert receipt.profile == "from_env"

    def test_generate_receipt_uses_lockfile_default(self, lockfile, sample_response):
        """Should use lockfile default when no explicit or env profile."""
        lockfile.default_profile = "lockfile_default"
        manager = ReceiptManager(lockfile)

        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
        )
        assert receipt.profile == "lockfile_default"

    def test_generate_receipt_handles_exception(self, manager, sample_response):
        """Should handle exceptions gracefully."""
        # Mock receipt generator to raise exception
        manager.receipt_generator = MagicMock()
        manager.receipt_generator.generate_receipt.side_effect = Exception("Test error")

        receipt = manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
        )

        assert receipt is None
        assert len(manager.receipts) == 0

    def test_generate_streaming_receipt_success(self, manager, sample_cost_info):
        """Should generate streaming receipt successfully."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        receipt = manager.generate_streaming_receipt(
            usage=usage,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
            profile="production",
        )

        assert receipt is not None
        assert receipt.alias == "fast"
        assert len(manager.receipts) == 1

    def test_generate_streaming_receipt_no_usage(self, manager, sample_cost_info):
        """Should return None when no usage."""
        receipt = manager.generate_streaming_receipt(
            usage=None,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )

        assert receipt is None
        assert len(manager.receipts) == 0

    def test_get_profile_name_explicit(self, manager):
        """Should return explicit profile."""
        assert manager._get_profile_name("explicit") == "explicit"

    @patch.dict("os.environ", {"LLMRING_PROFILE": "from_env"})
    def test_get_profile_name_env(self, manager):
        """Should return env var when no explicit."""
        assert manager._get_profile_name() == "from_env"

    def test_get_profile_name_lockfile_default(self, lockfile):
        """Should return lockfile default."""
        lockfile.default_profile = "lockfile_default"
        manager = ReceiptManager(lockfile)
        assert manager._get_profile_name() == "lockfile_default"

    def test_get_profile_name_fallback(self):
        """Should return 'default' as fallback."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.default_profile = None
        manager = ReceiptManager(lockfile)
        assert manager._get_profile_name() == "default"

    def test_get_zero_cost_info(self, manager):
        """Should return zero cost dict."""
        zero_cost = manager._get_zero_cost_info()
        assert zero_cost["input_cost"] == 0.0
        assert zero_cost["output_cost"] == 0.0
        assert zero_cost["total_cost"] == 0.0

    def test_clear_receipts(self, manager, sample_response, sample_cost_info):
        """Should clear all receipts."""
        # Generate some receipts
        manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )
        assert len(manager.receipts) == 1

        # Clear receipts
        manager.clear_receipts()
        assert len(manager.receipts) == 0

    def test_get_receipts(self, manager, sample_response, sample_cost_info):
        """Should return copy of receipts."""
        # Generate a receipt
        manager.generate_receipt(
            response=sample_response,
            original_alias="fast",
            provider_type="openai",
            model_name="gpt-4",
            cost_info=sample_cost_info,
        )

        receipts = manager.get_receipts()
        assert len(receipts) == 1
        # Should be a copy
        receipts.clear()
        assert len(manager.receipts) == 1  # Original not affected

    def test_update_lockfile(self, manager):
        """Should update lockfile reference."""
        new_lockfile = MagicMock(spec=Lockfile)
        new_lockfile.calculate_digest.return_value = "xyz789"

        manager.update_lockfile(new_lockfile)
        assert manager.lockfile == new_lockfile
