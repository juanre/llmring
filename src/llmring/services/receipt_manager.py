"""
Receipt management service for LLMRing.

Handles generation and storage of receipts for LLM API calls.
"""

import logging
import os
from typing import Dict, List, Optional

from llmring.lockfile_core import Lockfile
from llmring.receipts import Receipt, ReceiptGenerator
from llmring.schemas import LLMResponse

logger = logging.getLogger(__name__)


class ReceiptManager:
    """
    Manages receipt generation and storage for LLM API calls.

    Receipts provide an auditable record of:
    - Which alias/model was used
    - Which lockfile version was active
    - Token usage and costs
    - Timestamp and unique receipt ID
    """

    def __init__(self, lockfile: Optional[Lockfile] = None):
        """
        Initialize the receipt manager.

        Args:
            lockfile: Optional lockfile for digest calculation
        """
        self.lockfile = lockfile
        self.receipt_generator: Optional[ReceiptGenerator] = None
        self.receipts: List[Receipt] = []

    def generate_receipt(
        self,
        response: LLMResponse,
        original_alias: str,
        provider_type: str,
        model_name: str,
        cost_info: Optional[Dict[str, float]] = None,
        profile: Optional[str] = None,
    ) -> Optional[Receipt]:
        """
        Generate a receipt for an LLM API call.

        Args:
            response: The LLM response with usage information
            original_alias: The original alias or model string used
            provider_type: Provider type (e.g., "openai")
            model_name: Model name (e.g., "gpt-4")
            cost_info: Optional cost information
            profile: Optional profile name

        Returns:
            Generated receipt or None if generation fails
        """
        if not response.usage or not self.lockfile:
            logger.debug("Skipping receipt generation: no usage or lockfile")
            return None

        try:
            # Initialize receipt generator if not already done
            if not self.receipt_generator:
                self.receipt_generator = ReceiptGenerator()

            # Calculate lockfile digest
            lock_digest = self.lockfile.calculate_digest()

            # Determine profile used
            profile_name = self._get_profile_name(profile)

            # Determine alias (or mark as direct model)
            alias = original_alias if ":" not in original_alias else "direct_model"

            # Generate receipt
            receipt = self.receipt_generator.generate_receipt(
                alias=alias,
                profile=profile_name,
                lock_digest=lock_digest,
                provider=provider_type,
                model=model_name,
                usage=response.usage,
                costs=cost_info or self._get_zero_cost_info(),
            )

            # Store receipt locally
            self.receipts.append(receipt)
            logger.debug(f"Generated receipt {receipt.receipt_id} for {provider_type}:{model_name}")

            return receipt

        except Exception as e:
            logger.warning(f"Failed to generate receipt: {e}")
            return None

    def generate_streaming_receipt(
        self,
        usage: Dict,
        original_alias: str,
        provider_type: str,
        model_name: str,
        cost_info: Optional[Dict[str, float]] = None,
        profile: Optional[str] = None,
    ) -> Optional[Receipt]:
        """
        Generate a receipt for a streaming LLM API call.

        Args:
            usage: Token usage dictionary from accumulated chunks
            original_alias: The original alias or model string used
            provider_type: Provider type (e.g., "openai")
            model_name: Model name (e.g., "gpt-4")
            cost_info: Optional cost information
            profile: Optional profile name

        Returns:
            Generated receipt or None if generation fails
        """
        if not usage or not self.lockfile:
            logger.debug("Skipping streaming receipt generation: no usage or lockfile")
            return None

        try:
            # Initialize receipt generator if needed
            if not self.receipt_generator:
                self.receipt_generator = ReceiptGenerator()

            # Calculate lockfile digest
            lock_digest = self.lockfile.calculate_digest()

            # Determine profile used
            profile_name = self._get_profile_name(profile)

            # Determine alias (or mark as direct model)
            alias = original_alias if ":" not in original_alias else "direct_model"

            # Generate receipt
            receipt = self.receipt_generator.generate_receipt(
                alias=alias,
                profile=profile_name,
                lock_digest=lock_digest,
                provider=provider_type,
                model=model_name,
                usage=usage,
                costs=cost_info or self._get_zero_cost_info(),
            )

            # Store receipt locally
            self.receipts.append(receipt)
            logger.debug(
                f"Generated receipt {receipt.receipt_id} for streaming {provider_type}:{model_name}"
            )

            return receipt

        except Exception as e:
            logger.warning(f"Failed to generate receipt for streaming: {e}")
            return None

    def _get_profile_name(self, profile: Optional[str] = None) -> str:
        """
        Get the profile name from various sources.

        Priority:
        1. Explicit profile argument
        2. LLMRING_PROFILE environment variable
        3. Lockfile default profile
        4. "default" as fallback

        Args:
            profile: Explicit profile name

        Returns:
            Profile name to use
        """
        if profile:
            return profile
        if env_profile := os.getenv("LLMRING_PROFILE"):
            return env_profile
        if self.lockfile and self.lockfile.default_profile:
            return self.lockfile.default_profile
        return "default"

    @staticmethod
    def _get_zero_cost_info() -> Dict[str, float]:
        """
        Get a zero-cost info dictionary.

        Returns:
            Dictionary with all costs set to 0.0
        """
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    def clear_receipts(self):
        """Clear stored receipts."""
        self.receipts.clear()
        logger.debug("Cleared all stored receipts")

    def get_receipts(self) -> List[Receipt]:
        """
        Get all stored receipts.

        Returns:
            List of receipts
        """
        return self.receipts.copy()

    def update_lockfile(self, lockfile: Optional[Lockfile]):
        """
        Update the lockfile reference.

        Args:
            lockfile: New lockfile instance
        """
        self.lockfile = lockfile
        logger.debug("Updated receipt manager lockfile reference")
