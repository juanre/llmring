"""Usage tracking interface for LLM service."""

from typing import Any, Dict, Optional, Protocol


class UsageTracker(Protocol):
    """Interface for tracking LLM usage."""

    async def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record LLM usage.

        Args:
            model: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            user_id: Optional user identifier
            metadata: Optional additional metadata
        """
        ...
