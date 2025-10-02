"""
Receipt schema for LLMRing.

Receipts are generated and signed by llmring-server, not by the client.
This module only contains the Receipt model for deserializing server-issued receipts.

For receipt generation and signing, see llmring-server/src/llmring_server/models/receipts.py
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Receipt(BaseModel):
    """
    A receipt for an LLM API call.

    Receipts are issued by llmring-server and signed with Ed25519 over JCS-canonicalized JSON.
    This class is used for deserializing receipts received from the server.

    Phase 7.5: Supports both single and batch receipts.
    """

    # Identity
    receipt_id: str = Field(..., description="Unique receipt identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Receipt timestamp",
    )

    # Receipt type (Phase 7.5)
    receipt_type: str = Field(
        default="single",
        description="Type: 'single' (one call) or 'batch' (multiple calls)",
    )

    # Request info (for single receipts)
    alias: str = Field(..., description="Alias used for the request")
    profile: str = Field(..., description="Profile used")
    lock_digest: str = Field(..., description="SHA256 digest of lockfile")

    # Model info (for single receipts)
    provider: str = Field(..., description="Provider used")
    model: str = Field(..., description="Model used")

    # Usage
    prompt_tokens: int = Field(..., description="Input tokens")
    completion_tokens: int = Field(..., description="Output tokens")
    total_tokens: int = Field(..., description="Total tokens")

    # Cost
    input_cost: float = Field(..., description="Cost for input tokens (USD)")
    output_cost: float = Field(..., description="Cost for output tokens (USD)")
    total_cost: float = Field(..., description="Total cost (USD)")

    # Batch receipt fields (Phase 7.5)
    batch_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated statistics for batch receipts",
    )
    description: Optional[str] = Field(
        None, description="User-provided description for the receipt"
    )
    tags: Optional[List[str]] = Field(None, description="User-provided tags for categorization")

    # Signature (added by server)
    signature: Optional[str] = Field(None, description="Ed25519 signature (base64)")

    def to_canonical_json(self) -> str:
        """
        Convert receipt to canonical JSON for verification.

        Uses JCS (JSON Canonicalization Scheme) for deterministic output.
        """
        # Get dict without signature
        data = self.model_dump(exclude={"signature"})

        # Convert datetime to ISO format
        data["timestamp"] = self.timestamp.isoformat()

        # Sort keys and serialize with no whitespace
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def calculate_digest(self) -> str:
        """Calculate SHA256 digest of canonical JSON."""
        canonical = self.to_canonical_json()
        return hashlib.sha256(canonical.encode()).hexdigest()
