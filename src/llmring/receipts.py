"""
Receipt generation and verification for LLMRing.

Receipts are issued when connected to server/SaaS and include:
- Alias used
- Profile
- Lock digest
- Token counts
- Cost breakdown
- Ed25519 signature
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    """
    A receipt for an LLM API call.

    Receipts are signed with Ed25519 over JCS-canonicalized JSON.
    """

    # Identity
    receipt_id: str = Field(..., description="Unique receipt identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Receipt timestamp",
    )

    # Request info
    alias: str = Field(..., description="Alias used for the request")
    profile: str = Field(..., description="Profile used")
    lock_digest: str = Field(..., description="SHA256 digest of lockfile")

    # Model info
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

    # Signature (added after creation)
    signature: Optional[str] = Field(None, description="Ed25519 signature (base64)")

    def to_canonical_json(self) -> str:
        """
        Convert receipt to canonical JSON for signing.

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


class ReceiptSigner:
    """Signs and verifies receipts using Ed25519."""

    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey] = None):
        """
        Initialize the signer.

        Args:
            private_key: Ed25519 private key for signing (optional for verification only)
        """
        self.private_key = private_key
        self.public_key = private_key.public_key() if private_key else None

    @classmethod
    def generate_keypair(
        cls,
    ) -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
        """Generate a new Ed25519 keypair."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return private_key, public_key

    @classmethod
    def load_private_key(cls, key_data: bytes) -> ed25519.Ed25519PrivateKey:
        """Load a private key from bytes."""
        return ed25519.Ed25519PrivateKey.from_private_bytes(key_data)

    @classmethod
    def load_public_key(cls, key_data: bytes) -> ed25519.Ed25519PublicKey:
        """Load a public key from bytes."""
        return ed25519.Ed25519PublicKey.from_public_bytes(key_data)

    def sign_receipt(self, receipt: Receipt) -> str:
        """
        Sign a receipt and return the signature.

        Args:
            receipt: Receipt to sign

        Returns:
            Base64-encoded signature

        Raises:
            ValueError: If no private key is available
        """
        if not self.private_key:
            raise ValueError("No private key available for signing")

        # Get canonical JSON
        canonical = receipt.to_canonical_json()

        # Sign
        signature = self.private_key.sign(canonical.encode())

        # Encode as base64
        import base64

        return base64.b64encode(signature).decode()

    @staticmethod
    def verify_receipt(receipt: Receipt, public_key: ed25519.Ed25519PublicKey) -> bool:
        """
        Verify a receipt signature.

        Args:
            receipt: Receipt with signature
            public_key: Public key to verify with

        Returns:
            True if signature is valid
        """
        if not receipt.signature:
            return False

        try:
            # Decode signature
            import base64

            signature = base64.b64decode(receipt.signature)

            # Get canonical JSON
            canonical = receipt.to_canonical_json()

            # Verify
            public_key.verify(signature, canonical.encode())
            return True

        except Exception:
            return False

    def export_private_key(self) -> bytes:
        """Export private key as bytes."""
        if not self.private_key:
            raise ValueError("No private key to export")

        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def export_public_key(self) -> bytes:
        """Export public key as bytes."""
        if not self.public_key:
            raise ValueError("No public key to export")

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )


class ReceiptGenerator:
    """Generates receipts for LLM API calls."""

    def __init__(self, signer: Optional[ReceiptSigner] = None):
        """
        Initialize the generator.

        Args:
            signer: Optional signer for receipts
        """
        self.signer = signer

    def generate_receipt(
        self,
        alias: str,
        profile: str,
        lock_digest: str,
        provider: str,
        model: str,
        usage: Dict[str, int],
        costs: Dict[str, float],
        receipt_id: Optional[str] = None,
    ) -> Receipt:
        """
        Generate a receipt for an API call.

        Args:
            alias: Alias used
            profile: Profile used
            lock_digest: SHA256 digest of lockfile
            provider: Provider used
            model: Model used
            usage: Token usage dict with prompt_tokens, completion_tokens, total_tokens
            costs: Cost dict with input_cost, output_cost, total_cost
            receipt_id: Optional receipt ID (generated if not provided)

        Returns:
            Signed receipt (if signer available) or unsigned receipt
        """
        # Generate receipt ID if not provided
        if not receipt_id:
            import uuid

            receipt_id = str(uuid.uuid4())

        # Create receipt
        receipt = Receipt(
            receipt_id=receipt_id,
            alias=alias,
            profile=profile,
            lock_digest=lock_digest,
            provider=provider,
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            input_cost=costs.get("input_cost", 0.0),
            output_cost=costs.get("output_cost", 0.0),
            total_cost=costs.get("total_cost", 0.0),
        )

        # Sign if signer available
        if self.signer:
            receipt.signature = self.signer.sign_receipt(receipt)

        return receipt

    def calculate_costs(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Calculate costs for token usage.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model_info: Optional model info with pricing

        Returns:
            Dictionary with input_cost, output_cost, total_cost
        """
        # Default to zero if no pricing info
        input_cost = 0.0
        output_cost = 0.0

        if model_info:
            # Get pricing from model info
            input_price = model_info.get("dollars_per_million_tokens_input", 0.0)
            output_price = model_info.get("dollars_per_million_tokens_output", 0.0)

            input_cost = (prompt_tokens / 1_000_000) * input_price
            output_cost = (completion_tokens / 1_000_000) * output_price

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
