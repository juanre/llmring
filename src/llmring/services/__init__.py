"""
Service layer for LLMRing - extracted from monolithic LLMRing class.

This package contains focused service classes that handle specific responsibilities:
- AliasResolver: Resolve model aliases to concrete provider:model references
- SchemaAdapter: Adapt schemas and tools for provider-specific requirements
- CostCalculator: Calculate costs based on token usage and pricing
- ReceiptManager: Generate and manage receipts for LLM requests
- ValidationService: Validate requests against model capabilities and constraints
"""

from llmring.services.alias_resolver import AliasResolver
from llmring.services.cost_calculator import CostCalculator
from llmring.services.receipt_manager import ReceiptManager
from llmring.services.schema_adapter import SchemaAdapter
from llmring.services.validation_service import ValidationService

__all__ = [
    "AliasResolver",
    "CostCalculator",
    "ReceiptManager",
    "SchemaAdapter",
    "ValidationService",
]
