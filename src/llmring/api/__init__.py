"""LLM Service API module.

This module provides a complete API for interacting with LLM models without
exposing any internal database schema or implementation details.
"""

from llmring.api.service import LLMRingAPI
from llmring.api.types import (CostBreakdown, ModelInfo, ModelRequirements,
                                   ModelStatistics, ProviderInfo,
                                   ProviderStats, RefreshResult, ServiceHealth,
                                   ValidationResult)

__all__ = [
    # Types
    "ModelInfo",
    "ProviderInfo",
    "ProviderStats",
    "ModelStatistics",
    "CostBreakdown",
    "ModelRequirements",
    "RefreshResult",
    "ValidationResult",
    "ServiceHealth",
    # Service
    "LLMRingAPI",
]
