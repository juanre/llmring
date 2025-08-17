"""Data models for model refresh operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional


@dataclass
class ModelInfo:
    """Standardized model information from providers."""

    provider: str
    model_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None

    # Capabilities
    max_context: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_parallel_tool_calls: bool = False
    tool_call_format: Optional[str] = None

    # Cost information (dollars per million tokens)
    dollars_per_million_tokens_input: Optional[Decimal] = None
    dollars_per_million_tokens_output: Optional[Decimal] = None

    # Metadata
    is_active: bool = True
    inactive_from: Optional[datetime] = None
    source: str = "api"  # "api", "scrape", "manual"
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDiff:
    """Represents differences between discovered and existing models."""

    new_models: List[ModelInfo] = field(default_factory=list)
    updated_models: List[tuple[ModelInfo, ModelInfo]] = field(
        default_factory=list
    )  # (old, new)
    retired_models: List[ModelInfo] = field(default_factory=list)
    unchanged_models: List[ModelInfo] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes to apply."""
        return bool(self.new_models or self.updated_models or self.retired_models)

    @property
    def summary(self) -> str:
        """Human-readable summary of changes."""
        parts = []
        if self.new_models:
            parts.append(f"{len(self.new_models)} new models")
        if self.updated_models:
            parts.append(f"{len(self.updated_models)} updated models")
        if self.retired_models:
            parts.append(f"{len(self.retired_models)} retired models")
        if self.unchanged_models:
            parts.append(f"{len(self.unchanged_models)} unchanged models")

        return ", ".join(parts) if parts else "No changes"


@dataclass
class RefreshResult:
    """Result of a model refresh operation."""

    success: bool
    message: str
    diff: Optional[ModelDiff] = None
    backup_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Statistics
    models_added: int = 0
    models_updated: int = 0
    models_retired: int = 0
    duration_seconds: float = 0.0

    @classmethod
    def success_result(
        cls, message: str, diff: ModelDiff, backup_id: str, duration: float
    ) -> "RefreshResult":
        """Create a successful refresh result."""
        return cls(
            success=True,
            message=message,
            diff=diff,
            backup_id=backup_id,
            models_added=len(diff.new_models),
            models_updated=len(diff.updated_models),
            models_retired=len(diff.retired_models),
            duration_seconds=duration,
        )

    @classmethod
    def error_result(cls, message: str, errors: List[str]) -> "RefreshResult":
        """Create an error refresh result."""
        return cls(
            success=False,
            message=message,
            errors=errors,
        )
