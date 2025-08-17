"""Type definitions for the LLM Service API.

This module defines all the data types used by the LLM Service API to ensure
complete encapsulation of internal implementation details.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ModelInfo(BaseModel):
    """Complete information about an LLM model.

    This class provides a complete view of a model's capabilities, status, and pricing
    without exposing any internal database schema details. All pricing is normalized
    to dollars per million tokens for consistency.

    Attributes:
        provider: The model provider (e.g., 'openai', 'anthropic', 'google', 'ollama')
        model_name: The canonical model name as used by the provider
        display_name: Human-friendly display name for the model
        description: Optional description of the model's strengths or use cases
        is_active: Whether the model is currently active and available
        last_updated: When the model information was last updated
        added_date: When the model was first added to the registry
        inactive_since: When the model was deactivated (None if active)
        max_context_tokens: Maximum input context size in tokens
        max_output_tokens: Maximum output size in tokens
        supports_vision: Whether the model can process images
        supports_function_calling: Whether the model supports function/tool calling
        supports_json_mode: Whether the model can guarantee JSON output
        supports_parallel_tool_calls: Whether multiple tools can be called in one turn
        supports_streaming: Whether the model supports streaming responses
        tool_call_format: Format for tool calls (provider-specific)
        cost_per_million_input_tokens: Cost in USD per million input tokens
        cost_per_million_output_tokens: Cost in USD per million output tokens
    """

    # Identity
    provider: str = Field(
        ..., description="Model provider (openai, anthropic, google, ollama)"
    )
    model_name: str = Field(..., description="Canonical model name")
    display_name: str = Field(..., description="Human-friendly display name")
    description: Optional[str] = Field(None, description="Model description")

    # Status
    is_active: bool = Field(..., description="Whether model is currently active")
    last_updated: datetime = Field(..., description="Last update timestamp")
    added_date: datetime = Field(..., description="When model was added")
    inactive_since: Optional[datetime] = Field(
        None, description="Deactivation timestamp"
    )

    # Capabilities
    max_context_tokens: Optional[int] = Field(
        None, description="Max input context size"
    )
    max_output_tokens: Optional[int] = Field(None, description="Max output size")
    supports_vision: bool = Field(False, description="Can process images")
    supports_function_calling: bool = Field(
        False, description="Supports function calls"
    )
    supports_json_mode: bool = Field(False, description="Can guarantee JSON output")
    supports_parallel_tool_calls: bool = Field(
        False, description="Can call multiple tools"
    )
    supports_streaming: bool = Field(True, description="Supports streaming responses")
    tool_call_format: Optional[str] = Field(None, description="Tool call format")

    # Pricing (always in dollars per million tokens)
    cost_per_million_input_tokens: Optional[Decimal] = Field(
        None, description="Input cost per 1M tokens"
    )
    cost_per_million_output_tokens: Optional[Decimal] = Field(
        None, description="Output cost per 1M tokens"
    )

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def has_pricing(self) -> bool:
        """Check if model has complete pricing information.

        Returns:
            True if both input and output pricing are available, False otherwise.
        """
        return (
            self.cost_per_million_input_tokens is not None
            and self.cost_per_million_output_tokens is not None
        )

    def get_cost_per_unit(
        self, unit: Literal["token", "1k", "1m"] = "1k"
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get cost per specified unit.

        Args:
            unit: The unit to calculate cost for:
                - 'token': Cost per single token
                - '1k': Cost per thousand tokens
                - '1m': Cost per million tokens

        Returns:
            Tuple of (input_cost, output_cost) for the specified unit,
            or (None, None) if pricing is not available.
        """
        if not self.has_pricing:
            return (None, None)

        divisor = {"token": 1_000_000, "1k": 1_000, "1m": 1}[unit]

        return (
            float(self.cost_per_million_input_tokens) / divisor,
            float(self.cost_per_million_output_tokens) / divisor,
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate total cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD, or None if pricing is not available.
        """
        if not self.has_pricing:
            return None

        input_cost = (input_tokens / 1_000_000) * float(
            self.cost_per_million_input_tokens
        )
        output_cost = (output_tokens / 1_000_000) * float(
            self.cost_per_million_output_tokens
        )
        return input_cost + output_cost

    def format_cost_string(self, unit: Literal["token", "1k", "1m"] = "1k") -> str:
        """Format pricing as a human-readable display string.

        Args:
            unit: The unit to display pricing for (default: '1k')

        Returns:
            Formatted string like '$0.005/$0.015 per 1k tokens' or 'Pricing not available'
        """
        input_cost, output_cost = self.get_cost_per_unit(unit)
        if input_cost is None or output_cost is None:
            return "Pricing not available"

        unit_label = {"token": "token", "1k": "1k tokens", "1m": "1M tokens"}[unit]

        return f"${input_cost:.3f}/${output_cost:.3f} per {unit_label}"


class ProviderInfo(BaseModel):
    """Information about a model provider.

    Provides summary information about a provider including model counts,
    API key status, and supported features.

    Attributes:
        provider: Provider identifier (e.g., 'openai', 'anthropic')
        display_name: Human-friendly provider name
        total_models: Total number of models from this provider
        active_models: Number of currently active models
        has_api_key: Whether an API key is configured for this provider
        supports_streaming: Whether provider supports streaming responses
        features: List of supported features ('vision', 'functions', 'json')
    """

    provider: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-friendly provider name")
    total_models: int = Field(..., description="Total number of models")
    active_models: int = Field(..., description="Number of active models")
    has_api_key: bool = Field(..., description="Whether API key is configured")
    supports_streaming: bool = Field(True, description="Supports streaming")
    features: List[str] = Field(default_factory=list, description="Supported features")


class ProviderStats(BaseModel):
    """Detailed statistics for a specific provider.

    Provides comprehensive statistics about a provider's models including
    capability counts, pricing ranges, and context size information.

    Attributes:
        total_models: Total number of models from this provider
        active_models: Number of currently active models
        vision_models: Number of models supporting vision/image input
        function_calling_models: Number of models supporting function calls
        json_mode_models: Number of models supporting JSON output mode
        min_input_cost: Minimum input cost per million tokens (USD)
        max_input_cost: Maximum input cost per million tokens (USD)
        avg_input_cost: Average input cost per million tokens (USD)
        min_context_size: Minimum context size across all models
        max_context_size: Maximum context size across all models
    """

    total_models: int = Field(..., description="Total models")
    active_models: int = Field(..., description="Active models")
    vision_models: int = Field(..., description="Models with vision support")
    function_calling_models: int = Field(
        ..., description="Models with function calling"
    )
    json_mode_models: int = Field(..., description="Models with JSON mode")
    min_input_cost: Optional[float] = Field(
        None, description="Min input cost per 1M tokens"
    )
    max_input_cost: Optional[float] = Field(
        None, description="Max input cost per 1M tokens"
    )
    avg_input_cost: Optional[float] = Field(
        None, description="Avg input cost per 1M tokens"
    )
    min_context_size: Optional[int] = Field(None, description="Min context size")
    max_context_size: Optional[int] = Field(None, description="Max context size")


class ModelStatistics(BaseModel):
    """Overall statistics about all available models.

    Provides a comprehensive view of the model registry including counts,
    pricing availability, and per-provider breakdowns.

    Attributes:
        total_models: Total number of models in the registry
        active_models: Number of currently active models
        inactive_models: Number of inactive/deprecated models
        models_with_pricing: Number of models with pricing information
        models_without_pricing: Number of models without pricing
        last_refresh: When the model data was last refreshed
        providers: Per-provider statistics breakdown
    """

    total_models: int = Field(..., description="Total models in registry")
    active_models: int = Field(..., description="Currently active models")
    inactive_models: int = Field(..., description="Inactive models")
    models_with_pricing: int = Field(..., description="Models with pricing info")
    models_without_pricing: int = Field(..., description="Models without pricing")
    last_refresh: Optional[datetime] = Field(None, description="Last data refresh time")
    providers: Dict[str, ProviderStats] = Field(..., description="Per-provider stats")


class CostBreakdown(BaseModel):
    """Detailed cost calculation result.

    Provides a complete breakdown of costs for a specific model usage,
    including token counts, individual costs, and pricing information.

    Attributes:
        provider: Model provider
        model_name: Name of the model used
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
    """

    provider: str = Field(..., description="Model provider")
    model_name: str = Field(..., description="Model name")
    input_tokens: int = Field(..., description="Input token count")
    output_tokens: int = Field(..., description="Output token count")
    input_cost: float = Field(..., description="Cost for input tokens")
    output_cost: float = Field(..., description="Cost for output tokens")
    total_cost: float = Field(..., description="Total cost (input + output)")
    currency: str = Field("USD", description="Currency for costs")
    cost_per_million_input: float = Field(
        ..., description="Input pricing per 1M tokens"
    )
    cost_per_million_output: float = Field(
        ..., description="Output pricing per 1M tokens"
    )


class ModelRequirements(BaseModel):
    """Requirements for finding compatible models.

    Specifies criteria for filtering models based on capabilities,
    pricing constraints, and feature requirements.

    Attributes:
        min_context_size: Minimum required context window size
        max_input_cost_per_million: Maximum acceptable input cost per 1M tokens
        max_output_cost_per_million: Maximum acceptable output cost per 1M tokens
        requires_vision: Model must support image/vision input
        requires_function_calling: Model must support function/tool calling
        requires_json_mode: Model must support guaranteed JSON output
        requires_parallel_tools: Model must support parallel tool calls
        providers: List of acceptable providers (None = all providers)
        active_only: Only consider active models
    """

    min_context_size: Optional[int] = Field(
        None, description="Min required context size"
    )
    max_input_cost_per_million: Optional[float] = Field(
        None, description="Max input cost/1M tokens"
    )
    max_output_cost_per_million: Optional[float] = Field(
        None, description="Max output cost/1M tokens"
    )
    requires_vision: bool = Field(False, description="Must support vision")
    requires_function_calling: bool = Field(False, description="Must support functions")
    requires_json_mode: bool = Field(False, description="Must support JSON mode")
    requires_parallel_tools: bool = Field(
        False, description="Must support parallel tools"
    )
    providers: Optional[List[str]] = Field(None, description="Allowed providers")
    active_only: bool = Field(True, description="Only active models")


class RefreshResult(BaseModel):
    """Result of a model refresh operation.

    Contains detailed information about the outcome of refreshing
    model data from various sources.

    Attributes:
        success: Whether the refresh completed successfully
        message: Summary message describing the result
        models_added: Number of new models added
        models_updated: Number of existing models updated
        models_deactivated: Number of models marked inactive
        models_activated: Number of models reactivated
        errors: List of error messages encountered
        duration_seconds: Time taken for the refresh operation
        timestamp: When the refresh was performed
        source: Data source used ('api', 'json', 'pdf')
    """

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Result summary")
    models_added: int = Field(0, description="New models added")
    models_updated: int = Field(0, description="Models updated")
    models_deactivated: int = Field(0, description="Models deactivated")
    models_activated: int = Field(0, description="Models activated")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    duration_seconds: float = Field(..., description="Operation duration")
    timestamp: datetime = Field(..., description="Operation timestamp")
    source: str = Field(..., description="Data source (api/json/pdf)")


class ValidationResult(BaseModel):
    """Result of model validation.

    Contains the outcome of validating a model request including
    existence checks, status verification, and requirement matching.

    Attributes:
        valid: Overall validation result
        model_exists: Whether the model exists in the registry
        model_active: Whether the model is currently active
        meets_requirements: Whether model meets specified requirements
        issues: List of validation issues found
        suggestions: List of alternative model suggestions
    """

    valid: bool = Field(..., description="Overall validation result")
    model_exists: bool = Field(..., description="Model exists in registry")
    model_active: bool = Field(..., description="Model is active")
    meets_requirements: bool = Field(..., description="Meets all requirements")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(
        default_factory=list, description="Alternative suggestions"
    )


class ServiceHealth(BaseModel):
    """Health status of the LLM service.

    Provides comprehensive health information about the service
    including database connectivity and model availability.

    Attributes:
        status: Overall health status
        database_connected: Whether database connection is active
        models_loaded: Number of models successfully loaded
        last_refresh: When models were last refreshed
        version: API version
        issues: List of current issues affecting service health
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall status"
    )
    database_connected: bool = Field(..., description="Database connection status")
    models_loaded: int = Field(..., description="Number of models loaded")
    last_refresh: Optional[datetime] = Field(
        None, description="Last model refresh time"
    )
    version: str = Field(..., description="API version")
    issues: List[str] = Field(default_factory=list, description="Current issues")
