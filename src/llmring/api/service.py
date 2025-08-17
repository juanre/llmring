"""LLM Service API implementation.

This module provides a complete API for interacting with LLM models without
exposing any internal database schema or implementation details.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from llmring.api.types import (CostBreakdown, ModelInfo, ModelRequirements,
                                   ModelStatistics, ProviderInfo,
                                   ProviderStats, ServiceHealth,
                                   ValidationResult)
from llmring.db import LLMDatabase
from llmring.schemas import LLMModel

logger = logging.getLogger(__name__)


class LLMRingAPI:
    """Complete API for LLM Service operations.

    This class provides a comprehensive abstraction layer over the LLM service
    database, ensuring that consumers never need to know about internal schema
    details or implementation specifics. All methods return well-defined types
    from the api.types module.

    The API is organized into the following categories:
    - Model Discovery & Retrieval: Finding and getting model information
    - Model Statistics & Summaries: Aggregate data about models
    - Cost Calculations: Computing and comparing model costs
    - Model Filtering & Search: Finding models by criteria
    - Model Management: Activating/deactivating models
    - Model Validation: Checking model availability and requirements
    - Service Health: Monitoring service status
    - Utility Methods: Helper functions for common agents

    Example:
        api = LLMRingAPI(db)
        models = await api.list_models(provider="openai")
        cost = await api.calculate_cost("openai", "gpt-4o", 1000, 500)
    """

    def __init__(self, db: LLMDatabase):
        """Initialize the API service.

        Args:
            db: LLMDatabase instance for database operations

        Note:
            The API maintains an internal cache with a 5-minute TTL
            for frequently accessed data.
        """
        self.db = db
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes

    def _model_to_info(self, model: LLMModel) -> ModelInfo:
        """Convert internal LLMModel to API ModelInfo.

        This internal method handles the transformation from database
        schema objects to public API types, ensuring complete encapsulation.

        Args:
            model: Internal LLMModel from database

        Returns:
            ModelInfo object suitable for API consumers
        """
        return ModelInfo(
            provider=model.provider,
            model_name=model.model_name,
            display_name=model.display_name or model.model_name,
            description=model.description,
            is_active=model.inactive_from is None,
            last_updated=model.updated_at,
            added_date=model.created_at,
            inactive_since=model.inactive_from,
            max_context_tokens=model.max_context,
            max_output_tokens=model.max_output_tokens,
            supports_vision=model.supports_vision or False,
            supports_function_calling=model.supports_function_calling or False,
            supports_json_mode=model.supports_json_mode or False,
            supports_parallel_tool_calls=model.supports_parallel_tool_calls or False,
            supports_streaming=True,  # Default to True
            tool_call_format=model.tool_call_format,
            cost_per_million_input_tokens=model.dollars_per_million_tokens_input,
            cost_per_million_output_tokens=model.dollars_per_million_tokens_output,
        )

    # Model Discovery & Retrieval

    async def list_models(
        self,
        provider: Optional[str] = None,
        active_only: bool = True,
        include_pricing: bool = True,
        include_capabilities: bool = True,
        sort_by: str = "model_name",
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> List[ModelInfo]:
        """List all available models with optional filtering.

        Provides a comprehensive list of models with flexible filtering
        and sorting options. This is the primary method for discovering
        available models.

        Args:
            provider: Filter by specific provider (e.g., 'openai', 'anthropic').
                     None returns models from all providers.
            active_only: If True, only return currently active models.
                        If False, include inactive/deprecated models.
            include_pricing: Include pricing information in results (always True currently).
            include_capabilities: Include capability flags in results (always True currently).
            sort_by: Field to sort by. Options: 'model_name', 'provider', 'display_name', 'cost'.
            sort_order: Sort order - 'asc' for ascending, 'desc' for descending.

        Returns:
            List of ModelInfo objects matching the criteria, sorted as requested.

        Example:
            # Get all active OpenAI models sorted by cost
            models = await api.list_models(
                provider="openai",
                sort_by="cost",
                sort_order="asc"
            )
        """
        # Get models from database, passing active_only to the database query
        models = await self.db.list_models(provider=provider, active_only=active_only)

        # Convert to ModelInfo
        model_infos = [self._model_to_info(m) for m in models]

        # Sort
        reverse = sort_order == "desc"
        if sort_by in ["model_name", "provider", "display_name"]:
            model_infos.sort(key=lambda m: getattr(m, sort_by), reverse=reverse)
        elif sort_by == "cost":
            model_infos.sort(
                key=lambda m: float(m.cost_per_million_input_tokens or 0),
                reverse=reverse,
            )

        return model_infos

    async def get_model(self, provider: str, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information for a specific model.

        Retrieves complete information about a single model, including
        inactive models. This is useful for checking model details
        even after a model has been deprecated.

        Args:
            provider: The model provider (e.g., 'openai', 'anthropic')
            model_name: The exact model name (e.g., 'gpt-4o', 'claude-3-opus-20240229')

        Returns:
            ModelInfo object if found, None if the model doesn't exist.

        Example:
            model = await api.get_model("openai", "gpt-4o")
            if model and model.has_pricing:
                print(f"Cost: {model.format_cost_string()}")
        """
        # Always get model regardless of active status
        model = await self.db.get_model(provider, model_name, active_only=False)
        if model is None:
            return None
        return self._model_to_info(model)

    async def get_providers(self) -> List[ProviderInfo]:
        """Get information about all available providers.

        Returns a summary of each provider including model counts,
        API key status, and supported features. This is useful for
        understanding what providers are available and configured.

        Returns:
            List of ProviderInfo objects, one for each provider,
            sorted alphabetically by provider name.

        Note:
            API key status is determined by checking environment variables.
            The 'features' field aggregates capabilities across all models
            from that provider.

        Example:
            providers = await api.get_providers()
            for p in providers:
                if p.has_api_key:
                    print(f"{p.provider}: {p.active_models} active models")
        """
        models = await self.db.list_models(active_only=False)

        # Group by provider
        provider_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "active": 0, "has_key": False, "features": set()}
        )

        for model in models:
            data = provider_data[model.provider]
            data["total"] += 1
            if model.inactive_from is None:
                data["active"] += 1

            # Track features
            if model.supports_vision:
                data["features"].add("vision")
            if model.supports_function_calling:
                data["features"].add("functions")
            if model.supports_json_mode:
                data["features"].add("json")

        # Check API keys from environment variables
        import os

        api_key_status = {
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "google": bool(
                os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            ),
            "ollama": True,  # Always available locally
        }

        # Convert to ProviderInfo
        providers = []
        for provider, data in provider_data.items():
            providers.append(
                ProviderInfo(
                    provider=provider,
                    display_name=provider.title(),
                    total_models=data["total"],
                    active_models=data["active"],
                    has_api_key=api_key_status.get(provider, False),
                    supports_streaming=True,
                    features=sorted(list(data["features"])),
                )
            )

        return sorted(providers, key=lambda p: p.provider)

    async def batch_get_models(
        self, model_requests: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Optional[ModelInfo]]:
        """Get multiple models in a single request."""
        results = {}

        # Could optimize with bulk query, but for now iterate
        for provider, model_name in model_requests:
            model = await self.get_model(provider, model_name)
            results[(provider, model_name)] = model

        return results

    # Model Statistics & Summaries

    async def get_model_statistics(self) -> ModelStatistics:
        """Get comprehensive statistics about all models."""
        models = await self.db.list_models(active_only=False)

        # Initialize counters
        total = len(models)
        active = sum(1 for m in models if m.inactive_from is None)
        with_pricing = sum(
            1 for m in models if m.dollars_per_million_tokens_input is not None
        )

        # Provider statistics
        provider_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "active": 0,
                "vision": 0,
                "functions": 0,
                "json": 0,
                "costs": [],
                "contexts": [],
            }
        )

        for model in models:
            stats = provider_stats[model.provider]
            stats["total"] += 1

            if model.inactive_from is None:
                stats["active"] += 1

            if model.supports_vision:
                stats["vision"] += 1
            if model.supports_function_calling:
                stats["functions"] += 1
            if model.supports_json_mode:
                stats["json"] += 1

            if model.dollars_per_million_tokens_input:
                stats["costs"].append(float(model.dollars_per_million_tokens_input))
            if model.max_context:
                stats["contexts"].append(model.max_context)

        # Convert to ProviderStats
        providers = {}
        for provider, stats in provider_stats.items():
            providers[provider] = ProviderStats(
                total_models=stats["total"],
                active_models=stats["active"],
                vision_models=stats["vision"],
                function_calling_models=stats["functions"],
                json_mode_models=stats["json"],
                min_input_cost=min(stats["costs"]) if stats["costs"] else None,
                max_input_cost=max(stats["costs"]) if stats["costs"] else None,
                avg_input_cost=(
                    sum(stats["costs"]) / len(stats["costs"])
                    if stats["costs"]
                    else None
                ),
                min_context_size=min(stats["contexts"]) if stats["contexts"] else None,
                max_context_size=max(stats["contexts"]) if stats["contexts"] else None,
            )

        return ModelStatistics(
            total_models=total,
            active_models=active,
            inactive_models=total - active,
            models_with_pricing=with_pricing,
            models_without_pricing=total - with_pricing,
            last_refresh=None,  # Would need to track this
            providers=providers,
        )

    async def get_provider_summary(self, provider: str) -> Optional[ProviderStats]:
        """Get detailed statistics for a specific provider."""
        stats = await self.get_model_statistics()
        return stats.providers.get(provider)

    async def get_cost_statistics(
        self, provider: Optional[str] = None, active_only: bool = True
    ) -> Dict[str, Any]:
        """Get cost statistics across models."""
        models = await self.list_models(provider=provider, active_only=active_only)

        input_costs = []
        output_costs = []

        for model in models:
            if model.has_pricing:
                input_costs.append(float(model.cost_per_million_input_tokens))
                output_costs.append(float(model.cost_per_million_output_tokens))

        if not input_costs:
            return {
                "models_with_pricing": 0,
                "median_input_cost": None,
                "median_output_cost": None,
                "min_input_cost": None,
                "max_input_cost": None,
                "percentiles": {},
            }

        input_costs.sort()
        output_costs.sort()

        def get_percentile(lst, p):
            k = (len(lst) - 1) * p
            f = int(k)
            c = f + 1 if f < len(lst) - 1 else f
            return lst[f] + (k - f) * (lst[c] - lst[f])

        return {
            "models_with_pricing": len(input_costs),
            "median_input_cost": get_percentile(input_costs, 0.5),
            "median_output_cost": get_percentile(output_costs, 0.5),
            "min_input_cost": min(input_costs),
            "max_input_cost": max(input_costs),
            "min_output_cost": min(output_costs),
            "max_output_cost": max(output_costs),
            "percentiles": {
                "25th": {
                    "input": get_percentile(input_costs, 0.25),
                    "output": get_percentile(output_costs, 0.25),
                },
                "75th": {
                    "input": get_percentile(input_costs, 0.75),
                    "output": get_percentile(output_costs, 0.75),
                },
                "90th": {
                    "input": get_percentile(input_costs, 0.9),
                    "output": get_percentile(output_costs, 0.9),
                },
            },
        }

    # Cost Calculations

    async def calculate_cost(
        self,
        provider: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        include_breakdown: bool = True,
    ) -> Optional[Union[float, CostBreakdown]]:
        """Calculate the cost for a specific token usage.

        Computes the total cost for processing a given number of input
        and output tokens with a specific model. Can return either a
        simple total or a detailed breakdown.

        Args:
            provider: The model provider
            model_name: The model to calculate costs for
            input_tokens: Number of input tokens to process
            output_tokens: Number of output tokens to generate
            include_breakdown: If True, return detailed CostBreakdown.
                             If False, return just the total cost as float.

        Returns:
            If include_breakdown=True: CostBreakdown with detailed costs
            If include_breakdown=False: Total cost as float
            None if model not found or has no pricing

        Example:
            # Get detailed breakdown
            breakdown = await api.calculate_cost(
                "openai", "gpt-4o",
                input_tokens=1000,
                output_tokens=500
            )
            print(f"Total: ${breakdown.total_cost:.4f}")

            # Get just the total
            total = await api.calculate_cost(
                "openai", "gpt-4o",
                1000, 500,
                include_breakdown=False
            )
        """
        model = await self.get_model(provider, model_name)
        if not model or not model.has_pricing:
            return None

        cost = model.calculate_cost(input_tokens, output_tokens)

        if not include_breakdown:
            return cost

        return CostBreakdown(
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=(input_tokens / 1_000_000)
            * float(model.cost_per_million_input_tokens),
            output_cost=(output_tokens / 1_000_000)
            * float(model.cost_per_million_output_tokens),
            total_cost=cost,
            currency="USD",
            cost_per_million_input=float(model.cost_per_million_input_tokens),
            cost_per_million_output=float(model.cost_per_million_output_tokens),
        )

    async def estimate_cost_for_conversation(
        self,
        provider: str,
        model_name: str,
        messages: List[Dict[str, str]],
        expected_output_tokens: Optional[int] = None,
    ) -> Optional[CostBreakdown]:
        """Estimate cost for a conversation.

        Provides a rough cost estimate for a conversation based on
        message content. Uses simple heuristics for token counting
        as actual tokenization is provider-specific.

        Args:
            provider: The model provider
            model_name: The model to use
            messages: List of message dicts with 'role' and 'content' keys
            expected_output_tokens: Expected response length in tokens.
                                  If None, estimates based on input length.

        Returns:
            CostBreakdown with estimated costs, or None if model has no pricing

        Note:
            Token estimation uses a rough heuristic of 1 token per 4 characters.
            For accurate estimates, use provider-specific tokenizers.

        Example:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about programming."}
            ]
            estimate = await api.estimate_cost_for_conversation(
                "anthropic", "claude-3-opus-20240229",
                messages,
                expected_output_tokens=50
            )
        """
        model = await self.get_model(provider, model_name)
        if not model or not model.has_pricing:
            return None

        # Simple token estimation (would need provider-specific tokenizers)
        # Rough estimate: 1 token per 4 characters
        input_tokens = sum(len(msg.get("content", "")) // 4 for msg in messages)

        # Default output estimation
        if expected_output_tokens is None:
            expected_output_tokens = min(input_tokens // 2, 1000)

        return await self.calculate_cost(
            provider, model_name, input_tokens, expected_output_tokens
        )

    async def format_cost_display(
        self,
        amount: float,
        tokens: int,
        unit: Literal["token", "1k", "1m"] = "1k",
        currency: str = "USD",
    ) -> str:
        """Format a cost amount for display."""
        divisor = {"token": 1, "1k": 1_000, "1m": 1_000_000}[unit]

        unit_label = {"token": "token", "1k": "1k tokens", "1m": "1M tokens"}[unit]

        cost_per_unit = amount / (tokens / divisor)
        return f"${cost_per_unit:.3f}/{unit_label}"

    async def compare_model_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        providers: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[CostBreakdown]:
        """Compare costs across multiple models for given token counts."""
        models = await self.list_models(provider=None, active_only=True)

        if providers:
            models = [m for m in models if m.provider in providers]

        # Calculate costs
        costs = []
        for model in models:
            if model.has_pricing:
                breakdown = await self.calculate_cost(
                    model.provider,
                    model.model_name,
                    input_tokens,
                    output_tokens,
                    include_breakdown=True,
                )
                if breakdown:
                    costs.append(breakdown)

        # Sort by total cost
        costs.sort(key=lambda c: c.total_cost)

        return costs[:max_results]

    # Model Filtering & Search

    async def find_models_by_features(
        self,
        vision: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        json_mode: Optional[bool] = None,
        parallel_tools: Optional[bool] = None,
        streaming: Optional[bool] = None,
        active_only: bool = True,
        providers: Optional[List[str]] = None,
    ) -> List[ModelInfo]:
        """Find models with specific features."""
        models = await self.list_models(active_only=active_only)

        # Apply filters
        if providers:
            models = [m for m in models if m.provider in providers]

        if vision is not None:
            models = [m for m in models if m.supports_vision == vision]

        if function_calling is not None:
            models = [
                m for m in models if m.supports_function_calling == function_calling
            ]

        if json_mode is not None:
            models = [m for m in models if m.supports_json_mode == json_mode]

        if parallel_tools is not None:
            models = [
                m for m in models if m.supports_parallel_tool_calls == parallel_tools
            ]

        if streaming is not None:
            models = [m for m in models if m.supports_streaming == streaming]

        return models

    async def find_models_by_cost_range(
        self,
        max_input_cost_per_million: Optional[float] = None,
        max_output_cost_per_million: Optional[float] = None,
        min_input_cost_per_million: Optional[float] = None,
        min_output_cost_per_million: Optional[float] = None,
        active_only: bool = True,
    ) -> List[ModelInfo]:
        """Find models within a specific cost range."""
        models = await self.list_models(active_only=active_only)

        result = []
        for model in models:
            if not model.has_pricing:
                continue

            input_cost = float(model.cost_per_million_input_tokens)
            output_cost = float(model.cost_per_million_output_tokens)

            if (
                max_input_cost_per_million is not None
                and input_cost > max_input_cost_per_million
            ):
                continue
            if (
                max_output_cost_per_million is not None
                and output_cost > max_output_cost_per_million
            ):
                continue
            if (
                min_input_cost_per_million is not None
                and input_cost < min_input_cost_per_million
            ):
                continue
            if (
                min_output_cost_per_million is not None
                and output_cost < min_output_cost_per_million
            ):
                continue

            result.append(model)

        return result

    async def find_models_by_context_size(
        self,
        min_context: Optional[int] = None,
        max_context: Optional[int] = None,
        min_output: Optional[int] = None,
        active_only: bool = True,
    ) -> List[ModelInfo]:
        """Find models by context window size."""
        models = await self.list_models(active_only=active_only)

        result = []
        for model in models:
            if min_context and (
                not model.max_context_tokens or model.max_context_tokens < min_context
            ):
                continue
            if max_context and (
                model.max_context_tokens and model.max_context_tokens > max_context
            ):
                continue
            if min_output and (
                not model.max_output_tokens or model.max_output_tokens < min_output
            ):
                continue

            result.append(model)

        return result

    async def search_models(
        self,
        query: str,
        search_fields: Optional[List[str]] = None,
        active_only: bool = True,
        fuzzy: bool = True,
    ) -> List[ModelInfo]:
        """Search for models by text query."""
        if search_fields is None:
            search_fields = ["model_name", "display_name", "description"]

        models = await self.list_models(active_only=active_only)
        query_lower = query.lower()

        result = []
        for model in models:
            # Check each search field
            for field in search_fields:
                value = getattr(model, field, None)
                if value and query_lower in str(value).lower():
                    result.append(model)
                    break
            else:
                # Fuzzy matching for model names
                if fuzzy and "model_name" in search_fields:
                    if any(
                        part in model.model_name.lower() for part in query_lower.split()
                    ):
                        result.append(model)

        return result

    async def find_compatible_models(
        self, requirements: ModelRequirements
    ) -> List[ModelInfo]:
        """Find all models meeting specific requirements."""
        models = await self.list_models(active_only=requirements.active_only)

        # Apply provider filter
        if requirements.providers:
            models = [m for m in models if m.provider in requirements.providers]

        result = []
        for model in models:
            # Check context size
            if requirements.min_context_size:
                if (
                    not model.max_context_tokens
                    or model.max_context_tokens < requirements.min_context_size
                ):
                    continue

            # Check costs
            if model.has_pricing:
                input_cost = float(model.cost_per_million_input_tokens)
                output_cost = float(model.cost_per_million_output_tokens)

                if (
                    requirements.max_input_cost_per_million
                    and input_cost > requirements.max_input_cost_per_million
                ):
                    continue
                if (
                    requirements.max_output_cost_per_million
                    and output_cost > requirements.max_output_cost_per_million
                ):
                    continue

            # Check features
            if requirements.requires_vision and not model.supports_vision:
                continue
            if (
                requirements.requires_function_calling
                and not model.supports_function_calling
            ):
                continue
            if requirements.requires_json_mode and not model.supports_json_mode:
                continue
            if (
                requirements.requires_parallel_tools
                and not model.supports_parallel_tool_calls
            ):
                continue

            result.append(model)

        return result

    # Model Management

    async def activate_model(self, provider: str, model_name: str) -> bool:
        """Activate a specific model."""
        try:
            query = self.db.db._prepare_query(
                """
                UPDATE {{tables.llm_models}}
                SET inactive_from = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE provider = $1 AND model_name = $2 AND inactive_from IS NOT NULL
                """
            )
            await self.db.db.execute(query, provider, model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to activate model {provider}:{model_name}: {e}")
            return False

    async def deactivate_model(
        self, provider: str, model_name: str, reason: Optional[str] = None
    ) -> bool:
        """Deactivate a specific model."""
        try:
            query = self.db.db._prepare_query(
                """
                UPDATE {{tables.llm_models}}
                SET inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE provider = $1 AND model_name = $2 AND inactive_from IS NULL
                """
            )
            await self.db.db.execute(query, provider, model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate model {provider}:{model_name}: {e}")
            return False

    async def bulk_update_model_status(
        self, updates: List[Tuple[str, str, bool]], reason: Optional[str] = None
    ) -> Dict[Tuple[str, str], bool]:
        """Update multiple model statuses in one operation."""
        results = {}

        for provider, model_name, active in updates:
            if active:
                success = await self.activate_model(provider, model_name)
            else:
                success = await self.deactivate_model(provider, model_name, reason)
            results[(provider, model_name)] = success

        return results

    async def activate_all_models(self, provider: Optional[str] = None) -> int:
        """Activate all models for a provider or all providers."""
        try:
            if provider:
                query = self.db.db._prepare_query(
                    """
                    UPDATE {{tables.llm_models}}
                    SET inactive_from = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE provider = $1 AND inactive_from IS NOT NULL
                    """
                )
                result = await self.db.db.execute(query, provider)
            else:
                query = self.db.db._prepare_query(
                    """
                    UPDATE {{tables.llm_models}}
                    SET inactive_from = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE inactive_from IS NOT NULL
                    """
                )
                result = await self.db.db.execute(query)

            # Extract count from result
            count = int(result.split()[-1]) if result else 0
            return count
        except Exception as e:
            logger.error(f"Failed to activate models: {e}")
            return 0

    async def deactivate_models_without_pricing(self) -> int:
        """Deactivate all models that don't have pricing information."""
        try:
            query = self.db.db._prepare_query(
                """
                UPDATE {{tables.llm_models}}
                SET inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE (dollars_per_million_tokens_input IS NULL
                    OR dollars_per_million_tokens_output IS NULL)
                    AND inactive_from IS NULL
                """
            )
            result = await self.db.db.execute(query)

            count = int(result.split()[-1]) if result else 0
            return count
        except Exception as e:
            logger.error(f"Failed to deactivate models without pricing: {e}")
            return 0

    # Model Validation & Compatibility

    async def validate_model_request(
        self,
        provider: str,
        model_name: str,
        requirements: Optional[ModelRequirements] = None,
    ) -> ValidationResult:
        """Validate if a model exists and meets requirements.

        Comprehensive validation to check if a model request is valid,
        including existence, active status, and requirement matching.
        Provides detailed feedback on any issues found.

        Args:
            provider: The model provider
            model_name: The model name to validate
            requirements: Optional requirements the model must meet

        Returns:
            ValidationResult with detailed validation information including:
            - valid: Overall result (exists, active, meets requirements)
            - model_exists: Whether the model is in the registry
            - model_active: Whether the model is currently active
            - meets_requirements: Whether all requirements are satisfied
            - issues: List of specific problems found
            - suggestions: Alternative model suggestions if validation fails

        Example:
            requirements = ModelRequirements(
                min_context_size=100000,
                requires_vision=True,
                max_input_cost_per_million=10.0
            )
            result = await api.validate_model_request(
                "openai", "gpt-4-vision",
                requirements
            )
            if not result.valid:
                print("Issues:", result.issues)
                print("Try these instead:", result.suggestions)
        """
        model = await self.get_model(provider, model_name)

        if not model:
            return ValidationResult(
                valid=False,
                model_exists=False,
                model_active=False,
                meets_requirements=False,
                issues=[f"Model {provider}:{model_name} not found"],
                suggestions=await self._get_similar_model_names(provider, model_name),
            )

        issues = []

        # Check if active
        if not model.is_active:
            issues.append("Model is currently inactive")

        # Check requirements if provided
        meets_requirements = True
        if requirements:
            if requirements.min_context_size and (
                not model.max_context_tokens
                or model.max_context_tokens < requirements.min_context_size
            ):
                meets_requirements = False
                issues.append(
                    f"Context size {model.max_context_tokens} < required {requirements.min_context_size}"
                )

            if model.has_pricing:
                input_cost = float(model.cost_per_million_input_tokens)
                if (
                    requirements.max_input_cost_per_million
                    and input_cost > requirements.max_input_cost_per_million
                ):
                    meets_requirements = False
                    issues.append(
                        f"Input cost ${input_cost}/M > max ${requirements.max_input_cost_per_million}/M"
                    )

            if requirements.requires_vision and not model.supports_vision:
                meets_requirements = False
                issues.append("Model does not support vision")

            if (
                requirements.requires_function_calling
                and not model.supports_function_calling
            ):
                meets_requirements = False
                issues.append("Model does not support function calling")

        return ValidationResult(
            valid=model.is_active and meets_requirements,
            model_exists=True,
            model_active=model.is_active,
            meets_requirements=meets_requirements,
            issues=issues,
            suggestions=[],
        )

    async def _get_similar_model_names(
        self, provider: str, model_name: str
    ) -> List[str]:
        """Get similar model names for suggestions."""
        models = await self.list_models(provider=provider)
        suggestions = []

        model_lower = model_name.lower()
        for model in models:
            if (
                model_lower in model.model_name.lower()
                or model.model_name.lower() in model_lower
            ):
                suggestions.append(f"Did you mean: {provider}:{model.model_name}?")

        return suggestions[:3]

    async def suggest_alternative_models(
        self, provider: str, model_name: str, max_suggestions: int = 5
    ) -> List[ModelInfo]:
        """Suggest alternative models similar to a given model."""
        # Get the original model to understand its characteristics
        original = await self.get_model(provider, model_name)
        if not original:
            # If model doesn't exist, return some popular models
            models = await self.list_models(active_only=True)
            return models[:max_suggestions]

        # Find similar models
        all_models = await self.list_models(active_only=True)

        # Score models by similarity
        scored_models = []
        for model in all_models:
            if model.provider == provider and model.model_name == model_name:
                continue  # Skip the original

            score = 0

            # Same provider bonus
            if model.provider == provider:
                score += 10

            # Similar capabilities
            if model.supports_vision == original.supports_vision:
                score += 5
            if model.supports_function_calling == original.supports_function_calling:
                score += 5
            if model.supports_json_mode == original.supports_json_mode:
                score += 3

            # Similar context size
            if model.max_context_tokens and original.max_context_tokens:
                ratio = model.max_context_tokens / original.max_context_tokens
                if 0.5 <= ratio <= 2.0:
                    score += 5

            # Similar pricing
            if model.has_pricing and original.has_pricing:
                input_ratio = float(model.cost_per_million_input_tokens) / float(
                    original.cost_per_million_input_tokens
                )
                if 0.5 <= input_ratio <= 2.0:
                    score += 5

            scored_models.append((score, model))

        # Sort by score and return top suggestions
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return [model for score, model in scored_models[:max_suggestions]]

    async def get_model_recommendations(
        self,
        use_case: str,
        budget_per_million_tokens: Optional[float] = None,
        required_features: Optional[List[str]] = None,
        max_recommendations: int = 5,
    ) -> List[Tuple[ModelInfo, float]]:
        """Get model recommendations for a specific use case."""
        if required_features is None:
            required_features = []

        # Map use cases to requirements
        use_case_features = {
            "code_generation": ["function_calling"],
            "chat": ["function_calling"],
            "vision": ["vision"],
            "data_analysis": ["function_calling", "json"],
            "summarization": [],
            "translation": [],
        }

        # Get features for use case
        features = use_case_features.get(use_case.lower(), [])
        features.extend(required_features)

        # Build requirements
        requirements = ModelRequirements(
            requires_vision="vision" in features,
            requires_function_calling="function_calling" in features,
            requires_json_mode="json" in features,
            max_input_cost_per_million=budget_per_million_tokens,
            max_output_cost_per_million=(
                budget_per_million_tokens * 3 if budget_per_million_tokens else None
            ),
            active_only=True,
        )

        # Get compatible models
        models = await self.find_compatible_models(requirements)

        # Score models
        scored = []
        for model in models:
            score = 100.0

            # Prefer models with good context size for code
            if "code" in use_case.lower() and model.max_context_tokens:
                if model.max_context_tokens >= 100000:
                    score += 20
                elif model.max_context_tokens >= 50000:
                    score += 10

            # Prefer cheaper models if budget conscious
            if budget_per_million_tokens and model.has_pricing:
                cost = float(model.cost_per_million_input_tokens)
                if cost < budget_per_million_tokens * 0.5:
                    score += 15

            # Provider reputation
            provider_scores = {"openai": 10, "anthropic": 10, "google": 8, "ollama": 5}
            score += provider_scores.get(model.provider, 0)

            scored.append((model, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:max_recommendations]

    # Service Health & Monitoring

    async def get_service_health(self) -> ServiceHealth:
        """Get comprehensive health status of the LLM service."""
        issues = []

        # Check database connection
        try:
            db_connected = await self.check_database_connection()
        except:
            db_connected = False
            issues.append("Database connection failed")

        # Count models
        try:
            models = await self.db.list_models(active_only=False)
            models_loaded = len(models)
        except:
            models_loaded = 0
            issues.append("Failed to load models")

        # Determine status
        if not db_connected:
            status = "unhealthy"
        elif models_loaded == 0:
            status = "degraded"
            issues.append("No models loaded")
        else:
            status = "healthy"

        return ServiceHealth(
            status=status,
            database_connected=db_connected,
            models_loaded=models_loaded,
            last_refresh=None,  # Would need to track this
            version="1.0.0",
            issues=issues,
        )

    async def check_database_connection(self) -> bool:
        """Check if database connection is active."""
        try:
            await self.db.health_check()
            return True
        except:
            return False

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        # This would track real metrics in production
        return {
            "database_pool_size": 10,
            "database_pool_available": 8,
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "version": "1.0.0",
        }

    async def verify_model_data_integrity(self) -> Dict[str, Any]:
        """Verify integrity of model data."""
        issues = []
        models = await self.db.list_models(active_only=False)

        # Check for common issues
        for model in models:
            # Models with pricing should have both input and output
            if (
                model.dollars_per_million_tokens_input
                and not model.dollars_per_million_tokens_output
            ):
                issues.append(
                    f"{model.provider}:{model.model_name} has input pricing but no output pricing"
                )
            elif (
                model.dollars_per_million_tokens_output
                and not model.dollars_per_million_tokens_input
            ):
                issues.append(
                    f"{model.provider}:{model.model_name} has output pricing but no input pricing"
                )

            # Vision models should have reasonable context
            if model.supports_vision and model.max_context and model.max_context < 4096:
                issues.append(
                    f"{model.provider}:{model.model_name} supports vision but has small context"
                )

        return {
            "total_models": len(models),
            "issues": issues,
            "issue_count": len(issues),
            "status": "ok" if not issues else "issues_found",
        }

    # Utility Methods

    async def get_model_families(
        self, provider: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get model families/series for providers."""
        models = await self.list_models(provider=provider, active_only=False)

        families = defaultdict(list)

        # Group by common prefixes
        for model in models:
            # Extract family name (simplified logic)
            if "-" in model.model_name:
                family = model.model_name.split("-")[0]
            else:
                family = model.model_name

            families[family].append(model.model_name)

        return dict(families)

    async def normalize_model_name(
        self, provider: str, model_name: str
    ) -> Optional[str]:
        """Normalize a model name to its canonical form."""
        # Check if it exists as-is
        model = await self.get_model(provider, model_name)
        if model:
            return model.model_name

        # Try common aliases
        aliases = {
            ("anthropic", "claude-3-opus"): "claude-3-opus-20240229",
            ("anthropic", "claude-3-sonnet"): "claude-3-sonnet-20240229",
            ("anthropic", "claude-3-haiku"): "claude-3-haiku-20240307",
            ("anthropic", "claude-3.5-sonnet"): "claude-3-5-sonnet-20241022",
            ("openai", "gpt-4"): "gpt-4",
            ("openai", "gpt-4-turbo"): "gpt-4-turbo",
            ("openai", "gpt-3.5"): "gpt-3.5-turbo",
        }

        canonical = aliases.get((provider, model_name))
        if canonical:
            return canonical

        # Try partial match
        models = await self.list_models(provider=provider)
        for model in models:
            if model_name in model.model_name:
                return model.model_name

        return None

    async def estimate_tokens(self, text: str, provider: str, model_name: str) -> int:
        """Estimate token count for text using provider-specific tokenizer."""
        # Simplified estimation - in production would use actual tokenizers
        # OpenAI: ~1 token per 4 chars
        # Anthropic: ~1 token per 3.5 chars
        # Others: ~1 token per 4 chars

        char_per_token = {"openai": 4, "anthropic": 3.5, "google": 4, "ollama": 4}

        chars = char_per_token.get(provider, 4)
        return max(1, int(len(text) / chars))
