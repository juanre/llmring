"""
Mixin classes for provider implementations to reduce code duplication.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from llmring.registry import RegistryClient, RegistryModel

logger = logging.getLogger(__name__)


class RegistryModelSelectorMixin:
    """Mixin for providers that need to select default models from registry."""

    def __init__(self):
        """Initialize the mixin - should be called by provider __init__."""
        # Don't initialize _registry_client here - let providers set it
        # before calling this __init__
        if not hasattr(self, "_registry_client"):
            self._registry_client: Optional[RegistryClient] = None
        self.default_model: Optional[str] = None

    async def select_default_from_registry(
        self,
        provider_name: str,
        available_models: List[str],
        cost_range: Tuple[float, float] = (0.1, 10.0),
        fallback_model: Optional[str] = None,
    ) -> str:
        """
        Select default model from registry using a consistent scoring policy.

        Args:
            provider_name: Name of the provider (e.g., "anthropic", "openai")
            available_models: List of available model names from registry
            cost_range: Tuple of (min_cost, max_cost) for balanced cost scoring
            fallback_model: Optional fallback model if selection fails

        Returns:
            Selected default model name
        """
        try:
            if not self._registry_client:
                raise ValueError("Registry client not initialized")

            registry_models = await self._registry_client.fetch_current_models(provider_name)
            active_models = [
                m for m in registry_models if m.is_active and m.model_name in available_models
            ]

            if active_models:
                scored_models = self._score_models(active_models, cost_range)

                if scored_models:
                    # Select highest scoring model
                    best_model = max(scored_models, key=lambda x: x[1])
                    logger.info(
                        f"{provider_name.capitalize()}: Selected default model '{best_model[0]}' "
                        f"with score {best_model[1]} from registry analysis"
                    )
                    return best_model[0]

        except Exception as e:
            logger.warning(f"Could not use registry metadata for {provider_name} selection: {e}")

        # Fallback to first available or specified fallback
        if available_models:
            selected = available_models[0]
            logger.info(f"{provider_name.capitalize()}: Using first available model: {selected}")
            return selected

        if fallback_model:
            logger.warning(
                f"No models available from registry for {provider_name}, "
                f"using fallback: {fallback_model}"
            )
            return fallback_model

        raise ValueError(f"No models available for {provider_name}")

    def _score_models(
        self, models: List[RegistryModel], cost_range: Tuple[float, float]
    ) -> List[Tuple[str, int]]:
        """
        Score models based on cost, capabilities, and recency.

        Args:
            models: List of registry models to score
            cost_range: Tuple of (min_cost, max_cost) for balanced cost scoring

        Returns:
            List of (model_name, score) tuples
        """
        scored_models = []
        min_cost, max_cost = cost_range

        for model in models:
            score = 0

            # Prefer models with moderate cost (not cheapest, not most expensive)
            input_cost = model.dollars_per_million_tokens_input or 0
            if min_cost <= input_cost <= max_cost:
                score += 10
            elif input_cost < min_cost:
                score += 5  # Still give some score to cheaper models

            # Prefer models with good capabilities
            if model.supports_function_calling:
                score += 5
            if model.supports_vision:
                score += 3
            if hasattr(model, "supports_audio") and model.supports_audio:
                score += 2

            # Prefer more recent models (if added_date is available)
            if model.added_date:
                days_since_added = (datetime.now(timezone.utc) - model.added_date).days
                if days_since_added < 90:  # Less than 3 months old
                    score += 8
                elif days_since_added < 180:  # Less than 6 months old
                    score += 5
                elif days_since_added < 365:  # Less than 1 year old
                    score += 2

            scored_models.append((model.model_name, score))

        return scored_models


class ProviderLoggingMixin:
    """Mixin to standardize logging across providers."""

    def __init__(self, provider_name: str):
        """Initialize the logging mixin."""
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"llmring.providers.{provider_name}")

    def log_info(self, message: str):
        """Log info message with provider context."""
        self.logger.info(f"[{self.provider_name}] {message}")

    def log_warning(self, message: str):
        """Log warning message with provider context."""
        self.logger.warning(f"[{self.provider_name}] {message}")

    def log_error(self, message: str, exc_info: bool = False):
        """Log error message with provider context."""
        self.logger.error(f"[{self.provider_name}] {message}", exc_info=exc_info)

    def log_debug(self, message: str):
        """Log debug message with provider context."""
        self.logger.debug(f"[{self.provider_name}] {message}")
