"""Model curator to select the best models from each provider."""

import logging
from dataclasses import dataclass
from typing import List

from .pdf_parser import ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class CurationCriteria:
    """Criteria for model selection."""

    max_models_per_provider: int = 3
    require_vision: bool = False
    require_function_calling: bool = False
    exclude_deprecated: bool = True
    prefer_newer: bool = True
    cost_weight: float = 0.3
    capability_weight: float = 0.4
    recency_weight: float = 0.3


class ModelCurator:
    """Curate and select the best models from each provider."""

    def __init__(self, criteria: CurationCriteria = None):
        """Initialize curator with selection criteria."""
        self.criteria = criteria or CurationCriteria()

    def select_best_models(
        self, models: List[ModelInfo], provider: str
    ) -> List[ModelInfo]:
        """
        Select the best models based on criteria.

        Args:
            models: List of all available models
            provider: Provider name for specific rules

        Returns:
            List of selected models (up to max_models_per_provider)
        """
        logger.info(f"Selecting best models from {len(models)} {provider} models")

        # Filter out deprecated models if requested
        if self.criteria.exclude_deprecated:
            models = [m for m in models if m.deprecation_date is None]
            logger.info(f"After filtering deprecated: {len(models)} models")

        # Apply provider-specific rules
        models = self._apply_provider_rules(models, provider)

        # Score and rank models
        scored_models = []
        for model in models:
            score = self._calculate_model_score(model)
            scored_models.append((score, model))

        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[0], reverse=True)

        # Select top models
        selected = [
            model for _, model in scored_models[: self.criteria.max_models_per_provider]
        ]

        # Ensure diversity in selection
        selected = self._ensure_diversity(selected, scored_models)

        logger.info(f"Selected {len(selected)} models for {provider}")
        for model in selected:
            logger.info(f"  - {model.model_id}: {model.display_name}")

        return selected

    def _apply_provider_rules(
        self, models: List[ModelInfo], provider: str
    ) -> List[ModelInfo]:
        """Apply provider-specific filtering rules."""
        filtered = models.copy()

        if provider == "openai":
            # Exclude o1 models if o3 is available
            o3_models = [m for m in filtered if "o3" in m.model_id.lower()]
            if o3_models:
                filtered = [m for m in filtered if "o1" not in m.model_id.lower()]
                logger.info("Excluded o1 models since o3 is available")

            # Exclude older GPT-3.5 variants if newer exists
            gpt35_models = [m for m in filtered if "gpt-3.5" in m.model_id.lower()]
            if len(gpt35_models) > 1:
                # Keep only the most recent
                gpt35_models.sort(key=lambda x: x.release_date or "", reverse=True)
                keep_model = gpt35_models[0]
                filtered = [
                    m
                    for m in filtered
                    if "gpt-3.5" not in m.model_id.lower() or m == keep_model
                ]

        elif provider == "anthropic":
            # Prefer Claude 3.5 and 4 models over older versions
            has_newer = any(
                "claude-3-5" in m.model_id
                or "claude-4" in m.model_id
                or "opus-4" in m.model_id
                for m in filtered
            )
            if has_newer:
                # Keep one Claude 3 model for budget option
                claude3_models = [
                    m
                    for m in filtered
                    if "claude-3-" in m.model_id and "claude-3-5" not in m.model_id
                ]
                if claude3_models:
                    # Keep cheapest Claude 3
                    claude3_models.sort(
                        key=lambda x: x.dollars_per_million_tokens_input
                    )
                    cheapest_claude3 = claude3_models[0]
                    filtered = [
                        m
                        for m in filtered
                        if "claude-3-" not in m.model_id
                        or "claude-3-5" in m.model_id
                        or m == cheapest_claude3
                    ]

        elif provider == "google":
            # Prefer Gemini 2.0 models over 1.5 if available
            has_v2 = any("gemini-2" in m.model_id for m in filtered)
            if has_v2:
                # Keep one 1.5 model for long context
                v15_models = [m for m in filtered if "gemini-1.5" in m.model_id]
                if v15_models:
                    # Keep the one with longest context
                    v15_models.sort(key=lambda x: x.max_context, reverse=True)
                    best_v15 = v15_models[0]
                    filtered = [
                        m
                        for m in filtered
                        if "gemini-1.5" not in m.model_id or m == best_v15
                    ]

        return filtered

    def _calculate_model_score(self, model: ModelInfo) -> float:
        """Calculate a score for model selection."""
        score = 0.0

        # Cost score (lower is better, normalized)
        # Assume max reasonable cost is $100 per million tokens
        cost_avg = (
            model.dollars_per_million_tokens_input
            + model.dollars_per_million_tokens_output
        ) / 2
        cost_score = max(0, 1 - (cost_avg / 100))
        score += cost_score * self.criteria.cost_weight

        # Capability score
        capability_score = 0.0
        capability_count = 0

        if model.supports_vision:
            capability_score += 1
            capability_count += 1
        if model.supports_function_calling:
            capability_score += 1
            capability_count += 1
        if model.supports_json_mode:
            capability_score += 0.5
            capability_count += 1
        if model.supports_parallel_tool_calls:
            capability_score += 0.5
            capability_count += 1

        # Long context bonus
        if model.max_context >= 100000:
            capability_score += 1
            capability_count += 1
        elif model.max_context >= 32000:
            capability_score += 0.5
            capability_count += 1

        if capability_count > 0:
            capability_score = capability_score / capability_count

        score += capability_score * self.criteria.capability_weight

        # Recency score (if release date available)
        if model.release_date and self.criteria.prefer_newer:
            try:
                from datetime import datetime

                release = datetime.fromisoformat(model.release_date)
                days_old = (datetime.now() - release).days
                # Normalize: 0 days = 1.0, 365 days = 0.5, 730+ days = 0.0
                recency_score = max(0, 1 - (days_old / 730))
                score += recency_score * self.criteria.recency_weight
            except:
                pass

        return score

    def _ensure_diversity(
        self, selected: List[ModelInfo], all_scored: List[tuple]
    ) -> List[ModelInfo]:
        """Ensure diversity in model selection."""
        # Check if we have models with different characteristics
        has_vision = any(m.supports_vision for m in selected)
        has_cheap = any(m.dollars_per_million_tokens_input < 1.0 for m in selected)
        has_expensive = any(m.dollars_per_million_tokens_input > 10.0 for m in selected)

        # If we're missing diversity, try to add it
        if len(selected) < self.criteria.max_models_per_provider:
            for score, model in all_scored[len(selected) :]:
                if not has_vision and model.supports_vision:
                    selected.append(model)
                    has_vision = True
                elif not has_cheap and model.dollars_per_million_tokens_input < 1.0:
                    selected.append(model)
                    has_cheap = True
                elif (
                    not has_expensive and model.dollars_per_million_tokens_input > 10.0
                ):
                    selected.append(model)
                    has_expensive = True

                if len(selected) >= self.criteria.max_models_per_provider:
                    break

        return selected[: self.criteria.max_models_per_provider]
