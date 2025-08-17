"""Compare discovered models with existing database models."""

import logging
from decimal import Decimal
from typing import List

from .models import ModelDiff, ModelInfo

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compares discovered models with existing database models."""

    def __init__(self, price_change_threshold: float = 0.05):
        """
        Initialize comparator.

        Args:
            price_change_threshold: Threshold for automatic price updates (5% default)
        """
        self.price_change_threshold = price_change_threshold

    def compare_models(
        self, discovered: List[ModelInfo], existing: List[ModelInfo]
    ) -> ModelDiff:
        """
        Compare discovered models with existing models.

        Args:
            discovered: Models discovered from providers
            existing: Models currently in database

        Returns:
            ModelDiff with changes to apply
        """
        logger.info(
            f"Comparing {len(discovered)} discovered vs {len(existing)} existing models"
        )

        # Create lookup dictionaries
        discovered_map = {(m.provider, m.model_name): m for m in discovered}
        existing_map = {(m.provider, m.model_name): m for m in existing}

        discovered_keys = set(discovered_map.keys())
        existing_keys = set(existing_map.keys())

        # Find new models
        new_keys = discovered_keys - existing_keys
        new_models = [discovered_map[key] for key in new_keys]

        # Find potentially retired models (existing but not discovered)
        # Only consider active models for retirement
        retired_keys = existing_keys - discovered_keys
        retired_models = [
            existing_map[key] for key in retired_keys if existing_map[key].is_active
        ]

        # Find models that exist in both (potential updates)
        common_keys = discovered_keys & existing_keys
        updated_models = []
        unchanged_models = []

        for key in common_keys:
            discovered_model = discovered_map[key]
            existing_model = existing_map[key]

            if self._models_need_update(existing_model, discovered_model):
                updated_models.append((existing_model, discovered_model))
            else:
                unchanged_models.append(existing_model)

        diff = ModelDiff(
            new_models=new_models,
            updated_models=updated_models,
            retired_models=retired_models,
            unchanged_models=unchanged_models,
        )

        logger.info(f"Model comparison complete: {diff.summary}")
        return diff

    def _models_need_update(self, existing: ModelInfo, discovered: ModelInfo) -> bool:
        """
        Determine if a model needs to be updated.

        Args:
            existing: Current model in database
            discovered: Newly discovered model

        Returns:
            True if model should be updated
        """
        # Check if basic info changed
        if (
            existing.display_name != discovered.display_name
            or existing.description != discovered.description
            or existing.max_context != discovered.max_context
            or existing.max_output_tokens != discovered.max_output_tokens
        ):
            return True

        # Check if capabilities changed
        if (
            existing.supports_vision != discovered.supports_vision
            or existing.supports_function_calling
            != discovered.supports_function_calling
            or existing.supports_json_mode != discovered.supports_json_mode
            or existing.supports_parallel_tool_calls
            != discovered.supports_parallel_tool_calls
            or existing.tool_call_format != discovered.tool_call_format
        ):
            return True

        # Check pricing changes
        if self._pricing_changed(existing, discovered):
            return True

        # Check if model was inactive but is now available
        if not existing.is_active and discovered.is_active:
            return True

        return False

    def _pricing_changed(self, existing: ModelInfo, discovered: ModelInfo) -> bool:
        """
        Check if pricing has changed significantly.

        Args:
            existing: Current model in database
            discovered: Newly discovered model

        Returns:
            True if pricing changed beyond threshold
        """
        # If one has pricing and the other doesn't, it's a change
        if (existing.dollars_per_million_tokens_input is None) != (
            discovered.dollars_per_million_tokens_input is None
        ):
            return True
        if (existing.dollars_per_million_tokens_output is None) != (
            discovered.dollars_per_million_tokens_output is None
        ):
            return True

        # If both have no pricing, no change
        if (
            existing.dollars_per_million_tokens_input is None
            and discovered.dollars_per_million_tokens_input is None
            and existing.dollars_per_million_tokens_output is None
            and discovered.dollars_per_million_tokens_output is None
        ):
            return False

        # Calculate percentage changes
        input_changed = self._price_change_exceeds_threshold(
            existing.dollars_per_million_tokens_input,
            discovered.dollars_per_million_tokens_input,
        )

        output_changed = self._price_change_exceeds_threshold(
            existing.dollars_per_million_tokens_output,
            discovered.dollars_per_million_tokens_output,
        )

        return input_changed or output_changed

    def _price_change_exceeds_threshold(
        self, old_price: Decimal, new_price: Decimal
    ) -> bool:
        """
        Check if price change exceeds threshold.

        Args:
            old_price: Current price
            new_price: New price

        Returns:
            True if change exceeds threshold
        """
        if old_price is None or new_price is None:
            return old_price != new_price

        if old_price == 0:
            return new_price != 0

        # Convert to float for calculation
        old_price_float = float(old_price)
        new_price_float = float(new_price)
        change_ratio = abs((new_price_float - old_price_float) / old_price_float)
        return change_ratio > self.price_change_threshold

    def generate_detailed_report(self, diff: ModelDiff) -> str:
        """
        Generate a detailed human-readable report of changes.

        Args:
            diff: Model differences

        Returns:
            Detailed report string
        """
        report = []
        report.append("=== Model Refresh Report ===")
        report.append(f"Summary: {diff.summary}")
        report.append("")

        if diff.new_models:
            report.append(f"NEW MODELS ({len(diff.new_models)}):")
            for model in diff.new_models:
                price_info = ""
                if (
                    model.dollars_per_million_tokens_input
                    and model.dollars_per_million_tokens_output
                ):
                    price_info = f" (${model.dollars_per_million_tokens_input:.2f}/${model.dollars_per_million_tokens_output:.2f} per 1M tokens)"
                report.append(f"  + {model.provider}:{model.model_name}{price_info}")
            report.append("")

        if diff.updated_models:
            report.append(f"UPDATED MODELS ({len(diff.updated_models)}):")
            for old_model, new_model in diff.updated_models:
                changes = self._describe_changes(old_model, new_model)
                report.append(f"  ~ {new_model.provider}:{new_model.model_name}")
                for change in changes:
                    report.append(f"    - {change}")
            report.append("")

        if diff.retired_models:
            report.append(f"RETIRED MODELS ({len(diff.retired_models)}):")
            for model in diff.retired_models:
                report.append(f"  - {model.provider}:{model.model_name}")
            report.append("")

        if diff.unchanged_models:
            report.append(f"UNCHANGED MODELS: {len(diff.unchanged_models)}")

        return "\n".join(report)

    def _describe_changes(
        self, old_model: ModelInfo, new_model: ModelInfo
    ) -> List[str]:
        """Describe specific changes between models."""
        changes = []

        if old_model.display_name != new_model.display_name:
            changes.append(
                f"Display name: '{old_model.display_name}' → '{new_model.display_name}'"
            )

        if old_model.max_context != new_model.max_context:
            changes.append(
                f"Context: {old_model.max_context} → {new_model.max_context}"
            )

        if old_model.max_output_tokens != new_model.max_output_tokens:
            changes.append(
                f"Max output: {old_model.max_output_tokens} → {new_model.max_output_tokens}"
            )

        # Pricing changes
        if self._pricing_changed(old_model, new_model):
            old_input = (
                f"${old_model.dollars_per_million_tokens_input:.2f}/1M"
                if old_model.dollars_per_million_tokens_input
                else "None"
            )
            new_input = (
                f"${new_model.dollars_per_million_tokens_input:.2f}/1M"
                if new_model.dollars_per_million_tokens_input
                else "None"
            )
            old_output = (
                f"${old_model.dollars_per_million_tokens_output:.2f}/1M"
                if old_model.dollars_per_million_tokens_output
                else "None"
            )
            new_output = (
                f"${new_model.dollars_per_million_tokens_output:.2f}/1M"
                if new_model.dollars_per_million_tokens_output
                else "None"
            )

            changes.append(f"Input cost: {old_input} → {new_input}")
            changes.append(f"Output cost: {old_output} → {new_output}")

        # Capability changes
        capabilities = [
            ("Vision", "supports_vision"),
            ("Function calling", "supports_function_calling"),
            ("JSON mode", "supports_json_mode"),
            ("Parallel tools", "supports_parallel_tool_calls"),
        ]

        for name, attr in capabilities:
            old_val = getattr(old_model, attr)
            new_val = getattr(new_model, attr)
            if old_val != new_val:
                changes.append(f"{name}: {old_val} → {new_val}")

        return changes
