"""Complete model refresh manager that integrates discovery and pricing."""

import asyncio
import logging
from typing import Any, Dict, List

from llmring.base import LLMProviderFactory
from llmring.config import ModelRefreshConfig
from llmring.model_refresh.model_filter import ModelCategory, ModelFilter
from llmring.model_refresh.models import ModelInfo, RefreshResult
from llmring.model_refresh.refresh_manager import ModelRefreshManager
from llmring.pricing.anthropic_pricing import AnthropicPricingScraper
from llmring.pricing.google_pricing import GooglePricingScraper
from llmring.pricing.openai_pricing import OpenAIPricingScraper

logger = logging.getLogger(__name__)


class CompleteModelRefreshManager:
    """Manages complete model refresh including discovery and pricing."""

    def __init__(self, config: ModelRefreshConfig):
        """
        Initialize complete refresh manager.

        Args:
            config: Model refresh configuration
        """
        self.config = config
        self.refresh_manager = ModelRefreshManager(
            config.get_database_connection_params(), config.backup_directory
        )

        # Initialize pricing scrapers
        self.pricing_scrapers = {
            "openai": OpenAIPricingScraper(config.openai_api_key),
            "anthropic": AnthropicPricingScraper(
                config.openai_api_key
            ),  # Uses OpenAI for LLM processing
            "google": GooglePricingScraper(config.openai_api_key),
        }

        # Initialize model filter
        self.model_filter = ModelFilter(config.openai_api_key)

    async def discover_all_models(self) -> List[ModelInfo]:
        """
        Discover models from all providers.

        Returns:
            List of discovered models
        """
        all_models = []

        for provider_name in ["openai", "anthropic", "google", "ollama"]:
            try:
                logger.info(f"Discovering models from {provider_name}")

                # Get provider credentials
                credentials = self.config.get_provider_credentials()[provider_name]

                # Create provider instance
                provider = LLMProviderFactory.create_provider(
                    provider_name, **credentials
                )

                # Discover models
                models = await provider.discover_models()
                logger.info(f"Discovered {len(models)} models from {provider_name}")

                all_models.extend(models)

            except Exception as e:
                logger.error(
                    f"Failed to discover models from {provider_name}: {str(e)}"
                )
                continue

        logger.info(f"Total models discovered: {len(all_models)}")
        return all_models

    async def update_pricing_information(
        self, models: List[ModelInfo]
    ) -> List[ModelInfo]:
        """
        Update pricing information for discovered models.

        Args:
            models: List of models to update pricing for

        Returns:
            List of models with updated pricing
        """
        if not self.config.enable_price_scraping:
            logger.info("Price scraping disabled, skipping pricing updates")
            return models

        logger.info("Updating pricing information for discovered models")

        # Get pricing data for each provider
        pricing_data = {}

        for provider_name, scraper in self.pricing_scrapers.items():
            try:
                logger.info(f"Scraping pricing for {provider_name}")
                result = await scraper.get_pricing_with_cache()

                if result.success:
                    # Create pricing lookup
                    provider_pricing = {
                        model.model_name: model for model in result.models
                    }
                    pricing_data[provider_name] = provider_pricing
                    logger.info(
                        f"Got pricing for {len(result.models)} {provider_name} models"
                    )
                else:
                    logger.warning(
                        f"Pricing scraping failed for {provider_name}: {result.error}"
                    )
                    # Use fallback pricing
                    fallback_models = scraper.get_fallback_pricing()
                    provider_pricing = {
                        model.model_name: model for model in fallback_models
                    }
                    pricing_data[provider_name] = provider_pricing
                    logger.info(
                        f"Using fallback pricing for {len(fallback_models)} {provider_name} models"
                    )

            except Exception as e:
                logger.error(f"Failed to get pricing for {provider_name}: {str(e)}")
                continue

        # Update models with pricing information using smart matching
        updated_models = []
        models_without_pricing = []

        for model in models:
            # Look for pricing data using smart matching
            if model.provider in pricing_data:
                logger.debug(
                    f"Checking pricing for {model.provider}:{model.model_name}"
                )
                pricing_info = self._find_matching_pricing(
                    model, pricing_data[model.provider]
                )
                if pricing_info:
                    # Create a new model with updated pricing
                    from dataclasses import replace

                    updated_model = replace(
                        model,
                        cost_per_token_input=pricing_info.input_cost_per_token,
                        cost_per_token_output=pricing_info.output_cost_per_token,
                    )
                    logger.info(
                        f"Applied pricing to {model.provider}:{model.model_name}: ${float(pricing_info.input_cost_per_token)*1000000:.2f}/${float(pricing_info.output_cost_per_token)*1000000:.2f}"
                    )
                    updated_models.append(updated_model)
                else:
                    logger.debug(
                        f"No pricing match found for {model.provider}:{model.model_name}"
                    )
                    if (
                        model.provider != "ollama"
                    ):  # Ollama is free/local, others should have pricing
                        models_without_pricing.append(
                            f"{model.provider}:{model.model_name}"
                        )
                    updated_models.append(model)
            else:
                logger.debug(f"No pricing data available for provider {model.provider}")
                if (
                    model.provider != "ollama"
                ):  # Ollama is free/local, others should have pricing
                    models_without_pricing.append(
                        f"{model.provider}:{model.model_name}"
                    )
                updated_models.append(model)

        # Check for models without pricing (safety check)
        if models_without_pricing:
            logger.warning(
                f"Found {len(models_without_pricing)} non-Ollama models without pricing:"
            )
            for model_id in models_without_pricing[:10]:  # Show first 10
                logger.warning(f"  - {model_id}")
            if len(models_without_pricing) > 10:
                logger.warning(f"  ... and {len(models_without_pricing) - 10} more")

        # Note: Final pricing filtering will be done later in the complete refresh process

        # Count models with pricing
        models_with_pricing = sum(
            1 for m in updated_models if m.cost_per_token_input is not None
        )
        logger.info(
            f"Updated pricing for {models_with_pricing}/{len(updated_models)} models"
        )

        return updated_models

    async def perform_complete_refresh(
        self,
        dry_run: bool = False,
        discover_models: bool = True,
        update_pricing: bool = True,
        filter_models: bool = True,
    ) -> RefreshResult:
        """
        Perform a complete model refresh including discovery and pricing.

        Args:
            dry_run: If True, preview changes without applying
            discover_models: If True, discover models from provider APIs
            update_pricing: If True, update pricing information
            filter_models: If True, filter to production-ready models only

        Returns:
            RefreshResult with operation details
        """
        logger.info("Starting complete model refresh")

        try:
            # Discover models from providers
            if discover_models:
                discovered_models = await self.discover_all_models()
            else:
                logger.info("Skipping model discovery")
                discovered_models = []

            # Filter models to production-ready only
            if filter_models and discovered_models:
                logger.info(f"Filtering {len(discovered_models)} discovered models")
                filtered_models = await self.model_filter.filter_models(
                    discovered_models,
                    use_llm_analysis=True,
                    include_categories={ModelCategory.PRODUCTION},
                )

                # Generate filter summary
                filter_summary = self.model_filter.get_filter_summary(
                    discovered_models, filtered_models
                )
                logger.info(
                    f"Filtered models: {filter_summary['total_original']} → {filter_summary['total_filtered']} "
                    f"({filter_summary['reduction_percentage']:.1f}% reduction)"
                )

                discovered_models = filtered_models
            elif filter_models:
                logger.info("No models to filter")
            else:
                logger.info("Skipping model filtering")

            # Update pricing information
            if update_pricing and discovered_models:
                discovered_models = await self.update_pricing_information(
                    discovered_models
                )
            elif update_pricing:
                logger.info("No models discovered, skipping pricing updates")
            else:
                logger.info("Skipping pricing updates")

            # HARD RULE: NO non-Ollama models can be free
            if discovered_models:
                original_count = len(discovered_models)
                pricing_filtered_models = []
                violations = []

                for model in discovered_models:
                    if model.provider == "ollama":
                        # Ollama models are always free/local - this is allowed
                        pricing_filtered_models.append(model)
                    elif (
                        model.cost_per_token_input is not None
                        and model.cost_per_token_output is not None
                    ):
                        # Check for zero/free pricing (which should never happen for commercial APIs)
                        input_cost = float(model.cost_per_token_input)
                        output_cost = float(model.cost_per_token_output)

                        if input_cost <= 0 or output_cost <= 0:
                            violations.append(
                                f"{model.provider}:{model.model_name} (${input_cost*1000000:.2f}/${output_cost*1000000:.2f})"
                            )
                            logger.error(
                                f"VIOLATION: {model.provider}:{model.model_name} has zero/free pricing - this is not allowed!"
                            )
                        else:
                            # Valid pricing for commercial model
                            pricing_filtered_models.append(model)
                    else:
                        logger.info(
                            f"Excluded {model.provider}:{model.model_name} - no pricing data"
                        )

                # If there are any pricing violations, STOP immediately
                if violations:
                    error_msg = f"CRITICAL: Found {len(violations)} non-Ollama models with free/zero pricing: {violations[:5]}"
                    logger.error(error_msg)
                    return RefreshResult.error_result(error_msg, violations)

                discovered_models = pricing_filtered_models
                logger.info(
                    f"Final pricing filter: {original_count} → {len(discovered_models)} models (excluded {original_count - len(discovered_models)} without valid pricing)"
                )

            # Perform database refresh
            if discovered_models:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.refresh_manager.refresh_models,
                    discovered_models,
                    dry_run,
                    True,  # create_backup
                )
            else:
                result = RefreshResult.error_result(
                    "No models discovered to refresh",
                    ["Model discovery returned no results"],
                )

            if result.success:
                logger.info(f"Complete refresh successful: {result.message}")
            else:
                logger.error(f"Complete refresh failed: {result.message}")

            return result

        except Exception as e:
            error_msg = f"Complete model refresh failed: {str(e)}"
            logger.error(error_msg)
            return RefreshResult.error_result(error_msg, [str(e)])

    async def get_pricing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of pricing information availability.

        Returns:
            Dictionary with pricing summary
        """
        summary = {
            "pricing_enabled": self.config.enable_price_scraping,
            "providers": {},
        }

        if not self.config.enable_price_scraping:
            return summary

        for provider_name, scraper in self.pricing_scrapers.items():
            try:
                # Check cached pricing
                cache_info = scraper.get_cache_info()
                cache_valid = scraper._is_cache_valid(provider_name)

                summary["providers"][provider_name] = {
                    "scraper_available": True,
                    "cache_valid": cache_valid,
                    "cache_info": cache_info,
                }

            except Exception as e:
                summary["providers"][provider_name] = {
                    "scraper_available": False,
                    "error": str(e),
                }

        return summary

    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report.

        Returns:
            Dictionary with status information
        """
        base_status = self.refresh_manager.generate_status_report()

        # Add pricing information
        base_status.update(
            {
                "model_discovery_enabled": self.config.enable_model_refresh,
                "api_discovery_enabled": self.config.enable_api_discovery,
                "price_scraping_enabled": self.config.enable_price_scraping,
                "provider_credentials": {
                    provider: bool(creds.get("api_key") or creds.get("base_url"))
                    for provider, creds in self.config.get_provider_credentials().items()
                },
            }
        )

        return base_status

    def _find_matching_pricing(self, model, pricing_lookup):
        """Find pricing for a model using smart matching."""
        # Try exact match first
        if model.model_name in pricing_lookup:
            return pricing_lookup[model.model_name]

        # Try pattern matching for common cases
        model_lower = model.model_name.lower()

        if model.provider == "anthropic":
            # Match patterns like "claude-opus-4-20250514" to "Claude Opus 4"
            for pricing_key, pricing_model in pricing_lookup.items():
                pricing_lower = pricing_key.lower()

                # Debug the matching attempt
                logger.debug(
                    f"Trying to match '{model.model_name}' with '{pricing_key}'"
                )

                # Check for version matches (be more flexible with patterns)
                if ("opus" in model_lower and "4" in model_lower) and (
                    "opus" in pricing_lower and "4" in pricing_lower
                ):
                    logger.info(f"Matched {model.model_name} -> {pricing_key} (opus 4)")
                    return pricing_model
                elif ("sonnet" in model_lower and "4" in model_lower) and (
                    "sonnet" in pricing_lower and "4" in pricing_lower
                ):
                    logger.info(
                        f"Matched {model.model_name} -> {pricing_key} (sonnet 4)"
                    )
                    return pricing_model
                elif ("sonnet" in model_lower and "3-7" in model_lower) and (
                    "sonnet" in pricing_lower and "3.7" in pricing_lower
                ):
                    logger.info(
                        f"Matched {model.model_name} -> {pricing_key} (sonnet 3.7)"
                    )
                    return pricing_model
                elif ("sonnet" in model_lower and "3-5" in model_lower) and (
                    "sonnet" in pricing_lower and "3.5" in pricing_lower
                ):
                    logger.info(
                        f"Matched {model.model_name} -> {pricing_key} (sonnet 3.5)"
                    )
                    return pricing_model
                elif ("haiku" in model_lower and "3-5" in model_lower) and (
                    "haiku" in pricing_lower and "3.5" in pricing_lower
                ):
                    logger.info(
                        f"Matched {model.model_name} -> {pricing_key} (haiku 3.5)"
                    )
                    return pricing_model
                elif ("opus" in model_lower and "3" in model_lower) and (
                    "opus" in pricing_lower and "3" in pricing_lower
                ):
                    logger.info(f"Matched {model.model_name} -> {pricing_key} (opus 3)")
                    return pricing_model

        logger.debug(f"No pricing match found for {model.model_name}")
        return None
