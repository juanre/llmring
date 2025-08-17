"""Anthropic pricing scraper."""

import logging
from decimal import Decimal
from typing import List

from .llm_processor import LLMPriceProcessor
from .scraper_base import ModelPricing, PricingScraper, ScrapingResult

logger = logging.getLogger(__name__)


class AnthropicPricingScraper(PricingScraper):
    """Scraper for Anthropic pricing information."""

    def __init__(self, llm_api_key: str = None):
        """
        Initialize Anthropic pricing scraper.

        Args:
            llm_api_key: API key for LLM processing
        """
        super().__init__()
        self.llm_processor = LLMPriceProcessor(llm_api_key)

    def get_pricing_url(self) -> str:
        """Get Anthropic pricing URL."""
        return "https://docs.anthropic.com/en/docs/about-claude/models/overview"

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "anthropic"

    async def scrape_pricing(self) -> ScrapingResult:
        """
        Scrape Anthropic pricing information.

        Returns:
            ScrapingResult with pricing data
        """
        try:
            url = self.get_pricing_url()
            logger.info(f"Scraping Anthropic pricing from {url}")

            # Fetch page content
            html_content = await self.fetch_page_content(url)

            # Extract pricing using LLM
            pricing_data = await self.llm_processor.extract_pricing_from_html(
                html_content, self.get_provider_name(), url
            )

            # Convert to ModelPricing objects
            models = []
            for item in pricing_data:
                model = ModelPricing(
                    model_name=item["model_name"],
                    input_cost_per_token=item["input_cost_per_token"],
                    output_cost_per_token=item["output_cost_per_token"],
                    source_url=url,
                    confidence=item["confidence"],
                )
                models.append(model)

            logger.info(
                f"Successfully scraped pricing for {len(models)} Anthropic models"
            )
            return ScrapingResult(success=True, models=models)

        except Exception as e:
            error_msg = f"Failed to scrape Anthropic pricing: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(success=False, error=error_msg)

    def get_fallback_pricing(self) -> List[ModelPricing]:
        """
        Get fallback pricing data for Anthropic models.

        Returns:
            List of ModelPricing with known prices
        """
        # Known Anthropic pricing as of 2025 (per token)
        fallback_data = [
            {
                "model_name": "claude-4-opus",
                "input_cost": Decimal("0.000015"),  # $15 per 1M tokens
                "output_cost": Decimal("0.000075"),  # $75 per 1M tokens
            },
            {
                "model_name": "claude-4-sonnet",
                "input_cost": Decimal("0.000003"),  # $3 per 1M tokens
                "output_cost": Decimal("0.000015"),  # $15 per 1M tokens
            },
            {
                "model_name": "claude-3-7-sonnet-20250219",
                "input_cost": Decimal("0.000003"),  # $3 per 1M tokens
                "output_cost": Decimal("0.000015"),  # $15 per 1M tokens
            },
            {
                "model_name": "claude-3-5-sonnet-20241022",
                "input_cost": Decimal("0.000003"),  # $3 per 1M tokens
                "output_cost": Decimal("0.000015"),  # $15 per 1M tokens
            },
            {
                "model_name": "claude-3-5-haiku-20241022",
                "input_cost": Decimal("0.0000008"),  # $0.80 per 1M tokens
                "output_cost": Decimal("0.000004"),  # $4.00 per 1M tokens
            },
            {
                "model_name": "claude-3-opus-20240229",
                "input_cost": Decimal("0.000015"),  # $15 per 1M tokens
                "output_cost": Decimal("0.000075"),  # $75 per 1M tokens
            },
            {
                "model_name": "claude-3-sonnet-20240229",
                "input_cost": Decimal("0.000003"),  # $3 per 1M tokens
                "output_cost": Decimal("0.000015"),  # $15 per 1M tokens
            },
            {
                "model_name": "claude-3-haiku-20240307",
                "input_cost": Decimal("0.00000025"),  # $0.25 per 1M tokens
                "output_cost": Decimal("0.00000125"),  # $1.25 per 1M tokens
            },
        ]

        models = []
        for item in fallback_data:
            model = ModelPricing(
                model_name=item["model_name"],
                input_cost_per_token=item["input_cost"],
                output_cost_per_token=item["output_cost"],
                source_url="fallback",
                confidence=0.8,
            )
            models.append(model)

        return models
