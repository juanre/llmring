"""Google Gemini pricing scraper."""

import logging
from decimal import Decimal
from typing import List

from .llm_processor import LLMPriceProcessor
from .scraper_base import ModelPricing, PricingScraper, ScrapingResult

logger = logging.getLogger(__name__)


class GooglePricingScraper(PricingScraper):
    """Scraper for Google Gemini pricing information."""

    def __init__(self, llm_api_key: str = None):
        """
        Initialize Google pricing scraper.

        Args:
            llm_api_key: API key for LLM processing
        """
        super().__init__()
        self.llm_processor = LLMPriceProcessor(llm_api_key)

    def get_pricing_url(self) -> str:
        """Get Google Gemini API pricing URL."""
        return "https://ai.google.dev/gemini-api/docs/pricing"

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "google"

    async def scrape_pricing(self) -> ScrapingResult:
        """
        Scrape Google Gemini pricing information.

        Returns:
            ScrapingResult with pricing data
        """
        try:
            url = self.get_pricing_url()
            logger.info(f"Scraping Google pricing from {url}")

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

            logger.info(f"Successfully scraped pricing for {len(models)} Google models")
            return ScrapingResult(success=True, models=models)

        except Exception as e:
            error_msg = f"Failed to scrape Google pricing: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(success=False, error=error_msg)

    def get_fallback_pricing(self) -> List[ModelPricing]:
        """
        Get fallback pricing data for Google models.

        Returns:
            List of ModelPricing with known prices
        """
        # Known Google Gemini pricing as of 2025 (per token) - based on https://ai.google.dev/gemini-api/docs/pricing
        fallback_data = [
            {
                "model_name": "gemini-2.5-flash-preview",
                "input_cost": Decimal("0.00000015"),  # $0.15 per 1M tokens
                "output_cost": Decimal("0.0000006"),  # $0.60 per 1M tokens
            },
            {
                "model_name": "gemini-1.5-pro",
                "input_cost": Decimal("0.00000125"),  # $1.25 per 1M tokens
                "output_cost": Decimal("0.000005"),  # $5.00 per 1M tokens
            },
            {
                "model_name": "gemini-1.5-flash",
                "input_cost": Decimal("0.000000075"),  # $0.075 per 1M tokens
                "output_cost": Decimal("0.0000003"),  # $0.30 per 1M tokens
            },
            {
                "model_name": "gemini-1.5-flash-8b",
                "input_cost": Decimal("0.0000000375"),  # $0.0375 per 1M tokens
                "output_cost": Decimal("0.00000015"),  # $0.15 per 1M tokens
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
