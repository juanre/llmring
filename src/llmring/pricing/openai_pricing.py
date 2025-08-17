"""OpenAI pricing scraper."""

import logging
from decimal import Decimal
from typing import List

from .llm_processor import LLMPriceProcessor
from .scraper_base import ModelPricing, PricingScraper, ScrapingResult

logger = logging.getLogger(__name__)


class OpenAIPricingScraper(PricingScraper):
    """Scraper for OpenAI pricing information."""

    def __init__(self, llm_api_key: str = None):
        """
        Initialize OpenAI pricing scraper.

        Args:
            llm_api_key: API key for LLM processing
        """
        super().__init__()
        self.llm_processor = LLMPriceProcessor(llm_api_key)

    def get_pricing_url(self) -> str:
        """Get OpenAI pricing URL."""
        return "https://platform.openai.com/docs/pricing"

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"

    async def scrape_pricing(self) -> ScrapingResult:
        """
        Scrape OpenAI pricing information.

        Returns:
            ScrapingResult with pricing data
        """
        try:
            url = self.get_pricing_url()
            logger.info(f"Scraping OpenAI pricing from {url}")

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

            logger.info(f"Successfully scraped pricing for {len(models)} OpenAI models")
            return ScrapingResult(success=True, models=models)

        except Exception as e:
            error_msg = f"Failed to scrape OpenAI pricing: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(success=False, error=error_msg)

    def get_fallback_pricing(self) -> List[ModelPricing]:
        """
        Get fallback pricing data for OpenAI models.

        Returns:
            List of ModelPricing with known prices
        """
        # Known OpenAI pricing as of 2025 (per token)
        fallback_data = [
            {
                "model_name": "gpt-4o",
                "input_cost": Decimal("0.000005"),  # $5 per 1M tokens
                "output_cost": Decimal("0.000015"),  # $15 per 1M tokens
            },
            {
                "model_name": "gpt-4o-mini",
                "input_cost": Decimal("0.00000015"),  # $0.15 per 1M tokens
                "output_cost": Decimal("0.0000006"),  # $0.6 per 1M tokens
            },
            {
                "model_name": "gpt-4-turbo",
                "input_cost": Decimal("0.00001"),  # $10 per 1M tokens
                "output_cost": Decimal("0.00003"),  # $30 per 1M tokens
            },
            {
                "model_name": "gpt-4",
                "input_cost": Decimal("0.00003"),  # $30 per 1M tokens
                "output_cost": Decimal("0.00006"),  # $60 per 1M tokens
            },
            {
                "model_name": "gpt-3.5-turbo",
                "input_cost": Decimal("0.0000005"),  # $0.5 per 1M tokens
                "output_cost": Decimal("0.0000015"),  # $1.5 per 1M tokens
            },
            {
                "model_name": "o1",
                "input_cost": Decimal("0.000015"),  # $15 per 1M tokens
                "output_cost": Decimal("0.00006"),  # $60 per 1M tokens
            },
            {
                "model_name": "o1-mini",
                "input_cost": Decimal("0.000003"),  # $3 per 1M tokens
                "output_cost": Decimal("0.000012"),  # $12 per 1M tokens
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
