"""Base class for pricing scrapers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    model_name: str
    input_cost_per_token: Optional[Decimal] = None
    output_cost_per_token: Optional[Decimal] = None
    currency: str = "USD"
    source_url: str = ""
    scraped_at: datetime = None
    confidence: float = 1.0  # 0.0 to 1.0

    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.utcnow()


@dataclass
class ScrapingResult:
    """Result of a pricing scraping operation."""

    success: bool
    models: List[ModelPricing] = None
    error: str = ""
    cached: bool = False
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.models is None:
            self.models = []


class PricingScraper(ABC):
    """Base class for provider-specific pricing scrapers."""

    def __init__(self, cache_duration_hours: int = 6):
        """
        Initialize pricing scraper.

        Args:
            cache_duration_hours: How long to cache pricing data
        """
        self.cache_duration_hours = cache_duration_hours
        self._cache: Dict[str, ScrapingResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    @abstractmethod
    async def scrape_pricing(self) -> ScrapingResult:
        """
        Scrape pricing information from the provider's website.

        Returns:
            ScrapingResult with pricing data
        """
        pass

    @abstractmethod
    def get_pricing_url(self) -> str:
        """
        Get the URL to scrape pricing from.

        Returns:
            URL string
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the provider name for this scraper.

        Returns:
            Provider name (e.g., "openai", "anthropic")
        """
        pass

    async def get_pricing_with_cache(self) -> ScrapingResult:
        """
        Get pricing with caching support.

        Returns:
            ScrapingResult with pricing data
        """
        cache_key = self.get_provider_name()

        # Check cache
        if self._is_cache_valid(cache_key):
            result = self._cache[cache_key]
            result.cached = True
            logger.info(f"Using cached pricing for {cache_key}")
            return result

        # Scrape fresh data
        start_time = asyncio.get_event_loop().time()
        result = await self.scrape_pricing()
        result.duration_seconds = asyncio.get_event_loop().time() - start_time

        # Cache if successful
        if result.success:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.utcnow()
            logger.info(f"Cached pricing for {cache_key}: {len(result.models)} models")

        return result

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache or cache_key not in self._cache_timestamps:
            return False

        cache_age = datetime.utcnow() - self._cache_timestamps[cache_key]
        return cache_age < timedelta(hours=self.cache_duration_hours)

    async def fetch_page_content(self, url: str) -> str:
        """
        Fetch HTML content from a URL.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            Exception: If fetch fails
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(
                            f"Successfully fetched {len(content)} characters from {url}"
                        )
                        return content
                    else:
                        raise Exception(f"HTTP {response.status}: {response.reason}")

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            raise

    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content using BeautifulSoup.

        Args:
            html_content: HTML content to parse

        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html_content, "html.parser")

    def clean_cache(self):
        """Remove expired entries from cache."""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp >= timedelta(hours=self.cache_duration_hours):
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]
            logger.info(f"Removed expired cache entry for {key}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        return {
            "provider": self.get_provider_name(),
            "cache_duration_hours": self.cache_duration_hours,
            "cached_entries": list(self._cache.keys()),
            "cache_timestamps": {
                key: timestamp.isoformat()
                for key, timestamp in self._cache_timestamps.items()
            },
        }
