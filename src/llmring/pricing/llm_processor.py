"""LLM-powered price extraction from HTML content."""

import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from llmring.providers.openai_api import OpenAIProvider

logger = logging.getLogger(__name__)


class LLMPriceProcessor:
    """Uses LLM to extract pricing information from HTML content."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM processor.

        Args:
            api_key: OpenAI API key for processing
        """
        self.llm = OpenAIProvider(api_key=api_key, model="gpt-4o")

    async def extract_pricing_from_html(
        self, html_content: str, provider_name: str, url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extract pricing information from HTML using LLM.

        Args:
            html_content: HTML content to process
            provider_name: Name of the provider (openai, anthropic, google)
            url: Source URL for context

        Returns:
            List of pricing dictionaries
        """
        try:
            # Clean and truncate HTML content
            cleaned_html = self._clean_html_content(html_content)

            # Create extraction prompt
            prompt = self._create_extraction_prompt(provider_name, url)

            # Call LLM for extraction
            response = await self.llm.generate_response(prompt, cleaned_html)

            # Parse LLM response
            pricing_data = self._parse_llm_response(response)

            logger.info(
                f"Extracted {len(pricing_data)} model prices for {provider_name}"
            )
            return pricing_data

        except Exception as e:
            logger.error(f"LLM price extraction failed for {provider_name}: {str(e)}")
            return []

    def _clean_html_content(self, html_content: str) -> str:
        """Clean and truncate HTML content for LLM processing."""
        from bs4 import BeautifulSoup

        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        # Truncate to reasonable length for LLM
        max_length = 8000  # Leave room for prompt
        if len(text) > max_length:
            text = text[:max_length] + "\\n[Content truncated...]"

        return text

    def _create_extraction_prompt(self, provider_name: str, url: str) -> str:
        """Create prompt for LLM price extraction."""
        provider_specific_instructions = ""

        if provider_name == "anthropic":
            provider_specific_instructions = """
ANTHROPIC-SPECIFIC INSTRUCTIONS:
- Extract the EXACT API model names like "claude-3-opus-20240229", "claude-3-5-sonnet-20241022", NOT display names
- Look for the pricing table that shows API model names with dates/versions
- The model names should include version dates like "-20240229" or "-20241022"
- Example correct names: "claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"
- Do NOT use display names like "Claude 3 Opus" or "Claude 3.5 Sonnet"
"""
        elif provider_name == "openai":
            provider_specific_instructions = """
OPENAI-SPECIFIC INSTRUCTIONS:
- Extract the EXACT API model names like "gpt-4o", "gpt-4o-mini", "o1", "o1-mini"
- Look for the pricing table that shows API model identifiers
- Example correct names: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
- Do NOT use display names like "GPT-4 Omni" or "GPT-4 Turbo"
"""
        elif provider_name == "google":
            provider_specific_instructions = """
GOOGLE-SPECIFIC INSTRUCTIONS:
- Extract the EXACT API model names like "gemini-2.5-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"
- Look for the pricing table that shows "per 1M tokens in USD"
- CRITICAL: The pricing is already PER 1M TOKENS - do NOT multiply by 1000
- If you see "$0.15 per 1M tokens" then the input_cost_per_million_tokens should be 0.15
- Example correct names: "gemini-2.5-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"
- Do NOT use display names like "Gemini Pro" or "Gemini Flash"
- IGNORE any mention of "free tier" or "free quota" - we need PAID pricing only
"""

        return f"""Extract pricing information for {provider_name.title()} language models from the following webpage content.

SOURCE URL: {url}

{provider_specific_instructions}

I need you to find pricing information for language models and return it as a JSON array. Each model should have:
- model_name: The EXACT API model identifier (e.g., "gpt-4o", "claude-3-opus-20240229")
- input_cost_per_million_tokens: Cost per 1 million input tokens in USD
- output_cost_per_million_tokens: Cost per 1 million output tokens in USD
- confidence: Your confidence in this data (0.0 to 1.0)

CRITICAL REQUIREMENTS:
1. Use EXACT API model names, not display names or marketing names
2. Only extract pricing for chat/completion models (not embeddings, image generation, etc.)
3. UNIT CONVERSION: If prices are "per 1K tokens", multiply by 1000 to get per-million. If already "per 1M tokens", use as-is.
4. PAY ATTENTION TO UNITS: "$0.15 per 1M tokens" = 0.15, "$0.15 per 1K tokens" = 150.0
5. Only include models you're confident about (confidence >= 0.8)
6. Return valid JSON only, no explanation text
7. If no pricing found, return empty array []
8. ALL NON-OLLAMA MODELS MUST HAVE PRICING - if you see "free" for non-Ollama models, something is wrong

Example output format:
[
  {{
    "model_name": "claude-3-opus-20240229",
    "input_cost_per_million_tokens": 15.0,
    "output_cost_per_million_tokens": 75.0,
    "confidence": 0.95
  }},
  {{
    "model_name": "claude-3-5-sonnet-20241022",
    "input_cost_per_million_tokens": 3.0,
    "output_cost_per_million_tokens": 15.0,
    "confidence": 0.9
  }}
]

Extract from this content:"""

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract pricing data."""
        try:
            # Clean response (remove any markdown formatting)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            pricing_data = json.loads(response)

            # Validate and convert data
            validated_data = []
            for item in pricing_data:
                if self._validate_pricing_item(item):
                    # Convert to proper decimal values (per token, not per million)
                    validated_item = {
                        "model_name": item["model_name"],
                        "input_cost_per_token": Decimal(
                            str(item["input_cost_per_million_tokens"])
                        )
                        / Decimal("1000000"),
                        "output_cost_per_token": Decimal(
                            str(item["output_cost_per_million_tokens"])
                        )
                        / Decimal("1000000"),
                        "confidence": float(item["confidence"]),
                    }
                    validated_data.append(validated_item)

            return validated_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"Response was: {response}")
            return []
        except Exception as e:
            logger.error(f"Failed to process LLM response: {str(e)}")
            return []

    def _validate_pricing_item(self, item: Dict[str, Any]) -> bool:
        """Validate a pricing item from LLM response."""
        required_fields = [
            "model_name",
            "input_cost_per_million_tokens",
            "output_cost_per_million_tokens",
            "confidence",
        ]

        # Check required fields
        for field in required_fields:
            if field not in item:
                logger.warning(f"Missing field {field} in pricing item")
                return False

        # Validate types and ranges
        try:
            model_name = str(item["model_name"]).strip()
            if not model_name:
                return False

            input_cost = float(item["input_cost_per_million_tokens"])
            output_cost = float(item["output_cost_per_million_tokens"])
            confidence = float(item["confidence"])

            # Sanity checks
            if input_cost < 0 or input_cost > 1000:  # Max $1000 per million tokens
                logger.warning(f"Invalid input cost: {input_cost}")
                return False

            if output_cost < 0 or output_cost > 1000:  # Max $1000 per million tokens
                logger.warning(f"Invalid output cost: {output_cost}")
                return False

            if confidence < 0.0 or confidence > 1.0:
                logger.warning(f"Invalid confidence: {confidence}")
                return False

            if confidence < 0.8:  # Require high confidence
                logger.info(f"Low confidence ({confidence}) for {model_name}, skipping")
                return False

            return True

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid data types in pricing item: {str(e)}")
            return False
