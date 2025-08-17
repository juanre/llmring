"""Smart model filtering using provider documentation and LLM analysis."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Set

from llmring.model_refresh.models import ModelInfo
from llmring.providers.openai_api import OpenAIProvider

logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Model categories for filtering."""

    PRODUCTION = "production"
    PREVIEW = "preview"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    SPECIALIZED = "specialized"  # embeddings, moderation, etc.


@dataclass
class ModelClassification:
    """Classification result for a model."""

    model_name: str
    category: ModelCategory
    confidence: float  # 0.0 to 1.0
    reasoning: str
    recommended: bool


class ModelFilter:
    """Intelligent model filtering using documentation and LLM analysis."""

    def __init__(self, llm_api_key: str = None):
        """
        Initialize model filter.

        Args:
            llm_api_key: API key for LLM analysis
        """
        self.llm = (
            OpenAIProvider(api_key=llm_api_key, model="gpt-4o") if llm_api_key else None
        )

        # Provider documentation URLs for context
        self.provider_docs = {
            "openai": "https://platform.openai.com/docs/models",
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/overview",
            "google": "https://cloud.google.com/vertex-ai/generative-ai/docs/models",
            "ollama": "https://ollama.com/library",
        }

        # Known patterns for quick filtering
        self.quick_filters = {
            "openai": {
                "production_patterns": [
                    r"^gpt-4o$",
                    r"^gpt-4o-mini$",
                    r"^gpt-4-turbo$",
                    r"^gpt-4$",
                    r"^gpt-3\.5-turbo$",
                    r"^o1$",
                    r"^o1-mini$",
                ],
                "exclude_patterns": [
                    r"preview",
                    r"realtime",
                    r"audio",
                    r"transcribe",
                    r"search",
                    r"instruct",
                    r"2024-\d{2}-\d{2}",
                    r"davinci-002",
                    r"babbage-002",
                    r"embedding",
                    r"moderation",
                    r"whisper",
                    r"tts",
                    r"dall-e",
                ],
            },
            "anthropic": {
                "production_patterns": [
                    r"^claude-4-(opus|sonnet)$",
                    r"^claude-3\.7-sonnet",
                    r"^claude-3-5-(sonnet|haiku)",
                    r"^claude-3-(opus|sonnet|haiku)",
                ],
                "exclude_patterns": [
                    r"claude-2\.",
                    r"claude-instant",
                    r"preview",
                    r"experimental",
                ],
            },
            "google": {
                "production_patterns": [
                    r"^gemini-2\.5-pro$",
                    r"^gemini-2\.0-(flash|pro)$",
                    r"^gemini-1\.5-(pro|flash)$",
                    r"^gemini-pro$",
                ],
                "exclude_patterns": [
                    r"experimental",
                    r"preview",
                    r"lite",
                    r"embedding",
                    r"aqa",
                    r"text-bison",
                    r"chat-bison",
                ],
            },
            "ollama": {
                "production_patterns": [
                    r"^llama3\.(3|2|1)(:latest)?$",
                    r"^mistral(:latest)?$",
                    r"^qwen2\.5(:latest)?$",
                    r"^codellama(:latest)?$",
                ],
                "exclude_patterns": [
                    r"preview",
                    r"experimental",
                    r"test",
                    r"-instruct",
                ],
            },
        }

    async def filter_models(
        self,
        models: List[ModelInfo],
        use_llm_analysis: bool = True,
        include_categories: Set[ModelCategory] = None,
    ) -> List[ModelInfo]:
        """
        Filter models to include only production-ready ones.

        Args:
            models: List of models to filter
            use_llm_analysis: Whether to use LLM for classification
            include_categories: Categories to include (default: PRODUCTION only)

        Returns:
            Filtered list of models
        """
        if include_categories is None:
            include_categories = {ModelCategory.PRODUCTION}

        logger.info(f"Filtering {len(models)} models, categories: {include_categories}")

        # Group models by provider for batch processing
        by_provider = {}
        for model in models:
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model)

        filtered_models = []

        for provider, provider_models in by_provider.items():
            logger.info(f"Filtering {len(provider_models)} {provider} models")

            # Classify models for this provider
            classifications = await self._classify_provider_models(
                provider, provider_models, use_llm_analysis
            )

            # Filter based on categories
            for classification in classifications:
                if (
                    classification.category in include_categories
                    and classification.recommended
                ):
                    # Find the original model
                    original_model = next(
                        (
                            m
                            for m in provider_models
                            if m.model_name == classification.model_name
                        ),
                        None,
                    )
                    if original_model:
                        filtered_models.append(original_model)
                        logger.debug(
                            f"Included {classification.model_name}: {classification.category.value} "
                            f"(confidence: {classification.confidence:.2f})"
                        )

        logger.info(f"Filtered to {len(filtered_models)} production-ready models")
        return filtered_models

    async def _classify_provider_models(
        self, provider: str, models: List[ModelInfo], use_llm_analysis: bool
    ) -> List[ModelClassification]:
        """Classify models for a specific provider."""
        classifications = []

        # Quick pattern-based filtering first
        quick_results = self._apply_quick_filters(provider, models)

        if not use_llm_analysis or not self.llm:
            # Use only quick filters
            return quick_results

        # For ambiguous cases, use LLM analysis
        ambiguous_models = [
            result
            for result in quick_results
            if result.confidence < 0.8  # Low confidence from quick filters
        ]

        if ambiguous_models:
            logger.info(
                f"Using LLM analysis for {len(ambiguous_models)} ambiguous {provider} models"
            )
            llm_results = await self._llm_classify_models(provider, ambiguous_models)

            # Replace ambiguous results with LLM results
            final_results = []
            for result in quick_results:
                if result.confidence >= 0.8:
                    final_results.append(result)
                else:
                    # Find corresponding LLM result
                    llm_result = next(
                        (r for r in llm_results if r.model_name == result.model_name),
                        result,  # Fallback to quick result
                    )
                    final_results.append(llm_result)

            return final_results

        return quick_results

    def _apply_quick_filters(
        self, provider: str, models: List[ModelInfo]
    ) -> List[ModelClassification]:
        """Apply pattern-based quick filters."""
        import re

        classifications = []
        filters = self.quick_filters.get(provider, {})
        production_patterns = filters.get("production_patterns", [])
        exclude_patterns = filters.get("exclude_patterns", [])

        for model in models:
            model_name = model.model_name.lower()

            # Check if explicitly excluded
            is_excluded = any(
                re.search(pattern, model_name) for pattern in exclude_patterns
            )
            if is_excluded:
                classifications.append(
                    ModelClassification(
                        model_name=model.model_name,
                        category=ModelCategory.SPECIALIZED,
                        confidence=0.9,
                        reasoning="Matches exclusion pattern",
                        recommended=False,
                    )
                )
                continue

            # Check if explicitly production
            is_production = any(
                re.search(pattern, model_name) for pattern in production_patterns
            )
            if is_production:
                classifications.append(
                    ModelClassification(
                        model_name=model.model_name,
                        category=ModelCategory.PRODUCTION,
                        confidence=0.9,
                        reasoning="Matches production pattern",
                        recommended=True,
                    )
                )
                continue

            # Ambiguous case
            category = self._guess_category_from_name(model.model_name)
            classifications.append(
                ModelClassification(
                    model_name=model.model_name,
                    category=category,
                    confidence=0.5,  # Low confidence - needs LLM analysis
                    reasoning="Pattern-based guess",
                    recommended=(category == ModelCategory.PRODUCTION),
                )
            )

        return classifications

    def _guess_category_from_name(self, model_name: str) -> ModelCategory:
        """Guess category from model name patterns."""
        name_lower = model_name.lower()

        # Check for obvious indicators
        if any(
            word in name_lower for word in ["preview", "experimental", "alpha", "beta"]
        ):
            return ModelCategory.PREVIEW

        if any(word in name_lower for word in ["deprecated", "legacy", "old"]):
            return ModelCategory.DEPRECATED

        if any(
            word in name_lower for word in ["embedding", "moderation", "whisper", "tts"]
        ):
            return ModelCategory.SPECIALIZED

        # Default to production for main model names
        return ModelCategory.PRODUCTION

    async def _llm_classify_models(
        self, provider: str, classifications: List[ModelClassification]
    ) -> List[ModelClassification]:
        """Use LLM to classify ambiguous models."""
        try:
            model_names = [c.model_name for c in classifications]

            prompt = self._create_classification_prompt(provider, model_names)
            response = await self.llm.generate_response(prompt, "")

            # Parse LLM response
            llm_classifications = self._parse_llm_classification(
                response, classifications
            )
            return llm_classifications

        except Exception as e:
            logger.error(f"LLM classification failed for {provider}: {str(e)}")
            # Return original classifications as fallback
            return classifications

    def _create_classification_prompt(
        self, provider: str, model_names: List[str]
    ) -> str:
        """Create prompt for LLM model classification."""
        docs_url = self.provider_docs.get(provider, "")

        return f"""Analyze these {provider.upper()} language models and classify each as production-ready or not.

PROVIDER: {provider.upper()}
DOCUMENTATION: {docs_url}

MODELS TO CLASSIFY:
{chr(10).join(f"- {name}" for name in model_names)}

For each model, determine:
1. CATEGORY: production, preview, experimental, deprecated, or specialized
2. RECOMMENDED: true if suitable for production use, false otherwise
3. REASONING: Brief explanation

RULES:
- PRODUCTION: Stable, documented, recommended for production use
- PREVIEW: Beta/preview versions, limited availability
- EXPERIMENTAL: Early access, testing only
- DEPRECATED: Old versions, being phased out
- SPECIALIZED: Non-chat models (embeddings, moderation, etc.)

Only recommend models that are:
✓ Stable and well-documented
✓ Suitable for production chat/completion use
✓ Not preview/experimental versions
✓ Current generation (not deprecated)

Return JSON array format:
[
  {{
    "model_name": "exact-model-name",
    "category": "production|preview|experimental|deprecated|specialized",
    "recommended": true|false,
    "reasoning": "brief explanation",
    "confidence": 0.8
  }}
]

Focus on identifying the core, stable models that developers should actually use in production."""

    def _parse_llm_classification(
        self, response: str, original_classifications: List[ModelClassification]
    ) -> List[ModelClassification]:
        """Parse LLM classification response."""
        import json

        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            llm_results = json.loads(response)

            # Convert to ModelClassification objects
            classifications = []
            for item in llm_results:
                try:
                    classification = ModelClassification(
                        model_name=item["model_name"],
                        category=ModelCategory(item["category"]),
                        confidence=float(item.get("confidence", 0.8)),
                        reasoning=item["reasoning"],
                        recommended=bool(item["recommended"]),
                    )
                    classifications.append(classification)
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Invalid LLM classification item: {item}, error: {e}"
                    )
                    continue

            return classifications

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM classification response: {str(e)}")
            logger.debug(f"Response was: {response}")
            return original_classifications
        except Exception as e:
            logger.error(f"Error processing LLM classification: {str(e)}")
            return original_classifications

    def get_filter_summary(
        self, models: List[ModelInfo], filtered: List[ModelInfo]
    ) -> Dict[str, Any]:
        """Get summary of filtering results."""
        original_by_provider = {}
        filtered_by_provider = {}

        for model in models:
            original_by_provider[model.provider] = (
                original_by_provider.get(model.provider, 0) + 1
            )

        for model in filtered:
            filtered_by_provider[model.provider] = (
                filtered_by_provider.get(model.provider, 0) + 1
            )

        return {
            "total_original": len(models),
            "total_filtered": len(filtered),
            "reduction_percentage": (
                (1 - len(filtered) / len(models)) * 100 if models else 0
            ),
            "by_provider": {
                provider: {
                    "original": original_by_provider.get(provider, 0),
                    "filtered": filtered_by_provider.get(provider, 0),
                }
                for provider in set(original_by_provider.keys())
                | set(filtered_by_provider.keys())
            },
        }
