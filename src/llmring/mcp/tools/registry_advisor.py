"""
MCP tools for registry analysis and lockfile recommendations.

These tools power the intelligent lockfile creation system by providing
real-time registry data and analysis capabilities to the advisor LLM.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from llmring.registry import RegistryClient, RegistryModel

logger = logging.getLogger(__name__)


class RegistryAdvisorTools:
    """MCP tools for registry analysis."""

    def __init__(self):
        self.registry = RegistryClient()

    async def get_provider_models(self, provider: str) -> Dict[str, Any]:
        """
        Get all active models for a provider with capabilities and pricing.

        Args:
            provider: Provider name (openai, anthropic, google, ollama)

        Returns:
            Dict with provider models and metadata
        """
        try:
            models = await self.registry.fetch_current_models(provider)
            active_models = [m for m in models if m.is_active]

            return {
                "provider": provider,
                "total_active_models": len(active_models),
                "models": [
                    {
                        "name": model.model_name,
                        "display_name": model.display_name,
                        "description": model.description or f"{provider.title()} model",
                        "max_input_tokens": model.max_input_tokens,
                        "max_output_tokens": model.max_output_tokens,
                        "cost_per_million_input": model.dollars_per_million_tokens_input,
                        "cost_per_million_output": model.dollars_per_million_tokens_output,
                        "supports_vision": model.supports_vision,
                        "supports_function_calling": model.supports_function_calling,
                        "supports_json_mode": model.supports_json_mode,
                        "added_date": (model.added_date.isoformat() if model.added_date else None),
                        "is_active": model.is_active,
                    }
                    for model in active_models
                ],
                "capabilities_summary": {
                    "has_vision_models": any(m.supports_vision for m in active_models),
                    "has_function_calling": any(m.supports_function_calling for m in active_models),
                    "has_json_mode": any(m.supports_json_mode for m in active_models),
                    "cost_range": (
                        {
                            "min_input_cost": min(
                                (m.dollars_per_million_tokens_input or 0) for m in active_models
                            ),
                            "max_input_cost": max(
                                (m.dollars_per_million_tokens_input or 0) for m in active_models
                            ),
                        }
                        if active_models
                        else None
                    ),
                },
            }

        except Exception as e:
            logger.warning(f"Failed to fetch models for {provider}: {e}")
            return {
                "provider": provider,
                "error": str(e),
                "total_active_models": 0,
                "models": [],
            }

    async def compare_models_by_cost(self, model_refs: List[str]) -> Dict[str, Any]:
        """
        Compare models by cost efficiency for typical usage patterns.

        Args:
            model_refs: List of provider:model strings to compare

        Returns:
            Cost comparison with recommendations
        """
        comparisons = []

        for model_ref in model_refs:
            try:
                provider, model_name = model_ref.split(":", 1)
                models = await self.registry.fetch_current_models(provider)
                model = next((m for m in models if m.model_name == model_name), None)

                if model and model.is_active:
                    # Calculate cost for typical usage patterns
                    input_cost = model.dollars_per_million_tokens_input or 0
                    output_cost = model.dollars_per_million_tokens_output or 0

                    # Typical request: 1K input, 200 output tokens
                    typical_request_cost = (input_cost * 1 / 1000) + (output_cost * 0.2 / 1000)

                    comparisons.append(
                        {
                            "model_ref": model_ref,
                            "cost_per_million_input": input_cost,
                            "cost_per_million_output": output_cost,
                            "cost_per_typical_request": typical_request_cost,
                            "max_context_tokens": model.max_input_tokens,
                            "capabilities": {
                                "vision": model.supports_vision,
                                "function_calling": model.supports_function_calling,
                                "json_mode": model.supports_json_mode,
                            },
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to analyze model {model_ref}: {e}")

        # Sort by cost efficiency
        comparisons.sort(key=lambda x: x["cost_per_typical_request"])

        return {
            "comparison_date": datetime.now().isoformat(),
            "methodology": "Cost per typical request (1K input + 200 output tokens)",
            "comparisons": comparisons,
            "most_cost_effective": comparisons[0]["model_ref"] if comparisons else None,
            "most_expensive": comparisons[-1]["model_ref"] if comparisons else None,
        }

    async def recommend_for_use_case(
        self,
        use_case: str,
        budget_preference: str = "balanced",
        required_capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend models for specific use cases.

        Args:
            use_case: Use case category
            budget_preference: cost_conscious, balanced, or performance_first
            required_capabilities: List of required features

        Returns:
            Recommendations with rationale
        """
        # Define use case requirements
        use_case_requirements = {
            "data_analysis": {
                "reasoning_importance": "high",
                "function_calling": True,
                "json_mode": True,
                "context_size": "large",
            },
            "creative_writing": {
                "reasoning_importance": "high",
                "creativity": "high",
                "context_size": "large",
                "output_length": "long",
            },
            "code_generation": {
                "reasoning_importance": "high",
                "function_calling": True,
                "accuracy": "high",
                "context_size": "large",
            },
            "general_qa": {
                "reasoning_importance": "medium",
                "speed": "high",
                "cost_sensitivity": "high",
            },
            "document_processing": {
                "vision": True,
                "reasoning_importance": "high",
                "context_size": "very_large",
            },
            "research": {
                "reasoning_importance": "very_high",
                "context_size": "very_large",
                "accuracy": "high",
            },
        }

        requirements = use_case_requirements.get(use_case, {})
        if required_capabilities:
            for cap in required_capabilities:
                requirements[cap] = True

        # Get models from all providers
        all_models = []
        for provider in ["openai", "anthropic", "google"]:
            try:
                models = await self.registry.fetch_current_models(provider)
                all_models.extend([(provider, model) for model in models if model.is_active])
            except Exception:
                # Provider might not be available, skip
                continue

        # Score and rank models
        recommendations = []
        for provider, model in all_models:
            score = self._calculate_use_case_score(model, requirements, budget_preference)
            if score > 0:
                recommendations.append(
                    {
                        "model_ref": f"{provider}:{model.model_name}",
                        "score": score,
                        "reasoning": self._explain_recommendation(model, requirements, use_case),
                        "estimated_cost": self._estimate_use_case_cost(model, use_case),
                    }
                )

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return {
            "use_case": use_case,
            "budget_preference": budget_preference,
            "requirements": requirements,
            "top_recommendations": recommendations[:3],
            "analysis_date": datetime.now().isoformat(),
        }

    def _calculate_use_case_score(
        self, model: RegistryModel, requirements: Dict, budget_preference: str
    ) -> float:
        """Calculate how well a model fits the use case requirements."""
        score = 0.0

        # Base capability scoring
        if requirements.get("vision") and model.supports_vision:
            score += 30
        elif requirements.get("vision") and not model.supports_vision:
            return 0  # Required capability missing

        if requirements.get("function_calling") and model.supports_function_calling:
            score += 20

        if requirements.get("json_mode") and model.supports_json_mode:
            score += 15

        # Context size scoring
        context_requirement = requirements.get("context_size", "medium")
        if context_requirement == "very_large" and (model.max_input_tokens or 0) >= 1000000:
            score += 25
        elif context_requirement == "large" and (model.max_input_tokens or 0) >= 100000:
            score += 20
        elif context_requirement == "medium" and (model.max_input_tokens or 0) >= 32000:
            score += 15

        # Cost preference scoring
        input_cost = model.dollars_per_million_tokens_input or 0
        if budget_preference == "cost_conscious" and input_cost <= 1.0:
            score += 20
        elif budget_preference == "balanced" and 1.0 <= input_cost <= 10.0:
            score += 15
        elif budget_preference == "performance_first":
            score += 10  # Cost less important

        # Recency bonus (newer models often better)
        if model.added_date:
            days_old = (datetime.now() - model.added_date).days
            if days_old <= 90:  # Less than 3 months old
                score += 10

        return score

    def _explain_recommendation(
        self, model: RegistryModel, requirements: Dict, use_case: str
    ) -> str:
        """Generate human-readable explanation for model recommendation."""
        reasons = []

        if requirements.get("vision") and model.supports_vision:
            reasons.append("supports vision for document/image processing")

        if requirements.get("function_calling") and model.supports_function_calling:
            reasons.append("native function calling for tool integration")

        if model.max_input_tokens and model.max_input_tokens >= 100000:
            reasons.append(f"large context window ({model.max_input_tokens:,} tokens)")

        cost = model.dollars_per_million_tokens_input or 0
        if cost <= 1.0:
            reasons.append("very cost-effective")
        elif cost <= 5.0:
            reasons.append("good cost/performance balance")

        if not reasons:
            reasons.append("suitable for general use")

        return f"Excellent for {use_case}: " + ", ".join(reasons)

    def _estimate_use_case_cost(self, model: RegistryModel, use_case: str) -> float:
        """Estimate monthly cost for use case."""
        # Use case token patterns (input/output per request)
        patterns = {
            "data_analysis": (5000, 1000),  # Complex analysis
            "creative_writing": (2000, 2000),  # Long creative output
            "code_generation": (3000, 1500),  # Code context + generation
            "general_qa": (500, 200),  # Simple Q&A
            "document_processing": (10000, 1000),  # Large document analysis
            "research": (8000, 2000),  # Research synthesis
        }

        input_tokens, output_tokens = patterns.get(use_case, (1000, 200))

        input_cost = (model.dollars_per_million_tokens_input or 0) * input_tokens / 1000000
        output_cost = (model.dollars_per_million_tokens_output or 0) * output_tokens / 1000000

        return input_cost + output_cost


# Tool function implementations for MCP server
registry_tools = RegistryAdvisorTools()


async def tool_get_provider_models(provider: str) -> Dict[str, Any]:
    """MCP tool: Get provider models."""
    return await registry_tools.get_provider_models(provider)


async def tool_compare_models_by_cost(model_refs: List[str]) -> Dict[str, Any]:
    """MCP tool: Compare models by cost."""
    return await registry_tools.compare_models_by_cost(model_refs)


async def tool_recommend_for_use_case(
    use_case: str,
    budget_preference: str = "balanced",
    required_capabilities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """MCP tool: Recommend models for use case."""
    return await registry_tools.recommend_for_use_case(
        use_case, budget_preference, required_capabilities
    )
