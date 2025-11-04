# ABOUTME: Cost calculation service for token usage across providers.
# ABOUTME: Calculates costs using registry pricing data and token counts.
"""
Cost calculation service for LLMRing.

Calculates costs for LLM API calls based on token usage and registry pricing data.
"""

import logging
from typing import Dict, Optional

from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMResponse
from llmring.utils import parse_model_string

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Calculates costs for LLM API calls.

    Uses pricing information from the registry to calculate:
    - Input (prompt) token costs
    - Output (completion) token costs
    - Total cost breakdown
    """

    def __init__(self, registry: RegistryClient):
        """
        Initialize the cost calculator.

        Args:
            registry: Registry client for fetching pricing information
        """
        self.registry = registry

    async def calculate_cost(
        self, response: LLMResponse, registry_model: Optional[RegistryModel] = None
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the cost of an API call from the response.

        Args:
            response: LLMResponse object with model and usage information
            registry_model: Optional pre-fetched registry model (for performance)

        Returns:
            Cost breakdown dictionary with keys:
            - input_cost: Cost of prompt tokens
            - output_cost: Cost of completion tokens
            - total_cost: Total cost
            - cost_per_million_input: Pricing rate for input
            - cost_per_million_output: Pricing rate for output

            Returns None if:
            - No usage information available
            - Model not found in registry
            - Pricing information not available

        Example:
            >>> calculator = CostCalculator(registry)
            >>> response = LLMResponse(
            ...     content="Hello",
            ...     model="openai:gpt-4",
            ...     usage={"prompt_tokens": 100, "completion_tokens": 50}
            ... )
            >>> cost = await calculator.calculate_cost(response)
            >>> print(f"Total: ${cost['total_cost']:.4f}")
        """
        if not response.usage:
            logger.debug("No usage information available for cost calculation")
            return None

        # Parse model string to get provider and model name
        if ":" not in response.model:
            logger.warning(f"Invalid model format for cost calculation: {response.model}")
            return None

        provider, model_name = parse_model_string(response.model)

        # Get pricing info from registry if not provided
        if not registry_model:
            registry_model = await self._get_registry_model(provider, model_name)

        if not registry_model:
            logger.debug(f"Model not found in registry: {provider}:{model_name}")
            return None

        # Check if pricing information is available
        if (
            registry_model.dollars_per_million_tokens_input is None
            or registry_model.dollars_per_million_tokens_output is None
        ):
            logger.debug(f"Pricing not available for {provider}:{model_name}")
            return None

        # Extract token counts
        usage = response.usage
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        # Cache read metrics
        cache_read_tokens = int(
            usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0
        )

        # Cache write metrics
        cache_write_5m_tokens = int(usage.get("cache_creation_5m_tokens", 0) or 0)
        cache_write_1h_tokens = int(usage.get("cache_creation_1h_tokens", 0) or 0)
        cache_write_generic_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)

        # Reasoning tokens (if provider reports them)
        reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)

        base_input_rate = registry_model.dollars_per_million_tokens_input or 0.0
        base_output_rate = registry_model.dollars_per_million_tokens_output or 0.0
        cached_input_rate = registry_model.dollars_per_million_tokens_cached_input
        cache_write_5m_rate = registry_model.dollars_per_million_tokens_cache_write_5m
        cache_write_1h_rate = registry_model.dollars_per_million_tokens_cache_write_1h
        cache_read_rate = (
            registry_model.dollars_per_million_tokens_cache_read
            or cached_input_rate
            or base_input_rate
        )

        # Handle prompt tokens: remove cached reads from base billing
        non_cached_prompt_tokens = max(prompt_tokens - cache_read_tokens, 0)

        # Long-context pricing: models may charge different rates above a threshold
        # For example, Sonnet 4 charges $3/M for first 200k tokens, $6/M beyond that
        long_context_threshold = registry_model.long_context_threshold_tokens or 0
        supports_long_context = (
            registry_model.supports_long_context_pricing and long_context_threshold > 0
        )

        if supports_long_context and non_cached_prompt_tokens > long_context_threshold:
            # Split tokens: regular rate up to threshold, long-context rate beyond
            long_tokens = non_cached_prompt_tokens - long_context_threshold
            regular_prompt_tokens = long_context_threshold
        else:
            # All tokens billed at regular rate
            long_tokens = 0
            regular_prompt_tokens = non_cached_prompt_tokens

        regular_prompt_cost = self._calculate_token_cost(regular_prompt_tokens, base_input_rate)
        long_context_input_rate = (
            registry_model.dollars_per_million_tokens_input_long_context or base_input_rate
        )
        long_context_cost = self._calculate_token_cost(long_tokens, long_context_input_rate)

        cache_read_cost = self._calculate_token_cost(cache_read_tokens, cache_read_rate)

        # Cache writes
        cache_write_cost_5m = self._calculate_token_cost(cache_write_5m_tokens, cache_write_5m_rate)
        cache_write_cost_1h = self._calculate_token_cost(cache_write_1h_tokens, cache_write_1h_rate)

        # Remaining generic cache write tokens (if provider only reports a single total)
        consumed_write_tokens = cache_write_5m_tokens + cache_write_1h_tokens
        generic_write_tokens = max(cache_write_generic_tokens - consumed_write_tokens, 0)
        fallback_write_rate = (
            cache_write_5m_rate or cache_write_1h_rate or cached_input_rate or base_input_rate
        )
        cache_write_cost_generic = self._calculate_token_cost(
            generic_write_tokens, fallback_write_rate
        )

        # Output / reasoning costs
        # Note: Most providers include reasoning_tokens in completion_tokens total.
        # We subtract them here to avoid double-counting when applying separate pricing.
        # If completion_tokens < reasoning_tokens, the provider reports them separately,
        # so we skip the subtraction.
        reasoning_cost = 0.0
        reasoning_rate = registry_model.dollars_per_million_tokens_output_thinking or 0.0
        if reasoning_tokens > 0 and reasoning_rate:
            if completion_tokens >= reasoning_tokens:
                completion_tokens -= reasoning_tokens
            reasoning_cost = self._calculate_token_cost(reasoning_tokens, reasoning_rate)

        output_cost = self._calculate_token_cost(completion_tokens, base_output_rate)

        total_cost = (
            regular_prompt_cost
            + long_context_cost
            + cache_read_cost
            + cache_write_cost_5m
            + cache_write_cost_1h
            + cache_write_cost_generic
            + output_cost
            + reasoning_cost
        )

        logger.debug(
            "Cost for %s:%s: $%.6f (prompt: $%.6f, long_context: $%.6f, cache_read: $%.6f, "
            "cache_write_5m: $%.6f, cache_write_1h: $%.6f, cache_write_other: $%.6f, "
            "output: $%.6f, reasoning: $%.6f)",
            provider,
            model_name,
            total_cost,
            regular_prompt_cost,
            long_context_cost,
            cache_read_cost,
            cache_write_cost_5m,
            cache_write_cost_1h,
            cache_write_cost_generic,
            output_cost,
            reasoning_cost,
        )

        return {
            "input_cost": regular_prompt_cost + long_context_cost,
            "output_cost": output_cost,
            "cache_read_cost": cache_read_cost,
            "cache_write_cost_5m": cache_write_cost_5m,
            "cache_write_cost_1h": cache_write_cost_1h,
            "cache_write_cost_other": cache_write_cost_generic,
            "long_context_input_cost": long_context_cost,
            "reasoning_cost": reasoning_cost,
            "total_cost": total_cost,
            "cost_per_million_input": base_input_rate,
            "cost_per_million_output": base_output_rate,
            "cost_per_million_cached_input": cached_input_rate,
            "cost_per_million_cache_read": cache_read_rate,
            "cost_per_million_cache_write_5m": cache_write_5m_rate,
            "cost_per_million_cache_write_1h": cache_write_1h_rate,
            "cost_per_million_output_thinking": reasoning_rate or None,
        }

    @staticmethod
    def _calculate_token_cost(token_count: int, cost_per_million: Optional[float]) -> float:
        """
        Calculate cost for a given number of tokens.

        Args:
            token_count: Number of tokens
            cost_per_million: Cost per million tokens

        Returns:
            Cost in dollars
        """
        if not cost_per_million:
            return 0.0
        return (token_count / 1_000_000) * cost_per_million

    async def _get_registry_model(self, provider: str, model_name: str) -> Optional[RegistryModel]:
        """
        Get model information from the registry.

        Args:
            provider: Provider name (e.g., "openai")
            model_name: Model name (e.g., "gpt-4")

        Returns:
            Registry model or None if not found
        """
        try:
            models = await self.registry.fetch_current_models(provider)
            for model in models:
                if model.model_name == model_name:
                    return model
            logger.debug(f"Model {model_name} not found in {provider} registry")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch registry for {provider}: {e}")
            return None

    def add_cost_to_response(self, response: LLMResponse, cost_info: Dict[str, float]) -> None:
        """
        Add cost information to a response object.

        Modifies the response in-place by adding cost data to the usage dictionary.

        Args:
            response: LLMResponse to modify
            cost_info: Cost information from calculate_cost()
        """
        if not response.usage:
            response.usage = {}

        response.usage["cost"] = cost_info["total_cost"]
        breakdown = {
            "input": cost_info.get("input_cost", 0.0),
            "output": cost_info.get("output_cost", 0.0),
        }

        if cost_info.get("cache_read_cost"):
            breakdown["cache_read"] = cost_info["cache_read_cost"]
        if cost_info.get("cache_write_cost_5m"):
            breakdown["cache_write_5m"] = cost_info["cache_write_cost_5m"]
        if cost_info.get("cache_write_cost_1h"):
            breakdown["cache_write_1h"] = cost_info["cache_write_cost_1h"]
        if cost_info.get("cache_write_cost_other"):
            breakdown["cache_write_other"] = cost_info["cache_write_cost_other"]
        if cost_info.get("long_context_input_cost"):
            breakdown["long_context_input"] = cost_info["long_context_input_cost"]
        if cost_info.get("reasoning_cost"):
            breakdown["reasoning"] = cost_info["reasoning_cost"]

        response.usage["cost_breakdown"] = breakdown

    def get_zero_cost_info(self) -> Dict[str, float]:
        """
        Get a zero-cost info dictionary.

        Useful as a fallback when cost calculation is not available.

        Returns:
            Dictionary with all costs set to 0.0
        """
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }
