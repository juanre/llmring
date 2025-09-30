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
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)

        # Calculate costs
        input_cost = self._calculate_token_cost(
            prompt_tokens, registry_model.dollars_per_million_tokens_input
        )
        output_cost = self._calculate_token_cost(
            completion_tokens, registry_model.dollars_per_million_tokens_output
        )
        total_cost = input_cost + output_cost

        logger.debug(
            f"Cost for {provider}:{model_name}: "
            f"${total_cost:.6f} (input: ${input_cost:.6f}, output: ${output_cost:.6f})"
        )

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_million_input": registry_model.dollars_per_million_tokens_input,
            "cost_per_million_output": registry_model.dollars_per_million_tokens_output,
        }

    @staticmethod
    def _calculate_token_cost(token_count: int, cost_per_million: float) -> float:
        """
        Calculate cost for a given number of tokens.

        Args:
            token_count: Number of tokens
            cost_per_million: Cost per million tokens

        Returns:
            Cost in dollars
        """
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
        response.usage["cost_breakdown"] = {
            "input": cost_info["input_cost"],
            "output": cost_info["output_cost"],
        }

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
