"""
Validation service for LLMRing.

Validates LLM requests against model capabilities and constraints.
"""

import logging
from typing import Optional

from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMRequest
from llmring.utils import parse_model_string

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Validates LLM requests against model constraints.

    Checks:
    - Context window limits (input + output tokens)
    - Model capabilities (vision, function calling, etc.)
    - Schema validity
    """

    def __init__(self, registry: RegistryClient):
        """
        Initialize the validation service.

        Args:
            registry: Registry client for fetching model information
        """
        self.registry = registry

    async def validate_context_limit(
        self, request: LLMRequest, registry_model: Optional[RegistryModel] = None
    ) -> Optional[str]:
        """
        Validate that the request doesn't exceed model context limits.

        Args:
            request: The LLM request to validate
            registry_model: Optional pre-fetched registry model (for performance)

        Returns:
            Error message if validation fails, None if validation passes

        Example:
            >>> validator = ValidationService(registry)
            >>> error = await validator.validate_context_limit(request)
            >>> if error:
            ...     print(f"Validation failed: {error}")
        """
        if not request.model:
            logger.debug("No model specified, skipping context validation")
            return None

        # Parse model string
        if ":" not in request.model:
            logger.warning(f"Invalid model format for validation: {request.model}")
            return None

        provider_type, model_name = parse_model_string(request.model)

        # Get model info from registry if not provided
        if not registry_model:
            registry_model = await self._get_registry_model(provider_type, model_name)

        if not registry_model or not registry_model.max_input_tokens:
            # Can't validate without limits
            logger.debug(
                f"No input limits found for {provider_type}:{model_name}, skipping validation"
            )
            return None

        # Estimate input tokens
        estimated_input_tokens = self._estimate_input_tokens(
            request, provider_type, model_name, registry_model
        )

        # Check input limit
        if estimated_input_tokens > registry_model.max_input_tokens:
            return (
                f"Estimated input tokens ({estimated_input_tokens}) exceeds "
                f"model input limit ({registry_model.max_input_tokens})"
            )

        # Check output limit if specified
        if request.max_tokens and registry_model.max_output_tokens:
            if request.max_tokens > registry_model.max_output_tokens:
                return (
                    f"Requested max tokens ({request.max_tokens}) exceeds "
                    f"model output limit ({registry_model.max_output_tokens})"
                )

        return None

    def _estimate_input_tokens(
        self,
        request: LLMRequest,
        provider_type: str,
        model_name: str,
        registry_model: RegistryModel,
    ) -> int:
        """
        Estimate input token count for a request.

        Uses a two-stage approach:
        1. Quick character-based check for obviously too-large inputs
        2. Proper tokenization for inputs that might fit

        Args:
            request: The LLM request
            provider_type: Provider type (e.g., "openai")
            model_name: Model name (e.g., "gpt-4")
            registry_model: Registry model with limits

        Returns:
            Estimated token count
        """
        # First do a quick character-based check
        total_chars = sum(
            (
                len(message.content)
                if isinstance(message.content, str)
                else len(str(message.content))
            )
            for message in request.messages
        )

        # If we have way more characters than could possibly fit
        # (assuming worst case 1 char = 1 token), skip expensive tokenization
        if total_chars > registry_model.max_input_tokens * 2:
            logger.debug(
                f"Character count ({total_chars}) far exceeds limit, skipping tokenization"
            )
            return total_chars

        # Use proper tokenization for more accurate estimate
        try:
            from llmring.token_counter import count_tokens

            # Convert messages to dict format for token counting
            message_dicts = []
            for message in request.messages:
                msg_dict = {"role": message.role}
                if isinstance(message.content, str):
                    msg_dict["content"] = message.content
                elif isinstance(message.content, list):
                    msg_dict["content"] = message.content
                else:
                    msg_dict["content"] = str(message.content)
                message_dicts.append(msg_dict)

            token_count = count_tokens(message_dicts, provider_type, model_name)
            logger.debug(f"Estimated {token_count} tokens for input")
            return token_count

        except Exception as e:
            # Fall back to character count if tokenization fails
            logger.warning(f"Token counting failed, using character count: {e}")
            return total_chars

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

    async def validate_model_capabilities(
        self, request: LLMRequest, registry_model: Optional[RegistryModel] = None
    ) -> Optional[str]:
        """
        Validate that the model supports the requested features.

        Checks:
        - Vision support if images are present
        - Function calling support if tools are present
        - JSON mode support if JSON response requested

        Args:
            request: The LLM request to validate
            registry_model: Optional pre-fetched registry model

        Returns:
            Error message if validation fails, None if validation passes
        """
        if not request.model:
            return None

        if ":" not in request.model:
            return None

        provider_type, model_name = parse_model_string(request.model)

        # Get model info from registry if not provided
        if not registry_model:
            registry_model = await self._get_registry_model(provider_type, model_name)

        if not registry_model:
            # Can't validate without registry info
            return None

        # Check vision support
        has_images = any(
            isinstance(msg.content, list)
            and any(
                isinstance(part, dict) and part.get("type") in ["image", "image_url"]
                for part in msg.content
            )
            for msg in request.messages
        )
        if has_images and not registry_model.supports_vision:
            return f"Model {provider_type}:{model_name} does not support vision/images"

        # Check function calling support
        if request.tools and not registry_model.supports_function_calling:
            return f"Model {provider_type}:{model_name} does not support function calling"

        # Check JSON mode support
        if request.json_response and not registry_model.supports_json_mode:
            return f"Model {provider_type}:{model_name} does not support JSON mode"

        return None
