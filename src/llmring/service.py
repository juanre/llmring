"""
LLM service that manages providers and routes requests.
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from llmring.base import BaseLLMProvider
from llmring.lockfile import Lockfile
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.receipts import Receipt, ReceiptGenerator
from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMRequest, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)


class LLMRing:
    """LLM service that manages providers and routes requests."""

    def __init__(
        self,
        origin: str = "llmring",
        registry_url: Optional[str] = None,
        lockfile_path: Optional[str] = None,
    ):
        """
        Initialize the LLM service.

        Args:
            origin: Origin identifier for tracking
            registry_url: Optional custom registry URL
            lockfile_path: Optional path to lockfile
        """
        self.origin = origin
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self.registry = RegistryClient(registry_url=registry_url)
        self._registry_models: Dict[str, List[RegistryModel]] = {}

        # Initialize receipt generator (no signer for local mode)
        self.receipt_generator: Optional[ReceiptGenerator] = None
        self.receipts: List[Receipt] = []  # Store receipts locally for now

        # Load lockfile if available
        self.lockfile: Optional[Lockfile] = None
        if lockfile_path:
            from pathlib import Path

            self.lockfile = Lockfile.load(Path(lockfile_path))
        else:
            # Try to find lockfile in current directory or parents
            lockfile_path = Lockfile.find_lockfile()
            if lockfile_path:
                self.lockfile = Lockfile.load(lockfile_path)

        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all configured providers from environment variables."""
        logger.info("Initializing LLM providers")

        # Initialize Anthropic provider if API key is available
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.register_provider("anthropic", api_key=anthropic_key)
                logger.info("Successfully initialized Anthropic provider")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {e}")

        # Initialize OpenAI provider if API key is available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                self.register_provider("openai", api_key=openai_key)
                logger.info("Successfully initialized OpenAI provider")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")

        # Initialize Google provider if API key is available
        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if google_key:
            try:
                self.register_provider("google", api_key=google_key)
                logger.info("Successfully initialized Google provider")
            except Exception as e:
                logger.error(f"Failed to initialize Google provider: {e}")

        # Initialize Ollama provider (no API key required)
        try:
            self.register_provider("ollama")
            logger.info("Successfully initialized Ollama provider")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")

        logger.info(
            f"Initialized {len(self.providers)} providers: {list(self.providers.keys())}"
        )

    def register_provider(self, provider_type: str, **kwargs):
        """
        Register a provider instance.

        Args:
            provider_type: Type of provider (anthropic, openai, google, ollama)
            **kwargs: Provider-specific configuration
        """
        if provider_type == "anthropic":
            self.providers[provider_type] = AnthropicProvider(**kwargs)
        elif provider_type == "openai":
            self.providers[provider_type] = OpenAIProvider(**kwargs)
        elif provider_type == "google":
            self.providers[provider_type] = GoogleProvider(**kwargs)
        elif provider_type == "ollama":
            self.providers[provider_type] = OllamaProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def get_provider(self, provider_type: str) -> BaseLLMProvider:
        """
        Get a provider instance.

        Args:
            provider_type: Type of provider

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not found
        """
        if provider_type not in self.providers:
            raise ValueError(
                f"Provider '{provider_type}' not found. Available providers: {list(self.providers.keys())}"
            )
        return self.providers[provider_type]

    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """
        Parse a model string into provider and model name.

        Args:
            model: Model string (e.g., "anthropic:claude-3-opus-20240229" or just "gpt-4")

        Returns:
            Tuple of (provider_type, model_name)
        """
        if ":" in model:
            provider_type, model_name = model.split(":", 1)
            return provider_type, model_name
        else:
            # Try to infer provider from model name
            if model.startswith("gpt"):
                return "openai", model
            elif model.startswith("claude"):
                return "anthropic", model
            elif model.startswith("gemini"):
                return "google", model
            else:
                # Default to first available provider
                if self.providers:
                    return list(self.providers.keys())[0], model
                else:
                    raise ValueError("No providers available")

    def resolve_alias(self, alias_or_model: str, profile: Optional[str] = None) -> str:
        """
        Resolve an alias to a model string, or return the input if it's already a model.

        Args:
            alias_or_model: Either an alias or a model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Resolved model string (provider:model)
        """
        # If it looks like a model reference (contains colon), return as-is
        if ":" in alias_or_model:
            return alias_or_model

        # Try to resolve as alias from lockfile
        if self.lockfile:
            profile_name = profile or os.getenv("LLMRING_PROFILE")
            resolved = self.lockfile.resolve_alias(alias_or_model, profile_name)
            if resolved:
                logger.debug(f"Resolved alias '{alias_or_model}' to '{resolved}'")
                return resolved

        # If no lockfile or alias not found, assume it's a model name
        # and try to infer provider (backwards compatibility)
        logger.debug(
            f"Could not resolve alias '{alias_or_model}', treating as model name"
        )
        return alias_or_model

    async def chat(
        self, request: LLMRequest, profile: Optional[str] = None
    ) -> Union[LLMResponse, AsyncIterator[StreamChunk]]:
        """
        Send a chat request to the appropriate provider.

        Args:
            request: LLM request with messages and parameters
            profile: Optional profile name for alias resolution

        Returns:
            LLM response or async iterator of stream chunks if streaming
        """
        # Store original alias for receipt
        original_alias = request.model or ""

        # Resolve alias if needed
        resolved_model = self.resolve_alias(request.model or "", profile)

        # Parse model to get provider
        provider_type, model_name = self._parse_model_string(resolved_model)

        # Get provider
        provider = self.get_provider(provider_type)

        # Validate model if provider supports it
        if hasattr(provider, "validate_model") and not provider.validate_model(
            model_name
        ):
            raise ValueError(
                f"Model '{model_name}' not supported by {provider_type} provider"
            )

        # If no model specified, use provider's default
        if not model_name and hasattr(provider, "get_default_model"):
            model_name = provider.get_default_model()

        # Validate context limits if possible
        validation_error = await self.validate_context_limit(request)
        if validation_error:
            logger.warning(f"Context validation warning: {validation_error}")
            # We log but don't block - let the provider handle it

        # Check if streaming is requested
        if request.stream:
            # For streaming, we need to wrap the stream to handle receipts
            return self._create_streaming_wrapper(
                provider=provider,
                model_name=model_name,
                request=request,
                provider_type=provider_type,
                original_alias=original_alias,
                profile=profile
            )
        
        # Send non-streaming request to provider
        response = await provider.chat(
            messages=request.messages,
            model=model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream=False,
        )

        # Calculate and add cost information if available
        if response.usage:
            cost_info = await self.calculate_cost(response)
            if cost_info:
                # Add cost to usage dict
                response.usage["cost"] = cost_info["total_cost"]
                response.usage["cost_breakdown"] = {
                    "input": cost_info["input_cost"],
                    "output": cost_info["output_cost"],
                }
                logger.debug(
                    f"Calculated cost for {provider_type}:{model_name}: ${cost_info['total_cost']:.6f}"
                )

        # Generate receipt if we have usage information
        if response.usage and self.lockfile:
            try:
                # Initialize receipt generator if not already done
                if not self.receipt_generator:
                    self.receipt_generator = ReceiptGenerator()

                # Calculate lockfile digest
                lock_digest = self.lockfile.calculate_digest()

                # Determine profile used
                profile_name = (
                    profile
                    or os.getenv("LLMRING_PROFILE")
                    or self.lockfile.default_profile
                )

                # Generate receipt
                receipt = self.receipt_generator.generate_receipt(
                    alias=(
                        original_alias if ":" not in original_alias else "direct_model"
                    ),
                    profile=profile_name,
                    lock_digest=lock_digest,
                    provider=provider_type,
                    model=model_name,
                    usage=response.usage,
                    costs=(
                        cost_info
                        if cost_info
                        else {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
                    ),
                )

                # Store receipt locally
                self.receipts.append(receipt)
                logger.debug(
                    f"Generated receipt {receipt.receipt_id} for {provider_type}:{model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate receipt: {e}")

        return response

    async def _create_streaming_wrapper(
        self,
        provider: BaseLLMProvider,
        model_name: str,
        request: LLMRequest,
        provider_type: str,
        original_alias: str,
        profile: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Create a streaming wrapper that handles receipts and cost calculation.
        
        Args:
            provider: The provider instance
            model_name: The model name
            request: The original request
            provider_type: Type of provider (openai, anthropic, etc.)
            original_alias: Original alias used in request
            profile: Optional profile name
            
        Yields:
            Stream chunks from the provider with receipt handling
        """
        # Get the stream from provider
        stream = await provider.chat(
            messages=request.messages,
            model=model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream=True,
        )
        
        # Track usage for receipt generation
        accumulated_usage = None
        
        # Stream chunks to client
        async for chunk in stream:
            # If this chunk has usage info, store it
            if chunk.usage:
                accumulated_usage = chunk.usage
                
            # Yield the chunk to client
            yield chunk
        
        # After streaming completes, generate receipt if we have usage
        if accumulated_usage and self.lockfile:
            try:
                # Calculate cost if possible
                cost_info = None
                if accumulated_usage:
                    # Create a temporary response object for cost calculation
                    temp_response = LLMResponse(
                        content="",
                        model=f"{provider_type}:{model_name}",
                        usage=accumulated_usage,
                        finish_reason="stop"
                    )
                    cost_info = await self.calculate_cost(temp_response)
                
                # Initialize receipt generator if needed
                if not self.receipt_generator:
                    self.receipt_generator = ReceiptGenerator()
                
                # Calculate lockfile digest
                lock_digest = self.lockfile.calculate_digest()
                
                # Determine profile used
                profile_name = (
                    profile
                    or os.getenv("LLMRING_PROFILE")
                    or self.lockfile.default_profile
                )
                
                # Generate receipt
                receipt = self.receipt_generator.generate_receipt(
                    alias=(
                        original_alias if ":" not in original_alias else "direct_model"
                    ),
                    profile=profile_name,
                    lock_digest=lock_digest,
                    provider=provider_type,
                    model=model_name,
                    usage=accumulated_usage,
                    costs=(
                        cost_info
                        if cost_info
                        else {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
                    ),
                )
                
                # Store receipt locally
                self.receipts.append(receipt)
                logger.debug(
                    f"Generated receipt {receipt.receipt_id} for streaming {provider_type}:{model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate receipt for streaming: {e}")

    async def chat_with_alias(
        self,
        alias_or_model: str,
        messages: List[Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        profile: Optional[str] = None,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> Union[LLMResponse, AsyncIterator[StreamChunk]]:
        """
        Convenience method to chat using an alias or model string.

        Args:
            alias_or_model: Alias name or model string (provider:model)
            messages: List of messages
            temperature: Optional temperature
            max_tokens: Optional max tokens
            profile: Optional profile for alias resolution
            stream: Whether to stream the response
            **kwargs: Additional parameters for the request

        Returns:
            LLM response or async iterator of stream chunks if streaming
        """
        # Resolve alias
        model = self.resolve_alias(alias_or_model, profile)

        # Create request
        from llmring.schemas import LLMRequest

        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        return await self.chat(request, profile=profile)

    # Lockfile management methods

    def bind_alias(self, alias: str, model: str, profile: Optional[str] = None) -> None:
        """
        Bind an alias to a model in the lockfile.

        Args:
            alias: Alias name
            model: Model string (provider:model)
            profile: Optional profile name
        """
        if not self.lockfile:
            # Create a new lockfile if none exists
            self.lockfile = Lockfile.create_default()

        self.lockfile.set_binding(alias, model, profile)
        self.lockfile.save()
        logger.info(
            f"Bound alias '{alias}' to '{model}' in profile '{profile or self.lockfile.default_profile}'"
        )

    def unbind_alias(self, alias: str, profile: Optional[str] = None) -> None:
        """
        Remove an alias binding from the lockfile.

        Args:
            alias: Alias to remove
            profile: Optional profile name
        """
        if not self.lockfile:
            raise ValueError("No lockfile found")

        profile_config = self.lockfile.get_profile(profile)
        if profile_config.remove_binding(alias):
            self.lockfile.save()
            logger.info(
                f"Removed alias '{alias}' from profile '{profile or self.lockfile.default_profile}'"
            )
        else:
            logger.warning(
                f"Alias '{alias}' not found in profile '{profile or self.lockfile.default_profile}'"
            )

    def list_aliases(self, profile: Optional[str] = None) -> Dict[str, str]:
        """
        List all aliases in a profile.

        Args:
            profile: Optional profile name

        Returns:
            Dictionary of alias -> model mappings
        """
        if not self.lockfile:
            return {}

        profile_config = self.lockfile.get_profile(profile)
        return {binding.alias: binding.model_ref for binding in profile_config.bindings}

    def init_lockfile(self, force: bool = False) -> None:
        """
        Initialize a new lockfile with defaults.

        Args:
            force: Overwrite existing lockfile
        """
        from pathlib import Path

        lockfile_path = Path("llmring.lock")

        if lockfile_path.exists() and not force:
            raise FileExistsError(
                "Lockfile already exists. Use force=True to overwrite."
            )

        self.lockfile = Lockfile.create_default()
        self.lockfile.save(lockfile_path)
        logger.info(f"Created lockfile at {lockfile_path}")

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from registered providers.

        Returns:
            Dictionary mapping provider names to their supported models
        """
        models = {}
        for provider_name, provider in self.providers.items():
            if hasattr(provider, "get_supported_models"):
                models[provider_name] = provider.get_supported_models()
            else:
                models[provider_name] = []
        return models

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model: Model string (e.g., "openai:gpt-4")

        Returns:
            Model information dictionary
        """
        provider_type, model_name = self._parse_model_string(model)

        # Check cache first
        cache_key = f"{provider_type}:{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Get provider
        provider = self.get_provider(provider_type)

        # Build model info
        model_info = {
            "provider": provider_type,
            "model": model_name,
            "supported": hasattr(provider, "validate_model")
            and provider.validate_model(model_name),
        }

        # Add default model info if available
        if hasattr(provider, "get_default_model"):
            model_info["is_default"] = model_name == provider.get_default_model()

        # Cache and return
        self._model_cache[cache_key] = model_info
        return model_info

    async def get_model_from_registry(
        self, provider: str, model_name: str
    ) -> Optional[RegistryModel]:
        """
        Get model information from the registry.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            Registry model information or None if not found
        """
        # Fetch from registry if not cached
        if provider not in self._registry_models:
            try:
                models = await self.registry.fetch_current_models(provider)
                self._registry_models[provider] = models
            except Exception as e:
                logger.warning(f"Failed to fetch registry for {provider}: {e}")
                return None

        # Find the model
        for model in self._registry_models.get(provider, []):
            if model.model_name == model_name:
                return model

        return None

    async def validate_context_limit(self, request: LLMRequest) -> Optional[str]:
        """
        Validate that the request doesn't exceed model context limits.

        Args:
            request: The LLM request

        Returns:
            Error message if validation fails, None if ok
        """
        if not request.model:
            return None
        provider_type, model_name = self._parse_model_string(request.model)

        # Get model info from registry
        registry_model = await self.get_model_from_registry(provider_type, model_name)
        if not registry_model or not registry_model.max_input_tokens:
            # Can't validate without limits
            return None

        # Calculate token count for input using proper tokenization
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
        
        estimated_input_tokens = count_tokens(message_dicts, provider_type, model_name)

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

    async def calculate_cost(
        self, response: "LLMResponse"
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the cost of an API call from the response.

        Args:
            response: LLMResponse object with model and usage information

        Returns:
            Cost breakdown or None if pricing not available

        Example:
            response = await ring.chat("openai:gpt-4o-mini", messages)
            cost = await ring.calculate_cost(response)
            print(f"Total cost: ${cost['total_cost']:.4f}")
        """
        if not response.usage:
            return None

        provider, model_name = self._parse_model_string(response.model)
        usage = response.usage
        registry_model = await self.get_model_from_registry(provider, model_name)
        if not registry_model:
            return None

        if (
            registry_model.dollars_per_million_tokens_input is None
            or registry_model.dollars_per_million_tokens_output is None
        ):
            return None

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        input_cost = (
            prompt_tokens / 1_000_000
        ) * registry_model.dollars_per_million_tokens_input
        output_cost = (
            completion_tokens / 1_000_000
        ) * registry_model.dollars_per_million_tokens_output
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_million_input": registry_model.dollars_per_million_tokens_input,
            "cost_per_million_output": registry_model.dollars_per_million_tokens_output,
        }

    async def get_enhanced_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get enhanced model information including registry data.

        Args:
            model: Model string (e.g., "openai:gpt-4")

        Returns:
            Enhanced model information dictionary
        """
        provider_type, model_name = self._parse_model_string(model)

        # Get basic info
        model_info = self.get_model_info(model)

        # Enhance with registry data
        registry_model = await self.get_model_from_registry(provider_type, model_name)
        if registry_model:
            model_info.update(
                {
                    "display_name": registry_model.display_name,
                    "description": registry_model.description,
                    "max_input_tokens": registry_model.max_input_tokens,
                    "max_output_tokens": registry_model.max_output_tokens,
                    "supports_vision": registry_model.supports_vision,
                    "supports_function_calling": registry_model.supports_function_calling,
                    "supports_json_mode": registry_model.supports_json_mode,
                    "supports_parallel_tool_calls": registry_model.supports_parallel_tool_calls,
                    "dollars_per_million_tokens_input": registry_model.dollars_per_million_tokens_input,
                    "dollars_per_million_tokens_output": registry_model.dollars_per_million_tokens_output,
                    "is_active": registry_model.is_active,
                }
            )

        return model_info

    async def close(self):
        """Clean up resources."""
        # Clear registry cache
        self.registry.clear_cache()
        # Providers don't typically need cleanup, but we keep this for consistency
        pass
