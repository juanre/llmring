"""
LLM service that manages providers and routes requests.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from llmring.base import BaseLLMProvider
from llmring.constants import LOCKFILE_NAME
from llmring.exceptions import ProviderNotFoundError
from llmring.lockfile_core import Lockfile
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.receipts import Receipt, ReceiptGenerator
from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMRequest, LLMResponse, StreamChunk
from llmring.services.alias_resolver import AliasResolver
from llmring.services.cost_calculator import CostCalculator
from llmring.services.receipt_manager import ReceiptManager
from llmring.services.schema_adapter import SchemaAdapter
from llmring.services.validation_service import ValidationService
from llmring.utils import parse_model_string
from llmring.validation import InputValidator

logger = logging.getLogger(__name__)


class LLMRing:
    """LLM service that manages providers and routes requests."""

    def __init__(
        self,
        origin: str = "llmring",
        registry_url: Optional[str] = None,
        lockfile_path: Optional[str] = None,
        alias_cache_size: int = 100,
        alias_cache_ttl: int = 3600,
    ):
        """
        Initialize the LLM service.

        Args:
            origin: Origin identifier for tracking
            registry_url: Optional custom registry URL
            lockfile_path: Optional path to lockfile
            alias_cache_size: Maximum number of cached alias resolutions (default: 100)
            alias_cache_ttl: TTL for alias cache entries in seconds (default: 3600)
        """
        self.origin = origin
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self.registry = RegistryClient(registry_url=registry_url)
        self._registry_models: Dict[str, List[RegistryModel]] = {}
        # O(1) alias lookup: provider -> alias -> concrete model name
        self._alias_to_model: Dict[str, Dict[str, str]] = {}

        # Alias resolution service (will be initialized after providers)
        self._alias_resolver: Optional[AliasResolver] = None
        self._alias_cache_size = alias_cache_size
        self._alias_cache_ttl = alias_cache_ttl

        # Schema adapter service
        self._schema_adapter = SchemaAdapter()

        # Cost calculator service
        self._cost_calculator = CostCalculator(self.registry)

        # Receipt manager service (will be initialized with lockfile)
        self._receipt_manager: Optional[ReceiptManager] = None

        # Validation service
        self._validation_service = ValidationService(self.registry)

        # Load lockfile with explicit resolution strategy
        self.lockfile: Optional[Lockfile] = None
        self.lockfile_path: Optional[Path] = None  # Remember where lockfile was loaded from

        # Resolution order:
        # 1. Explicit path parameter
        # 2. Environment variable
        # 3. Current working directory
        # 4. Package's bundled lockfile (fallback)

        if lockfile_path:
            # Explicit path provided - must exist
            self.lockfile_path = Path(lockfile_path)
            if not self.lockfile_path.exists():
                raise FileNotFoundError(f"Specified lockfile not found: {self.lockfile_path}")
            self.lockfile = Lockfile.load(self.lockfile_path)
        elif env_path := os.getenv("LLMRING_LOCKFILE_PATH"):
            # Environment variable - must exist
            self.lockfile_path = Path(env_path)
            if not self.lockfile_path.exists():
                raise FileNotFoundError(f"Lockfile from env var not found: {self.lockfile_path}")
            self.lockfile = Lockfile.load(self.lockfile_path)
        elif Path(LOCKFILE_NAME).exists():
            # Current directory
            self.lockfile_path = Path(LOCKFILE_NAME).resolve()
            self.lockfile = Lockfile.load(self.lockfile_path)
        else:
            # Fallback to package's bundled lockfile
            try:
                self.lockfile = Lockfile.load_package_lockfile()
                self.lockfile_path = Lockfile.get_package_lockfile_path()
                logger.info(f"Using bundled lockfile from package: {self.lockfile_path}")
            except Exception as e:
                logger.warning(f"Could not load any lockfile: {e}")
                # Continue without lockfile - some operations may fail

        # Initialize receipt manager with lockfile
        self._receipt_manager = ReceiptManager(self.lockfile)

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
        google_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GEMINI_API_KEY")
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

        logger.info(f"Initialized {len(self.providers)} providers: {list(self.providers.keys())}")

        # Initialize alias resolver with available providers
        self._alias_resolver = AliasResolver(
            lockfile=self.lockfile,
            available_providers=set(self.providers.keys()),
            cache_size=self._alias_cache_size,
            cache_ttl=self._alias_cache_ttl,
        )

    def register_provider(self, provider_type: str, **kwargs):
        """
        Register a provider instance.

        Args:
            provider_type: Type of provider (anthropic, openai, google, ollama)
            **kwargs: Provider-specific configuration
        """
        # Create provider instance
        if provider_type == "anthropic":
            provider = AnthropicProvider(**kwargs)
        elif provider_type == "openai":
            provider = OpenAIProvider(**kwargs)
        elif provider_type == "google":
            provider = GoogleProvider(**kwargs)
        elif provider_type == "ollama":
            provider = OllamaProvider(**kwargs)
        else:
            raise ProviderNotFoundError(f"Unknown provider type: {provider_type}")

        # Set the registry client to use the same one as the service
        if hasattr(provider, "_registry_client"):
            provider._registry_client = self.registry

        self.providers[provider_type] = provider

        # Update alias resolver with new provider set
        if self._alias_resolver:
            self._alias_resolver.update_available_providers(set(self.providers.keys()))

    async def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from configured providers via the registry.

        Returns:
            Dictionary mapping provider names to lists of model names

        Example:
            >>> models = await ring.get_available_models()
            >>> print(models["anthropic"])
            ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', ...]
        """
        models = {}
        for provider_name in self.providers.keys():
            try:
                # Fetch models from registry for this provider
                registry_models = await self.registry.fetch_current_models(provider_name)

                # Extract model names from registry models
                models[provider_name] = [
                    model.model_name
                    for model in registry_models
                    if model.is_active  # Only include active models
                ]
            except Exception as e:
                logger.debug(f"Could not fetch models for {provider_name} from registry: {e}")
                models[provider_name] = []
        return models

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
            raise ProviderNotFoundError(
                f"Provider '{provider_type}' not found. Available providers: {list(self.providers.keys())}"
            )
        return self.providers[provider_type]

    @property
    def receipts(self) -> List[Receipt]:
        """Get all stored receipts (for backward compatibility)."""
        return self._receipt_manager.receipts if self._receipt_manager else []

    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """
        Parse a model string into provider and model name.

        Args:
            model: Must be in provider:model format (e.g., "anthropic:claude-3-opus")

        Returns:
            Tuple of (provider_type, model_name)

        Raises:
            ValueError: If model string is not in provider:model format
        """
        return parse_model_string(model)

    def resolve_alias(self, alias_or_model: str, profile: Optional[str] = None) -> str:
        """
        Resolve an alias to a model string, or return the input if it's already a model.

        Args:
            alias_or_model: Either an alias or a model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Resolved model string (provider:model) - first available from fallback list
        """
        if not self._alias_resolver:
            raise RuntimeError("Alias resolver not initialized")
        return self._alias_resolver.resolve(alias_or_model, profile)

    def clear_alias_cache(self):
        """Clear the alias resolution cache."""
        if self._alias_resolver:
            self._alias_resolver.clear_cache()

    async def chat(self, request: LLMRequest, profile: Optional[str] = None) -> LLMResponse:
        """
        Send a chat request to the appropriate provider.

        Args:
            request: LLM request with messages and parameters
            profile: Optional profile name for alias resolution

        Returns:
            LLM response with complete generated content

        Raises:
            ValueError: If message content is invalid or too large

        Note:
            For streaming responses, use chat_stream() instead.
        """
        # Validate message content
        messages_dict = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in request.messages
        ]
        InputValidator.validate_message_content(messages_dict)

        # Store original alias for receipt
        original_alias = request.model or ""

        # Resolve alias if needed
        resolved_model = self.resolve_alias(request.model or "", profile)

        # Parse model to get provider
        provider_type, model_name = self._parse_model_string(resolved_model)

        # Get provider
        provider = self.get_provider(provider_type)

        # Check if we should use pinned registry version
        if self.lockfile and profile:
            profile_config = self.lockfile.get_profile(profile)
            if provider_type in profile_config.registry_versions:
                pinned_version = profile_config.registry_versions[provider_type]
                # Set the pinned version on the provider's registry client
                if hasattr(provider, "_registry_client") and provider._registry_client:
                    # Store the pinned version for this validation
                    provider._registry_client._pinned_version = pinned_version

        # Get model info from registry (cached)
        registry_model = None
        try:
            registry_model = await self.get_model_from_registry(provider_type, model_name)
            if not registry_model:
                logger.warning(
                    f"Model '{provider_type}:{model_name}' not found in registry. "
                    f"Cost tracking and token limits unavailable."
                )
        except Exception as e:
            logger.debug(f"Could not check registry for model {provider_type}:{model_name}: {e}")

        # If no model specified, use provider's default
        if not model_name and hasattr(provider, "get_default_model"):
            model_name = await provider.get_default_model()

        # Validate context limits if possible
        # Create a temporary request with the resolved model for validation
        validation_request = request.model_copy()
        validation_request.model = f"{provider_type}:{model_name}"
        validation_error = await self.validate_context_limit(validation_request)
        if validation_error:
            logger.warning(f"Context validation warning: {validation_error}")
            # We log but don't block - let the provider handle it

        # Apply structured output adapter for non-OpenAI providers
        adapted_request = await self._apply_structured_output_adapter(
            request, provider_type, provider
        )

        # Filter out unsupported parameters based on model capabilities
        if registry_model:
            if not registry_model.supports_temperature and adapted_request.temperature is not None:
                logger.debug(
                    f"Model {provider_type}:{model_name} doesn't support temperature, removing parameter"
                )
                adapted_request.temperature = None

            # Could add more capability checks here in the future (streaming, etc.)

        # Send non-streaming request to provider
        response = await provider.chat(
            messages=adapted_request.messages,
            model=model_name,
            temperature=adapted_request.temperature,
            max_tokens=adapted_request.max_tokens,
            response_format=adapted_request.response_format,
            tools=adapted_request.tools,
            tool_choice=adapted_request.tool_choice,
            json_response=adapted_request.json_response,
            cache=adapted_request.cache,
            extra_params=adapted_request.extra_params,
        )

        # Post-process structured output if adapter was used
        response = await self._schema_adapter.post_process_structured_output(
            response, adapted_request, provider_type
        )

        # Ensure response has the full provider:model format
        if response.model and ":" not in response.model:
            response.model = f"{provider_type}:{response.model}"

        # Calculate and add cost information if available
        cost_info = None
        if response.usage:
            cost_info = await self.calculate_cost(response)
            if cost_info:
                self._cost_calculator.add_cost_to_response(response, cost_info)
                logger.debug(
                    f"Calculated cost for {provider_type}:{model_name}: ${cost_info['total_cost']:.6f}"
                )

        # Generate receipt if we have usage information
        if self._receipt_manager:
            self._receipt_manager.generate_receipt(
                response=response,
                original_alias=original_alias,
                provider_type=provider_type,
                model_name=model_name,
                cost_info=cost_info,
                profile=profile,
            )

        return response

    async def chat_stream(
        self, request: LLMRequest, profile: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a streaming chat request to the appropriate provider.

        Args:
            request: LLM request with messages and parameters
            profile: Optional profile name for alias resolution

        Yields:
            Stream chunks from the provider

        Raises:
            ValueError: If message content is invalid or too large

        Example:
            >>> async for chunk in llmring.chat_stream(request):
            ...     print(chunk.delta, end="", flush=True)
        """
        # Validate message content
        messages_dict = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in request.messages
        ]
        InputValidator.validate_message_content(messages_dict)

        # Store original alias for receipt
        original_alias = request.model or ""

        # Resolve alias if needed
        resolved_model = self.resolve_alias(request.model or "", profile)

        # Parse model to get provider
        provider_type, model_name = self._parse_model_string(resolved_model)

        # Get provider
        provider = self.get_provider(provider_type)

        # Check if we should use pinned registry version
        if self.lockfile and profile:
            profile_config = self.lockfile.get_profile(profile)
            if provider_type in profile_config.registry_versions:
                pinned_version = profile_config.registry_versions[provider_type]
                # Set the pinned version on the provider's registry client
                if hasattr(provider, "_registry_client") and provider._registry_client:
                    # Store the pinned version for this validation
                    provider._registry_client._pinned_version = pinned_version

        # Get model info from registry (cached)
        registry_model = None
        try:
            registry_model = await self.get_model_from_registry(provider_type, model_name)
            if not registry_model:
                logger.warning(
                    f"Model '{provider_type}:{model_name}' not found in registry. "
                    f"Cost tracking and token limits unavailable."
                )
        except Exception as e:
            logger.debug(f"Could not check registry for model {provider_type}:{model_name}: {e}")

        # If no model specified, use provider's default
        if not model_name and hasattr(provider, "get_default_model"):
            model_name = await provider.get_default_model()

        # Validate context limits if possible
        # Create a temporary request with the resolved model for validation
        validation_request = request.model_copy()
        validation_request.model = f"{provider_type}:{model_name}"
        validation_error = await self.validate_context_limit(validation_request)
        if validation_error:
            logger.warning(f"Context validation warning: {validation_error}")
            # We log but don't block - let the provider handle it

        # Apply structured output adapter for non-OpenAI providers
        adapted_request = await self._apply_structured_output_adapter(
            request, provider_type, provider
        )

        # Filter out unsupported parameters based on model capabilities
        if registry_model:
            if not registry_model.supports_temperature and adapted_request.temperature is not None:
                logger.debug(
                    f"Model {provider_type}:{model_name} doesn't support temperature, removing parameter"
                )
                adapted_request.temperature = None

            # Could add more capability checks here in the future

        # Get the stream from provider
        stream = await provider.chat_stream(
            messages=adapted_request.messages,
            model=model_name,
            temperature=adapted_request.temperature,
            max_tokens=adapted_request.max_tokens,
            response_format=adapted_request.response_format,
            tools=adapted_request.tools,
            tool_choice=adapted_request.tool_choice,
            json_response=adapted_request.json_response,
            cache=adapted_request.cache,
            extra_params=adapted_request.extra_params,
        )

        # Track usage for receipt generation
        accumulated_usage = None

        # Stream chunks to client
        async for chunk in stream:
            # If this chunk has usage info, store it
            if chunk.usage:
                accumulated_usage = chunk.usage

            # Ensure chunk has the full provider:model format
            if chunk.model and ":" not in chunk.model:
                chunk.model = f"{provider_type}:{chunk.model}"

            # Yield the chunk to client
            yield chunk

        # After streaming completes, generate receipt if we have usage
        if accumulated_usage and self._receipt_manager:
            # Calculate cost if possible
            cost_info = None
            if accumulated_usage:
                # Create a temporary response object for cost calculation
                temp_response = LLMResponse(
                    content="",
                    model=f"{provider_type}:{model_name}",
                    usage=accumulated_usage,
                    finish_reason="stop",
                )
                cost_info = await self.calculate_cost(temp_response)

            # Generate streaming receipt
            self._receipt_manager.generate_streaming_receipt(
                usage=accumulated_usage,
                original_alias=original_alias,
                provider_type=provider_type,
                model_name=model_name,
                cost_info=cost_info,
                profile=profile,
            )

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
            **kwargs,
        )

        if stream:
            return self.chat_stream(request, profile=profile)
        else:
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
        # Save to the original path if we have one, otherwise use default
        self.lockfile.save(self.lockfile_path)
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
            from llmring.exceptions import LockfileNotFoundError

            raise LockfileNotFoundError("No lockfile found")

        profile_config = self.lockfile.get_profile(profile)
        if profile_config.remove_binding(alias):
            # Save to the original path if we have one, otherwise use default
            self.lockfile.save(self.lockfile_path)
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

        lockfile_path = Path(LOCKFILE_NAME)

        if lockfile_path.exists() and not force:
            raise FileExistsError("Lockfile already exists. Use force=True to overwrite.")

        self.lockfile = Lockfile.create_default()
        self.lockfile.save(lockfile_path)
        logger.info(f"Created lockfile at {lockfile_path}")

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model: Model alias (e.g., "fast", "balanced") or provider:model string (e.g., "openai:gpt-4")

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
        # Since we removed validation gatekeeping, all models are "supported"
        # (the provider will fail naturally if it doesn't support the model)
        model_info = {
            "provider": provider_type,
            "model": model_name,
            "supported": True,  # No gatekeeping - providers decide at runtime
        }

        # Add default model info if available
        if hasattr(provider, "get_default_model"):
            try:
                default_model = await provider.get_default_model()
                model_info["is_default"] = model_name == default_model
            except Exception:
                # Registry might be unavailable - that's OK
                model_info["is_default"] = False

        # Cache and return
        self._model_cache[cache_key] = model_info
        return model_info

    async def _apply_structured_output_adapter(
        self, request: LLMRequest, provider_type: str, provider: BaseLLMProvider
    ) -> LLMRequest:
        """
        Apply structured output adapter for non-OpenAI providers.

        Delegates to SchemaAdapter service.
        """
        return await self._schema_adapter.apply_structured_output_adapter(
            request, provider_type, provider
        )

    async def get_model_from_registry(
        self, provider: str, model_name: str
    ) -> Optional[RegistryModel]:
        """
        Get model information from the registry.

        This method resolves aliases to concrete model names and returns the registry
        information for the concrete model.

        Args:
            provider: Provider name
            model_name: Model name (can be an alias or concrete name)

        Returns:
            Registry model information or None if not found
        """
        # Fetch from registry and build alias lookup if not cached
        if provider not in self._registry_models:
            try:
                models = await self.registry.fetch_current_models(provider)
                self._registry_models[provider] = models

                # Build O(1) alias lookup for this provider
                self._build_alias_lookup(provider, models)
            except Exception as e:
                logger.warning(f"Failed to fetch registry for {provider}: {e}")
                return None

        # First check if this is an alias (O(1) lookup)
        if provider in self._alias_to_model and model_name in self._alias_to_model[provider]:
            # Resolve alias to concrete model name
            concrete_name = self._alias_to_model[provider][model_name]
            logger.debug(f"Resolved alias '{model_name}' to concrete model '{concrete_name}'")
            model_name = concrete_name

        # Now find the concrete model
        for model in self._registry_models.get(provider, []):
            if model.model_name == model_name:
                return model

        return None

    def _build_alias_lookup(self, provider: str, models: List[RegistryModel]) -> None:
        """
        Build O(1) alias lookup dictionary for a provider.

        When multiple models have the same alias, the most recent (lexicographically
        largest) model name is chosen.

        Args:
            provider: Provider name
            models: List of registry models
        """
        # Only rebuild if not already cached
        if provider in self._alias_to_model:
            logger.debug(f"Alias lookup already cached for {provider}")
            return

        self._alias_to_model[provider] = {}
        alias_map = self._alias_to_model[provider]

        for model in models:
            if not model.is_active:
                continue

            # Process aliases if they exist
            if hasattr(model, "model_aliases") and model.model_aliases:
                aliases = model.model_aliases
                if not isinstance(aliases, list):
                    aliases = [aliases]

                for alias in aliases:
                    if alias:  # Skip empty aliases
                        # If alias already exists, keep the more recent (larger) model name
                        if alias in alias_map:
                            existing = alias_map[alias]
                            # Choose lexicographically larger (more recent) model
                            if model.model_name > existing:
                                logger.debug(
                                    f"Alias '{alias}' conflict: choosing '{model.model_name}' "
                                    f"over '{existing}' (more recent)"
                                )
                                alias_map[alias] = model.model_name
                        else:
                            alias_map[alias] = model.model_name
                            logger.debug(f"Mapped alias '{alias}' -> '{model.model_name}'")

    def clear_alias_cache(self, provider: Optional[str] = None) -> None:
        """
        Clear the alias cache for a provider or all providers.

        Args:
            provider: Provider name to clear, or None to clear all
        """
        if provider:
            if provider in self._alias_to_model:
                del self._alias_to_model[provider]
                logger.info(f"Cleared alias cache for {provider}")
        else:
            self._alias_to_model.clear()
            logger.info("Cleared all alias caches")

    async def validate_context_limit(self, request: LLMRequest) -> Optional[str]:
        """
        Validate that the request doesn't exceed model context limits.

        Args:
            request: The LLM request

        Returns:
            Error message if validation fails, None if ok
        """
        # Get registry model for better performance (avoid double fetch)
        if request.model and ":" in request.model:
            provider_type, model_name = parse_model_string(request.model)
            registry_model = await self.get_model_from_registry(provider_type, model_name)
        else:
            registry_model = None

        return await self._validation_service.validate_context_limit(request, registry_model)

    async def calculate_cost(self, response: "LLMResponse") -> Optional[Dict[str, float]]:
        """
        Calculate the cost of an API call from the response.

        Args:
            response: LLMResponse object with model and usage information

        Returns:
            Cost breakdown or None if pricing not available

        Example:
            response = await ring.chat("fast", messages)  # Use alias instead of direct model
            cost = await ring.calculate_cost(response)
            print(f"Total cost: ${cost['total_cost']:.4f}")
        """
        # Get registry model for better performance (avoid double fetch)
        if ":" in response.model:
            provider, model_name = parse_model_string(response.model)
            registry_model = await self.get_model_from_registry(provider, model_name)
        else:
            registry_model = None

        return await self._cost_calculator.calculate_cost(response, registry_model)

    async def get_enhanced_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get enhanced model information including registry data.

        Args:
            model: Model alias (e.g., "fast", "balanced") or provider:model string (e.g., "openai:gpt-4")

        Returns:
            Enhanced model information dictionary
        """
        provider_type, model_name = self._parse_model_string(model)

        # Get basic info
        model_info = await self.get_model_info(model)

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
        # Close all providers to clean up httpx clients
        for provider in self.providers.values():
            if hasattr(provider, "aclose"):
                await provider.aclose()

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        await self.close()
