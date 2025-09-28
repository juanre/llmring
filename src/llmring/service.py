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

        # Alias resolution cache
        self._alias_cache: Dict[tuple[str, Optional[str]], tuple[str, float]] = {}
        self._alias_cache_size = alias_cache_size
        self._alias_cache_ttl = alias_cache_ttl

        # Initialize receipt generator (no signer for local mode)
        self.receipt_generator: Optional[ReceiptGenerator] = None
        self.receipts: List[Receipt] = []  # Store receipts locally for now

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

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from configured providers via the registry.

        Returns:
            Dictionary mapping provider names to lists of model names
        """
        models = {}
        for provider_name in self.providers.keys():
            try:
                # Fetch models from registry for this provider
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a task
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, self.registry.fetch_current_models(provider_name)
                        )
                        registry_models = future.result(timeout=5)
                else:
                    # If no loop is running, we can run directly
                    registry_models = asyncio.run(self.registry.fetch_current_models(provider_name))

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
        if ":" not in model:
            raise ValueError(
                f"Invalid model format: '{model}'. "
                f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
                f"If you meant to use an alias, ensure it's defined in your lockfile."
            )

        provider_type, model_name = model.split(":", 1)
        return provider_type, model_name

    def resolve_alias(self, alias_or_model: str, profile: Optional[str] = None) -> str:
        """
        Resolve an alias to a model string, or return the input if it's already a model.

        Args:
            alias_or_model: Either an alias or a model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Resolved model string (provider:model) - first available from fallback list
        """
        # If it looks like a model reference (contains colon), return as-is
        if ":" in alias_or_model:
            return alias_or_model

        # Check cache first
        cache_key = (alias_or_model, profile)
        if cache_key in self._alias_cache:
            cached_value, cached_time = self._alias_cache[cache_key]
            import time

            if time.time() - cached_time < self._alias_cache_ttl:
                logger.debug(
                    f"Using cached resolution for alias '{alias_or_model}': '{cached_value}'"
                )
                return cached_value
            else:
                # Cache entry expired, remove it
                del self._alias_cache[cache_key]

        # Try to resolve as alias from lockfile
        if self.lockfile:
            profile_name = profile or os.getenv("LLMRING_PROFILE")
            model_refs = self.lockfile.resolve_alias(alias_or_model, profile_name)

            if model_refs:
                # Try each model in order until we find one with an available provider
                unavailable_models = []
                for model_ref in model_refs:
                    try:
                        provider_type, _ = self._parse_model_string(model_ref)
                        if provider_type in self.providers:
                            logger.debug(
                                f"Resolved alias '{alias_or_model}' to '{model_ref}' (provider available)"
                            )
                            # Add to cache
                            self._add_to_alias_cache(cache_key, model_ref)
                            return model_ref
                        else:
                            unavailable_models.append(f"{model_ref} (no {provider_type} API key)")
                            logger.debug(
                                f"Skipping '{model_ref}' - provider '{provider_type}' not available"
                            )
                    except ValueError:
                        # Invalid model reference format
                        logger.warning(
                            f"Invalid model reference in alias '{alias_or_model}': {model_ref}"
                        )
                        continue

                # No available providers found
                if unavailable_models:
                    raise ValueError(
                        f"No available providers for alias '{alias_or_model}'. "
                        f"Tried models: {', '.join(unavailable_models)}. "
                        f"Please configure the required API keys."
                    )

        # If no lockfile or alias not found, this is an error
        # We require explicit provider:model format or valid aliases
        raise ValueError(
            f"Invalid model format: '{alias_or_model}'. "
            f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
            f"If you meant to use an alias, ensure it's defined in your lockfile."
        )

    def _add_to_alias_cache(self, cache_key: tuple[str, Optional[str]], value: str):
        """Add an entry to the alias cache, respecting size limits."""
        import time

        # If cache is at capacity, remove oldest entry
        if len(self._alias_cache) >= self._alias_cache_size:
            # Find and remove the oldest entry
            oldest_key = min(self._alias_cache.keys(), key=lambda k: self._alias_cache[k][1])
            del self._alias_cache[oldest_key]

        # Add new entry
        self._alias_cache[cache_key] = (value, time.time())

    def clear_alias_cache(self):
        """Clear the alias resolution cache."""
        self._alias_cache.clear()
        logger.debug("Alias cache cleared")

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

        # Check if streaming is requested
        if adapted_request.stream:
            # For streaming, we need to wrap the stream to handle receipts
            return self._create_streaming_wrapper(
                provider=provider,
                model_name=model_name,
                request=adapted_request,
                provider_type=provider_type,
                original_alias=original_alias,
                profile=profile,
            )

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
            stream=False,
            extra_params=adapted_request.extra_params,
        )

        # Post-process structured output if adapter was used
        response = await self._post_process_structured_output(
            response, adapted_request, provider_type
        )

        # Ensure response has the full provider:model format
        if response.model and ":" not in response.model:
            response.model = f"{provider_type}:{response.model}"

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
                    profile or os.getenv("LLMRING_PROFILE") or self.lockfile.default_profile
                )

                # Generate receipt
                receipt = self.receipt_generator.generate_receipt(
                    alias=(original_alias if ":" not in original_alias else "direct_model"),
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
        profile: Optional[str] = None,
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
            json_response=request.json_response,
            cache=request.cache,
            stream=True,
            extra_params=request.extra_params,
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
                        finish_reason="stop",
                    )
                    cost_info = await self.calculate_cost(temp_response)

                # Initialize receipt generator if needed
                if not self.receipt_generator:
                    self.receipt_generator = ReceiptGenerator()

                # Calculate lockfile digest
                lock_digest = self.lockfile.calculate_digest()

                # Determine profile used
                profile_name = (
                    profile or os.getenv("LLMRING_PROFILE") or self.lockfile.default_profile
                )

                # Generate receipt
                receipt = self.receipt_generator.generate_receipt(
                    alias=(original_alias if ":" not in original_alias else "direct_model"),
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

        Converts json_schema requests to tool-based approaches for Anthropic/Google.
        """
        # Only adapt if we have a json_schema request and no existing tools
        if (
            not request.response_format
            or request.response_format.get("type") != "json_schema"
            or request.tools
        ):
            return request

        schema = request.response_format.get("json_schema", {}).get("schema", {})
        if not schema:
            return request

        # Create a copy of the request to modify
        from copy import deepcopy

        adapted_request = deepcopy(request)

        # Import Message for use in adapter
        from llmring.schemas import Message

        if provider_type == "anthropic":
            # Anthropic: Use tool injection approach
            respond_tool = {
                "type": "function",
                "function": {
                    "name": "respond_with_structure",
                    "description": "Respond with structured data matching the required schema",
                    "parameters": schema,
                },
            }
            adapted_request.tools = [respond_tool]
            adapted_request.tool_choice = {"type": "any"}  # Force tool use

        elif provider_type == "google":
            # Google: Use function declaration approach with JSON Schema normalization
            normalized_schema, notes = self._normalize_json_schema_for_google(schema)

            if notes:
                try:
                    logger.warning(
                        "Normalized JSON Schema for Google; potential downgrades: %s",
                        "; ".join(notes),
                    )
                except Exception:
                    # Avoid failing on logging issues
                    pass

            respond_tool = {
                "type": "function",
                "function": {
                    "name": "respond_with_structure",
                    "description": "Respond with structured data matching the required schema",
                    "parameters": normalized_schema,
                },
            }
            adapted_request.tools = [respond_tool]
            adapted_request.tool_choice = "any"  # Force function calling
            adapted_request.metadata = adapted_request.metadata or {}
            if notes:
                adapted_request.metadata["_schema_normalization_notes"] = notes

        elif provider_type == "ollama":
            # Ollama: Best effort with format and schema hinting
            adapted_request.json_response = True

            # Add schema as system instruction
            schema_instruction = f"\n\nIMPORTANT: Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

            # Add to system message or create one
            messages = list(adapted_request.messages)
            if messages and messages[0].role == "system":
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + schema_instruction,
                    metadata=messages[0].metadata,
                )
            else:
                messages.insert(
                    0,
                    Message(
                        role="system",
                        content=f"You are a helpful assistant.{schema_instruction}",
                    ),
                )
            adapted_request.messages = messages

        # Mark request as adapted for post-processing
        adapted_request.metadata = adapted_request.metadata or {}
        adapted_request.metadata["_structured_output_adapted"] = True
        adapted_request.metadata["_original_schema"] = schema

        return adapted_request

    def _normalize_json_schema_for_google(
        self, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Normalize a JSON Schema into a form acceptable by Google Gemini function
        declarations. Removes/adjusts unsupported features.

        - Converts union types like ["boolean", "null"] to a single type, removing
          "null" and recording a note. If multiple non-null types are present,
          falls back to "string" and records a note.
        - Removes unsupported keywords such as additionalProperties, anyOf, oneOf,
          allOf, not, patternProperties, if/then/else, pattern, format (records notes).
        - Recursively normalizes properties and items.

        Returns a tuple of (normalized_schema, notes).
        """

        notes: List[str] = []

        def normalize(node: Any, path: str) -> Any:
            # Primitives or non-dict structures are returned as-is
            if not isinstance(node, dict):
                return node

            result: Dict[str, Any] = {}

            # Handle type normalization first
            node_type = node.get("type")
            if isinstance(node_type, list):
                # Remove null if present, pick a remaining type
                non_null_types = [t for t in node_type if t != "null"]
                if len(non_null_types) == 1:
                    result["type"] = non_null_types[0]
                    notes.append(f"{path or '<root>'}: removed 'null' from union type {node_type}")
                elif len(non_null_types) == 0:
                    # Only null provided; fallback to string
                    result["type"] = "string"
                    notes.append(
                        f"{path or '<root>'}: union type {node_type} normalized to 'string'"
                    )
                else:
                    # Multiple non-null types unsupported; fallback to string
                    result["type"] = "string"
                    notes.append(
                        f"{path or '<root>'}: multi-type union {node_type} normalized to 'string'"
                    )
            elif isinstance(node_type, str):
                result["type"] = node_type

            # Copy supported basic fields cautiously
            # Preserve description/title/default/enum when present
            for key in [
                "title",
                "description",
                "default",
                "enum",
                "const",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "minItems",
                "maxItems",
            ]:
                if key in node:
                    result[key] = node[key]

            # Remove/ignore unsupported or risky keywords
            removed_keywords = []
            for key in [
                "additionalProperties",
                "anyOf",
                "oneOf",
                "allOf",
                "not",
                "patternProperties",
                "if",
                "then",
                "else",
                "pattern",
                "format",
                "dependencies",
            ]:
                if key in node:
                    removed_keywords.append(key)
            if removed_keywords:
                notes.append(f"{path or '<root>'}: removed unsupported keywords {removed_keywords}")

            # Object handling
            effective_type = result.get("type") or node.get("type")
            if effective_type == "object":
                # Normalize properties
                properties = node.get("properties", {})
                if isinstance(properties, dict):
                    norm_props: Dict[str, Any] = {}
                    for prop_name, prop_schema in properties.items():
                        norm_props[prop_name] = normalize(
                            prop_schema,
                            f"{path + '.' if path else ''}properties.{prop_name}",
                        )
                    result["properties"] = norm_props

                # Keep required list as-is
                if "required" in node and isinstance(node["required"], list):
                    result["required"] = [str(x) for x in node["required"]]

            # Array handling
            if effective_type == "array":
                items = node.get("items")
                if isinstance(items, list) and items:
                    # Tuple typing not supported; choose first
                    result["items"] = normalize(items[0], f"{path or '<root>'}.items[0]")
                    notes.append(
                        f"{path or '<root>'}: tuple-typed 'items' normalized to first schema"
                    )
                elif isinstance(items, dict):
                    result["items"] = normalize(items, f"{path or '<root>'}.items")

            return result

        normalized = normalize(schema, "")
        return normalized, notes

    async def _post_process_structured_output(
        self, response: LLMResponse, request: LLMRequest, provider_type: str
    ) -> LLMResponse:
        """
        Post-process response from structured output adapter.

        Extracts JSON from tool calls and validates against schema.
        """
        import json

        # Only process if request was adapted
        if not request.metadata or not request.metadata.get("_structured_output_adapted"):
            return response

        original_schema = request.metadata.get("_original_schema", {})

        try:
            if provider_type == "openai":
                # OpenAI native: Parse JSON from content
                try:
                    parsed_data = json.loads(response.content)
                    response.parsed = parsed_data

                    # Validate against schema if strict mode
                    if request.response_format and request.response_format.get("strict"):
                        self._validate_json_schema(parsed_data, original_schema)

                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from OpenAI response")

            elif provider_type in ["anthropic", "google"] and response.tool_calls:
                # Extract JSON from tool call arguments
                for tool_call in response.tool_calls:
                    if tool_call["function"]["name"] == "respond_with_structure":
                        # Parse the arguments as our structured response
                        tool_args = tool_call["function"]["arguments"]
                        if isinstance(tool_args, str):
                            parsed_data = json.loads(tool_args)
                        else:
                            parsed_data = tool_args

                        # Set content to JSON string and parsed to dict
                        response.content = json.dumps(parsed_data, indent=2)
                        response.parsed = parsed_data

                        # Validate against schema if strict mode
                        if request.response_format and request.response_format.get("strict"):
                            self._validate_json_schema(parsed_data, original_schema)

                        break

            elif provider_type == "ollama":
                # Try to parse JSON from content
                try:
                    parsed_data = json.loads(response.content)
                    response.parsed = parsed_data

                    # Validate against schema if strict mode
                    if request.response_format and request.response_format.get("strict"):
                        self._validate_json_schema(parsed_data, original_schema)

                except json.JSONDecodeError:
                    # If JSON parsing fails and strict mode, try one repair attempt
                    if (
                        request.response_format
                        and request.response_format.get("strict")
                        and request.extra_params.get("retry_on_json_failure", True)
                    ):
                        logger.info(f"JSON parsing failed for {provider_type}, attempting repair")

                        # Single retry with repair prompt
                        from copy import deepcopy

                        from llmring.schemas import Message

                        repair_request = deepcopy(request)
                        repair_prompt = f"The previous response was not valid JSON. Please provide ONLY valid JSON matching this schema:\n{json.dumps(original_schema, indent=2)}\n\nOriginal content to fix:\n{response.content}"

                        repair_request.messages = [Message(role="user", content=repair_prompt)]
                        repair_request.metadata["_retry_attempt"] = True

                        try:
                            # Get provider and retry (avoid infinite recursion)
                            if not request.metadata.get("_retry_attempt"):
                                provider = self.get_provider(provider_type)
                                repair_response = await provider.chat(
                                    messages=repair_request.messages,
                                    model=(
                                        repair_request.model.split(":", 1)[1]
                                        if ":" in repair_request.model
                                        else repair_request.model
                                    ),
                                    temperature=0.1,  # Lower temperature for better JSON
                                    max_tokens=repair_request.max_tokens,
                                    json_response=True,
                                    extra_params=repair_request.extra_params,
                                )

                                # Try parsing the repaired response
                                repaired_data = json.loads(repair_response.content)
                                response.content = repair_response.content
                                response.parsed = repaired_data
                                self._validate_json_schema(repaired_data, original_schema)
                                logger.info(f"JSON repair successful for {provider_type}")

                        except Exception as repair_error:
                            logger.warning(f"JSON repair attempt failed: {repair_error}")
                    else:
                        logger.warning(f"Failed to parse JSON from {provider_type} response")

        except Exception as e:
            logger.warning(f"Structured output post-processing failed: {e}")

        return response

    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate data against JSON schema.

        Raises ValidationError if data doesn't match schema.
        """
        try:
            import jsonschema

            jsonschema.validate(instance=data, schema=schema)
        except ImportError:
            # If jsonschema not available, skip validation
            logger.warning("jsonschema not installed, skipping schema validation")
        except jsonschema.ValidationError as e:
            from llmring.exceptions import ValidationError

            raise ValidationError(
                f"Response does not match required schema: {e.message}",
                field=e.absolute_path[-1] if e.absolute_path else None,
                value=e.instance,
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
        if not request.model:
            return None
        provider_type, model_name = self._parse_model_string(request.model)

        # Get model info from registry
        registry_model = await self.get_model_from_registry(provider_type, model_name)
        if not registry_model or not registry_model.max_input_tokens:
            # Can't validate without limits
            return None

        # Calculate token count for input using proper tokenization
        # First do a quick character-based check to avoid expensive tokenization for obviously too-large inputs
        total_chars = sum(
            (
                len(message.content)
                if isinstance(message.content, str)
                else len(str(message.content))
            )
            for message in request.messages
        )

        # If we have way more characters than could possibly fit (assuming worst case 1 char = 1 token)
        # Skip expensive tokenization
        if total_chars > registry_model.max_input_tokens * 2:
            estimated_input_tokens = total_chars  # Use char count as rough estimate
        else:
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

        input_cost = (prompt_tokens / 1_000_000) * registry_model.dollars_per_million_tokens_input
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
