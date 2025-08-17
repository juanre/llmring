"""
LLM service that manages providers and routes requests.
"""

import logging
import os
import time
import asyncio
from typing import Any, Dict, List, Optional

from pgdbm import AsyncDatabaseManager
from llmring.api import LLMRingAPI
from llmring.base import BaseLLMProvider
from llmring.db import LLMDatabase
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMRequest, LLMResponse


logger = logging.getLogger(__name__)


class LLMRing:
    """LLM service that manages providers and routes requests."""

    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        db_manager: Optional[AsyncDatabaseManager] = None,  # NEW
        origin: str = "llmring",
        enable_db_logging: bool = True,
    ):
        """
        Initialize the LLM service.

        Args:
            db_connection_string: Connection string (for standalone mode)
            db_manager: External database manager (for integrated mode)
            origin: Origin identifier for database logging
            enable_db_logging: Whether to enable database logging
        """
        self.origin = origin
        self.enable_db_logging = enable_db_logging
        self._db_initialized = False

        # Initialize database if enabled
        if enable_db_logging:
            if db_manager:
                # Use shared connection with our schema
                self.db = LLMDatabase.from_manager(db_manager, schema="llmring")
            elif db_connection_string:
                # Create own pool (standalone mode)
                self.db = LLMDatabase(
                    connection_string=db_connection_string, schema="llmring"
                )
            else:
                # Default connection for development
                self.db = LLMDatabase(schema="llmring")
            self.api = None  # Will be initialized after db is initialized
        else:
            self.db = None
            self.api = None

        self.providers: Dict[str, BaseLLMProvider] = {}
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        # Background DB logging
        self._log_queue: Optional[asyncio.Queue] = None
        self._log_task: Optional[asyncio.Task] = None
        self._initialize_providers()

    async def _ensure_db_initialized(self):
        """Ensure database is initialized (async operation)."""
        if self.enable_db_logging and self.db and not self._db_initialized:
            try:
                logger.info("Initializing LLM service database connection")
                await self.db.initialize()
                logger.debug("Database connection established")

                # Apply migrations on startup
                logger.info("Applying database migrations")
                await self.db.apply_migrations()
                logger.debug("Database migrations applied successfully")

                self._db_initialized = True

                # Initialize API after database is ready
                self.api = LLMRingAPI(self.db)

                logger.info("LLM service database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}", exc_info=True)
                logger.warning("Database logging will be disabled")
                self.db = None
                self.enable_db_logging = False

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

    async def chat(
        self, request: LLMRequest, id_at_origin: Optional[str] = None
    ) -> LLMResponse:
        """
        Send a chat request to the appropriate provider.

        Args:
            request: LLM request with messages and parameters
            id_at_origin: User identifier at the origin for database logging

        Returns:
            LLM response
        """
        # Ensure database is initialized
        await self._ensure_db_initialized()

        # Parse model to get provider
        provider_type, model_name = self._parse_model_string(request.model or "")

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

        # Track timing for database logging
        start_time = time.time()
        status = "success"
        error_type = None
        error_message = None

        try:
            # Send request to provider
            response = await provider.chat(
                messages=request.messages,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=request.response_format,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            error_message = str(e)
            raise
        finally:
            # Log to database if enabled (background queue)
            if self.enable_db_logging and self.db and id_at_origin:
                try:
                    response_time_ms = int((time.time() - start_time) * 1000)

                    # Extract token usage from response
                    prompt_tokens = 0
                    completion_tokens = 0
                    if response and response.usage:
                        prompt_tokens = response.usage.get("prompt_tokens", 0)
                        completion_tokens = response.usage.get("completion_tokens", 0)

                    # System prompt logging removed for privacy

                    # Extract tool names if tools are used
                    tools_used = None
                    if request.tools:
                        tools_used = [
                            tool.get("function", {}).get("name", "unknown")
                            for tool in request.tools
                        ]

                    # Enqueue record for background logging
                    await self._ensure_log_worker()
                    record = {
                        "origin": self.origin,
                        "id_at_origin": id_at_origin,
                        "provider": provider_type,
                        "model_name": model_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "response_time_ms": response_time_ms,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "tools_used": [
                            _truncate(t, 256) for t in tools_used
                        ]
                        if tools_used
                        else None,
                        "status": status,
                        "error_type": error_type,
                        "error_message": _prepare_error_message(error_message),
                    }
                    try:
                        assert self._log_queue is not None
                        self._log_queue.put_nowait(record)
                    except asyncio.QueueFull:
                        logger.warning("DB log queue full; dropping API call record")
                except Exception as db_error:
                    # Don't let database errors break the main flow
                    logger.warning(
                        "Failed to log API call to database", exc_info=True
                    )

        return response

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

    async def get_usage_stats(self, id_at_origin: str, days: int = 30):
        """Get usage statistics for a user.

        Args:
            id_at_origin: User identifier at the origin
            days: Number of days to look back

        Returns:
            UsageStats instance or None if no data found
        """
        if not self.db:
            return None
        await self._ensure_db_initialized()
        return await self.db.get_usage_stats(self.origin, id_at_origin, days)

    async def list_recent_calls(
        self, id_at_origin: Optional[str] = None, limit: int = 100
    ):
        """List recent API calls.

        Args:
            id_at_origin: Filter by user (optional)
            limit: Maximum number of records

        Returns:
            List of CallRecord instances
        """
        if not self.db:
            return []
        await self._ensure_db_initialized()
        return await self.db.list_recent_calls(
            origin=self.origin, id_at_origin=id_at_origin, limit=limit
        )

    async def get_models_from_db(
        self, provider: Optional[str] = None, active_only: bool = True
    ):
        """Get models from database.

        Args:
            provider: Filter by provider (optional)
            active_only: Only return active models (default: True)

        Returns:
            List of LLMModel instances
        """
        if not self.db:
            logger.warning("Database not configured, returning empty model list")
            return []

        try:
            logger.info(
                f"Fetching models from database - provider: {provider}, active_only: {active_only}"
            )
            await self._ensure_db_initialized()

            models = await self.db.list_models(
                provider=provider, active_only=active_only
            )
            logger.info(f"Successfully fetched {len(models)} models from database")
            return models

        except Exception as e:
            logger.error(f"Failed to fetch models from database: {e}", exc_info=True)
            # Re-raise to let the caller handle it appropriately
            raise

    async def get_model_from_db(self, provider: str, model_name: str) -> Optional[Any]:
        """Get a specific model from database.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model_name: Model name (e.g., 'gpt-4', 'claude-3-opus')

        Returns:
            LLMModel instance or None if not found
        """
        if not self.db:
            logger.warning(
                f"Database not configured, cannot fetch model {provider}:{model_name}"
            )
            return None

        try:
            logger.info(
                f"Fetching model from database - provider: {provider}, model: {model_name}"
            )
            await self._ensure_db_initialized()

            model = await self.db.get_model(provider, model_name)
            if model:
                logger.debug(f"Successfully fetched model {provider}:{model_name}")
            else:
                logger.info(f"Model {provider}:{model_name} not found in database")
            return model

        except Exception as e:
            logger.error(
                f"Failed to fetch model {provider}:{model_name} from database: {e}",
                exc_info=True,
            )
            # Re-raise to let the caller handle it appropriately
            raise

    async def get_usage_hints(self, use_case: str) -> List[Dict[str, Any]]:
        """Deprecated. Use application-side heuristics on model metadata instead."""
        raise NotImplementedError("Usage hints are no longer provided by the service")

    async def get_provider_usage_hints(
        self, provider: str
    ) -> Dict[str, Dict[str, Any]]:
        """Deprecated. Use application-side heuristics on model metadata instead."""
        raise NotImplementedError(
            "Provider usage hints are no longer provided by the service"
        )

    async def close(self):
        """Close database connections."""
        # Stop log worker gracefully
        if self._log_queue is not None:
            try:
                # Signal shutdown
                self._log_queue.put_nowait(None)  # type: ignore[arg-type]
            except Exception:
                pass
        if self._log_task is not None:
            try:
                await asyncio.wait_for(self._log_task, timeout=5.0)
            except Exception:
                self._log_task.cancel()
            finally:
                self._log_task = None
        if self.db and self._db_initialized:
            await self.db.close()
            self._db_initialized = False

    async def _ensure_log_worker(self):
        if self._log_queue is None:
            maxsize = int(os.getenv("LLMRING_LOG_QUEUE_MAX", "1000"))
            self._log_queue = asyncio.Queue(maxsize=maxsize)
        if self._log_task is None or self._log_task.done():
            self._log_task = asyncio.create_task(self._log_worker())

    async def _log_worker(self):
        assert self._log_queue is not None
        while True:
            item = await self._log_queue.get()
            if item is None:
                # Drain remaining quickly
                while not self._log_queue.empty():
                    leftover = self._log_queue.get_nowait()
                    if leftover is None:
                        continue
                    await self._write_log_record(leftover)
                break
            try:
                await self._write_log_record(item)
            except Exception:
                logger.warning("Failed to write API call record", exc_info=True)
            finally:
                self._log_queue.task_done()

    async def _write_log_record(self, rec: Dict[str, Any]):
        if not self.db:
            return
        await self.db.record_api_call(
            origin=rec["origin"],
            id_at_origin=rec["id_at_origin"],
            provider=rec["provider"],
            model_name=rec["model_name"],
            prompt_tokens=rec["prompt_tokens"],
            completion_tokens=rec["completion_tokens"],
            response_time_ms=rec["response_time_ms"],
            temperature=rec.get("temperature"),
            max_tokens=rec.get("max_tokens"),
            system_prompt=rec.get("system_prompt"),
            tools_used=rec.get("tools_used"),
            status=rec.get("status", "success"),
            error_type=rec.get("error_type"),
            error_message=rec.get("error_message"),
        )


def _truncate(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _redact_and_truncate(s: Optional[str], max_len: int = 2048) -> Optional[str]:
    if s is None:
        return None
    redacted = s
    # Basic redactions for common secrets; keep simple to avoid false negatives
    redacted = redacted.replace("sk-", "sk-***")
    redacted = redacted.replace("AIza", "AIza***")
    redacted = redacted.replace("ya29.", "ya29.***")
    return _truncate(redacted, max_len)


def _prepare_error_message(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    # Only include a minimal, sanitized error summary by default
    if os.getenv("LLMRING_LOG_ERROR_DETAIL", "false").lower() not in {"1", "true", "yes", "on"}:
        # Keep a short prefix and redact tokens/keys-like sequences
        summary = s.split("\n", 1)[0]
        return _redact_and_truncate(summary, max_len=256)
    # If detail explicitly enabled, still redact and truncate
    return _redact_and_truncate(s, max_len=1024)
