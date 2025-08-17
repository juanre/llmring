"""
LLM service with SQLite support for local development.

This is a simplified alternative to the main service.py for when you want to use SQLite
instead of PostgreSQL. For production use with pgdbm, use the main service.py.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llmring.base import BaseLLMProvider
from llmring.db_sqlite import SQLiteDatabase
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMRequest, LLMResponse

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LLMRingSQLite:
    """LLM service with SQLite database for local development."""

    def __init__(
        self,
        db_path: str = "llmring.db",
        origin: str = "llmring",
        enable_db_logging: bool = True,
    ):
        """
        Initialize the LLM service with SQLite database.

        Args:
            db_path: Path to SQLite database file
            origin: Origin identifier for database logging
            enable_db_logging: Whether to enable database logging
        """
        self.origin = origin
        self.enable_db_logging = enable_db_logging
        self._db_initialized = False

        # Initialize SQLite database if enabled
        if enable_db_logging:
            self.db = SQLiteDatabase(db_path)
        else:
            self.db = None

        self.providers: Dict[str, BaseLLMProvider] = {}
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._initialize_providers()

    async def _ensure_db_initialized(self):
        """Ensure database is initialized (async operation)."""
        if self.enable_db_logging and self.db and not self._db_initialized:
            try:
                logger.info("Initializing SQLite database connection")
                await self.db.initialize()
                self._db_initialized = True
                logger.info("SQLite database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}", exc_info=True)
                logger.warning("Database logging will be disabled")
                self.db = None
                self.enable_db_logging = False

    def _initialize_providers(self):
        """Initialize all configured providers from environment variables."""
        logger.info("Initializing LLM providers")

        # OpenAI
        if openai_key := os.getenv("OPENAI_API_KEY"):
            try:
                self.providers["openai"] = OpenAIProvider(api_key=openai_key)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")

        # Anthropic
        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {e}")

        # Google/Gemini
        if google_key := os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            try:
                self.providers["google"] = GoogleProvider(api_key=google_key)
                logger.info("Google provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google provider: {e}")

        # Ollama (local)
        if os.getenv("OLLAMA_API_BASE") or os.getenv("ENABLE_OLLAMA", "false").lower() == "true":
            try:
                self.providers["ollama"] = OllamaProvider()
                logger.info("Ollama provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama provider: {e}")

        logger.info(f"Initialized {len(self.providers)} providers")

    def register_provider(self, name: str, provider: Optional[BaseLLMProvider] = None, **kwargs):
        """
        Register a provider instance.

        Args:
            name: Provider name (e.g., 'openai', 'anthropic')
            provider: Provider instance (if None, will create based on name)
            **kwargs: Additional arguments for provider initialization
        """
        if provider:
            self.providers[name] = provider
        else:
            # Create provider based on name
            if name == "openai":
                self.providers[name] = OpenAIProvider(**kwargs)
            elif name == "anthropic":
                self.providers[name] = AnthropicProvider(**kwargs)
            elif name == "google":
                self.providers[name] = GoogleProvider(**kwargs)
            elif name == "ollama":
                self.providers[name] = OllamaProvider(**kwargs)
            else:
                raise ValueError(f"Unknown provider: {name}")

        logger.info(f"Registered provider: {name}")

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider."""
        result = {}
        for name, provider in self.providers.items():
            result[name] = provider.get_supported_models()
        return result

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        Route chat request to appropriate provider.

        Args:
            request: LLM request with messages and parameters

        Returns:
            LLM response from the provider
        """
        # Ensure database is initialized
        await self._ensure_db_initialized()

        start_time = time.time()

        # Extract provider from model name (e.g., "openai:gpt-4" -> "openai")
        if ":" in request.model:
            provider_name, model_name = request.model.split(":", 1)
        else:
            # Try to determine provider from model name
            provider_name = self._determine_provider(request.model)
            model_name = request.model

        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")

        provider = self.providers[provider_name]

        # Log the request start
        logger.info(f"Routing request to {provider_name} with model {model_name}")

        try:
            # Call the provider with only non-None, supported kwargs
            kwargs = {
                "messages": request.messages,
                "model": model_name,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.tools is not None:
                kwargs["tools"] = request.tools
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
            if request.response_format is not None:
                kwargs["response_format"] = request.response_format

            response = await provider.chat(**kwargs)

            # Log to database if enabled
            if self.enable_db_logging and self.db:
                try:
                    from decimal import Decimal
                    from uuid import uuid4
                    from datetime import datetime, timezone
                    from llmring.schemas import CallRecord

                    # Get model info for cost calculation
                    model_info = await self.db.get_model(provider_name, model_name)

                    # Calculate cost
                    cost = Decimal("0")
                    if response.usage and model_info:
                        prompt_tokens = response.usage.get("prompt_tokens", 0)
                        completion_tokens = response.usage.get("completion_tokens", 0)

                        if model_info.dollars_per_million_tokens_input:
                            cost += Decimal(str(prompt_tokens)) * model_info.dollars_per_million_tokens_input / Decimal("1000000")
                        if model_info.dollars_per_million_tokens_output:
                            cost += Decimal(str(completion_tokens)) * model_info.dollars_per_million_tokens_output / Decimal("1000000")

                    record = CallRecord(
                        id=uuid4(),
                        origin=self.origin,
                        id_at_origin="default",  # TODO: Add user tracking
                        provider=provider_name,
                        model_name=model_name,
                        prompt_tokens=response.usage.get("prompt_tokens", 0) if response.usage else 0,
                        completion_tokens=response.usage.get("completion_tokens", 0) if response.usage else 0,
                        total_tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                        estimated_cost=cost,
                        dollars_per_million_tokens_input_used=model_info.dollars_per_million_tokens_input if model_info else None,
                        dollars_per_million_tokens_output_used=model_info.dollars_per_million_tokens_output if model_info else None,
                        called_at=datetime.now(timezone.utc),
                    )
                    await self.db.record_api_call(record)
                except Exception as e:
                    logger.error(f"Failed to log API call: {e}")

            elapsed = time.time() - start_time
            logger.info(f"Request completed in {elapsed:.2f}s")

            return response

        except Exception as e:
            # Log error to database if enabled
            if self.enable_db_logging and self.db:
                try:
                    from uuid import uuid4
                    from decimal import Decimal
                    from llmring.schemas import CallRecord

                    record = CallRecord(
                        id=uuid4(),
                        origin=self.origin,
                        id_at_origin="default",
                        provider=provider_name,
                        model_name=model_name,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost=Decimal("0"),
                        status="error",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    await self.db.record_api_call(record)
                except Exception as log_error:
                    logger.error(f"Failed to log API error: {log_error}")

            logger.error(f"Request failed: {e}")
            raise

    def _determine_provider(self, model: str) -> str:
        """Determine provider from model name."""
        # Common model prefixes
        if model.startswith(("gpt-", "o1-")):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif model.startswith(("gemini-", "models/")):
            return "google"
        elif model.startswith(("llama", "mistral", "qwen")):
            return "ollama"

        # Check which providers support this model
        for name, provider in self.providers.items():
            if provider.validate_model(model):
                return name

        raise ValueError(f"No provider found for model: {model}")

    async def close(self):
        """Close database connections."""
        if self.db:
            await self.db.close()
