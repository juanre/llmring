"""
Information Service for MCP Client

This module provides comprehensive information about providers, models, costs,
usage, and data storage. It mirrors and extends llmring functionality
to give modules full transparency into what's happening behind the scenes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from llmring.exceptions import ConfigurationError, ModelError, ProviderError, ServerConnectionError

# LLMDatabase is not in llmring, will need to adapt
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""

    name: str
    available: bool
    model_count: int
    total_cost_last_30_days: float | None = None
    total_calls_last_30_days: int | None = None
    api_key_configured: bool = False
    description: str | None = None


@dataclass
class ModelInfo:
    """Detailed information about a specific model."""

    provider: str
    model_name: str
    display_name: str
    description: str | None
    max_context_tokens: int | None
    max_output_tokens: int | None
    supports_vision: bool
    supports_function_calling: bool
    supports_json_mode: bool
    supports_parallel_tool_calls: bool
    cost_per_input_token: Decimal | None
    cost_per_output_token: Decimal | None
    cost_per_1k_input_tokens: float | None  # Convenience field
    cost_per_1k_output_tokens: float | None  # Convenience field
    is_active: bool
    last_used: datetime | None = None
    total_usage_last_30_days: int | None = None
    total_cost_last_30_days: float | None = None


@dataclass
class UsageStats:
    """Comprehensive usage statistics."""

    user_id: str
    origin: str
    period_days: int
    total_calls: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    avg_cost_per_call: float
    most_used_model: str | None
    success_rate: float
    avg_response_time_ms: int | None
    models_used: list[str]
    daily_breakdown: list[dict[str, Any]]
    cost_by_model: dict[str, float]
    calls_by_model: dict[str, int]


@dataclass
class DataStorageInfo:
    """Information about what data is stored where."""

    mcp_client_tables: list[dict[str, Any]]
    llm_service_tables: list[dict[str, Any]]  # Keep backward compatibility name
    user_data_locations: dict[str, list[str]]
    retention_policies: dict[str, str]
    privacy_measures: list[str]


class MCPClientInfoService:
    """
    Comprehensive information service for MCP Client.

    Provides transparency into providers, models, costs, usage, and data storage.
    Modules can use this to understand exactly what's happening with their data.
    """

    def __init__(
        self,
        llmring: LLMRing | None = None,
        llmring_server_url: str | None = None,
        api_key: str | None = None,
        origin: str = "mcp-client-info",
    ):
        """
        Initialize the info service.

        Args:
            llmring: Optional LLM service instance
            llmring_server_url: LLMRing server URL for queries
            api_key: Optional API key for LLMRing server
            origin: Origin identifier for filtering data
        """
        self.llmring = llmring
        self.origin = origin
        self.llmring_server_url = llmring_server_url
        self.api_key = api_key

        # Backward compatibility attributes for tests
        self.llm_service = llmring
        self.llm_db = None  # No direct database access in HTTP architecture

        # Initialize HTTP client if server URL provided
        self.http_client = None
        if llmring_server_url:
            from llmring.mcp.http_client import MCPHttpClient

            self.http_client = MCPHttpClient(
                base_url=llmring_server_url,
                api_key=api_key,
            )

    def get_available_providers(self) -> list[ProviderInfo]:
        """
        Get information about all available LLM providers.

        Returns:
            List of ProviderInfo objects with provider details
        """
        providers = []

        if self.llmring:
            # Get provider information from LLM service
            available_models = self.llmring.get_available_models()

            for provider_name, models in available_models.items():
                # Check if provider is actually available (has API key)
                api_key_configured = provider_name in self.llmring.providers

                # Get usage stats if database available
                total_cost = None
                total_calls = None
                if self.llm_db:
                    try:
                        stats = self._get_provider_usage_stats(provider_name)
                        if stats:
                            total_cost = stats.get("total_cost")
                            total_calls = stats.get("total_calls")
                    except (ServerConnectionError, ProviderError) as e:
                        logger.warning(f"Error getting provider usage stats: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error getting provider usage stats: {e}")

                # Provider descriptions
                descriptions = {
                    "anthropic": "Anthropic's Claude models - excellent for reasoning and safety",
                    "openai": "OpenAI's GPT models - versatile and widely supported",
                    "google": "Google's Gemini models - large context and multimodal",
                    "ollama": "Local models via Ollama - private and cost-free",
                }

                providers.append(
                    ProviderInfo(
                        name=provider_name,
                        available=api_key_configured,
                        model_count=len(models),
                        total_cost_last_30_days=total_cost,
                        total_calls_last_30_days=total_calls,
                        api_key_configured=api_key_configured,
                        description=descriptions.get(provider_name),
                    )
                )

        return providers

    def get_models_for_provider(self, provider: str) -> list[ModelInfo]:
        """
        Get detailed information about models for a specific provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')

        Returns:
            List of ModelInfo objects with model details
        """
        models = []

        if self.llm_db:
            try:
                # Get models from database (authoritative source)
                db_models = self.llm_db.list_models(provider=provider)

                for model in db_models:
                    # Get usage stats for this model
                    usage_stats = self._get_model_usage_stats(provider, model.model_name)

                    # Convert costs to convenient formats using Decimal for precision
                    cost_per_1k_input = None
                    cost_per_1k_output = None
                    if model.cost_per_token_input:
                        # Use Decimal arithmetic to avoid floating point precision errors
                        cost_per_1k_input = float(
                            Decimal(str(model.cost_per_token_input)) * Decimal("1000")
                        )
                    if model.cost_per_token_output:
                        # Use Decimal arithmetic to avoid floating point precision errors
                        cost_per_1k_output = float(
                            Decimal(str(model.cost_per_token_output)) * Decimal("1000")
                        )

                    models.append(
                        ModelInfo(
                            provider=model.provider,
                            model_name=model.model_name,
                            display_name=model.display_name or model.model_name,
                            description=model.description,
                            max_context_tokens=model.max_context,
                            max_output_tokens=model.max_output_tokens,
                            supports_vision=model.supports_vision,
                            supports_function_calling=model.supports_function_calling,
                            supports_json_mode=model.supports_json_mode,
                            supports_parallel_tool_calls=model.supports_parallel_tool_calls,
                            cost_per_input_token=model.cost_per_token_input,
                            cost_per_output_token=model.cost_per_token_output,
                            cost_per_1k_input_tokens=cost_per_1k_input,
                            cost_per_1k_output_tokens=cost_per_1k_output,
                            is_active=model.is_active,
                            last_used=(usage_stats.get("last_used") if usage_stats else None),
                            total_usage_last_30_days=(
                                usage_stats.get("total_tokens") if usage_stats else None
                            ),
                            total_cost_last_30_days=(
                                usage_stats.get("total_cost") if usage_stats else None
                            ),
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not fetch models from database: {e}")

        # Fall back to LLM service if database not available
        if not models and self.llmring:
            available_models = self.llmring.get_available_models()
            if provider in available_models:
                for model_name in available_models[provider]:
                    models.append(
                        ModelInfo(
                            provider=provider,
                            model_name=model_name,
                            display_name=model_name,
                            description=None,
                            max_context_tokens=None,
                            max_output_tokens=None,
                            supports_vision=False,
                            supports_function_calling=False,
                            supports_json_mode=False,
                            supports_parallel_tool_calls=False,
                            cost_per_input_token=None,
                            cost_per_output_token=None,
                            cost_per_1k_input_tokens=None,
                            cost_per_1k_output_tokens=None,
                            is_active=True,
                        )
                    )

        return models

    def get_model_cost_info(self, model_identifier: str) -> dict[str, Any] | None:
        """
        Get cost information for a specific model.

        Args:
            model_identifier: Either "provider:model" or just "model"

        Returns:
            Dictionary with cost information or None if not found
        """
        if ":" in model_identifier:
            provider, model_name = model_identifier.split(":", 1)
        else:
            # Try to find the model across all providers
            provider = None
            model_name = model_identifier

            if self.llm_db:
                try:
                    models = self.llm_db.list_models()
                    for model in models:
                        if model.model_name == model_name:
                            provider = model.provider
                            break
                except (ModelError, ServerConnectionError) as e:
                    logger.warning(f"Error finding model provider: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error finding model provider: {e}")

        if not provider:
            return None

        models = self.get_models_for_provider(provider)
        for model in models:
            if model.model_name == model_name:
                return {
                    "provider": model.provider,
                    "model_name": model.model_name,
                    "display_name": model.display_name,
                    "cost_per_input_token": (
                        float(model.cost_per_input_token) if model.cost_per_input_token else None
                    ),
                    "cost_per_output_token": (
                        float(model.cost_per_output_token) if model.cost_per_output_token else None
                    ),
                    "cost_per_1k_input_tokens": model.cost_per_1k_input_tokens,
                    "cost_per_1k_output_tokens": model.cost_per_1k_output_tokens,
                    "max_context_tokens": model.max_context_tokens,
                    "supports_function_calling": model.supports_function_calling,
                    "supports_vision": model.supports_vision,
                }

        return None

    def get_usage_stats(
        self, user_id: str, days: int = 30, include_daily_breakdown: bool = True
    ) -> UsageStats | None:
        """
        Get comprehensive usage statistics for a user.

        Args:
            user_id: User identifier
            days: Number of days to look back
            include_daily_breakdown: Whether to include daily usage breakdown

        Returns:
            UsageStats object or None if no data available
        """
        # We need either a database or LLM service to get stats
        if not self.llm_db and not self.llmring:
            return None

        try:
            # Get overall stats from LLM service
            if self.llmring:
                service_stats = self.llmring.get_usage_stats(user_id, days=days)
                if not service_stats:
                    return None
            else:
                # Get stats directly from database
                service_stats = self._get_direct_usage_stats(user_id, days)
                if not service_stats:
                    return None

            # Get detailed breakdown
            models_used = self._get_models_used_by_user(user_id, days)
            cost_by_model = self._get_cost_by_model(user_id, days)
            calls_by_model = self._get_calls_by_model(user_id, days)
            daily_breakdown = []

            if include_daily_breakdown:
                daily_breakdown = self._get_daily_usage_breakdown(user_id, days)

            return UsageStats(
                user_id=user_id,
                origin=self.origin,
                period_days=days,
                total_calls=service_stats.total_calls,
                total_tokens=service_stats.total_tokens,
                total_input_tokens=getattr(service_stats, "total_input_tokens", 0),
                total_output_tokens=getattr(service_stats, "total_output_tokens", 0),
                total_cost=float(service_stats.total_cost),
                avg_cost_per_call=float(service_stats.avg_cost_per_call),
                most_used_model=service_stats.most_used_model,
                success_rate=float(service_stats.success_rate),
                avg_response_time_ms=service_stats.avg_response_time_ms,
                models_used=models_used,
                daily_breakdown=daily_breakdown,
                cost_by_model=cost_by_model,
                calls_by_model=calls_by_model,
            )
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return None

    def get_data_storage_info(self) -> DataStorageInfo:
        """
        Get comprehensive information about what data is stored where.

        Returns:
            DataStorageInfo object describing all data storage
        """
        mcp_client_tables = [
            {
                "table": "conversations",
                "endpoint": "/conversations",
                "purpose": "Conversation metadata and configuration via HTTP API",
                "user_identifiable_fields": ["project_id"],
                "retention": "Indefinite (until manually deleted)",
                "privacy_notes": "Contains conversation titles and settings, no message content",
            },
            {
                "table": "messages",
                "endpoint": "/conversations/{id}/messages",
                "purpose": "Individual conversation messages via HTTP API",
                "user_identifiable_fields": ["project_id"],
                "retention": "Indefinite (until conversation deleted)",
                "privacy_notes": "Contains full message content including system prompts",
            },
            {
                "table": "mcp_servers",
                "endpoint": "/api/v1/mcp/servers",
                "purpose": "MCP server connection information via HTTP API",
                "user_identifiable_fields": ["project_id"],
                "retention": "Indefinite (until manually deleted)",
                "privacy_notes": "Contains server URLs and connection metadata",
            },
            {
                "table": "conversation_templates",
                "endpoint": "/api/v1/templates",
                "purpose": "Reusable conversation templates via HTTP API",
                "user_identifiable_fields": ["project_id", "created_by"],
                "retention": "Indefinite (until manually deleted)",
                "privacy_notes": "Contains template configurations and system prompts",
            },
        ]

        llmring_tables = [
            {
                "table": "usage_logs",
                "endpoint": "/api/v1/log",
                "purpose": "Detailed log of every LLM API call via HTTP API",
                "user_identifiable_fields": ["project_id"],
                "retention": "Configurable (default: indefinite)",
                "privacy_notes": "System prompts are SHA-256 hashed for privacy, includes token counts and costs",
            },
            {
                "table": "receipts",
                "endpoint": "/receipts",
                "purpose": "Cryptographic receipts for usage verification",
                "user_identifiable_fields": ["project_id"],
                "retention": "Configurable (default: indefinite)",
                "privacy_notes": "Contains signed usage receipts for audit purposes",
            },
            {
                "table": "registry",
                "endpoint": "/registry",
                "purpose": "LLM model registry with cost and capability information",
                "user_identifiable_fields": [],
                "retention": "Indefinite (reference data)",
                "privacy_notes": "No user data - only model specifications and pricing",
            },
        ]

        user_data_locations = {
            "conversation_content": ["mcp_client.chat_messages"],
            "conversation_metadata": ["mcp_client.chat_sessions"],
            "llm_usage_tracking": [
                "llmring.llm_api_calls",
                "llmring.usage_analytics_daily",
            ],
            "mcp_server_connections": ["mcp_client.mcp_servers"],
            "cost_tracking": ["llmring.llm_api_calls", "llmring.usage_analytics_daily"],
        }

        retention_policies = {
            "conversation_data": "Retained indefinitely until manually deleted by user",
            "llm_usage_logs": "Configurable cleanup via llmring.cleanup_old_data()",
            "analytics_data": "Configurable retention (default: indefinite)",
            "model_registry": "Permanent reference data (no user content)",
        }

        privacy_measures = [
            "System prompts are SHA-256 hashed in LLM usage logs",
            "User messages are stored only in conversation tables under user control",
            "Database schema isolation between MCP client and LLM service",
            "User data is always tied to origin and user_id for traceability",
            "No cross-user data leakage due to proper user_id isolation",
            "Optional database encryption at rest (PostgreSQL configuration)",
            "Database access controlled via connection strings and authentication",
        ]

        return DataStorageInfo(
            mcp_client_tables=mcp_client_tables,
            llm_service_tables=llmring_tables,  # Use backward compatibility name
            user_data_locations=user_data_locations,
            retention_policies=retention_policies,
            privacy_measures=privacy_measures,
        )

    def get_user_data_summary(self, user_id: str) -> dict[str, Any]:
        """
        Get a summary of all data stored for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary summarizing all stored user data
        """
        summary = {
            "user_id": user_id,
            "origin": self.origin,
            "data_locations": {},
            "usage_summary": None,
            "conversation_summary": None,
            "privacy_info": {},
        }

        # Get usage summary
        usage_stats = self.get_usage_stats(user_id, days=30)
        if usage_stats:
            summary["usage_summary"] = {
                "total_calls_30_days": usage_stats.total_calls,
                "total_cost_30_days": usage_stats.total_cost,
                "models_used": usage_stats.models_used,
                "success_rate": usage_stats.success_rate,
            }

        # Get conversation summary (would need database access)
        # This would require adding conversation_manager to the info service
        summary["conversation_summary"] = {
            "note": "Conversation data summary requires conversation_manager access"
        }

        # Privacy information
        summary["privacy_info"] = {
            "system_prompts_hashed": True,
            "user_messages_stored": "In conversation tables only",
            "data_retention": "User controlled for conversations, configurable for usage logs",
            "data_isolation": "Complete isolation by user_id and origin",
        }

        return summary

    def _get_provider_usage_stats(self, provider: str, days: int = 30) -> dict[str, Any] | None:
        """Get usage statistics for a specific provider."""
        if not self.llm_db:
            return None

        try:
            # Get aggregated stats for the provider
            datetime.now() - timedelta(days=days)

            # Use the LLM database to get usage stats for models from this provider
            models = self.llm_db.list_models(provider=provider)
            if not models:
                return None

            model_names = [f"{provider}:{model.model_name}" for model in models]

            # Get usage stats from llmbridge if available
            if self.llmring:
                total_cost = 0.0
                total_calls = 0

                for _model_name in model_names:
                    try:
                        # This would require direct database query or enhanced llmbridge API
                        # For now, we'll use basic aggregation
                        stats = self.llmring.get_usage_stats(self.origin, days=days)
                        if stats:
                            # Filter by provider if possible (this is a simplified approach)
                            total_cost += float(stats.total_cost or 0)
                            total_calls += stats.total_calls or 0
                    except (ServerConnectionError, ProviderError) as e:
                        logger.warning(f"Error getting usage stats for provider model: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error getting usage stats: {e}")
                        continue

                return {
                    "total_cost": total_cost,
                    "total_calls": total_calls,
                    "provider": provider,
                    "period_days": days,
                }

            return None
        except (ServerConnectionError, ProviderError) as e:
            logger.warning(f"Error getting provider usage stats: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in provider usage stats: {e}")
            return None

    def _get_model_usage_stats(
        self, provider: str, model_name: str, days: int = 30
    ) -> dict[str, Any] | None:
        """Get usage statistics for a specific model."""
        if not self.llm_db and not self.llmring:
            return None

        try:
            model_identifier = f"{provider}:{model_name}"
            datetime.now() - timedelta(days=days)

            # Try to get usage stats for this specific model
            if self.llmring:
                try:
                    # Get overall stats and filter for this model if possible
                    stats = self.llmring.get_usage_stats(self.origin, days=days)
                    if (
                        stats
                        and hasattr(stats, "most_used_model")
                        and stats.most_used_model == model_identifier
                    ):
                        return {
                            "total_cost": float(stats.total_cost or 0),
                            "total_tokens": stats.total_tokens or 0,
                            "total_calls": stats.total_calls or 0,
                            "last_used": getattr(stats, "last_used", None),
                            "model": model_identifier,
                            "period_days": days,
                        }
                except (ServerConnectionError, ModelError) as e:
                    logger.warning(f"Error getting model usage stats: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error getting model usage stats: {e}")

            # Return empty stats if no data available
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_calls": 0,
                "last_used": None,
                "model": model_identifier,
                "period_days": days,
            }
        except (ServerConnectionError, ModelError) as e:
            logger.warning(f"Error getting model usage stats: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in model usage stats: {e}")
            return None

    def _get_direct_usage_stats(self, user_id: str, days: int = 30) -> Any | None:
        """Get usage stats directly from database when LLM service not available."""
        if not self.llm_db:
            return None

        try:
            # Create a simple stats object with basic information
            # This would need to be enhanced with actual database queries
            class BasicStats:
                def __init__(self):
                    self.total_calls = 0
                    self.total_tokens = 0
                    self.total_cost = 0.0
                    self.avg_cost_per_call = 0.0
                    self.most_used_model = None
                    self.success_rate = 1.0
                    self.avg_response_time_ms = None

            # For now, return empty stats when LLM service is not available
            # In a full implementation, this would query the llm_api_calls table directly
            return BasicStats()
        except (ConfigurationError, ServerConnectionError) as e:
            logger.warning(f"Error getting direct usage stats: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting direct usage stats: {e}")
            return None

    def _get_models_used_by_user(self, user_id: str, days: int = 30) -> list[str]:
        """Get list of models used by user."""
        if not self.llmring:
            return []

        try:
            # Get usage stats and extract model information
            stats = self.llmring.get_usage_stats(user_id, days=days)
            if stats and hasattr(stats, "most_used_model") and stats.most_used_model:
                return [stats.most_used_model]

            # In a full implementation, this would query the database for all models
            # used by this user in the time period
            return []
        except (ServerConnectionError, ModelError) as e:
            logger.warning(f"Error getting models used by user: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting models used: {e}")
            return []

    def _get_cost_by_model(self, user_id: str, days: int = 30) -> dict[str, float]:
        """Get cost breakdown by model."""
        if not self.llmring:
            return {}

        try:
            # Get usage stats and extract cost information
            stats = self.llmring.get_usage_stats(user_id, days=days)
            if stats and hasattr(stats, "most_used_model") and stats.most_used_model:
                return {stats.most_used_model: float(stats.total_cost or 0)}

            # In a full implementation, this would aggregate costs by model
            return {}
        except (ServerConnectionError, ModelError) as e:
            logger.warning(f"Error getting cost by model: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting cost by model: {e}")
            return {}

    def _get_calls_by_model(self, user_id: str, days: int = 30) -> dict[str, int]:
        """Get call count breakdown by model."""
        if not self.llmring:
            return {}

        try:
            # Get usage stats and extract call information
            stats = self.llmring.get_usage_stats(user_id, days=days)
            if stats and hasattr(stats, "most_used_model") and stats.most_used_model:
                return {stats.most_used_model: stats.total_calls or 0}

            # In a full implementation, this would count calls by model
            return {}
        except (ServerConnectionError, ModelError) as e:
            logger.warning(f"Error getting calls by model: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting calls by model: {e}")
            return {}

    def _get_daily_usage_breakdown(self, user_id: str, days: int = 30) -> list[dict[str, Any]]:
        """Get daily usage breakdown."""
        if not self.llmring:
            return []

        try:
            # Get usage stats and create a simple daily breakdown
            stats = self.llmring.get_usage_stats(user_id, days=days)
            if not stats:
                return []

            # For now, create a simple breakdown with total stats
            # In a full implementation, this would group by date
            breakdown = []
            for i in range(min(days, 7)):  # Show last 7 days as sample
                date = datetime.now() - timedelta(days=i)
                breakdown.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "calls": stats.total_calls // max(days, 1) if i == 0 else 0,
                        "tokens": stats.total_tokens // max(days, 1) if i == 0 else 0,
                        "cost": (float(stats.total_cost or 0) / max(days, 1) if i == 0 else 0.0),
                    }
                )

            return breakdown
        except (ServerConnectionError, ModelError) as e:
            logger.warning(f"Error getting daily usage breakdown: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting daily usage breakdown: {e}")
            return []

    def to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert dataclass or object to dictionary for JSON serialization."""
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, datetime | Decimal):
                    result[key] = str(value)
                elif isinstance(value, list):
                    result[key] = [
                        self.to_dict(item) if hasattr(item, "__dict__") else item for item in value
                    ]
                elif hasattr(value, "__dict__"):
                    result[key] = self.to_dict(value)
                else:
                    result[key] = value
            return result
        return obj


# Convenience function for easy access
def create_info_service(
    llmring: LLMRing | None = None,
    llmring_server_url: str | None = None,
    api_key: str | None = None,
    origin: str = "mcp-client-info",
    llm_service: LLMRing | None = None,  # Backward compatibility
) -> MCPClientInfoService:
    """
    Create an MCP Client info service instance.

    Args:
        llmring: Optional LLM service instance
        llmring_server_url: LLMRing server URL for queries
        api_key: Optional API key for LLMRing server
        origin: Origin identifier
        llm_service: Optional LLM service (backward compatibility alias)

    Returns:
        MCPClientInfoService instance
    """
    # Use llm_service if provided for backward compatibility
    service = llm_service or llmring

    return MCPClientInfoService(
        llmring=service,
        llmring_server_url=llmring_server_url,
        api_key=api_key,
        origin=origin,
    )
