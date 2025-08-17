"""Database manager for LLM service using async-db-utils."""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from pgdbm import AsyncDatabaseManager, AsyncMigrationManager, DatabaseConfig
from pgdbm.monitoring import MonitoredAsyncDatabaseManager

from .schemas import CallRecord, LLMModel, UsageStats

logger = logging.getLogger(__name__)


class LLMDatabase:
    """Async database manager for LLM service with model registry and call tracking."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        db_manager: Optional[AsyncDatabaseManager] = None,  # NEW
        schema: str = "llmring",
        min_connections: int = 2,
        max_connections: int = 5,
        enable_monitoring: bool = True,
    ):
        """Initialize database manager.

        Args:
            connection_string: PostgreSQL connection string (for standalone mode).
            db_manager: External AsyncDatabaseManager (for integrated mode).
            schema: Database schema name for the LLM service.
            min_connections: Min connections (only used in standalone mode).
            max_connections: Max connections (only used in standalone mode).
            enable_monitoring: Enable query monitoring (only used in standalone mode).
        """
        self.schema = schema
        self._external_db = db_manager is not None
        self._initialized = False
        self._prepared_statements_added = False

        if db_manager:
            # Use provided manager - it already has its own config
            self.db = db_manager
            self.config = db_manager.config
        else:
            # Create own manager (standalone mode)
            if connection_string is None:
                connection_string = "postgresql://postgres:postgres@localhost/postgres"

            self.config = DatabaseConfig(
                connection_string=connection_string,
                schema=schema,
                min_connections=min_connections,
                max_connections=max_connections,
                max_queries=50000,  # Recycle connections after this many queries
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                command_timeout=60.0,  # 60 seconds default timeout
            )

            # Use monitored database manager for production observability
            if enable_monitoring:
                self.db = MonitoredAsyncDatabaseManager(self.config)
                # These are set from kwargs in MonitoredAsyncDatabaseManager.__init__
                self.db._slow_query_threshold_ms = 100  # Log queries over 100ms
                self.db._max_history_size = 1000
            else:
                self.db = AsyncDatabaseManager(self.config)

    async def initialize(self):
        """Initialize database connection and ensure schema exists."""
        if self._initialized:
            return

        # Only connect if we created our own manager
        if not self._external_db:
            await self.db.connect()

        # Add prepared statements for frequently used queries
        if not self._prepared_statements_added:
            self._add_prepared_statements()
            self._prepared_statements_added = True

        # Ensure our schema exists
        await self._ensure_schema()

        # Find migrations relative to this module
        migrations_path = Path(__file__).parent / "migrations"

        # Initialize migrations
        self.migration_manager = AsyncMigrationManager(
            self.db, migrations_path=str(migrations_path), module_name="llmring"
        )

        self._initialized = True

    async def _ensure_schema(self):
        """Ensure our schema exists."""
        # Check if schema exists
        query = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.schemata
            WHERE schema_name = $1
        )
        """
        exists = await self.db.fetch_value(query, self.schema)

        if not exists:
            # Create schema
            await self.db.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')

    def _add_prepared_statements(self):
        """Add prepared statements for frequently executed queries."""
        # Model lookup - most frequent query
        self.db.add_prepared_statement(
            "get_model",
            """
            SELECT id, provider, model_name, display_name, description,
                   max_context, max_output_tokens, supports_vision,
                   supports_function_calling, supports_json_mode,
                   supports_parallel_tool_calls, tool_call_format,
                   dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                   inactive_from, created_at, updated_at
            FROM {{tables.llm_models}}
            WHERE provider = $1 AND model_name = $2 AND inactive_from IS NULL
            """,
        )

        # API call recording - second most frequent (without system prompt fields)
        self.db.add_prepared_statement(
            "record_api_call",
            """
            INSERT INTO {{tables.llm_api_calls}} (
                origin, id_at_origin, provider, model_name,
                prompt_tokens, completion_tokens, total_tokens,
                response_time_ms, temperature, max_tokens,
                top_p, stream, stop_sequences,
                tools_used, json_mode, response_format, seed,
                tool_choice, parallel_tool_calls, status,
                error_type, error_message, estimated_cost
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23
            ) RETURNING id
            """,
        )

        # Usage stats queries
        self.db.add_prepared_statement(
            "get_usage_stats",
            """
            WITH recent_calls AS (
                SELECT
                    COUNT(*) as total_calls,
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(estimated_cost) as total_cost,
                    AVG(response_time_ms) as avg_response_time_ms,
                    COUNT(CASE WHEN status = 'success' THEN 1 END)::float / COUNT(*) as success_rate,
                    COUNT(DISTINCT provider) as providers_used,
                    COUNT(DISTINCT model_name) as models_used
                FROM {{tables.llm_api_calls}}
                WHERE origin = $1
                AND id_at_origin = $2
                AND called_at >= CURRENT_TIMESTAMP - $3 * INTERVAL '1 day'
            ),
            most_used AS (
                SELECT model_name, COUNT(*) as usage_count
                FROM {{tables.llm_api_calls}}
                WHERE origin = $1
                AND id_at_origin = $2
                AND called_at >= CURRENT_TIMESTAMP - $3 * INTERVAL '1 day'
                GROUP BY model_name
                ORDER BY usage_count DESC
                LIMIT 1
            )
            SELECT
                rc.*,
                mu.model_name as most_used_model
            FROM recent_calls rc
            LEFT JOIN most_used mu ON true
            """,
        )

    async def close(self):
        """Close database connection if we own it."""
        if self._initialized and not self._external_db:
            await self.db.disconnect()
            self._initialized = False

    async def apply_migrations(self) -> Dict:
        """Apply database migrations."""
        if not self._initialized:
            await self.initialize()
        return await self.migration_manager.apply_pending_migrations()

    @classmethod
    def from_manager(cls, db_manager: AsyncDatabaseManager, schema: str = "llmring"):
        """Create instance from existing AsyncDatabaseManager.

        This is a convenience method for cleaner initialization.

        Args:
            db_manager: Existing AsyncDatabaseManager to use
            schema: Database schema name

        Returns:
            LLMDatabase instance using the provided manager
        """
        return cls(db_manager=db_manager, schema=schema)

    # ===== MODEL REGISTRY =====

    async def get_model(
        self, provider: str, model_name: str, active_only: bool = True
    ) -> Optional[LLMModel]:
        """Get model information by provider and name.

        Args:
            provider: Model provider (anthropic, openai, google, ollama)
            model_name: Model name
            active_only: Only return if model is active

        Returns:
            LLMModel instance or None if not found
        """
        if not self._initialized:
            await self.initialize()

        base_query = """
            SELECT id, provider, model_name, display_name, description,
                   max_context, max_output_tokens, supports_vision,
                   supports_function_calling, supports_json_mode,
                   supports_parallel_tool_calls, tool_call_format,
                   dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                   inactive_from, created_at, updated_at
            FROM {{tables.llm_models}}
            WHERE provider = $1 AND model_name = $2
        """

        if active_only:
            base_query += " AND inactive_from IS NULL"

        query = self.db._prepare_query(base_query)
        result = await self.db.fetch_one(query, provider, model_name)

        if not result:
            return None

        return LLMModel(
            id=result["id"],
            provider=result["provider"],
            model_name=result["model_name"],
            display_name=result["display_name"],
            description=result["description"],
            max_context=result["max_context"],
            max_output_tokens=result["max_output_tokens"],
            supports_vision=result["supports_vision"],
            supports_function_calling=result["supports_function_calling"],
            supports_json_mode=result["supports_json_mode"],
            supports_parallel_tool_calls=result["supports_parallel_tool_calls"],
            tool_call_format=result["tool_call_format"],
            dollars_per_million_tokens_input=result["dollars_per_million_tokens_input"],
            dollars_per_million_tokens_output=result[
                "dollars_per_million_tokens_output"
            ],
            inactive_from=result["inactive_from"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )

    async def list_models(
        self, provider: Optional[str] = None, active_only: bool = True
    ) -> List[LLMModel]:
        """List available models.

        Args:
            provider: Filter by provider (optional)
            active_only: Only return active models

        Returns:
            List of LLMModel instances
        """
        if not self._initialized:
            await self.initialize()

        base_query = """
            SELECT id, provider, model_name, display_name, description,
                   max_context, max_output_tokens, supports_vision,
                   supports_function_calling, supports_json_mode,
                   supports_parallel_tool_calls, tool_call_format,
                   dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                   inactive_from, created_at, updated_at
            FROM {{tables.llm_models}}
            WHERE 1=1
        """
        params = []
        param_count = 0

        if provider:
            param_count += 1
            base_query += f" AND provider = ${param_count}"
            params.append(provider)

        if active_only:
            base_query += " AND inactive_from IS NULL"

        base_query += " ORDER BY provider, model_name"

        query = self.db._prepare_query(base_query)
        results = await self.db.fetch_all(query, *params)

        return [
            LLMModel(
                id=row["id"],
                provider=row["provider"],
                model_name=row["model_name"],
                display_name=row["display_name"],
                description=row["description"],
                max_context=row["max_context"],
                max_output_tokens=row["max_output_tokens"],
                supports_vision=row["supports_vision"],
                supports_function_calling=row["supports_function_calling"],
                supports_json_mode=row["supports_json_mode"],
                supports_parallel_tool_calls=row["supports_parallel_tool_calls"],
                tool_call_format=row["tool_call_format"],
                dollars_per_million_tokens_input=row[
                    "dollars_per_million_tokens_input"
                ],
                dollars_per_million_tokens_output=row[
                    "dollars_per_million_tokens_output"
                ],
                inactive_from=row["inactive_from"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in results
        ]

    async def add_model(self, model: LLMModel) -> int:
        """Add a new model to the registry.

        Args:
            model: LLMModel instance

        Returns:
            Model ID
        """
        if not self._initialized:
            await self.initialize()

        query = self.db._prepare_query(
            """
            INSERT INTO {{tables.llm_models}} (
                provider, model_name, display_name, description,
                max_context, max_output_tokens, supports_vision,
                supports_function_calling, supports_json_mode,
                supports_parallel_tool_calls, tool_call_format,
                dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                inactive_from
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
        """
        )

        model_id = await self.db.fetch_value(
            query,
            model.provider,
            model.model_name,
            model.display_name,
            model.description,
            model.max_context,
            model.max_output_tokens,
            model.supports_vision,
            model.supports_function_calling,
            model.supports_json_mode,
            model.supports_parallel_tool_calls,
            model.tool_call_format,
            model.dollars_per_million_tokens_input,
            model.dollars_per_million_tokens_output,
            model.inactive_from,  # Pass inactive_from timestamp
        )

        return model_id

    async def update_model_costs(
        self,
        provider: str,
        model_name: str,
        dollars_per_million_tokens_input: Decimal,
        dollars_per_million_tokens_output: Decimal,
    ) -> bool:
        """Update model costs.

        Args:
            provider: Model provider
            model_name: Model name
            dollars_per_million_tokens_input: Cost in dollars per million input tokens
            dollars_per_million_tokens_output: Cost in dollars per million output tokens

        Returns:
            True if updated, False if model not found
        """
        if not self._initialized:
            await self.initialize()

        query = self.db._prepare_query(
            """
            UPDATE {{tables.llm_models}}
            SET dollars_per_million_tokens_input = $3,
                dollars_per_million_tokens_output = $4,
                updated_at = CURRENT_TIMESTAMP
            WHERE provider = $1 AND model_name = $2
        """
        )

        result = await self.db.execute(
            query,
            provider,
            model_name,
            dollars_per_million_tokens_input,
            dollars_per_million_tokens_output,
        )

        return result != "UPDATE 0"

    # ===== API CALL TRACKING =====

    async def record_api_call(
        self,
        origin: str,
        id_at_origin: str,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: int,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        json_mode: bool = False,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        status: str = "success",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> UUID:
        """Record an API call for tracking and analytics.

        Args:
            origin: The application or service making the call
            id_at_origin: User/session identifier in the origin system
            provider: LLM provider (anthropic, openai, google, ollama)
            model_name: Model used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            response_time_ms: Response time in milliseconds
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            top_p: Top-p parameter
            stream: Whether streaming was used
            stop_sequences: Stop sequences used
            system_prompt: System prompt used
            tools_used: List of tool names used
            json_mode: Whether JSON mode was enabled
            response_format: Response format configuration
            seed: Random seed used
            tool_choice: Tool choice configuration
            parallel_tool_calls: Whether parallel tool calls were enabled
            status: Call status (success, error)
            error_type: Type of error if any
            error_message: Error message if any

        Returns:
            UUID of the recorded call
        """
        if not self._initialized:
            await self.initialize()

        # Calculate total tokens
        total_tokens = prompt_tokens + completion_tokens

        # Get model cost information, guarding for None values
        model = await self.get_model(provider, model_name)
        if model and model.dollars_per_million_tokens_input is not None and model.dollars_per_million_tokens_output is not None:
            estimated_cost = float(
                (
                    prompt_tokens * model.dollars_per_million_tokens_input
                    + completion_tokens * model.dollars_per_million_tokens_output
                )
                / 1_000_000
            )
        else:
            estimated_cost = 0.0

        # Prepare JSON fields
        stop_sequences_json = json.dumps(stop_sequences) if stop_sequences else None
        tools_used_json = json.dumps(tools_used) if tools_used else None
        response_format_json = json.dumps(response_format) if response_format else None

        query = self.db._prepare_query(
            """
            INSERT INTO {{tables.llm_api_calls}} (
                origin, id_at_origin, provider, model_name,
                prompt_tokens, completion_tokens, total_tokens,
                response_time_ms, temperature, max_tokens,
                top_p, stream, stop_sequences,
                tools_used, json_mode, response_format, seed,
                tool_choice, parallel_tool_calls, status,
                error_type, error_message, estimated_cost
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23
            ) RETURNING id
        """
        )

        call_id = await self.db.fetch_value(
            query,
            origin,
            id_at_origin,
            provider,
            model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            response_time_ms,
            temperature,
            max_tokens,
            top_p,
            stream,
            stop_sequences_json,
            tools_used_json,
            json_mode,
            response_format_json,
            seed,
            tool_choice,
            parallel_tool_calls,
            status,
            error_type,
            error_message,
            estimated_cost,
        )

        return UUID(str(call_id))

    async def get_call_record(self, call_id: UUID) -> Optional[CallRecord]:
        """Get a specific call record by ID.

        Args:
            call_id: UUID of the call

        Returns:
            CallRecord instance or None if not found
        """
        if not self._initialized:
            await self.initialize()

        query = self.db._prepare_query(
            """
            SELECT * FROM {{tables.llm_api_calls}} WHERE id = $1
        """
        )
        result = await self.db.fetch_one(query, str(call_id))

        if not result:
            return None

        return CallRecord(
            id=UUID(str(result["id"])),
            origin=result["origin"],
            id_at_origin=result["id_at_origin"],
            provider=result["provider"],
            model_name=result["model_name"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
            response_time_ms=result["response_time_ms"],
            temperature=result["temperature"],
            max_tokens=result["max_tokens"],
            top_p=result["top_p"],
            stream=result["stream"],
            stop_sequences=(
                json.loads(result["stop_sequences"])
                if result["stop_sequences"]
                else None
            ),
            system_prompt_hash=None,
            tools_used=(
                json.loads(result["tools_used"]) if result["tools_used"] else None
            ),
            json_mode=result["json_mode"],
            response_format=(
                json.loads(result["response_format"])
                if result["response_format"]
                else None
            ),
            seed=result["seed"],
            tool_choice=result["tool_choice"],
            parallel_tool_calls=result["parallel_tool_calls"],
            status=result["status"],
            error_type=result["error_type"],
            error_message=result["error_message"],
            estimated_cost=result["estimated_cost"],
            called_at=result["called_at"],
        )

    async def get_usage_stats(
        self, origin: str, id_at_origin: str, days: int = 30
    ) -> Optional[UsageStats]:
        """Get usage statistics for a specific user.

        Args:
            origin: The application or service
            id_at_origin: User/session identifier
            days: Number of days to look back

        Returns:
            UsageStats instance or None if no calls found
        """
        if not self._initialized:
            await self.initialize()

        query = self.db._prepare_query(
            """
            WITH recent_calls AS (
                SELECT
                    COUNT(*) as total_calls,
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(estimated_cost) as total_cost,
                    AVG(response_time_ms) as avg_response_time_ms,
                    COUNT(CASE WHEN status = 'success' THEN 1 END)::float / NULLIF(COUNT(*), 0) as success_rate,
                    COUNT(DISTINCT provider) as providers_used,
                    COUNT(DISTINCT model_name) as models_used
                FROM {{tables.llm_api_calls}}
                WHERE origin = $1
                AND id_at_origin = $2
                AND called_at >= CURRENT_TIMESTAMP - $3 * INTERVAL '1 day'
            ),
            most_used AS (
                SELECT model_name, COUNT(*) as usage_count
                FROM {{tables.llm_api_calls}}
                WHERE origin = $1
                AND id_at_origin = $2
                AND called_at >= CURRENT_TIMESTAMP - $3 * INTERVAL '1 day'
                GROUP BY model_name
                ORDER BY usage_count DESC
                LIMIT 1
            )
            SELECT
                rc.*,
                mu.model_name as most_used_model
            FROM recent_calls rc
            LEFT JOIN most_used mu ON true
        """
        )

        result = await self.db.fetch_one(query, origin, id_at_origin, days)

        if not result or result["total_calls"] == 0:
            return None

        total_cost = float(result["total_cost"])
        total_calls = int(result["total_calls"])
        avg_cost = total_cost / total_calls if total_calls > 0 else 0.0

        return UsageStats(
            total_calls=total_calls,
            total_tokens=int(result["total_tokens"]),
            total_cost=Decimal(str(total_cost)),
            avg_cost_per_call=Decimal(str(avg_cost)),
            avg_response_time_ms=(
                int(round(result["avg_response_time_ms"]))
                if result["avg_response_time_ms"]
                else None
            ),
            success_rate=Decimal(str(result["success_rate"])),
            most_used_model=result["most_used_model"],
        )

    async def list_recent_calls(
        self,
        origin: str,
        id_at_origin: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CallRecord]:
        """List recent API calls.

        Args:
            origin: The application or service
            id_at_origin: Optional user/session filter
            limit: Maximum number of records
            offset: Pagination offset

        Returns:
            List of CallRecord instances
        """
        if not self._initialized:
            await self.initialize()

        if id_at_origin:
            query = self.db._prepare_query(
                """
                SELECT * FROM {{tables.llm_api_calls}}
                WHERE origin = $1 AND id_at_origin = $2
                ORDER BY called_at DESC
                LIMIT $3 OFFSET $4
            """
            )
            results = await self.db.fetch_all(
                query, origin, id_at_origin, limit, offset
            )
        else:
            query = self.db._prepare_query(
                """
                SELECT * FROM {{tables.llm_api_calls}}
                WHERE origin = $1
                ORDER BY called_at DESC
                LIMIT $2 OFFSET $3
            """
            )
            results = await self.db.fetch_all(query, origin, limit, offset)

        call_records = []
        for row in results:
            call_records.append(
                CallRecord(
                    id=UUID(str(row["id"])),
                    origin=row["origin"],
                    id_at_origin=row["id_at_origin"],
                    provider=row["provider"],
                    model_name=row["model_name"],
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    total_tokens=row["total_tokens"],
                    response_time_ms=row["response_time_ms"],
                    temperature=row["temperature"],
                    max_tokens=row["max_tokens"],
                    top_p=row["top_p"],
                    stream=row["stream"],
                    stop_sequences=(
                        json.loads(row["stop_sequences"])
                        if row["stop_sequences"]
                        else None
                    ),
                    system_prompt_hash=None,
                    tools_used=(
                        json.loads(row["tools_used"]) if row["tools_used"] else None
                    ),
                    json_mode=row["json_mode"],
                    response_format=(
                        json.loads(row["response_format"])
                        if row["response_format"]
                        else None
                    ),
                    seed=row["seed"],
                    tool_choice=row["tool_choice"],
                    parallel_tool_calls=row["parallel_tool_calls"],
                    status=row["status"],
                    error_type=row["error_type"],
                    error_message=row["error_message"],
                    estimated_cost=row["estimated_cost"],
                    called_at=row["called_at"],
                )
            )

        return call_records

    # ===== ANALYTICS =====

    async def aggregate_daily_analytics(self, date: Optional[str] = None) -> None:
        """Aggregate daily analytics.

        Args:
            date: Date to aggregate (YYYY-MM-DD format). If None, yesterday.
        """
        if not self._initialized:
            await self.initialize()

        if date:
            await self.db.execute("SELECT aggregate_daily_analytics($1)", date)
        else:
            await self.db.execute("SELECT aggregate_daily_analytics()")

    async def cleanup_old_data(
        self, days_to_keep_calls: int = 90, days_to_keep_analytics: int = 365
    ) -> Tuple[str, str]:
        """Clean up old data.

        Args:
            days_to_keep_calls: Days to keep individual API call records
            days_to_keep_analytics: Days to keep analytics data

        Returns:
            Tuple of (deleted_calls, deleted_analytics)
        """
        if not self._initialized:
            await self.initialize()

        # Delete old API calls
        query_calls = self.db._prepare_query(
            """
            DELETE FROM {{tables.llm_api_calls}}
            WHERE called_at < CURRENT_TIMESTAMP - $1 * INTERVAL '1 day'
        """
        )
        deleted_calls = await self.db.execute(query_calls, days_to_keep_calls)

        # Delete old analytics
        query_analytics = self.db._prepare_query(
            """
            DELETE FROM {{tables.usage_analytics_daily}}
            WHERE date < CURRENT_DATE - $1 * INTERVAL '1 day'
        """
        )
        deleted_analytics = await self.db.execute(
            query_analytics, days_to_keep_analytics
        )

        return deleted_calls, deleted_analytics

    # ===== MONITORING AND HEALTH =====

    async def get_pool_stats(self) -> Dict[str, any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        return await self.db.get_pool_stats()

    async def get_slow_queries(self, threshold_ms: Optional[int] = None) -> List[Dict]:
        """Get slow queries from monitoring.

        Args:
            threshold_ms: Override the default threshold (100ms)

        Returns:
            List of slow query records if monitoring is enabled
        """
        if not self._initialized:
            return []

        if isinstance(self.db, MonitoredAsyncDatabaseManager):
            return self.db.get_slow_queries(threshold_ms)
        return []

    async def get_query_metrics(self) -> Optional[Dict]:
        """Get query performance metrics.

        Returns:
            Dictionary with metrics if monitoring is enabled
        """
        if not self._initialized:
            return None

        if isinstance(self.db, MonitoredAsyncDatabaseManager):
            return await self.db.get_metrics()
        return None

    async def health_check(self) -> Dict[str, any]:
        """Perform a health check on the database.

        Returns:
            Dictionary with health status
        """
        try:
            if not self._initialized:
                return {"status": "unhealthy", "error": "Database not initialized"}

            # Test database connectivity
            await self.db.execute("SELECT 1")

            # Get pool stats
            pool_stats = await self.get_pool_stats()

            # Get metrics if available
            metrics = await self.get_query_metrics()

            return {
                "status": "healthy",
                "pool": pool_stats,
                "metrics": metrics,
                "schema": getattr(self.db, "schema", None),
                "monitoring_enabled": isinstance(
                    self.db, MonitoredAsyncDatabaseManager
                ),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
