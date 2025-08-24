"""
Database models for MCP client using pgdbm-utils.

This module provides async database operations for the MCP client, focusing only on
client-specific data (chat sessions, messages, MCP servers). LLM model information
is managed by llmbridge.
"""

import asyncio
import contextlib
import json
import logging
import os
import uuid
from typing import Any

import asyncpg
from pgdbm import AsyncDatabaseManager, AsyncMigrationManager, DatabaseConfig
from pgdbm.monitoring import MonitoredAsyncDatabaseManager
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llmring.mcp.client.pool_config import STANDALONE_POOL

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""

    pass


class MCPClientDB:
    """Async database access for MCP client using pgdbm-utils."""

    @classmethod
    def from_manager(cls, db_manager: AsyncDatabaseManager) -> "MCPClientDB":
        """
        Create MCPClientDB instance from an existing database manager (shared pool pattern).

        Note: The db_manager should already have the correct schema configured.
        If you need a different schema, create a new AsyncDatabaseManager instance
        with the shared pool:

        Example:
            shared_pool = await create_shared_pool(config)
            mcp_db_manager = AsyncDatabaseManager(pool=shared_pool, schema="mcp_client")
            mcp_client = MCPClientDB.from_manager(mcp_db_manager)

        Args:
            db_manager: Existing AsyncDatabaseManager instance with correct schema

        Returns:
            MCPClientDB instance using the shared pool
        """
        return cls(db_manager=db_manager)

    def __init__(
        self,
        connection_string: str | None = None,
        db_manager: AsyncDatabaseManager | None = None,
        schema: str = "mcp_client",
        min_connections: int | None = None,
        max_connections: int | None = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the async database manager.

        Args:
            connection_string: Database connection string (for standalone mode)
            db_manager: External database manager (for integrated mode with shared pool)
            schema: Schema name for database operations
            min_connections: Minimum number of connections in the pool (standalone mode only)
            max_connections: Maximum number of connections in the pool (standalone mode only)
            enable_monitoring: Enable query monitoring and metrics (standalone mode only)
        """
        if db_manager:
            # Use provided manager (integrated mode with shared pool)
            self.db = db_manager
            self._external_db = True
            self.config = self.db.config
            # Note: Schema is managed by the caller who creates the db_manager.
            # If a different schema is needed, the caller should create a new
            # AsyncDatabaseManager instance with the desired schema.
        else:
            # Create own manager (standalone mode)
            if connection_string is None:
                connection_string = os.getenv(
                    "DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres"
                )

            # Use standardized standalone pool configuration if not specified
            if min_connections is None:
                min_connections = STANDALONE_POOL.min_connections
            if max_connections is None:
                max_connections = STANDALONE_POOL.max_connections

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
                self.db = MonitoredAsyncDatabaseManager(
                    self.config, slow_query_threshold_ms=100, max_history_size=1000
                )
            else:
                self.db = AsyncDatabaseManager(self.config)

            self._external_db = False

        self._initialized = False
        self._prepared_statements_added = False
        self._shutdown_event = asyncio.Event()
        self._monitor_task = None

    async def initialize(self):
        """Initialize database connection and ensure schema exists."""
        if self._initialized:
            return

        try:
            # Only connect if we own the database manager
            if not self._external_db:
                await self.db.connect()

            # Add prepared statements for frequently used queries
            if not self._prepared_statements_added:
                self._add_prepared_statements()
                self._prepared_statements_added = True

            # Initialize migrations
            import os

            pkg_dir = os.path.dirname(os.path.dirname(__file__))
            migrations_path = os.path.join(pkg_dir, "migrations")

            self.migration_manager = AsyncMigrationManager(
                self.db, migrations_path=migrations_path, module_name="mcp_client"
            )

            # Start monitoring agent if enabled and we own the db
            if not self._external_db and isinstance(self.db, MonitoredAsyncDatabaseManager):
                self._monitor_task = asyncio.create_task(self._monitor_pool_health())

            self._initialized = True

        except asyncpg.exceptions.PostgresError as e:
            raise DatabaseError(f"Failed to initialize database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise

    def _add_prepared_statements(self):
        """Add prepared statements for frequently executed queries."""
        # Chat message insertion
        self.db.add_prepared_statement(
            "save_chat_message",
            """
            INSERT INTO {{tables.chat_messages}} (
                session_id, role, content, metadata, token_count
            ) VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
        )

        # Session retrieval
        self.db.add_prepared_statement(
            "get_chat_session",
            """
            SELECT id, created_by as user_id, title, system_prompt, model,
                   temperature, max_tokens, tool_config, created_at, updated_at
            FROM {{tables.chat_sessions}}
            WHERE id = $1
            """,
        )

        # MCP server lookup
        self.db.add_prepared_statement(
            "get_mcp_server",
            """
            SELECT id, name, base_url, transport_type, auth_type,
                   auth_config, capabilities, metadata, is_active,
                   created_at, updated_at
            FROM {{tables.mcp_servers}}
            WHERE id = $1
            """,
        )

    async def _monitor_pool_health(self):
        """Monitor connection pool health and log warnings."""
        while not self._shutdown_event.is_set():
            try:
                stats = await self.get_pool_stats()

                # Warn if pool is exhausted
                if stats.get("free_size", 0) == 0:
                    logger.warning("Connection pool exhausted!")

                # Warn if pool is near capacity
                pool_usage = stats.get("used_size", 0) / stats.get("size", 1)
                if pool_usage > 0.8:
                    logger.warning(f"Connection pool near capacity: {pool_usage:.1%}")

                # Log slow queries
                if isinstance(self.db, MonitoredAsyncDatabaseManager):
                    slow_queries = self.db.get_slow_queries(threshold_ms=100)
                    for query in slow_queries[-5:]:  # Last 5 slow queries
                        logger.warning(
                            f"Slow query ({query.duration_ms}ms): {query.query[:100]}..."
                        )

            except Exception as e:
                logger.error(f"Error in pool health monitoring: {e}")

            # Check every 30 seconds
            await asyncio.sleep(30)

    async def close(self):
        """Close database connection with graceful shutdown."""
        if self._initialized:
            # Signal shutdown
            self._shutdown_event.set()

            # Cancel monitoring agent if running
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitor_task

            # Only disconnect if we own the database manager
            if not self._external_db:
                # Wait for ongoing queries
                pool_stats = await self.get_pool_stats()
                retry_count = 0
                while pool_stats.get("used_size", 0) > 0 and retry_count < 30:
                    await asyncio.sleep(0.1)
                    pool_stats = await self.get_pool_stats()
                    retry_count += 1

                if retry_count >= 30:
                    logger.warning("Timeout waiting for connections to close")

                await self.db.disconnect()

            self._initialized = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(asyncpg.exceptions.PostgresConnectionError),
    )
    async def execute_with_retry(self, query: str, *args):
        """Execute query with automatic retry on connection errors."""
        if not self._initialized:
            await self.initialize()
        return await self.db.execute(query, *args)

    async def run_migrations(self) -> dict[str, list[str]]:
        """
        Run database migrations.

        Returns:
            Dictionary with applied and pending migrations
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self.migration_manager.apply_pending_migrations()
            return result
        except Exception as e:
            raise DatabaseError(f"Failed to run migrations: {e}")

    async def init_schema(self) -> dict[str, list[str]]:
        """
        Initialize database schema by running all migrations.
        Legacy alias for run_migrations.

        Returns:
            Dictionary with applied migrations info
        """
        return await self.run_migrations()

    async def reset_database(self) -> None:
        """
        Reset the database by dropping all tables in the schema and recreating them.
        This is destructive and will delete all data.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get the actual schema name from config
            # The config.schema_name is the field name, we need to get the value
            schema_name = "mcp_client"  # Default
            if self.config and hasattr(self.config, "schema_name"):
                schema_name = self.config.schema_name

            # Drop schema (cascade will drop all objects)
            query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
            await self.db.execute(query)

            # Recreate schema
            query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
            await self.db.execute(query)

            # Run migrations
            await self.run_migrations()

        except Exception as e:
            raise DatabaseError(f"Failed to reset database: {e}")

    # MCP server methods
    async def add_mcp_server(
        self,
        name: str,
        base_url: str,
        transport_type: str = "http",
        auth_type: str = "none",
        auth_config: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        is_active: bool = True,
        created_by: str = "system",
    ) -> str:
        """
        Add a new MCP server configuration.

        Returns:
            ID of the new server
        """
        if not self._initialized:
            await self.initialize()

        server_id = str(uuid.uuid4())

        query = """
        INSERT INTO {{tables.mcp_servers}} (
            id, name, base_url, transport_type, auth_type,
            auth_config, capabilities, metadata, is_active, created_by
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (base_url) DO UPDATE
        SET
            name = EXCLUDED.name,
            transport_type = EXCLUDED.transport_type,
            auth_type = EXCLUDED.auth_type,
            auth_config = EXCLUDED.auth_config,
            capabilities = EXCLUDED.capabilities,
            metadata = EXCLUDED.metadata,
            is_active = EXCLUDED.is_active,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id;
        """

        try:
            result = await self.db.fetch_one(
                query,
                server_id,
                name,
                base_url,
                transport_type,
                auth_type,
                json.dumps(auth_config) if auth_config else None,
                json.dumps(capabilities) if capabilities else None,
                json.dumps(metadata) if metadata else None,
                is_active,
                created_by,
            )

            return result["id"] if result else server_id

        except asyncpg.exceptions.UniqueViolationError:
            raise DatabaseError(f"MCP server with URL {base_url} already exists")
        except Exception as e:
            raise DatabaseError(f"Failed to add MCP server: {e}")

    async def get_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        """
        Get MCP server configuration by ID.

        Args:
            server_id: Server ID to retrieve

        Returns:
            Server configuration dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use prepared statement
            result = await self.db.fetch_one(
                """
                SELECT id, name, base_url, transport_type, auth_type,
                       auth_config, capabilities, metadata, is_active,
                       created_at, updated_at
                FROM {{tables.mcp_servers}}
                WHERE id = $1
                """,
                server_id,
            )

            if not result:
                return None

            server = dict(result)
            # Parse JSON fields
            for field in ["auth_config", "capabilities", "metadata"]:
                if server[field] and isinstance(server[field], str):
                    try:
                        server[field] = json.loads(server[field])
                    except json.JSONDecodeError:
                        server[field] = None

            return server

        except Exception as e:
            raise DatabaseError(f"Failed to get MCP server: {e}")

    async def get_mcp_servers(self, active_only: bool = True) -> list[dict[str, Any]]:
        """
        Get all MCP server configurations.

        Args:
            active_only: Only return active servers

        Returns:
            List of server configuration dictionaries
        """
        if not self._initialized:
            await self.initialize()

        query = """
        SELECT id, name, base_url, transport_type, auth_type,
               auth_config, capabilities, metadata, is_active,
               created_at, updated_at
        FROM {{tables.mcp_servers}}
        """

        if active_only:
            query += " WHERE is_active = true"

        query += " ORDER BY name"

        try:
            results = await self.db.fetch_all(query)

            servers = []
            for row in results:
                server = dict(row)

                # Parse JSON fields
                for field in ["auth_config", "capabilities", "metadata"]:
                    if server[field] and isinstance(server[field], str):
                        try:
                            server[field] = json.loads(server[field])
                        except json.JSONDecodeError:
                            server[field] = None

                servers.append(server)

            return servers

        except Exception as e:
            raise DatabaseError(f"Failed to get MCP servers: {e}")

    # Chat session methods
    async def create_chat_session(
        self,
        user_id: str | None = None,
        title: str | None = None,
        system_prompt: str | None = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new chat session.

        Args:
            user_id: User ID (required)
            title: Optional session title
            system_prompt: Optional system prompt
            model: Model to use
            temperature: Temperature setting
            max_tokens: Max tokens setting
            tool_config: Tool configuration

        Returns:
            ID of the new chat session
        """
        if not self._initialized:
            await self.initialize()

        if not user_id:
            user_id = "anonymous"

        session_id = str(uuid.uuid4())

        query = """
        INSERT INTO {{tables.chat_sessions}} (
            id, created_by, title, system_prompt, model,
            temperature, max_tokens, tool_config
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """

        try:
            result = await self.db.fetch_one(
                query,
                session_id,
                user_id,
                title,
                system_prompt,
                model,
                temperature,
                max_tokens,
                json.dumps(tool_config) if tool_config else None,
            )
            return result["id"] if result else session_id

        except Exception as e:
            raise DatabaseError(f"Failed to create chat session: {e}")

    async def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> int:
        """
        Save a chat message to history.

        Args:
            session_id: Chat session ID
            role: Message role (system, user, assistant)
            content: Message content
            metadata: Optional message metadata
            token_count: Optional token count

        Returns:
            ID of the saved message
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use prepared statement for better performance
            result = await self.db.fetch_one(
                """
                INSERT INTO {{tables.chat_messages}} (
                    session_id, role, content, metadata, token_count
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                session_id,
                role,
                content,
                json.dumps(metadata) if metadata else None,
                token_count,
            )
            return result["id"] if result else None

        except asyncpg.exceptions.ForeignKeyViolationError:
            raise DatabaseError(f"Chat session {session_id} not found")
        except Exception as e:
            # Normalize FK violations that may be wrapped by database manager
            msg = str(e).lower()
            if "violates foreign key constraint" in msg and (
                "chat_messages_session_id_fkey" in msg
                or 'not present in table "chat_sessions"' in msg
            ):
                raise DatabaseError(f"Chat session {session_id} not found")
            raise DatabaseError(f"Failed to save chat message: {e}")

    async def get_chat_messages(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get all messages for a chat session.

        Args:
            session_id: Chat session ID

        Returns:
            List of message dictionaries
        """
        if not self._initialized:
            await self.initialize()

        query = """
        SELECT id, role, content, metadata, token_count, timestamp
        FROM {{tables.chat_messages}}
        WHERE session_id = $1
        ORDER BY timestamp
        """

        try:
            results = await self.db.fetch_all(query, session_id)

            messages = []
            for row in results:
                message = dict(row)

                # Parse metadata JSON
                if message["metadata"] and isinstance(message["metadata"], str):
                    try:
                        message["metadata"] = json.loads(message["metadata"])
                    except json.JSONDecodeError:
                        message["metadata"] = None

                messages.append(message)

            return messages

        except Exception as e:
            raise DatabaseError(f"Failed to get chat messages: {e}")

    async def get_chat_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Get a chat session with its messages.

        Args:
            session_id: Chat session ID

        Returns:
            Session dictionary with messages or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use prepared statement
            session = await self.db.fetch_one(
                """
                SELECT id, created_by as user_id, title, system_prompt, model,
                       temperature, max_tokens, tool_config, created_at, updated_at
                FROM {{tables.chat_sessions}}
                WHERE id = $1
                """,
                session_id,
            )

            if not session:
                return None

            # Get messages
            session = dict(session)

            # Parse tool_config JSON
            if session["tool_config"] and isinstance(session["tool_config"], str):
                try:
                    session["tool_config"] = json.loads(session["tool_config"])
                except json.JSONDecodeError:
                    session["tool_config"] = None

            session["messages"] = await self.get_chat_messages(session_id)

            return session

        except Exception as e:
            raise DatabaseError(f"Failed to get chat session: {e}")

    # Usage tracking methods
    async def record_usage(
        self,
        user_id: str | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float = 0.0,
        conversation_id: str | None = None,
    ) -> None:
        """
        Record usage for analytics and billing.

        Args:
            user_id: User ID (optional)
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            estimated_cost: Estimated cost in dollars
            conversation_id: Optional conversation ID
        """
        if not self._initialized:
            await self.initialize()

        # Extract provider from model name
        provider = model.split(":")[0] if ":" in model else "unknown"

        query = """
        INSERT INTO {{tables.usage_analytics}} (
            user_id, conversation_id, model, provider,
            input_tokens, output_tokens, total_tokens,
            input_cost, output_cost, total_cost
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        total_tokens = input_tokens + output_tokens

        try:
            await self.db.execute(
                query,
                user_id or "anonymous",
                conversation_id,
                model,
                provider,
                input_tokens,
                output_tokens,
                total_tokens,
                (estimated_cost * (input_tokens / total_tokens) if total_tokens > 0 else 0),
                (estimated_cost * (output_tokens / total_tokens) if total_tokens > 0 else 0),
                estimated_cost,
            )
        except Exception as e:
            # Don't let usage tracking errors break the main flow
            logger.error(f"Failed to record usage: {e}")

    # Health and monitoring methods
    async def get_pool_stats(self) -> dict[str, any]:
        """Get connection pool statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return await self.db.get_pool_stats()

    async def get_slow_queries(self, threshold_ms: int | None = None) -> list[dict]:
        """Get slow queries from monitoring."""
        if not self._initialized:
            return []

        if isinstance(self.db, MonitoredAsyncDatabaseManager):
            return self.db.get_slow_queries(threshold_ms)
        return []

    async def get_query_metrics(self) -> dict | None:
        """Get query performance metrics."""
        if not self._initialized:
            return None

        if isinstance(self.db, MonitoredAsyncDatabaseManager):
            return await self.db.get_metrics()
        return None

    async def health_check(self) -> dict[str, any]:
        """Perform a comprehensive health check on the database."""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "error": "Database not initialized"}

            # Test database connectivity
            await self.db.execute("SELECT 1")

            # Determine schema to inspect (avoid BaseModel.schema() collision)
            schema_name = None
            # First, try to get a safe value from config if present
            if hasattr(self, "config") and self.config is not None:
                try:
                    if hasattr(self.config, "model_dump"):
                        dumped = self.config.model_dump()
                        schema_name = dumped.get("schema") or dumped.get("schema_name")
                except Exception:
                    schema_name = None
                if not schema_name:
                    candidate = getattr(self.config, "schema", None)
                    if not callable(candidate):
                        schema_name = candidate
                    if not schema_name:
                        candidate2 = getattr(self.config, "schema_name", None)
                        if not callable(candidate2):
                            schema_name = candidate2
            # Fall back to db manager's schema
            if not schema_name:
                schema_name = getattr(self.db, "schema", None)

            if not schema_name:
                schema_name = "public"

            # Check schema exists (inline literal to avoid parameterizing a callable)
            schema_literal = str(schema_name).replace("'", "''")
            schema_exists = await self.db.fetch_value(
                f"SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = '{schema_literal}')"
            )

            if not schema_exists:
                return {
                    "status": "unhealthy",
                    "error": f"Schema {schema_name} does not exist",
                }

            # Check critical tables
            critical_tables = ["chat_sessions", "chat_messages", "mcp_servers"]
            for table in critical_tables:
                table_exists = await self.db.table_exists(f"{schema_name}.{table}")
                if not table_exists:
                    return {
                        "status": "unhealthy",
                        "error": f"Missing critical table: {table}",
                    }

            # Get pool stats
            pool_stats = await self.get_pool_stats()

            # Get metrics if available
            metrics = await self.get_query_metrics()

            # Check current schema version
            schema_version = await self.db.fetch_value(
                "SELECT MAX(filename) FROM {{schema}}.schema_migrations WHERE module_name = $1",
                "mcp_client",
            )

            return {
                "status": "healthy",
                "pool": pool_stats,
                "metrics": metrics,
                "schema": schema_name,
                "schema_version": schema_version,
                "monitoring_enabled": isinstance(self.db, MonitoredAsyncDatabaseManager),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
