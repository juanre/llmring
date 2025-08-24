"""
Shared database pool utilities for MCP client.

This module provides utilities for creating and managing a shared database
connection pool that can be used by multiple components (MCPClientDB, LLMBridge, etc.)
to avoid connection pool exhaustion.
"""

import os
from contextlib import asynccontextmanager

from pgdbm import AsyncDatabaseManager, DatabaseConfig
from pgdbm.monitoring import MonitoredAsyncDatabaseManager

from llmring.mcp.server.client.pool_config import SHARED_POOL


async def create_shared_pool(
    connection_string: str | None = None,
    min_connections: int | None = None,
    max_connections: int | None = None,
    enable_monitoring: bool = True,
    slow_query_threshold_ms: int = 100,
    max_history_size: int = 1000,
) -> AsyncDatabaseManager:
    """
    Create a shared database connection pool.

    This creates a single pool that can be shared by multiple services
    to avoid connection pool exhaustion.

    Args:
        connection_string: Database connection string, defaults to environment variable
        min_connections: Minimum connections in shared pool (defaults to SHARED_POOL config)
        max_connections: Maximum connections in shared pool (defaults to SHARED_POOL config)
        enable_monitoring: Enable query monitoring
        slow_query_threshold_ms: Threshold for slow query logging (milliseconds)
        max_history_size: Maximum number of queries to store in history

    Returns:
        AsyncDatabaseManager instance with shared pool
    """
    if connection_string is None:
        connection_string = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres"
        )

    # Use standard shared pool configuration if not specified
    if min_connections is None:
        min_connections = SHARED_POOL.min_connections
    if max_connections is None:
        max_connections = SHARED_POOL.max_connections

    config = DatabaseConfig(
        connection_string=connection_string,
        min_connections=min_connections,
        max_connections=max_connections,
        max_queries=50000,
        max_inactive_connection_lifetime=300.0,
        command_timeout=60.0,
        schema="public",  # Default schema, will be overridden by managers
    )

    if enable_monitoring:
        db = MonitoredAsyncDatabaseManager(
            config,
            slow_query_threshold_ms=slow_query_threshold_ms,
            max_history_size=max_history_size,
        )
    else:
        db = AsyncDatabaseManager(config)

    await db.connect()
    return db


@asynccontextmanager
async def shared_pool_context(
    connection_string: str | None = None,
    min_connections: int | None = None,
    max_connections: int | None = None,
    enable_monitoring: bool = True,
    slow_query_threshold_ms: int = 100,
    max_history_size: int = 1000,
):
    """
    Context manager for shared database pool lifecycle.

    Usage:
        async with shared_pool_context() as pool:
            # Create schema-specific managers
            mcp_db_manager = AsyncDatabaseManager(pool=pool, schema="mcp_client")
            llm_db_manager = AsyncDatabaseManager(pool=pool, schema="llmbridge")

            # Create services using the schema-specific managers
            mcp_db = MCPClientDB.from_manager(mcp_db_manager)
            llmbridge = LLMBridge(db_manager=llm_db_manager)

            # Use services...

    The pool is automatically closed when exiting the context.
    """
    pool = await create_shared_pool(
        connection_string=connection_string,
        min_connections=min_connections,
        max_connections=max_connections,
        enable_monitoring=enable_monitoring,
        slow_query_threshold_ms=slow_query_threshold_ms,
        max_history_size=max_history_size,
    )

    try:
        yield pool
    finally:
        await pool.disconnect()
