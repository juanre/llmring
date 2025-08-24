"""Database wrapper for mcp-client using pgdbm-utils following best practices."""

import os

from pgdbm import AsyncDatabaseManager, DatabaseConfig
from pgdbm.monitoring import MonitoredAsyncDatabaseManager

__all__ = [
    "AsyncDatabaseManager",
    "DatabaseConfig",
    "create_mcp_db_config",
    "create_mcp_db_manager",
]


def create_mcp_db_config(
    connection_string: str | None = None,
    schema: str = "mcp_client",
    min_connections: int = 10,
    max_connections: int = 20,
) -> DatabaseConfig:
    """
    Create a DatabaseConfig for MCP client.

    Args:
        connection_string: Database connection string, defaults to environment variable
        schema: Database schema name, defaults to 'mcp_client'
        min_connections: Minimum pool connections
        max_connections: Maximum pool connections

    Returns:
        DatabaseConfig instance
    """
    if connection_string is None:
        connection_string = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres"
        )

    return DatabaseConfig(
        connection_string=connection_string,
        schema=schema,
        min_connections=min_connections,
        max_connections=max_connections,
        max_queries=50000,
        max_inactive_connection_lifetime=300.0,
        command_timeout=60.0,
    )


def create_mcp_db_manager(
    connection_string: str | None = None,
    schema: str = "mcp_client",
    enable_monitoring: bool = True,
) -> AsyncDatabaseManager:
    """
    Create an AsyncDatabaseManager for MCP client (without connecting).

    Args:
        connection_string: Database connection string, defaults to environment variable
        schema: Database schema name, defaults to 'mcp_client'
        enable_monitoring: Enable query monitoring

    Returns:
        AsyncDatabaseManager instance configured for MCP client

    Note:
        The caller is responsible for calling connect() and disconnect()
    """
    config = create_mcp_db_config(connection_string, schema)

    if enable_monitoring:
        db = MonitoredAsyncDatabaseManager(config)
        # Set monitoring parameters after initialization
        db._slow_query_threshold_ms = 100
        db._max_history_size = 1000
        return db
    else:
        return AsyncDatabaseManager(config)
