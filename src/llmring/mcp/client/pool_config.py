"""
Standardized connection pool configurations for MCP client.

This module defines standard pool sizes based on use case to ensure
consistent connection pool management across the codebase.
"""

from dataclasses import dataclass


@dataclass
class PoolConfig:
    """Database connection pool configuration."""

    min_connections: int
    max_connections: int
    description: str


# Standard pool configurations
STANDALONE_POOL = PoolConfig(
    min_connections=10,
    max_connections=10,
    description="For standalone CLI usage (single user)",
)

SHARED_POOL = PoolConfig(
    min_connections=50,
    max_connections=200,
    description="For shared pool (multiple services)",
)

CHAT_APP_POOL = PoolConfig(
    min_connections=20,
    max_connections=50,
    description="For interactive chat application",
)

TEST_POOL = PoolConfig(
    min_connections=2,
    max_connections=5,
    description="For testing with minimal connections",
)

# Pool configurations for specific use cases
CLI_QUERY_POOL = PoolConfig(
    min_connections=10, max_connections=30, description="For CLI query commands"
)

CLI_LIST_POOL = PoolConfig(
    min_connections=5, max_connections=10, description="For CLI listing operations"
)
