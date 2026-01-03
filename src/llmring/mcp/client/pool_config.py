"""Connection pool configuration for MCP clients. Manages multiple MCP server connections efficiently."""

from dataclasses import dataclass


@dataclass
class PoolConfig:
    """Database connection pool configuration (placeholder)."""

    min_connections: int
    max_connections: int
    description: str


# Placeholder configurations - not actually used
CHAT_APP_POOL = PoolConfig(
    min_connections=20,
    max_connections=50,
    description="For interactive chat application (not used)",
)

CLI_QUERY_POOL = PoolConfig(
    min_connections=10,
    max_connections=30,
    description="For CLI query commands (not used)",
)

CLI_LIST_POOL = PoolConfig(
    min_connections=10,
    max_connections=30,
    description="For CLI list commands (not used)",
)
