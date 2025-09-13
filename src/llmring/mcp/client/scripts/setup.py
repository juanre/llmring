#!/usr/bin/env python3
"""
MCP Client Setup Script

This script sets up the MCP client environment by:
1. Creating the necessary database tables
2. Adding default LLM model configurations
3. Configuring basic MCP server connections
4. Checking for required environment variables
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llmring.mcp.client import MCPClient
# Database model removed - now using HTTP-based architecture
# from llmring.mcp.client.models.db import MCPClientDB


def create_env_file(path: Path) -> None:
    """Create a template .env file if it doesn't exist."""
    if path.exists():
        print(f"Found existing .env file at {path}")
        return

    template = """# MCP Client environment variables
# LLM API Keys
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
OLLAMA_HOST=http://localhost:11434

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost/postgres
# Or use individual settings:
# MCPP_DB_HOST=localhost
# MCPP_DB_PORT=5432
# MCPP_DB_USER=postgres
# MCPP_DB_PASSWORD=postgres
# MCPP_DB_NAME=mcpp

# Default MCP Server
MCP_SERVER_URL=
MCP_SERVER_AUTH_TYPE=none  # none, bearer, api_key, oauth2
MCP_SERVER_AUTH_TOKEN=
"""

    with open(path, "w") as f:
        f.write(template)

    print(f"Created template .env file at {path}")
    print("Please edit this file to add your API keys and configuration")


async def setup_database(db_path: str | None = None) -> bool:
    """Set up the database with required tables."""
    print("Setting up MCP Client database...")

    # Initialize database connection
    db = MCPClientDB(connection_string=db_path)

    # Execute migrations
    try:
        await db.initialize()
        result = await db.run_migrations()
        print("✅ Database migrations completed successfully")
        print(f"   Applied: {len(result.get('applied', []))} migrations")
        print(f"   Skipped: {len(result.get('skipped', []))} migrations")
        return True
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False
    finally:
        await db.close()


async def check_llmring_setup() -> None:
    """Check if llmring is properly configured."""
    print("Checking LLM service configuration...")

    # Check if llmring is available
    try:
        from llmring.service import LLMRing

        service = LLMRing(origin="mcp-setup-check")
        models = service.get_available_models()

        print("✅ LLM service is available")
        print(f"   Available providers: {', '.join(models.keys())}")

        total_models = sum(len(m) for m in models.values())
        print(f"   Total models available: {total_models}")

    except ImportError:
        print("❌ LLM service not found. Please install llmring:")
        print("   uv add llmring")
    except Exception as e:
        print(f"⚠️ Error checking LLM service: {e}")


def check_env_variables():
    """Check for required environment variables and report status."""
    print("Checking environment variables...")

    vars_to_check = {
        "LLM API Keys": {
            "ANTHROPIC_API_KEY": False,
            "OPENAI_API_KEY": False,
            "GOOGLE_API_KEY": False,
            "OLLAMA_HOST": "http://localhost:11434",
        },
        "Database": {
            "DATABASE_URL": False,
            "MCPP_DB_HOST": "localhost",
            "MCPP_DB_PORT": "5432",
            "MCPP_DB_USER": "postgres",
            "MCPP_DB_PASSWORD": "postgres",
            "MCPP_DB_NAME": "mcpp",
        },
        "MCP Server": {
            "MCP_SERVER_URL": False,
            "MCP_SERVER_AUTH_TYPE": "none",
            "MCP_SERVER_AUTH_TOKEN": False,
        },
    }

    for category, variables in vars_to_check.items():
        print(f"\n{category}:")
        for var, default in variables.items():
            value = os.environ.get(var)
            if value:
                # Hide full values for API keys and auth tokens
                if "API_KEY" in var or "TOKEN" in var or "PASSWORD" in var:
                    shown_value = (
                        f"{value[:3]}...{value[-3:]}" if len(value) > 10 else "***"
                    )
                    print(f"  ✅ {var}: {shown_value}")
                else:
                    print(f"  ✅ {var}: {value}")
            elif default:
                print(f"  ℹ️ {var}: Using default ({default})")
            else:
                print(f"  ⚠️ {var}: Not set")


def test_mcp_connection():
    """Test connection to configured MCP server."""
    server_url = os.environ.get("MCP_SERVER_URL")
    if not server_url:
        print("\nℹ️ MCP server URL not configured. Skipping connection test.")
        return

    print(f"\nTesting connection to MCP server at {server_url}...")
    try:
        client = MCPClient(base_url=server_url)

        # Test connection by listing tools
        tools = client.list_tools()
        print("✅ Successfully connected to MCP server")
        print(f"  Available tools: {len(tools)}")
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")


async def main():
    parser = argparse.ArgumentParser(description="MCP Client Setup Script")
    parser.add_argument(
        "--db-path",
        help="Database connection string (uses DATABASE_URL env var if not specified)",
    )
    parser.add_argument("--env-file", help="Path to the .env file", default=".env")
    parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    parser.add_argument(
        "--skip-models", action="store_true", help="Skip adding default LLM models"
    )
    parser.add_argument(
        "--skip-env-check", action="store_true", help="Skip environment variable check"
    )
    parser.add_argument(
        "--skip-mcp-test", action="store_true", help="Skip MCP server connection test"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database and recreate from scratch",
    )

    args = parser.parse_args()

    # Create .env file if it doesn't exist
    env_path = Path(args.env_file)
    create_env_file(env_path)

    # Load environment variables
    load_dotenv(env_path)

    # Run setup steps
    if not args.skip_env_check:
        check_env_variables()

    if args.reset:
        print("\nResetting database...")
        db = MCPClientDB(connection_string=args.db_path)
        await db.initialize()
        await db.reset_database()
        print("✅ Database reset complete")
        await db.close()
    elif not args.skip_db:
        await setup_database(args.db_path)

    # Check LLM service instead of adding models
    await check_llmring_setup()

    if not args.skip_mcp_test:
        test_mcp_connection()

    print("\n✨ MCP Client setup completed! ✨")
    print("To start the chat interface, run: 'mcp-chat'")


if __name__ == "__main__":
    asyncio.run(main())
