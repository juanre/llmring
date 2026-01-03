"""Deprecated script for listing models.

This script originally depended on a DB-backed MCP client implementation that no longer exists.
It is kept as a placeholder for future updates and currently exits with an explanatory message.
"""

import argparse
import asyncio

from dotenv import load_dotenv

# Database model removed - now using HTTP-based architecture
# MCPClientDB functionality has been removed
# This script is deprecated and needs to be rewritten for the new architecture


async def list_models(
    db_path: str | None = None, provider_filter: str | None = None, verbose: bool = False
) -> None:
    """List all LLM models in the database."""
    print("Listing LLM models in database...\n")

    # Database functionality removed - this script needs updating
    print("ERROR: Database functionality has been removed.")
    print("This script needs to be updated to work with the new HTTP-based architecture.")
    return
    # Original code preserved for reference:
    # db = MCPClientDB(db_path)

    # Code removed - database functionality no longer available
    pass


def get_all_provider_models() -> dict:
    """Legacy helper retained for backward compatibility.

    Returns an empty mapping because provider enumeration used to rely on SDK-specific
    discovery calls that are no longer part of the provider interface.
    """
    return {}


async def check_missing_models(db_path: str | None = None, update: bool = False) -> None:
    """Check for missing models in the database."""
    print("Checking for missing models...\n")

    # Database functionality removed - this script needs updating
    print("ERROR: Database functionality has been removed.")
    print("This script needs to be updated to work with the new HTTP-based architecture.")
    return
    # Original code preserved for reference:
    # db = MCPClientDB(db_path)

    # Code removed - database functionality no longer available
    pass


async def main():
    parser = argparse.ArgumentParser(description="MCP Client LLM Models List Script")
    parser.add_argument(
        "--db-path",
        help="Database connection string (uses DATABASE_URL env var if not specified)",
    )
    parser.add_argument("--env-file", help="Path to the .env file", default=".env")
    parser.add_argument(
        "--provider",
        help="Filter models by provider (e.g., anthropic, openai, google, ollama)",
        choices=["anthropic", "openai", "google", "ollama"],
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed model information"
    )
    parser.add_argument("--check", "-c", action="store_true", help="Check for missing models")
    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Update database with missing models (implies --check)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    if args.update:
        # Update implies check
        await check_missing_models(args.db_path, update=True)
    elif args.check:
        await check_missing_models(args.db_path)
    else:
        # Default: list models
        await list_models(args.db_path, args.provider, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
