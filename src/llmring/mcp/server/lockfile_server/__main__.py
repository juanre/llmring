#!/usr/bin/env python3
"""
Entry point for running the lockfile MCP server.

This allows running the server with:
    python -m llmring.mcp.server.lockfile_server
"""

import asyncio
import sys

from dotenv import load_dotenv

from llmring.mcp.server.lockfile_server.server import main

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...exiting")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)