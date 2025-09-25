#!/usr/bin/env python3
"""
MCP Server for conversational lockfile management.

This server provides MCP tools for managing LLMRing lockfiles through
natural conversation, allowing users to interactively configure their
LLM aliases and bindings.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LockfileServer:
    """MCP server for conversational lockfile management."""
    
    def __init__(self, lockfile_path: Optional[Path] = None):
        """
        Initialize the lockfile server.

        Args:
            lockfile_path: Path to the lockfile (defaults to llmring.lock)
        """
        # Initialize lockfile tools
        self.tools = LockfileManagerTools(
            lockfile_path=lockfile_path
        )
        
        # Create MCP server
        self.server = MCPServer(
            name="LLMRing Lockfile Manager",
            version="1.0.0"
        )
        
        # Register all lockfile management tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all lockfile management tools with the MCP server."""
        
        # Add alias tool
        self.server.function_registry.register(
            name="add_alias",
            func=self._wrap_async(self.tools.add_alias),
            schema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "The alias name (e.g., 'fast', 'deep', 'coder')"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model reference (e.g., 'openai:gpt-4o-mini')"
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Optional use case description for automatic recommendation"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile to add the alias to (default: 'default')"
                    }
                },
                "required": ["alias"]
            },
            description="Add or update an alias in the lockfile. Can auto-recommend models based on use case."
        )
        
        # Remove alias tool
        self.server.function_registry.register(
            name="remove_alias",
            func=self._wrap_async(self.tools.remove_alias),
            schema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "The alias name to remove"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile to remove from (default: 'default')"
                    }
                },
                "required": ["alias"]
            },
            description="Remove an alias from the lockfile."
        )
        
        # List aliases tool
        self.server.function_registry.register(
            name="list_aliases",
            func=self._wrap_async(self.tools.list_aliases),
            schema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Profile to list aliases from"
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Include detailed model information"
                    }
                }
            },
            description="List all configured aliases and their bindings."
        )
        
        # Assess model tool
        self.server.function_registry.register(
            name="assess_model",
            func=self._wrap_async(self.tools.assess_model),
            schema={
                "type": "object",
                "properties": {
                    "model_ref": {
                        "type": "string",
                        "description": "Model to assess (alias or provider:model format)"
                    }
                },
                "required": ["model_ref"]
            },
            description="Assess a model's capabilities, costs, and suitability."
        )
        
        # Recommend alias tool
        self.server.function_registry.register(
            name="recommend_alias",
            func=self._wrap_async(self.tools.recommend_alias),
            schema={
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "description": "Description of what you need the model for"
                    },
                    "budget": {
                        "type": "string",
                        "enum": ["low", "balanced", "high"],
                        "description": "Budget preference"
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities (e.g., 'vision', 'function_calling')"
                    }
                },
                "required": ["use_case"]
            },
            description="Get model recommendations for a specific use case."
        )
        
        # Analyze costs tool
        self.server.function_registry.register(
            name="analyze_costs",
            func=self._wrap_async(self.tools.analyze_costs),
            schema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Profile to analyze"
                    },
                    "monthly_volume": {
                        "type": "object",
                        "properties": {
                            "input_tokens": {"type": "integer"},
                            "output_tokens": {"type": "integer"}
                        },
                        "description": "Expected monthly token usage"
                    }
                }
            },
            description="Analyze estimated costs for current configuration."
        )
        
        # Save lockfile tool
        self.server.function_registry.register(
            name="save_lockfile",
            func=self._wrap_async(self.tools.save_lockfile),
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to save to (defaults to current lockfile path)"
                    }
                }
            },
            description="Save the current lockfile configuration to disk."
        )
        
        # Get current configuration
        self.server.function_registry.register(
            name="get_configuration",
            func=self._wrap_async(self.tools.get_current_configuration),
            schema={
                "type": "object",
                "properties": {}
            },
            description="Get the complete current lockfile configuration."
        )
        
        logger.info(f"Registered {len(self.server.function_registry.functions)} lockfile management tools")
        
    def _wrap_async(self, async_func):
        """Wrap async function for synchronous call from MCP server."""
        def wrapper(**kwargs):
            # Run the async function in the event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_func(**kwargs))
        return wrapper
        
    async def run(self, transport=None):
        """Run the MCP server.

        Args:
            transport: Optional transport to use (defaults to STDIO)
        """
        if transport is None:
            transport = StdioTransport()

        # Run the server
        logger.info(f"Starting LLMRing Lockfile MCP Server with {transport.__class__.__name__}...")
        await self.server.run(transport)


async def main():
    """Main entry point for the lockfile MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLMRing Lockfile MCP Server")
    parser.add_argument("--port", type=int, help="Port for HTTP server")
    parser.add_argument("--host", default="localhost", help="Host for HTTP server")
    parser.add_argument("--lockfile", help="Path to lockfile")
    args = parser.parse_args()

    # Get paths from environment or args
    lockfile_path = args.lockfile or os.getenv("LLMRING_LOCKFILE_PATH")
    if lockfile_path:
        lockfile_path = Path(lockfile_path)

    # Create server
    server = LockfileServer(
        lockfile_path=lockfile_path
    )

    # Use STDIO transport for now
    transport = StdioTransport()
    logger.info("Starting STDIO server")

    await server.run(transport)


if __name__ == "__main__":
    asyncio.run(main())