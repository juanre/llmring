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
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        self.tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Create MCP server
        self.server = MCPServer(name="LLMRing Lockfile Manager", version="1.0.0")

        # Register all lockfile management tools
        self._register_tools()

    def _register_tools(self):
        """Register all lockfile management tools with the MCP server."""

        # Add alias tool
        self.server.register_tool(
            name="add_alias",
            handler=self._wrap_async(self.tools.add_alias),
            description="Add or update an alias with a model pool. The 'models' parameter accepts any numnber of comma-separated models forming a prioritized pool. This tool needs all the models in the pool, remember that it does not append to the existing pool.",
            input_schema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "REQUIRED: The alias name to create or update (e.g., 'fast', 'deep', 'coder', 'advisor')",
                    },
                    "models": {
                        "type": "string",
                        "description": "REQUIRED: Model(s) to bind to this alias separated by commas, for example 'openai:gpt-4o' for a single-model pool or 'anthropic:claude-3-haiku,openai:gpt-4o-mini' for a two-model pool.",
                    },
                    "profile": {
                        "type": "string",
                        "description": "OPTIONAL: Profile to add the alias to (defaults to 'default' if not specified). Profiles are strings like 'dev', 'prod', or 'test' that allow different models to be assigned to an alias depending on whether we are developing, testing, staging, in production, etc.",
                    },
                },
                "required": ["alias", "models"],
                "examples": [
                    {"alias": "fast", "models": "openai:gpt-4o-mini"},
                    {
                        "alias": "fast",
                        "models": "anthropic:claude-3-haiku,openai:gpt-4o-mini",
                        "_note": "LLMRing will use Claude when available and GPT-4o-mini as alternative",
                    },
                    {
                        "alias": "advisor",
                        "models": "anthropic:claude-opus-4-1-20250805,openai:gpt-4.1",
                        "_note": "High-quality model pool with cross-provider alternatives",
                    },
                    {
                        "alias": "deep",
                        "models": "anthropic:claude-3-opus,openai:gpt-4,google:gemini-ultra",
                        "_note": "Maximum availability with three provider alternatives",
                    },
                ],
            },
        )

        # Remove alias tool
        self.server.register_tool(
            name="remove_alias",
            handler=self._wrap_async(self.tools.remove_alias),
            description="Remove an alias from the lockfile.",
            input_schema={
                "type": "object",
                "properties": {
                    "alias": {"type": "string", "description": "The alias name to remove"},
                    "profile": {
                        "type": "string",
                        "description": "Profile to remove from (default: 'default')",
                    },
                },
                "required": ["alias"],
            },
        )

        # List aliases tool
        self.server.register_tool(
            name="list_aliases",
            handler=self._wrap_async(self.tools.list_aliases),
            description="List all configured aliases and their bindings.",
            input_schema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile to list aliases from"},
                    "verbose": {
                        "type": "boolean",
                        "description": "Include detailed model information",
                    },
                },
            },
        )

        # Assess model tool
        self.server.register_tool(
            name="assess_model",
            handler=self._wrap_async(self.tools.assess_model),
            description="Assess a model's capabilities, costs, and suitability.",
            input_schema={
                "type": "object",
                "properties": {
                    "model_ref": {
                        "type": "string",
                        "description": "Model to assess (alias or provider:model format)",
                    }
                },
                "required": ["model_ref"],
            },
        )

        # Analyze costs tool
        self.server.register_tool(
            name="analyze_costs",
            handler=self._wrap_async(self.tools.analyze_costs),
            description="Analyze estimated costs for current or hypothetical configuration.",
            input_schema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile to analyze"},
                    "monthly_volume": {
                        "type": "object",
                        "properties": {
                            "input_tokens": {"type": "integer"},
                            "output_tokens": {"type": "integer"},
                        },
                        "description": "Expected monthly token usage",
                    },
                    "hypothetical_models": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Optional hypothetical alias:model mappings for what-if analysis",
                    },
                },
            },
        )

        # Save lockfile tool
        self.server.register_tool(
            name="save_lockfile",
            handler=self._wrap_async(self.tools.save_lockfile),
            description="Save the current lockfile configuration to disk.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to save to (defaults to current lockfile path)",
                    }
                },
            },
        )

        # Get current configuration
        self.server.register_tool(
            name="get_configuration",
            handler=self._wrap_async(self.tools.get_current_configuration),
            description="Get the complete current lockfile configuration.",
            input_schema={"type": "object", "properties": {}},
        )

        # Get available providers
        self.server.register_tool(
            name="get_available_providers",
            handler=self._wrap_async(self.tools.get_available_providers),
            description="Check which providers have API keys configured in environment variables.",
            input_schema={"type": "object", "properties": {}},
        )

        # List models
        self.server.register_tool(
            name="list_models",
            handler=self._wrap_async(self.tools.list_models),
            description="List all available models with their specifications from the registry.",
            input_schema={
                "type": "object",
                "properties": {
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific providers",
                    },
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Include inactive/deprecated models",
                    },
                },
            },
        )

        # Filter models by requirements
        self.server.register_tool(
            name="filter_models_by_requirements",
            handler=self._wrap_async(self.tools.filter_models_by_requirements),
            description="Filter models based on specific requirements like context size, cost, and capabilities.",
            input_schema={
                "type": "object",
                "properties": {
                    "min_context": {
                        "type": "integer",
                        "description": "Minimum context window size required",
                    },
                    "max_input_cost": {
                        "type": "number",
                        "description": "Maximum cost per million input tokens",
                    },
                    "max_output_cost": {
                        "type": "number",
                        "description": "Maximum cost per million output tokens",
                    },
                    "required_capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities (e.g., vision, function_calling)",
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific providers",
                    },
                },
            },
        )

        # Get model details
        self.server.register_tool(
            name="get_model_details",
            handler=self._wrap_async(self.tools.get_model_details),
            description="Get complete details for specific models including pricing, capabilities, and specifications.",
            input_schema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model references to get details for",
                    }
                },
                "required": ["models"],
            },
        )

        logger.info(
            f"Registered {len(self.server.function_registry.functions)} lockfile management tools"
        )

    def _wrap_async(self, async_func):
        """Wrap async function for synchronous call from MCP server with enhanced error handling."""
        import concurrent.futures
        import threading

        def wrapper(**kwargs):
            # Extract timeout if provided in kwargs (with _ prefix to avoid conflicts)
            timeout = kwargs.pop("_timeout", 30)

            # Check if we're in an async context
            try:
                # Try to get the running loop
                loop = asyncio.get_running_loop()

                # We're in an async context, but need to run synchronously
                # Use a thread to avoid blocking
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, async_func(**kwargs))
                    try:
                        return future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        logger.error(f"Tool {async_func.__name__} timed out after {timeout}s")
                        raise TimeoutError(f"Tool execution timed out after {timeout}s")
                    except Exception as e:
                        logger.error(
                            f"Tool {async_func.__name__} execution error: {e}", exc_info=True
                        )
                        raise

            except RuntimeError:
                # No loop running, we can run normally
                try:
                    return asyncio.run(async_func(**kwargs))
                except Exception as e:
                    logger.error(
                        f"Tool {async_func.__name__} execution error (new loop): {e}", exc_info=True
                    )
                    raise

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
    server = LockfileServer(lockfile_path=lockfile_path)

    # Use STDIO transport for now
    transport = StdioTransport()
    logger.info("Starting STDIO server")

    await server.run(transport)


if __name__ == "__main__":
    asyncio.run(main())
