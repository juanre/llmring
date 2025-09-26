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

from dotenv import load_dotenv

from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools

# Load environment variables from .env file
load_dotenv()

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
            description="Add or update an alias in the lockfile. REQUIRES both 'alias' and 'model' parameters.",
            schema={
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "REQUIRED: The alias name to create (e.g., 'fast', 'deep', 'coder', 'pdf_converter')"
                    },
                    "model": {
                        "type": "string",
                        "description": "REQUIRED: Model reference in format provider:model (e.g., 'openai:gpt-4o-mini', 'anthropic:claude-3-haiku')"
                    },
                    "profile": {
                        "type": "string",
                        "description": "OPTIONAL: Profile to add the alias to (defaults to 'default' if not specified)"
                    }
                },
                "required": ["alias", "model"],
                "examples": [
                    {"alias": "fast", "model": "openai:gpt-5-nano"},
                    {"alias": "pdf_converter", "model": "openai:gpt-4o-mini"},
                    {"alias": "deep", "model": "anthropic:claude-3-opus", "profile": "production"}
                ]
            }
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
                    },
                    "hypothetical_models": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Optional hypothetical alias:model mappings for what-if analysis"
                    }
                }
            },
            description="Analyze estimated costs for current or hypothetical configuration."
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

        # Get available providers
        self.server.function_registry.register(
            name="get_available_providers",
            func=self._wrap_async(self.tools.get_available_providers),
            schema={
                "type": "object",
                "properties": {}
            },
            description="Check which providers have API keys configured in environment variables."
        )

        # List models
        self.server.function_registry.register(
            name="list_models",
            func=self._wrap_async(self.tools.list_models),
            schema={
                "type": "object",
                "properties": {
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific providers"
                    },
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Include inactive/deprecated models"
                    }
                }
            },
            description="List all available models with their specifications from the registry."
        )

        # Filter models by requirements
        self.server.function_registry.register(
            name="filter_models_by_requirements",
            func=self._wrap_async(self.tools.filter_models_by_requirements),
            schema={
                "type": "object",
                "properties": {
                    "min_context": {
                        "type": "integer",
                        "description": "Minimum context window size required"
                    },
                    "max_input_cost": {
                        "type": "number",
                        "description": "Maximum cost per million input tokens"
                    },
                    "max_output_cost": {
                        "type": "number",
                        "description": "Maximum cost per million output tokens"
                    },
                    "required_capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities (e.g., vision, function_calling)"
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific providers"
                    }
                }
            },
            description="Filter models based on specific requirements like context size, cost, and capabilities."
        )

        # Get model details
        self.server.function_registry.register(
            name="get_model_details",
            func=self._wrap_async(self.tools.get_model_details),
            schema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model references to get details for"
                    }
                },
                "required": ["models"]
            },
            description="Get complete details for specific models including pricing, capabilities, and specifications."
        )

        logger.info(f"Registered {len(self.server.function_registry.functions)} lockfile management tools")
        
    def _wrap_async(self, async_func):
        """Wrap async function for synchronous call from MCP server with enhanced error handling."""
        import concurrent.futures
        import threading

        def wrapper(**kwargs):
            # Extract timeout if provided in kwargs (with _ prefix to avoid conflicts)
            timeout = kwargs.pop('_timeout', 30)

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
                        logger.error(f"Tool {async_func.__name__} execution error: {e}", exc_info=True)
                        raise

            except RuntimeError:
                # No loop running, we can run normally
                try:
                    return asyncio.run(async_func(**kwargs))
                except Exception as e:
                    logger.error(f"Tool {async_func.__name__} execution error (new loop): {e}", exc_info=True)
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
    server = LockfileServer(
        lockfile_path=lockfile_path
    )

    # Use STDIO transport for now
    transport = StdioTransport()
    logger.info("Starting STDIO server")

    await server.run(transport)


if __name__ == "__main__":
    asyncio.run(main())