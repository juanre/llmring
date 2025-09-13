"""
CLI entry point for mcp-server command.

This CLI allows users to run MCP servers using JSON configuration files
rather than requiring Python module imports. This follows standard patterns
where tools, resources, and prompts are defined declaratively.
"""

import argparse
import asyncio
import sys
import json
from pathlib import Path
from typing import Any, Dict, List
import logging

from llmring.mcp.server.mcp_server import MCPServer
from llmring.mcp.server.transport import StdioServerTransport


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load MCP server configuration from JSON file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate basic structure
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def register_static_tools(
    server: MCPServer, tools_config: List[Dict[str, Any]]
) -> None:
    """Register static tools that return predefined content."""
    for tool in tools_config:
        name = tool.get("name")
        if not name:
            logger.warning("Skipping tool without name")
            continue

        description = tool.get("description", f"Tool: {name}")
        input_schema = tool.get("input_schema", {"type": "object"})
        response_content = tool.get("response", "Tool executed successfully")

        # Create static handler
        async def create_static_handler(content=response_content):
            return content

        server.register_tool(
            name=name,
            handler=create_static_handler,
            description=description,
            input_schema=input_schema,
        )

        logger.info(f"Registered static tool: {name}")


def register_resources(
    server: MCPServer, resources_config: List[Dict[str, Any]]
) -> None:
    """Register static resources."""
    for resource in resources_config:
        uri = resource.get("uri")
        if not uri:
            logger.warning("Skipping resource without URI")
            continue

        name = resource.get("name", uri)
        description = resource.get("description", f"Resource: {name}")
        content = resource.get("content", "")
        mime_type = resource.get("mime_type", "text/plain")

        server.register_static_resource(
            uri=uri,
            name=name,
            description=description,
            content=content,
            mime_type=mime_type,
        )

        logger.info(f"Registered resource: {uri}")


def register_prompts(server: MCPServer, prompts_config: List[Dict[str, Any]]) -> None:
    """Register static prompts."""
    for prompt in prompts_config:
        name = prompt.get("name")
        if not name:
            logger.warning("Skipping prompt without name")
            continue

        description = prompt.get("description", f"Prompt: {name}")
        content = prompt.get("content", "")
        arguments = prompt.get("arguments", [])

        server.register_static_prompt(
            name=name, description=description, content=content, arguments=arguments
        )

        logger.info(f"Registered prompt: {name}")


def create_server_from_config(config: Dict[str, Any]) -> MCPServer:
    """Create and configure MCP server from configuration."""
    # Server metadata
    server_name = config.get("name", "MCP Server")
    server_version = config.get("version", "1.0.0")

    # Create server
    server = MCPServer(name=server_name, version=server_version)

    # Register tools
    tools = config.get("tools", [])
    if tools:
        register_static_tools(server, tools)

    # Register resources
    resources = config.get("resources", [])
    if resources:
        register_resources(server, resources)

    # Register prompts
    prompts = config.get("prompts", [])
    if prompts:
        register_prompts(server, prompts)

    return server


async def run_server(config_path: str):
    """Run the MCP server from configuration file."""
    try:
        # Load configuration
        config = load_config(config_path)

        # Create server from config
        server = create_server_from_config(config)

        # Create STDIO transport
        transport = StdioServerTransport()

        logger.info(
            f"Starting {config.get('name', 'MCP Server')} v{config.get('version', '1.0.0')}"
        )

        # Run server
        await server.run(transport)

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise


def main():
    """CLI entry point for mcp-server command."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MCP Server CLI - Run MCP servers from JSON configuration files",
        epilog="""
Examples:
  mcp-server --config server.json
  mcp-server --config examples/math_tools.json

Configuration file format:
  {
    "name": "My MCP Server",
    "version": "1.0.0",
    "tools": [
      {
        "name": "hello",
        "description": "Say hello",
        "input_schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"}
          },
          "required": ["name"]
        },
        "response": "Hello, {name}!"
      }
    ],
    "resources": [
      {
        "uri": "config://server/info",
        "name": "Server Info",
        "description": "Server information",
        "content": "MCP Server v1.0.0",
        "mime_type": "text/plain"
      }
    ],
    "prompts": [
      {
        "name": "greeting",
        "description": "Generate a greeting",
        "content": "Hello, {name}! Welcome to our service.",
        "arguments": [
          {"name": "name", "description": "Person's name", "required": true}
        ]
      }
    ]
  }

Note: For complex tools that need to execute code, use this library programmatically
instead of the CLI. See examples/simple_server.py for guidance.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON configuration file defining tools, resources, and prompts",
    )

    args = parser.parse_args()

    # Run the server
    try:
        asyncio.run(run_server(args.config))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
