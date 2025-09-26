#!/usr/bin/env python3
"""
Example: Using the MCP Chat Client

This example demonstrates how to use the MCP chat client both from the command
line and programmatically in Python code.
"""

import asyncio
from pathlib import Path
from llmring.mcp.client.chat.app import MCPChatApp
from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport


# Example 1: Creating a simple MCP server for the chat client
def create_example_server():
    """Create a simple MCP server with custom tools."""
    server = MCPServer(
        name="Example Tools Server",
        version="1.0.0"
    )

    # Register some example tools
    @server.function_registry.register(
        name="get_time",
        description="Get the current time"
    )
    def get_time() -> dict:
        from datetime import datetime
        return {"time": datetime.now().isoformat()}

    @server.function_registry.register(
        name="calculate",
        description="Perform a calculation"
    )
    def calculate(expression: str) -> dict:
        try:
            # Note: Use ast.literal_eval in production for safety
            result = eval(expression)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e)}

    @server.function_registry.register(
        name="save_note",
        description="Save a note to memory"
    )
    def save_note(title: str, content: str) -> dict:
        # In a real application, this would persist to a database
        return {
            "success": True,
            "note": {
                "title": title,
                "content": content,
                "saved_at": datetime.now().isoformat()
            }
        }

    return server


# Example 2: Using the chat client programmatically
async def programmatic_chat_example():
    """Example of using the MCP chat client in Python code."""

    # Create chat app connected to the lockfile server
    app = MCPChatApp(
        mcp_server_url="stdio://python -m llmring.mcp.server.lockfile_server",
        llm_model="fast",  # Use the fast model for quick responses
        enable_telemetry=False
    )

    try:
        # Initialize the app
        await app.initialize_async()
        print("Chat client initialized successfully")

        # Send a message and get response
        print("\nSending message to assistant...")
        response = await app.send_message("What tools do you have available?")
        print(f"Assistant: {response}")

        # Send another message
        response = await app.send_message("Can you list my current aliases?")
        print(f"Assistant: {response}")

        # Demonstrate tool usage
        response = await app.send_message(
            "Add an alias called 'coder' for anthropic:claude-3-5-sonnet"
        )
        print(f"Assistant: {response}")

    finally:
        # Clean up
        await app.cleanup()
        print("\nChat client cleaned up")


# Example 3: Custom server runner for stdio
async def run_example_server():
    """Run the example MCP server."""
    server = create_example_server()
    transport = StdioTransport()
    await server.run(transport)


# Example 4: Interactive chat session
async def interactive_chat_example():
    """
    Example of running an interactive chat session.

    This would typically be run from the command line:
    llmring mcp chat --server "stdio://python mcp_chat_example.py --serve"
    """
    app = MCPChatApp(
        mcp_server_url="stdio://python -m llmring.mcp.server.lockfile_server",
        llm_model="advisor",
        enable_telemetry=False
    )

    await app.initialize_async()

    # Run the interactive session
    # This provides the full chat interface with history, commands, etc.
    await app.run_interactive()

    await app.cleanup()


def main():
    """Main entry point with examples."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        # Run as MCP server
        print("Starting example MCP server...")
        asyncio.run(run_example_server())
    else:
        # Show usage examples
        print("""
MCP Chat Client Examples
========================

1. Command Line Usage:
----------------------
# Connect to the lockfile management server
llmring lock chat

# Connect to a custom MCP server
llmring mcp chat --server "stdio://python mcp_chat_example.py --serve"

# Connect to an HTTP server
llmring mcp chat --server "http://localhost:8080"

# Use a specific model
llmring mcp chat --model fast --server "stdio://your-server"


2. Programmatic Usage:
----------------------
Run the programmatic example:
python mcp_chat_example.py


3. Create Your Own MCP Server:
-------------------------------
# Run this script as a server
python mcp_chat_example.py --serve

# Then connect to it
llmring mcp chat --server "stdio://python mcp_chat_example.py --serve"


4. Available Chat Commands:
----------------------------
/help       - Show all commands
/history    - Display conversation history
/sessions   - List saved sessions
/load <id>  - Load a previous session
/clear      - Clear current conversation
/model <alias> - Switch model
/tools      - List available tools
/exit       - Exit chat


5. Persistent History:
----------------------
All conversations are saved in ~/.llmring/mcp_chat/
- conversation_*.json: Individual conversations
- sessions.json: Session metadata
- command_history.txt: Command line history
        """)

        # Run the programmatic example
        print("\nRunning programmatic example...")
        print("=" * 50)
        asyncio.run(programmatic_chat_example())


if __name__ == "__main__":
    main()