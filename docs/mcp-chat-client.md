# MCP Chat Client

The LLMRing MCP Chat Client is a powerful, generic interactive terminal application that enables natural language conversations with any MCP (Model Context Protocol) server. It provides persistent history, rich commands, and seamless tool integration.

## Overview

The MCP Chat Client is designed to be completely generic and flexible:
- Connect to **any** MCP-compliant server
- Use **any** LLM model defined in your lockfile
- Persistent conversation history across sessions
- Rich terminal interface with syntax highlighting
- Automatic tool discovery and execution

## Installation

The MCP Chat Client is included with LLMRing:

```bash
pip install llmring
# or
uv add llmring
```

## Basic Usage

### Starting the Chat Client

```bash
# Connect to the built-in lockfile management server
llmring lock chat

# Connect to a custom MCP server via stdio
llmring mcp chat --server "stdio://python -m your.mcp.server"

# Connect to an HTTP MCP server
llmring mcp chat --server "http://localhost:8080"

# Connect to a WebSocket MCP server
llmring mcp chat --server "ws://localhost:8080"
```

### Command Line Options

```bash
llmring mcp chat [OPTIONS]

Options:
  --server TEXT       MCP server URL (stdio://, http://, ws://)
  --model TEXT        LLM model alias to use (default: advisor)
  --no-telemetry      Disable telemetry
  --debug             Enable debug logging
  --help              Show help message
```

## Chat Interface Features

### Built-in Commands

The chat client provides several commands that start with `/`:

| Command | Description |
|---------|-------------|
| `/help` | Display all available commands |
| `/history` | Show current conversation history |
| `/sessions` | List all saved chat sessions |
| `/load <session_id>` | Load and resume a previous session |
| `/clear` | Clear the current conversation |
| `/model <alias>` | Switch to a different model |
| `/tools` | List available MCP tools from the server |
| `/exit` or `/quit` | Exit the chat client |

### Persistent History

All conversations are automatically saved for later reference:

```
~/.llmring/mcp_chat/
├── command_history.txt           # Terminal command history
├── conversation_<session_id>.json # Individual conversations
└── sessions.json                 # Session metadata and index
```

#### Session Management

Each session includes:
- Unique session ID
- Timestamp of creation
- Complete message history
- Tool calls and their results
- Model used for each response

#### Loading Previous Sessions

```bash
# In the chat interface
You: /sessions
System: Available sessions:
  - abc123 (2024-01-15 10:30:00): "Configured fast model for API responses"
  - def456 (2024-01-15 14:20:00): "Set up vision models for image analysis"

You: /load abc123
System: Loaded session abc123. Continuing from previous conversation...
```

## Connecting to Different MCP Servers

### Stdio Servers (Local Processes)

Most common for development and local tools:

```bash
# Python MCP server
llmring mcp chat --server "stdio://python -m mypackage.mcp_server"

# Node.js MCP server
llmring mcp chat --server "stdio://node my-mcp-server.js"

# Any executable
llmring mcp chat --server "stdio:///usr/local/bin/my-mcp-tool"
```

### HTTP Servers

For REST API-based MCP servers:

```bash
# Local development
llmring mcp chat --server "http://localhost:8080"

# Remote server with authentication
llmring mcp chat --server "https://api.example.com/mcp"
```

### WebSocket Servers

For real-time, bidirectional communication:

```bash
# WebSocket connection
llmring mcp chat --server "ws://localhost:8080"

# Secure WebSocket
llmring mcp chat --server "wss://mcp.example.com"
```

## Using Different LLM Models

The chat client can use any model defined in your lockfile:

```bash
# Use the fast model for quick responses
llmring mcp chat --model fast

# Use a deep reasoning model
llmring mcp chat --model deep

# Use the default advisor model (Claude Opus 4.1)
llmring mcp chat  # defaults to 'advisor'
```

## Creating Custom MCP Servers

The chat client can connect to any MCP-compliant server you create:

### Simple Python Example

```python
#!/usr/bin/env python3
"""my_mcp_server.py - Custom MCP server example"""

import asyncio
from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport

# Create server
server = MCPServer(
    name="My Custom Tools",
    version="1.0.0"
)

# Register tools
@server.function_registry.register(
    name="get_weather",
    description="Get weather for a location"
)
def get_weather(location: str) -> dict:
    # Your implementation here
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny"
    }

@server.function_registry.register(
    name="calculate",
    description="Perform calculations"
)
def calculate(expression: str) -> dict:
    try:
        result = eval(expression)  # Simple example - use ast.literal_eval in production
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Run server
async def main():
    transport = StdioTransport()
    await server.run(transport)

if __name__ == "__main__":
    asyncio.run(main())
```

Then connect to it:

```bash
llmring mcp chat --server "stdio://python my_mcp_server.py"
```

### Advanced Server with State

```python
from llmring.mcp.server import MCPServer
from typing import Dict, List, Any

class StatefulMCPServer:
    def __init__(self):
        self.server = MCPServer(name="Stateful Server", version="1.0.0")
        self.state: Dict[str, Any] = {}
        self.register_tools()

    def register_tools(self):
        @self.server.function_registry.register(
            name="set_value",
            description="Store a value"
        )
        def set_value(key: str, value: Any) -> dict:
            self.state[key] = value
            return {"success": True, "key": key}

        @self.server.function_registry.register(
            name="get_value",
            description="Retrieve a value"
        )
        def get_value(key: str) -> dict:
            return {"value": self.state.get(key), "exists": key in self.state}

        @self.server.function_registry.register(
            name="list_keys",
            description="List all stored keys"
        )
        def list_keys() -> dict:
            return {"keys": list(self.state.keys())}

    async def run(self):
        from llmring.mcp.server.transport.stdio import StdioTransport
        transport = StdioTransport()
        await self.server.run(transport)

if __name__ == "__main__":
    import asyncio
    server = StatefulMCPServer()
    asyncio.run(server.run())
```

## Programmatic Usage

You can also use the MCP Chat Client programmatically in your Python code:

```python
import asyncio
from llmring.mcp.client.chat.app import MCPChatApp

async def main():
    # Create chat application
    app = MCPChatApp(
        mcp_server_url="stdio://python -m my_mcp_server",
        llm_model="advisor",
        enable_telemetry=False
    )

    # Initialize
    await app.initialize_async()

    # Run interactive session
    await app.run_interactive()

    # Or send a single message
    response = await app.send_message("What tools are available?")
    print(response)

    # Clean up
    await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Server Design Philosophy

When creating MCP servers for use with the chat client:

1. **Data-Focused Tools**: Design tools to provide data and perform actions, not make decisions
2. **LLM in Driver's Seat**: Let the LLM decide how to use tools based on user intent
3. **Clear Tool Names**: Use descriptive, action-oriented names
4. **Comprehensive Schemas**: Provide detailed parameter descriptions
5. **Error Handling**: Return informative error messages

### Security Considerations

1. **Validate Input**: Always validate tool parameters
2. **Limit Scope**: Tools should have minimal necessary permissions
3. **Secure Transport**: Use HTTPS/WSS in production
4. **Authentication**: Implement proper auth for production servers
5. **Audit Logging**: Log tool usage for security monitoring

### Performance Tips

1. **Use Appropriate Models**: Choose models based on task complexity
2. **Cache Results**: Implement caching in your MCP server for expensive operations
3. **Streaming Responses**: Use streaming for long-running operations
4. **Batch Operations**: Design tools to handle batch requests when possible

## Troubleshooting

### Common Issues

**Server won't start:**
- Check the server path is correct
- Ensure the server has proper permissions
- Verify Python/Node.js environment is activated

**Tools not appearing:**
- Run `/tools` to refresh tool list
- Check server logs for registration errors
- Verify tool schemas are valid

**History not saving:**
- Check permissions on `~/.llmring/mcp_chat/`
- Ensure enough disk space
- Look for errors in debug mode (`--debug`)

**Connection errors:**
- Verify server is running
- Check firewall/network settings
- Ensure correct protocol (stdio/http/ws)

### Debug Mode

Run with debug logging to troubleshoot issues:

```bash
llmring mcp chat --debug --server "stdio://python my_server.py"
```

## Examples

### Example 1: File Management Server

Connect to a file management MCP server:

```bash
llmring mcp chat --server "stdio://python -m file_manager_mcp"

You: Show me all Python files in the current directory
Assistant: I'll search for Python files in the current directory.

[Calling tool: list_files with pattern="*.py"]

I found 5 Python files:
- main.py (2.3 KB)
- utils.py (1.1 KB)
- config.py (890 B)
- test_main.py (3.4 KB)
- setup.py (456 B)

You: Read the config.py file
[Tool executes and shows file contents...]
```

### Example 2: Database Query Server

```bash
llmring mcp chat --server "http://localhost:8080/db-mcp"

You: How many users signed up last month?
Assistant: I'll query the database to find user signups from last month.

[Calling tool: execute_query]

Based on the database query, 1,247 new users signed up last month. This represents
a 15% increase from the previous month.
```

### Example 3: Multi-Tool Workflow

```bash
You: Analyze the performance of our API endpoint and create a report