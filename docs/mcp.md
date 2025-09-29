# Model Context Protocol (MCP) Integration

## Overview

MCP (Model Context Protocol) enables LLMs to interact with external tools and resources through a standardized protocol. LLMRing provides comprehensive MCP support including a powerful chat client, enhanced LLM instances with tool capabilities, and the ability to create custom MCP servers.

## Key Features

- **MCP Chat Client**: Interactive terminal application with persistent history
- **Enhanced LLM**: Automatic tool discovery and execution
- **Lockfile Management**: Conversational configuration via MCP tools
- **Multiple Transports**: HTTP, WebSocket, and stdio connections
- **Streaming Support**: Tool calls work seamlessly with streaming responses
- **Custom Servers**: Build your own MCP servers for specific use cases

## Choosing Your MCP Interface

LLMRing provides two ways to work with MCP:

| Feature | MCP Chat Client | Enhanced LLM |
|---------|----------------|--------------|
| **Best For** | Interactive terminal sessions, configuration | Programmatic integration, applications |
| **Interface** | Command-line chat application | Python API |
| **History** | Automatic persistent history in `~/.llmring/mcp_chat/` | Custom management needed |
| **Session Management** | Built-in session saving/loading | Manual implementation |
| **Tool Discovery** | Automatic with `/tools` command | Automatic via API |
| **Streaming** | Real-time terminal output | AsyncIterator for custom handling |
| **Use Cases** | - Lockfile configuration<br>- Interactive exploration<br>- CLI tools<br>- Quick testing | - Production applications<br>- Automated workflows<br>- Custom integrations<br>- Batch processing |
| **Example Commands** | `llmring lock chat`<br>`llmring mcp chat --server ...` | `create_enhanced_llm()`<br>`await llm.chat(messages)` |

**Quick Decision Guide:**
- **Use Chat Client** when you want to interactively configure, explore, or test MCP servers
- **Use Enhanced LLM** when you need MCP capabilities in your Python application

---

## Quick Start

### MCP Chat Client (Recommended for Interactive Use)

The easiest way to use MCP interactively is through the chat client:

```bash
# Conversational lockfile configuration (built-in MCP server)
llmring lock chat

# Connect to a custom MCP server
llmring mcp chat --server "stdio://python -m your_mcp_server"

# HTTP server
llmring mcp chat --server "http://localhost:8080"

# WebSocket server
llmring mcp chat --server "ws://localhost:8080"
```

### Enhanced LLM (Recommended for Programmatic Use)

```python
from llmring.mcp.client import create_enhanced_llm

# Create an enhanced LLM with MCP tools
llm = await create_enhanced_llm(
    model="fast",
    mcp_server_path="/path/to/mcp/server"
)

# Chat with automatic tool execution
messages = [{"role": "user", "content": "Help me with my files"}]
response = await llm.chat(messages)

print(response.content)
if response.tool_calls:
    print(f"Used tools: {[call['function']['name'] for call in response.tool_calls]}")
```

## MCP Chat Client

### Features

The MCP Chat Client is a generic, flexible terminal application that:
- Connects to **any** MCP-compliant server
- Uses **any** LLM model from your lockfile
- Maintains persistent conversation history
- Provides rich terminal interface with syntax highlighting
- Automatically discovers and executes tools

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

### Built-in Commands

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

All conversations are automatically saved:

```
~/.llmring/mcp_chat/
├── command_history.txt           # Terminal command history
├── conversation_<session_id>.json # Individual conversations
└── sessions.json                 # Session metadata and index
```

Each session includes:
- Unique session ID and timestamp
- Complete message history
- Tool calls and their results
- Model used for each response

#### Session Management Example

```bash
You: /sessions
System: Available sessions:
  - abc123 (2024-01-15 10:30): "Configured fast model for API"
  - def456 (2024-01-15 14:20): "Set up vision models"

You: /load abc123
System: Loaded session abc123. Continuing from previous conversation...
```

## Lockfile Management via MCP

The most powerful feature of MCP in LLMRing is conversational lockfile configuration:

```bash
llmring lock chat
```

This starts an interactive session where you can:
- Describe your requirements in natural language
- Get recommendations based on current registry
- Understand cost implications
- Configure aliases with fallback models
- Set up environment-specific profiles

### Example Conversation

```
You: I need a configuration for a coding assistant that prioritizes accuracy
Assistant: I'll help you configure an accurate coding assistant. Based on the registry,
I recommend using Claude 3.5 Sonnet as the primary model with GPT-4o as fallback.

[Calling tool: add_alias]
Added alias 'coder' with models: anthropic:claude-3-5-sonnet, openai:gpt-4o

This configuration prioritizes accuracy while providing fallback for availability.
Monthly cost estimate: ~$50-100 for moderate usage.

You: Add a cheaper option for simple tasks
Assistant: I'll add a cost-effective alias for simpler coding tasks.

[Calling tool: add_alias]
Added alias 'coder-fast' with model: openai:gpt-4o-mini

This model is 10x cheaper and perfect for simple completions, syntax fixes, and
basic code generation.
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

# Remote server
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

## Creating Custom MCP Servers

Build your own MCP servers to expose tools to LLMs:

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
        # Use ast.literal_eval in production for safety
        result = eval(expression)
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

Connect to your server:

```bash
llmring mcp chat --server "stdio://python my_mcp_server.py"
```

### Stateful Server Example

```python
from llmring.mcp.server import MCPServer
from typing import Dict, Any

class StatefulMCPServer:
    def __init__(self):
        self.server = MCPServer(name="Stateful Server", version="1.0.0")
        self.state: Dict[str, Any] = {}
        self.register_tools()

    def register_tools(self):
        @self.server.function_registry.register(
            name="set_value",
            description="Store a value with a key"
        )
        def set_value(key: str, value: Any) -> dict:
            self.state[key] = value
            return {"success": True, "key": key}

        @self.server.function_registry.register(
            name="get_value",
            description="Retrieve a value by key"
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

## Enhanced LLM with MCP

For programmatic usage with automatic tool execution:

### Basic Usage

```python
from llmring.mcp.client.enhanced_llm import create_enhanced_llm

# Create enhanced LLM
llm = await create_enhanced_llm(
    model="balanced",
    mcp_server_path="stdio://python -m my_mcp_server"
)

# Chat with automatic tool execution
messages = [{"role": "user", "content": "What's the weather in NYC?"}]
response = await llm.chat(messages)

# Tools are called automatically and results integrated
print(response.content)  # "The weather in NYC is 72°F and sunny."
```

### Streaming with Tools

```python
# Streaming works seamlessly with tool calls
messages = [{"role": "user", "content": "Analyze this file and summarize it"}]

async for chunk in await llm.chat_stream(messages):
    if chunk.type == "content":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "tool_call":
        print(f"\n[Calling tool: {chunk.tool_call.name}]")
    elif chunk.type == "tool_result":
        print(f"\n[Tool result received]")
```

### Direct MCP Client Usage

```python
from llmring.mcp.client.mcp_client import MCPClient

# Connect to MCP server
client = MCPClient("http://localhost:8000")
await client.initialize()

# List available tools
tools = await client.list_tools()
for tool in tools:
    print(f"- {tool['name']}: {tool['description']}")

# Execute a tool directly
result = await client.call_tool(
    "read_file",
    {"path": "/path/to/file.txt"}
)
print(result)

# Clean up
await client.close()
```

## Programmatic Chat Client

Use the MCP chat client in your Python code:

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

    # Send a single message
    response = await app.send_message("What tools are available?")
    print(response)

    # Or run interactive session
    await app.run_interactive()

    # Clean up
    await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Server Design Philosophy

When creating MCP servers:

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

1. **Choose Appropriate Models**: Match model capability to task complexity
2. **Cache Results**: Implement caching in MCP servers for expensive operations
3. **Streaming Responses**: Use streaming for long-running operations
4. **Batch Operations**: Design tools to handle batch requests when possible

## Troubleshooting

### Common Issues

**Server won't start:**
- Check the server path is correct
- Ensure proper permissions
- Verify Python/Node.js environment is activated

**Tools not appearing:**
- Run `/tools` to refresh tool list
- Check server logs for registration errors
- Verify tool schemas are valid

**History not saving:**
- Check permissions on `~/.llmring/mcp_chat/`
- Ensure enough disk space
- Look for errors in debug mode

**Connection errors:**
- Verify server is running
- Check firewall/network settings
- Ensure correct protocol (stdio/http/ws)

### Debug Mode

Enable detailed logging:

```bash
llmring mcp chat --debug --server "stdio://python my_server.py"
```

## Examples

### File Management

```bash
llmring mcp chat --server "stdio://python -m file_manager_mcp"

You: Show me all Python files in the current directory
Assistant: I'll search for Python files in the current directory.

[Calling tool: list_files with pattern="*.py"]

Found 5 Python files:
- main.py (2.3 KB)
- utils.py (1.1 KB)
- config.py (890 B)
- test_main.py (3.4 KB)
- setup.py (456 B)

You: Read config.py
[Tool executes and shows file contents...]
```

### Database Queries

```bash
llmring mcp chat --server "http://localhost:8080/db-mcp"

You: How many users signed up last month?
Assistant: I'll query the database for last month's signups.

[Calling tool: execute_query]

Based on the query, 1,247 new users signed up last month,
representing a 15% increase from the previous month.
```

### Multi-Tool Workflow

```bash
You: Analyze our API performance and create a report
Assistant: I'll analyze the API performance and create a comprehensive report.

[Calling tool: get_metrics]
[Calling tool: analyze_logs]
[Calling tool: generate_report]

Here's the performance analysis:
- Average response time: 142ms (↓ 8% from last week)
- Success rate: 99.97%
- Peak traffic: 1,200 req/s at 2 PM daily
- Most used endpoint: /api/v1/users

I've generated a detailed report saved as "api_performance_2024-01.pdf"
```

## Further Resources

- Example MCP servers: `examples/mcp/`
- MCP specification: [Model Context Protocol](https://github.com/anthropics/model-context-protocol)
- LLMRing MCP source: `src/llmring/mcp/`
