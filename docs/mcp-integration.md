# MCP (Model Context Protocol) Integration

LLMRing provides comprehensive MCP integration for building tool-enabled LLM applications.

## Overview

MCP allows LLMs to interact with external tools and resources through a standardized protocol. LLMRing's MCP integration provides:

- **MCP Chat Client**: Generic interactive chat client with persistent history
- **Enhanced LLM**: LLM instances with built-in tool capabilities
- **Tool Execution**: Automatic tool discovery and execution
- **Streaming Support**: Tool calls work with streaming responses
- **Multiple Transports**: HTTP, WebSocket, and stdio transports
- **Custom MCP Servers**: Build your own MCP servers for specific use cases

## Quick Start

### Basic MCP Integration

```python
from llmring.mcp.client.enhanced_llm import create_enhanced_llm

# Create an enhanced LLM with MCP tools
llm = await create_enhanced_llm(
    model="fast",
    mcp_server_path="/path/to/mcp/server"
)

# Chat with tool access
messages = [{"role": "user", "content": "Help me with my files"}]
response = await llm.chat(messages)

print(response.content)
if response.tool_calls:
    print(f"Used tools: {[call['function']['name'] for call in response.tool_calls]}")
```

### MCP Client

```python
from llmring.mcp.client.mcp_client import MCPClient

# Connect to MCP server
client = MCPClient("http://localhost:8000")
await client.initialize()

# List available tools
tools = await client.list_tools()
print(f"Available tools: {[tool['name'] for tool in tools]}")

# Execute a tool
result = await client.call_tool("read_file", {"path": "/path/to/file.txt"})
print(result)
```

## MCP Chat Client

LLMRing includes a powerful, generic MCP chat client that can connect to any MCP server.

### Features

- **Persistent History**: Conversations are saved and can be resumed
- **Multiple Transports**: Supports HTTP, WebSocket, and stdio connections
- **Generic Design**: Works with any MCP-compliant server
- **Rich Commands**: Built-in commands for managing sessions and history

### Running the Chat Client

```bash
# Connect to a local MCP server via stdio
llmring mcp chat --server "stdio://python -m your_mcp_server"

# Connect to an HTTP MCP server
llmring mcp chat --server "http://localhost:8080"

# Connect to a WebSocket MCP server
llmring mcp chat --server "ws://localhost:8080"

# Use a specific model for the chat
llmring mcp chat --model advisor --server "stdio://your-server"
```

### Chat Commands

The chat client provides several built-in commands:

- `/help` - Show available commands
- `/history` - Display current conversation history
- `/sessions` - List all saved chat sessions
- `/load <session_id>` - Load a previous session
- `/clear` - Clear current conversation
- `/model <alias>` - Switch to a different model
- `/tools` - List available MCP tools
- `/exit` or `/quit` - Exit the chat

### Persistent History

Chat history is automatically saved in `~/.llmring/mcp_chat/`:

```
~/.llmring/mcp_chat/
├── command_history.txt        # Command line history
├── conversation_<id>.json     # Individual conversation files
└── sessions.json              # Session metadata
```

Each conversation is saved with:
- Session ID and timestamp
- Complete message history
- Tool calls and results
- Model used for each response

### Creating Custom MCP Servers

You can create your own MCP servers and use them with the chat client:

```python
from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport

# Create your custom server
server = MCPServer(name="My Custom Server", version="1.0.0")

# Register your tools
@server.function_registry.register(
    name="my_tool",
    description="My custom tool"
)
def my_tool(param: str) -> dict:
    return {"result": f"Processed: {param}"}

# Run the server
async def main():
    transport = StdioTransport()
    await server.run(transport)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Then connect to it with the chat client:

```bash
llmring mcp chat --server "stdio://python my_server.py"
```

## Advanced Usage

### Streaming with Tools

```python
# Enhanced LLM supports streaming with automatic tool execution
messages = [{"role": "user", "content": "Analyze this file and summarize it"}]

async for chunk in await llm.chat_stream(messages):
    if chunk.type == "content":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "tool_call":
        print(f"\n[Calling tool: {chunk.tool_call.name}]")
    elif chunk.type == "tool_result":
        print(f"\n[Tool result received]")
```

### Using MCP Chat Client Programmatically

```python
from llmring.mcp.client.chat.app import MCPChatApp

# Create chat app with custom server
app = MCPChatApp(
    mcp_server_url="stdio://python -m my_mcp_server",
    llm_model="advisor",
    enable_telemetry=False
)

# Initialize and run
await app.initialize_async()
await app.run_interactive()
```

## Best Practices

### Performance
- Use streaming for long tool-assisted responses
- Cache tool results when appropriate
- Choose models based on tool complexity
- Use persistent history to avoid re-running expensive operations

### Security
- Validate tool parameters thoroughly
- Use secure transport (HTTPS/WSS) in production
- Implement proper authentication for MCP servers
- Be cautious with tools that access sensitive data

### Custom Servers
- Design tools to be data-focused rather than intelligent
- Let the LLM be in the driver's seat for decision making
- Provide clear, descriptive tool names and schemas
- Handle errors gracefully and return informative messages

For detailed examples, see the `examples/mcp/` directory.
