# MCP (Model Context Protocol) Integration

LLMRing provides comprehensive MCP integration for building tool-enabled LLM applications.

## Overview

MCP allows LLMs to interact with external tools and resources through a standardized protocol. LLMRing's MCP integration provides:

- **Enhanced LLM**: LLM instances with built-in tool capabilities
- **Tool Execution**: Automatic tool discovery and execution
- **Streaming Support**: Tool calls work with streaming responses
- **Multiple Transports**: HTTP, WebSocket, and stdio transports

## Quick Start

### Basic MCP Integration

```python
from llmring.mcp.client.enhanced_llm import create_enhanced_llm

# Create an enhanced LLM with MCP tools
llm = await create_enhanced_llm(
    model="openai:gpt-4o",
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

## Best Practices

### Performance
- Use streaming for long tool-assisted responses
- Cache tool results when appropriate
- Choose models based on tool complexity

### Security
- Validate tool parameters thoroughly
- Use secure transport (HTTPS/WSS) in production
- Implement proper authentication for MCP servers

For detailed examples, see the `examples/mcp/` directory.