# MCP (Model Context Protocol) Integration

## Overview

LLMRing provides experimental support for the Model Context Protocol (MCP), enabling rich interactions between LLMs and external tools, resources, and prompts. The implementation follows a clean architecture where llmring remains database-agnostic while llmring-server handles all persistence.

## Architecture

### Separation of Concerns

```
┌─────────────┐     HTTP      ┌──────────────┐
│   llmring   │───────────────▶│llmring-server│
│  (no DB)    │                │  (all DB)    │
└─────────────┘                └──────────────┘
      │                               │
      ▼                               ▼
┌─────────────┐                ┌──────────────┐
│ MCP Client  │                │  PostgreSQL  │
└─────────────┘                └──────────────┘
```

- **llmring**: Core library, database-agnostic, uses HTTP for persistence
- **llmring-server**: REST API backend, handles ALL database operations
- **MCP Integration**: Optional feature, cleanly separated

## MCP Client

### Basic Client Usage

```python
from llmring.mcp.client import AsyncMCPClient

async def main():
    # Connect to an MCP server
    client = AsyncMCPClient("http://localhost:8080")
    
    # Initialize connection
    await client.initialize()
    
    # List available tools
    tools = await client.list_tools()
    for tool in tools:
        print(f"Tool: {tool['name']} - {tool['description']}")
    
    # Execute a tool
    result = await client.call_tool(
        "calculator",
        {"expression": "15 * 23"}
    )
    print(f"Result: {result}")
    
    # List resources
    resources = await client.list_resources()
    
    # Read a resource
    content = await client.read_resource("file://docs/guide.md")
```

### Enhanced LLM with Tools

The `EnhancedLLM` class provides automatic tool usage during conversations:

```python
from llmring.mcp.client.enhanced_llm import EnhancedLLM

# Create enhanced LLM
llm = EnhancedLLM(
    llmring_server_url="http://localhost:8000",  # Optional
    default_model="openai:gpt-4o"
)

# Register tools
llm.register_tool(
    name="weather",
    description="Get current weather",
    parameters={
        "location": {"type": "string", "description": "City name"}
    },
    handler=get_weather_function
)

llm.register_tool(
    name="calculate",
    description="Perform calculations",
    parameters={
        "expression": {"type": "string", "description": "Math expression"}
    },
    handler=calculate_function
)

# Chat with automatic tool usage
response = await llm.chat([
    {"role": "user", "content": "What's the weather in Paris?"}
])
# The LLM will automatically call the weather tool

response = await llm.chat([
    {"role": "user", "content": "Calculate 15 * 23 for me"}
])
# The LLM will automatically call the calculate tool
```

### MCP Client with LLM Sampling

The `MCPClientWithLLM` supports server-initiated LLM sampling:

```python
from llmring.mcp.client.llm_client import MCPClientWithLLM

# Client that can handle sampling requests from the server
client = MCPClientWithLLM(
    "http://localhost:8080",
    default_model="claude-3-sonnet",
    sampling_config={
        "enabled": True,
        "max_tokens": 1000,
        "temperature": 0.7,
        "allowed_models": ["gpt-4", "claude-3-sonnet"]
    }
)

# When the server requests LLM sampling, this client can fulfill it
await client.initialize()
```

## MCP Server

### Creating an MCP Server

```python
from llmring.mcp.server import Server
from llmring.mcp.types import Tool, Resource, Prompt

# Create server
server = Server(name="my-tools-server")

# Add a tool
@server.tool()
async def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Add a resource
@server.resource("file://config.json")
async def get_config():
    """Provide configuration data."""
    return {
        "contents": [
            {
                "mimeType": "application/json",
                "text": '{"setting": "value"}'
            }
        ]
    }

# Add a prompt
@server.prompt()
async def code_review_prompt(language: str = "python") -> Prompt:
    """Generate a code review prompt."""
    return Prompt(
        name="code_review",
        description="Review code for best practices",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert {language} code reviewer..."
            }
        ]
    )

# Run server (STDIO transport)
from llmring.mcp.server.stdio import StdioServerTransport

transport = StdioServerTransport()
await server.run(transport)
```

### Transport Options

MCP servers support multiple transport mechanisms:

```python
# STDIO Transport (for subprocess communication)
from llmring.mcp.server.stdio import StdioServerTransport
transport = StdioServerTransport()

# HTTP Transport (for network communication)
from llmring.mcp.server.http import HTTPServerTransport
transport = HTTPServerTransport(port=8080)

# SSE Transport (for server-sent events)
from llmring.mcp.server.sse import SSEServerTransport
transport = SSEServerTransport(port=8081)

await server.run(transport)
```

## Persistence via llmring-server

All MCP data persistence goes through llmring-server REST API:

### Available Endpoints

```
POST   /mcp/servers           # Register MCP server
GET    /mcp/servers           # List servers
DELETE /mcp/servers/{id}      # Remove server

GET    /mcp/tools             # List tools
POST   /mcp/tools/{id}/execute # Execute tool

GET    /mcp/resources         # List resources
GET    /mcp/resources/{id}    # Read resource

GET    /mcp/prompts           # List prompts
GET    /mcp/prompts/{id}      # Get prompt

POST   /conversations         # Create conversation
GET    /conversations/{id}    # Get conversation
POST   /conversations/{id}/messages # Add message
```

### HTTP Client

The `MCPHttpClient` handles all communication with llmring-server:

```python
from llmring.mcp.http_client import MCPHttpClient

client = MCPHttpClient(
    base_url="http://localhost:8000",
    api_key="optional-api-key"
)

# Register an MCP server
server_data = await client.register_server(
    name="my-server",
    url="http://localhost:8080",
    transport_type="http"
)

# List available tools
tools = await client.list_tools()

# Execute a tool
result = await client.execute_tool(
    tool_id="uuid-here",
    input={"param": "value"}
)

# Create a conversation
conv_id = await client.create_conversation(
    title="New Chat",
    system_prompt="You are helpful"
)

# Add messages
await client.add_message(
    conversation_id=conv_id,
    role="user",
    content="Hello!"
)
```

## Testing

### Running MCP Tests

```bash
# Run all MCP tests
pytest tests/mcp/

# Run client tests only
pytest tests/mcp/client/

# Run server tests only
pytest tests/mcp/server/

# Run with coverage
pytest tests/mcp/ --cov=llmring.mcp
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test client-server interactions
- **Transport Tests**: Test STDIO, HTTP, SSE transports
- **LLM Sampling Tests**: Test server-initiated LLM requests
- **File Processing Tests**: Test handling of images and documents

## Limitations

- MCP support is experimental and the API may change
- Not all MCP features are fully implemented yet
- Usage statistics via MCP require llmring-server
- Some advanced MCP features like subscriptions are partial

## Examples

### Weather Tool Server

```python
import aiohttp
from llmring.mcp.server import Server

server = Server(name="weather-server")

@server.tool()
async def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    async with aiohttp.ClientSession() as session:
        url = f"https://api.weather.com/v1/current?q={location}"
        async with session.get(url) as response:
            data = await response.json()
            return {
                "location": location,
                "temperature": data["temp"],
                "conditions": data["conditions"]
            }
```

### Database Query Tool

```python
@server.tool()
async def query_database(sql: str, params: list = None) -> dict:
    """Execute a read-only SQL query."""
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries allowed"}
    
    # Execute query (example with asyncpg)
    conn = await asyncpg.connect("postgresql://localhost/db")
    try:
        rows = await conn.fetch(sql, *(params or []))
        return {"rows": [dict(row) for row in rows]}
    finally:
        await conn.close()
```

### File System Resource Provider

```python
import os
from pathlib import Path

@server.resource("file://docs/*")
async def provide_docs(uri: str):
    """Provide documentation files."""
    path = uri.replace("file://docs/", "")
    file_path = Path("docs") / path
    
    if not file_path.exists():
        raise ValueError(f"File not found: {path}")
    
    content = file_path.read_text()
    return {
        "contents": [{
            "mimeType": "text/plain",
            "text": content
        }]
    }
```

## Future Enhancements

- Full subscription support for resources
- WebSocket transport implementation
- Tool composition and chaining
- Enhanced error handling and retries
- Performance optimizations for large-scale deployments
- Better integration with llmring-api for authenticated scenarios