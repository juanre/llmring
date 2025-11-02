---
name: tools
description: Use when implementing function calling, tool use, or agents with LLMs - unified tool API works across OpenAI, Anthropic, Google, and Ollama with consistent tool definition and execution patterns
---

# Function Calling and Tool Use

## Installation

```bash
# With uv (recommended)
uv add llmring

# With pip
pip install llmring
```

**Provider SDKs (install what you need):**
```bash
uv add openai>=1.0      # OpenAI
uv add anthropic>=0.67   # Anthropic
uv add google-genai      # Google Gemini
uv add ollama>=0.4       # Ollama (prompt-based tools)
```

## API Overview

This skill covers:
- Tool definition format (JSON Schema)
- `tools` parameter in LLMRequest
- `tool_choice` parameter for controlling tool selection
- `tool_calls` in LLMResponse
- Multi-turn tool execution pattern
- Tool result messages

## Quick Start

```python
from llmring import LLMRing, LLMRequest, Message

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}]

async with LLMRing() as service:
    request = LLMRequest(
        model="tool-user",  # Your alias for tool-using tasks
        messages=[Message(role="user", content="What's the weather in NYC?")],
        tools=tools
    )

    response = await service.chat(request)

    # Check if model wants to call a tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

## Complete API Documentation

### Tool Definition Format

Tools use JSON Schema format for function definitions.

**Structure:**
```python
tool = {
    "type": "function",
    "function": {
        "name": str,           # Function name
        "description": str,    # What the function does
        "parameters": {        # JSON Schema for parameters
            "type": "object",
            "properties": {
                "param_name": {
                    "type": str,      # "string", "number", "boolean", "array", "object"
                    "description": str,
                    "enum": List[str]  # Optional: allowed values
                }
            },
            "required": List[str]  # Required parameter names
        }
    }
}
```

**Example:**
```python
get_weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'New York' or 'London'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}
```

### LLMRequest with Tools

**Parameters:**
- `tools` (list, optional): List of available tools
- `tool_choice` (str/dict, optional): Control tool selection
  - `"auto"`: Model decides whether to use tools (default)
  - `"none"`: Force no tool use
  - `"required"`: Force tool use (OpenAI) / `"any"` (Anthropic equivalent)
  - `{"type": "function", "function": {"name": "tool_name"}}`: Force specific tool

**Note:** Provider differences - OpenAI uses `"required"`, Anthropic uses `"any"` for the same behavior. LLMRing handles the translation automatically.

**Example:**
```python
from llmring import LLMRequest, Message

# Let model decide
request = LLMRequest(
    model="tool-user",  # Your alias for tool-using tasks
    messages=[Message(role="user", content="What's the weather?")],
    tools=[get_weather_tool]
)

# Force tool use
request = LLMRequest(
    model="tool-user",  # Your alias for tool-using tasks
    messages=[Message(role="user", content="Check NYC weather")],
    tools=[get_weather_tool],
    tool_choice="required"  # Must use a tool
)

# Force specific tool
request = LLMRequest(
    model="tool-user",  # Your alias for tool-using tasks
    messages=[Message(role="user", content="Get weather")],
    tools=[get_weather_tool, search_tool],
    tool_choice={
        "type": "function",
        "function": {"name": "get_weather"}
    }
)

# Disable tools
request = LLMRequest(
    model="tool-user",  # Your alias for tool-using tasks
    messages=[Message(role="user", content="Just chat")],
    tools=[get_weather_tool],
    tool_choice="none"  # Don't use tools
)
```

### Tool Calls in Response

When the model wants to call a tool, `response.tool_calls` is populated.

**Structure:**
```python
tool_call = {
    "id": str,              # Unique tool call ID
    "type": "function",
    "function": {
        "name": str,        # Function name
        "arguments": str    # JSON string of arguments
    }
}
```

**Example:**
```python
response = await service.chat(request)

if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])
        tool_id = tool_call["id"]

        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_args}")
```

### Tool Result Messages

After executing a tool, send the result back with role="tool".

**Structure:**
```python
tool_result = Message(
    role="tool",
    content: str,         # Tool execution result (JSON string)
    tool_call_id: str     # ID from the tool_call
)
```

**Example:**
```python
import json
from llmring import Message

# Execute tool
tool_result = {"temperature": 72, "condition": "sunny"}

# Send result back
result_message = Message(
    role="tool",
    content=json.dumps(tool_result),
    tool_call_id=tool_call["id"]
)
```

## Complete Tool Execution Pattern

```python
import json
from llmring import LLMRing, LLMRequest, Message

def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Mock weather function."""
    return {
        "location": location,
        "temperature": 72,
        "unit": unit,
        "condition": "sunny"
    }

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}]

async with LLMRing() as service:
    messages = [
        Message(role="user", content="What's the weather in San Francisco?")
    ]

    # First request: Model decides to use tool
    request = LLMRequest(model="tool-user",  # Your alias for tool-using tasks messages=messages, tools=tools)
    response = await service.chat(request)

    # Add assistant's tool call to history
    messages.append(Message(
        role="assistant",
        content=response.content or "",
        tool_calls=response.tool_calls
    ))

    # Execute tools
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Parse arguments
            args = json.loads(tool_call["function"]["arguments"])

            # Execute function
            result = get_weather(**args)

            # Add tool result to messages
            messages.append(Message(
                role="tool",
                content=json.dumps(result),
                tool_call_id=tool_call["id"]
            ))

    # Second request: Model uses tool results to answer
    request = LLMRequest(model="tool-user",  # Your alias for tool-using tasks messages=messages, tools=tools)
    response = await service.chat(request)

    print(response.content)
    # "The weather in San Francisco is 72°F and sunny."
```

## Common Patterns

### Multiple Tools

```python
from llmring import LLMRing, LLMRequest, Message

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

async with LLMRing() as service:
    request = LLMRequest(
        model="tool-user",  # Your alias for tool-using tasks
        messages=[Message(role="user", content="Weather in Paris and latest news")],
        tools=tools
    )

    response = await service.chat(request)

    # Model may call multiple tools
    if response.tool_calls:
        print(f"Model wants to call {len(response.tool_calls)} tools")
```

### Tool Execution Loop

```python
import json
from llmring import LLMRing, LLMRequest, Message

# Available functions
FUNCTIONS = {
    "get_weather": lambda location: {"temp": 72, "condition": "sunny"},
    "search_web": lambda query: {"results": ["Result 1", "Result 2"]}
}

async with LLMRing() as service:
    messages = [
        Message(role="user", content="What's the weather in NYC?")
    ]

    tools = [...]  # Your tool definitions

    # Loop until model stops calling tools
    while True:
        request = LLMRequest(model="tool-user",  # Your alias for tool-using tasks messages=messages, tools=tools)
        response = await service.chat(request)

        # Add assistant response
        messages.append(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        ))

        # If no tool calls, we're done
        if not response.tool_calls:
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            # Execute function
            result = FUNCTIONS[func_name](**args)

            # Add result
            messages.append(Message(
                role="tool",
                content=json.dumps(result),
                tool_call_id=tool_call["id"]
            ))

    # Final response
    print(response.content)
```

### Parallel Tool Calls

Some models can call multiple tools in parallel:

```python
import json
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="tool-user",  # Your alias for tool-using tasks
        messages=[Message(role="user", content="Weather in NYC and Paris")],
        tools=tools
    )

    response = await service.chat(request)

    # Model may request multiple tool calls at once
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call['function']['name']}")
            print(f"Args: {tool_call['function']['arguments']}")
            # Execute all tools, then send all results back
```

### Streaming with Tools

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="tool-user",  # Your alias for tool-using tasks
        messages=[Message(role="user", content="What's the weather?")],
        tools=tools
    )

    tool_calls_accumulated = []

    async for chunk in service.chat_stream(request):
        print(chunk.delta, end="", flush=True)

        # Accumulate tool calls (streamed incrementally)
        if chunk.tool_calls:
            tool_calls_accumulated = chunk.tool_calls

    # After streaming, check for tool calls
    if tool_calls_accumulated:
        print("\nModel wants to call tools")
```

### Error Handling in Tools

```python
import json
from llmring import LLMRing, LLMRequest, Message

def execute_tool_safely(func_name: str, args: dict) -> dict:
    """Execute tool with error handling."""
    try:
        result = FUNCTIONS[func_name](**args)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

async with LLMRing() as service:
    # ... after getting tool_calls ...

    for tool_call in response.tool_calls:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        # Execute with error handling
        result = execute_tool_safely(func_name, args)

        # Send result (including errors)
        messages.append(Message(
            role="tool",
            content=json.dumps(result),
            tool_call_id=tool_call["id"]
        ))
```

## Provider Support

| Provider | Tool Support | Notes |
|----------|--------------|-------|
| **OpenAI** | Native | Full support, parallel calls |
| **Anthropic** | Native | Full support |
| **Google** | Native | Full support |
| **Ollama** | Prompt-based | Tools via prompt engineering |

**Note:** Ollama uses prompt-based tool calling. LLMRing handles the adaptation automatically.

## Common Mistakes

### Wrong: Not Including Assistant Message with tool_calls

```python
# DON'T DO THIS - skip assistant message
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append(Message(role="tool", content=result))
# Missing assistant message!
```

**Right: Include Assistant Message**

```python
# DO THIS - add assistant message with tool_calls
messages.append(Message(
    role="assistant",
    content=response.content or "",
    tool_calls=response.tool_calls  # Include this!
))

# Then add tool results
for tool_call in response.tool_calls:
    result = execute_tool(tool_call)
    messages.append(Message(
        role="tool",
        content=json.dumps(result),
        tool_call_id=tool_call["id"]
    ))
```

### Wrong: Forgetting tool_call_id

```python
# DON'T DO THIS - no tool_call_id
messages.append(Message(
    role="tool",
    content=json.dumps(result)
))
```

**Right: Include tool_call_id**

```python
# DO THIS - include tool_call_id
messages.append(Message(
    role="tool",
    content=json.dumps(result),
    tool_call_id=tool_call["id"]  # Required!
))
```

### Wrong: Tool Result Not JSON String

```python
# DON'T DO THIS - Python dict
result = {"temperature": 72}
messages.append(Message(
    role="tool",
    content=result,  # Should be string!
    tool_call_id=tool_id
))
```

**Right: JSON String**

```python
# DO THIS - JSON string
result = {"temperature": 72}
messages.append(Message(
    role="tool",
    content=json.dumps(result),  # Convert to string
    tool_call_id=tool_id
))
```

### Wrong: Poor Tool Descriptions

```python
# DON'T DO THIS - vague description
{
    "name": "get_data",
    "description": "Gets data",
    "parameters": {...}
}
```

**Right: Clear Descriptions**

```python
# DO THIS - specific, actionable description
{
    "name": "get_weather",
    "description": "Get the current weather conditions for a specific city, including temperature and general conditions",
    "parameters": {...}
}
```

## Best Practices

1. **Clear tool descriptions:** Model uses descriptions to decide when to call tools
2. **Include tool_calls in assistant message:** Required for proper conversation flow
3. **Always include tool_call_id:** Links results to requests
4. **Use JSON strings for results:** Tool content must be string, not dict
5. **Handle errors gracefully:** Return error info as JSON to let model respond
6. **Validate tool arguments:** Parse and validate before executing
7. **Loop until done:** Continue conversation until no more tool calls

## Related Skills

- `llmring-chat` - Basic chat without tools
- `llmring-streaming` - Streaming tool calls
- `llmring-structured` - Combine tools with structured output
- `llmring-lockfile` - Configure models for tool use
- `llmring-providers` - Provider-specific tool features
