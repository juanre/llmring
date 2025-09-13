#!/usr/bin/env python
"""
Example of using MCP Enhanced LLM with streaming support.

This example demonstrates:
1. Basic streaming without tools
2. Streaming with tool execution
3. Registering custom tools
"""

import asyncio
import json
from datetime import datetime

from llmring.mcp.client.enhanced_llm import EnhancedLLM, create_enhanced_llm
from llmring.schemas import Message


# Example custom tool
def get_current_time():
    """Get the current date and time."""
    return datetime.now().isoformat()


def calculate(expression: str):
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        The result of the calculation
    """
    # Safe evaluation of math expressions
    allowed_names = {
        "abs": abs,
        "round": round,
        "pow": pow,
        "max": max,
        "min": min,
    }
    
    try:
        # Use eval with restricted namespace for safety
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


async def example_streaming_without_tools():
    """Example of streaming without tools."""
    print("=" * 60)
    print("Example 1: Streaming without tools")
    print("=" * 60)
    
    # Create enhanced LLM
    llm = create_enhanced_llm(
        llm_model="fast",  # Use semantic alias
        origin="streaming-example"
    )
    
    # Simple streaming request
    messages = [
        Message(role="user", content="Count from 1 to 5 slowly")
    ]
    
    print("User: Count from 1 to 5 slowly")
    print("Assistant: ", end="", flush=True)
    
    # Stream the response
    stream = await llm.chat(messages, stream=True, max_tokens=50)
    
    async for chunk in stream:
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    
    print("\n")


async def example_streaming_with_tools():
    """Example of streaming with tool execution."""
    print("=" * 60)
    print("Example 2: Streaming with tools")
    print("=" * 60)
    
    # Create enhanced LLM
    llm = create_enhanced_llm(
        llm_model="fast",  # Use semantic alias
        origin="streaming-example"
    )
    
    # Register custom tools
    llm.register_tool(
        name="get_current_time",
        description="Get the current date and time",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        handler=get_current_time,
        module_name="time_tools"
    )
    
    llm.register_tool(
        name="calculate",
        description="Safely evaluate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        handler=calculate,
        module_name="math_tools"
    )
    
    # Request that will trigger tool use
    messages = [
        Message(
            role="user",
            content="What's the current time? Also, what's 42 * 17 + 89?"
        )
    ]
    
    print("User: What's the current time? Also, what's 42 * 17 + 89?")
    print("Assistant: ", end="", flush=True)
    
    # Stream the response (will handle tool calls automatically)
    stream = await llm.chat(messages, stream=True)
    
    async for chunk in stream:
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    
    print("\n")


async def example_non_streaming_with_tools():
    """Example of non-streaming with tools for comparison."""
    print("=" * 60)
    print("Example 3: Non-streaming with tools (for comparison)")
    print("=" * 60)
    
    # Create enhanced LLM
    llm = create_enhanced_llm(
        llm_model="fast",  # Use semantic alias
        origin="streaming-example"
    )
    
    # Register the calculate tool
    llm.register_tool(
        name="calculate",
        description="Safely evaluate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        handler=calculate,
        module_name="math_tools"
    )
    
    # Request that will trigger tool use
    messages = [
        Message(
            role="user",
            content="Calculate the sum of squares: (3**2) + (4**2) + (5**2)"
        )
    ]
    
    print("User: Calculate the sum of squares: (3**2) + (4**2) + (5**2)")
    
    # Non-streaming response
    response = await llm.chat(messages, stream=False)
    
    print(f"Assistant: {response.content}")
    
    if response.usage:
        print(f"\nUsage: {response.usage.get('total_tokens', 0)} tokens")
    
    print()


async def main():
    """Run all examples."""
    try:
        # Example 1: Simple streaming
        await example_streaming_without_tools()
        
        # Example 2: Streaming with tools
        await example_streaming_with_tools()
        
        # Example 3: Non-streaming with tools for comparison
        await example_non_streaming_with_tools()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run examples
    asyncio.run(main())