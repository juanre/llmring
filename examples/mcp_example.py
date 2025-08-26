#!/usr/bin/env python
"""
Example of using MCP (Model Context Protocol) with LLMRing.

This example demonstrates:
1. Creating an MCP client
2. Connecting to an MCP server
3. Using tools and resources
4. Managing conversations with context
"""

import asyncio
import os
from typing import List, Dict, Any

# Import MCP components from llmring
from llmring.mcp.client import AsyncMCPClient, MCPClient
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.service import LLMRing


async def basic_mcp_example():
    """Basic example of using MCP client."""
    
    # Create an MCP client
    client = AsyncMCPClient()
    
    # In production, you would connect to an MCP server
    # await client.connect("ws://localhost:8080/mcp")
    
    print("MCP Client created successfully")
    
    # List available tools (if connected to a server)
    # tools = await client.list_tools()
    # print(f"Available tools: {tools}")
    
    # Execute a tool (if available)
    # result = await client.execute_tool("calculator", {"expression": "2 + 2"})
    # print(f"Tool result: {result}")


async def conversation_with_mcp():
    """Example of managing conversations with MCP."""
    
    # Create a conversation manager
    manager = AsyncConversationManager(
        mcp_client=AsyncMCPClient(),
        enable_persistence=False  # Use in-memory storage for this example
    )
    
    # Create a new conversation
    conversation_id = await manager.create_conversation(
        title="Example Chat",
        system_prompt="You are a helpful assistant.",
    )
    
    print(f"Created conversation: {conversation_id}")
    
    # Add a message to the conversation
    await manager.add_message(
        conversation_id,
        role="user",
        content="Hello! What is MCP?"
    )
    
    # Get conversation history
    messages = await manager.get_messages(conversation_id)
    print(f"Conversation has {len(messages)} messages")
    
    # In production, you would process the conversation with an LLM
    # and add the assistant's response back to the conversation


async def llmring_with_mcp():
    """Example of using LLMRing with MCP for enhanced capabilities."""
    
    # Create LLMRing service
    llmring = LLMRing(origin="mcp-example")
    
    # Create a conversation manager with MCP
    conversation_manager = AsyncConversationManager(
        mcp_client=AsyncMCPClient(),
        enable_persistence=False
    )
    
    # Create a conversation
    conv_id = await conversation_manager.create_conversation(
        title="LLMRing + MCP Demo",
        system_prompt="You are an AI assistant with access to tools via MCP.",
    )
    
    # Add user message
    await conversation_manager.add_message(
        conv_id,
        role="user", 
        content="What's the weather like today?"
    )
    
    # Get messages for LLM processing
    messages = await conversation_manager.get_messages(conv_id)
    
    # Process with LLMRing (using your configured model alias)
    # In production, you would have a model alias configured
    # response = await llmring.chat(
    #     alias="claude-3-sonnet",
    #     messages=[{"role": msg.role, "content": msg.content} for msg in messages]
    # )
    
    print(f"Conversation {conv_id} ready for LLM processing")
    print(f"Messages in conversation: {len(messages)}")


async def mcp_server_example():
    """Example of creating an MCP server."""
    
    from llmring.mcp.server import MCPServer
    
    # Create an MCP server
    server = MCPServer(
        name="example-mcp-server",
        version="1.0.0"
    )
    
    # Register a tool
    @server.tool("calculator")
    async def calculator_tool(expression: str) -> float:
        """Simple calculator tool."""
        try:
            # In production, use a safe expression evaluator
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    # Register a resource
    @server.resource("config")
    async def get_config() -> Dict[str, Any]:
        """Get configuration resource."""
        return {
            "version": "1.0.0",
            "features": ["calculator", "weather"],
            "max_tokens": 1000
        }
    
    print("MCP Server created with tools and resources")
    
    # In production, you would start the server
    # await server.start(host="localhost", port=8080)


def main():
    """Run all examples."""
    
    print("=" * 60)
    print("MCP (Model Context Protocol) Examples with LLMRing")
    print("=" * 60)
    
    # Run async examples
    asyncio.run(basic_mcp_example())
    print("-" * 60)
    
    asyncio.run(conversation_with_mcp())
    print("-" * 60)
    
    asyncio.run(llmring_with_mcp())
    print("-" * 60)
    
    asyncio.run(mcp_server_example())
    print("-" * 60)
    
    print("\nMCP integration complete!")
    print("For production use:")
    print("1. Configure your MCP server endpoint")
    print("2. Set up proper authentication")
    print("3. Enable persistence with a database")
    print("4. Configure your LLM aliases in llmring")


if __name__ == "__main__":
    main()