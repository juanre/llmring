#!/usr/bin/env python3
"""
Demo script showcasing MCP conversational lockfile management.

This demonstrates the complete MCP implementation with:
- Rate limiting (built into transports)
- Telemetry (via llmring-server if available)
- Logging (ProviderLoggingMixin)
- Tool execution with retry logic
- Connection recovery
"""

import asyncio
import os
from pathlib import Path

# Ensure environment is set up
os.environ.setdefault("LLMRING_LOG_LEVEL", "INFO")


async def demo_features():
    """Demonstrate key MCP features."""

    print("=" * 70)
    print("🚀 LLMRing MCP Conversational Lockfile Management Demo")
    print("=" * 70)

    # Show existing infrastructure
    print("\n📋 Existing Infrastructure:")
    print("  ✅ Rate Limiting: Built into SSE/HTTP transports (60 req/min default)")
    print("  ✅ Telemetry: Via llmring-server integration")
    print("  ✅ Logging: Standardized with ProviderLoggingMixin")
    print("  ✅ Persistence: AsyncConversationManager")
    print("  ✅ Retry Logic: Automatic with exponential backoff")
    print("  ✅ Tool Execution: Recursive loop with depth limiting")

    # Check for llmring-server
    if os.getenv("LLMRING_SERVER_URL"):
        print(f"\n📊 Telemetry Backend: {os.getenv('LLMRING_SERVER_URL')}")
        print("  - Conversation tracking enabled")
        print("  - Tool execution logging enabled")
        print("  - Metrics collection enabled")
    else:
        print("\n📊 Telemetry: Disabled (set LLMRING_SERVER_URL to enable)")

    print("\n" + "=" * 70)
    print("🎯 Demo Scenarios")
    print("=" * 70)

    from unittest.mock import AsyncMock

    from llmring.mcp.client.chat.app import MCPChatApp
    from llmring.schemas import LLMResponse, Message

    # Create chat app with all features enabled
    app = MCPChatApp(
        mcp_server_url="stdio://python -m llmring.mcp.server.lockfile_server",
        llm_model="advisor",
        enable_telemetry=True,  # Will auto-detect based on env
    )

    # Initialize
    await app.initialize_async()

    print("\n1️⃣ Tool Discovery")
    print("-" * 40)

    if app.available_tools:
        print(f"Found {len(app.available_tools)} tools:")
        for i, (name, tool) in enumerate(list(app.available_tools.items())[:5], 1):
            print(f"  {i}. {name}: {tool.get('description', '')[:50]}...")
    else:
        print("No tools found (server may not be running)")

    print("\n2️⃣ Rate Limiting Demo")
    print("-" * 40)
    print("The MCP client transports have built-in rate limiting:")
    print("  - SSE Transport: 60 requests/minute (configurable)")
    print("  - HTTP Transport: Automatic retry with backoff")
    print("  - STDIO Transport: No rate limiting needed")

    print("\n3️⃣ Connection Recovery Demo")
    print("-" * 40)
    print("The chat app now includes:")
    print("  - Automatic reconnection on connection errors")
    print("  - Exponential backoff (1s, 2s, 4s)")
    print("  - Tool retry after successful reconnection")

    print("\n4️⃣ Tool Execution Loop")
    print("-" * 40)
    print("Features:")
    print("  - Native LLM function calling")
    print("  - JSON fallback parser")
    print("  - Recursive execution (max depth: 5)")
    print("  - Continues until final answer")

    # Simulate a conversation
    print("\n" + "=" * 70)
    print("💬 Simulated Conversation")
    print("=" * 70)

    conversations = [
        ("List my aliases", "list_aliases", "Shows current alias configuration"),
        ("Add a fast alias", "add_alias", "Adds alias with specified model"),
        ("Show model details", "assess_model", "Shows model capabilities"),
        ("Analyze my costs", "analyze_costs", "Estimates monthly expenses"),
    ]

    for query, tool, description in conversations:
        print(f"\n👤 User: {query}")
        print(f"🤖 Assistant: [Would call {tool} tool]")
        print(f"   → {description}")

    # Show telemetry if available
    if app.integration:
        print("\n" + "=" * 70)
        print("📈 Telemetry Active")
        print("=" * 70)
        print("All interactions are being logged to llmring-server:")
        print("  - Conversation history preserved")
        print("  - Tool execution tracked")
        print("  - Performance metrics collected")

    # Clean up
    await app.cleanup()

    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\nTo run the full interactive chat:")
    print("  llmring lock chat")
    print("\nWith custom model:")
    print("  llmring lock chat --model advisor")
    print("\nWith telemetry:")
    print("  LLMRING_SERVER_URL=http://localhost:8000 llmring lock chat")


async def main():
    """Main entry point."""
    try:
        await demo_features()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
