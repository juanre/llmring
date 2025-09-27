#!/usr/bin/env python3
"""
Script to replicate the Google tool calling loop issue.

This script simulates exactly what happens in the MCP chat interface when
Google's Gemini model keeps calling the same tool repeatedly instead of
providing a final answer.

To run:
    python replicate_google_loop.py

Expected behavior:
    The model should call get_configuration once, receive the result,
    and then provide a final answer describing the lockfile contents.

Actual behavior:
    The model keeps calling get_configuration repeatedly until hitting
    the maximum depth limit.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Ensure Google API key is available
if os.environ.get("GOOGLE_GEMINI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_GEMINI_API_KEY"]

from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

# Simulated MCP server response for get_configuration tool
MOCK_LOCKFILE_CONFIG = {
    "profiles": {
        "default": {
            "name": "default",
            "bindings": [
                {
                    "alias": "advisor",
                    "models": [
                        "google:gemini-2.5-pro",
                        "anthropic:claude-3.5-haiku-20241022",
                        "openai:gpt-4o",
                    ],
                },
                {"alias": "writer", "models": ["anthropic:claude-3-5-sonnet-20241022"]},
                {"alias": "fast", "models": ["openai:gpt-4o-mini"]},
            ],
        },
        "dev": {
            "name": "dev",
            "bindings": [
                {
                    "alias": "advisor",
                    "models": [
                        "anthropic:claude-3-5-haiku-20241022",
                        "openai:gpt-4o-mini",
                        "google:gemini-flash",
                    ],
                }
            ],
        },
    },
    "default_profile": "default",
    "version": "1.0.0",
}


async def simulate_chat_conversation():
    """Simulate the exact conversation flow from MCP chat interface."""

    # Initialize LLMRing
    ring = LLMRing()

    # Check if Google provider is available
    if "google" not in ring.providers:
        print("❌ Google provider not initialized. Please set GEMINI_API_KEY or GOOGLE_API_KEY")
        return

    # System message from MCP chat
    system_message = """You are the LLMRing Lockfile Manager assistant. You help users manage their LLM aliases and model configurations.

You have access to MCP tools to manage the lockfile. When the user asks about aliases, models, or configurations, use the appropriate tool to help them.

Current tools available:
- get_configuration: Get the complete current lockfile configuration
- add_alias: Add a new alias with model bindings
- update_alias: Update an existing alias
- remove_alias: Remove an alias
- list_models: List available models from providers
- analyze_cost: Analyze potential costs for current configuration
- bump_registry: Update registry versions in lockfile

Be helpful and informative. When showing configurations, format them clearly."""

    # Define the MCP tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_configuration",
                "description": "Get the complete current lockfile configuration.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    # Start with the user's question
    messages = [
        Message(role="system", content=system_message),
        Message(role="user", content="Do we have a lockfile? What does it contain?"),
    ]

    print("=" * 70)
    print("REPLICATING GOOGLE TOOL CALLING LOOP ISSUE")
    print("=" * 70)
    print("\n🧑 User: Do we have a lockfile? What does it contain?\n")

    # Track recursion depth
    depth = 0
    max_depth = 5

    while depth < max_depth:
        depth += 1
        print(f"--- Iteration {depth}/{max_depth} ---")

        # Create request with tools
        request = LLMRequest(
            messages=messages, model="google:gemini-2.5-pro", tools=tools, tool_choice="auto"
        )

        # Get response from Google
        print("🤖 Sending request to Google Gemini...")
        response = await ring.chat(request)

        # Show response details
        if response.content:
            print(f"   Content: {response.content[:100]}...")
        else:
            print("   Content: (empty)")

        if response.tool_calls:
            print(f"   Tool calls: YES - {len(response.tool_calls)} call(s)")
            for call in response.tool_calls:
                func_name = call.get("function", {}).get("name", "unknown")
                print(f"   → Calling tool: {func_name} ✓")
        else:
            print("   Tool calls: NO")
            print("\n✅ Model provided final answer without calling tools!")
            break

        # If no tool calls, we're done
        if not response.tool_calls:
            break

        # Add assistant's response with tool calls
        messages.append(
            Message(
                role="assistant", content=response.content or "", tool_calls=response.tool_calls
            )
        )

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get("id", f"call_{depth}")
            function_name = tool_call.get("function", {}).get("name")

            # Simulate tool execution (always return the same config)
            if function_name == "get_configuration":
                tool_result = MOCK_LOCKFILE_CONFIG
            else:
                tool_result = {"error": f"Unknown tool: {function_name}"}

            # Add tool result to messages
            messages.append(
                Message(role="tool", content=json.dumps(tool_result), tool_call_id=tool_call_id)
            )

        print(f"   Messages in conversation: {len(messages)}")

        # Check if we're about to loop
        if depth < max_depth:
            print("   ⚠️  Continuing with tools still available...")

    if depth >= max_depth:
        print("\n❌ ISSUE REPRODUCED: Model called tools in every iteration!")
        print("   The model never provided a final answer about the lockfile contents.")
        print("   Instead, it kept calling get_configuration repeatedly.")

    print("\n" + "=" * 70)
    print("END OF REPLICATION")
    print("=" * 70)


async def main():
    """Main entry point."""
    try:
        await simulate_chat_conversation()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Google tool calling loop replication...")
    print("This simulates the exact issue seen in MCP chat interface.\n")
    asyncio.run(main())
