#!/usr/bin/env python3
"""
Quick start example for LLMRing.

Shows the most common use cases:
1. Basic chat without database
2. Chat with SQLite logging
3. Using different providers
4. Caching responses

Run with: python examples/quick_start.py
"""

import asyncio
import os
from llmring import LLMRing, LLMRequest, Message
from llmring.service_sqlite import LLMRingSQLite


async def example_no_database():
    """Example 1: Simple chat without database."""
    print("\n=== Example 1: No Database ===")

    # Create service without database
    service = LLMRing(enable_db_logging=False)

    # Make a request
    request = LLMRequest(
        messages=[Message(role="user", content="What is 2+2?")],
        model="gpt-4o-mini",  # or "claude-3-haiku-20240307"
        temperature=0
    )

    try:
        response = await service.chat(request)
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have OPENAI_API_KEY set")


async def example_with_sqlite():
    """Example 2: Chat with SQLite logging."""
    print("\n=== Example 2: With SQLite Database ===")

    # Create service with SQLite (auto-creates llmring.db)
    service = LLMRingSQLite()

    # Make a request (automatically logged)
    request = LLMRequest(
        messages=[Message(role="user", content="Tell me a short joke")],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100
    )

    try:
        response = await service.chat(request)
        print(f"Response: {response.content}")

        # Check usage stats
        stats = await service.get_usage_stats("user123", days=1)
        if stats:
            print(f"Total calls today: {stats.total_calls}")
            print(f"Total cost: ${stats.total_cost:.4f}")
    except Exception as e:
        print(f"Error: {e}")


async def example_with_caching():
    """Example 3: Using response caching."""
    print("\n=== Example 3: With Caching ===")

    service = LLMRing(enable_db_logging=False)

    # First request (not cached)
    request = LLMRequest(
        messages=[Message(role="user", content="What is the capital of France?")],
        model="gpt-4o-mini",
        temperature=0,  # Must be <= 0.1 for caching
        cache={"enabled": True, "ttl_seconds": 300}  # Cache for 5 minutes
    )

    print("First request (will hit API)...")
    response1 = await service.chat(request)
    print(f"Response: {response1.content}")

    # Second identical request (should be cached)
    print("Second request (should be cached)...")
    response2 = await service.chat(request)
    print(f"Response: {response2.content}")


async def example_different_providers():
    """Example 4: Using different providers."""
    print("\n=== Example 4: Different Providers ===")

    service = LLMRing(enable_db_logging=False)

    providers = [
        ("openai:gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic:claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        # ("google:gemini-1.5-flash", "GOOGLE_API_KEY"),
        # ("ollama:llama2", None),  # Requires local Ollama
    ]

    for model, env_var in providers:
        if env_var and not os.getenv(env_var):
            print(f"Skipping {model} (no {env_var})")
            continue

        try:
            request = LLMRequest(
                messages=[Message(role="user", content="Say 'hello' in one word")],
                model=model,
                temperature=0,
                max_tokens=10
            )
            response = await service.chat(request)
            print(f"{model}: {response.content}")
        except Exception as e:
            print(f"{model}: Error - {e}")


async def main():
    """Run all examples."""
    print("🚀 LLMRing Quick Start Examples")
    print("=" * 50)

    # Run examples
    await example_no_database()
    await example_with_sqlite()
    await example_with_caching()
    await example_different_providers()

    print("\n✨ Done!")


if __name__ == "__main__":
    # Set up your API keys as environment variables:
    # export OPENAI_API_KEY=sk-...
    # export ANTHROPIC_API_KEY=sk-ant-...
    # export GOOGLE_API_KEY=...

    asyncio.run(main())
