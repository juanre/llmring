"""
Example: Using LLMRing logging decorators with provider SDKs.

This example demonstrates how to use @log_llm_call and @log_llm_stream
decorators to add LLMRing logging to existing code using provider SDKs.
"""

import asyncio
import os

from llmring import log_llm_call, log_llm_stream

# -----------------------------------------------------------------------------
# Example 1: Logging OpenAI SDK calls
# -----------------------------------------------------------------------------


async def example_openai_logging():
    """Example of logging OpenAI SDK calls with LLMRing."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Wrap your existing function with the decorator
    @log_llm_call(
        server_url="http://localhost:8000",  # Your llmring-server URL
        provider="openai",  # Or use "auto" for auto-detection
        log_conversations=True,  # Log full conversations
        alias="my_chat_function",  # Optional alias for tracking
    )
    async def chat_with_gpt(messages, model="gpt-4o-mini"):
        """Your existing chat function."""
        return await client.chat.completions.create(
            model=model,
            messages=messages,
        )

    # Use the function normally - logging happens automatically
    response = await chat_with_gpt(
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )

    print(f"OpenAI response: {response.choices[0].message.content}")
    print("✓ Logged to llmring-server")


# -----------------------------------------------------------------------------
# Example 2: Logging Anthropic SDK calls
# -----------------------------------------------------------------------------


async def example_anthropic_logging():
    """Example of logging Anthropic SDK calls with LLMRing."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @log_llm_call(
        server_url="http://localhost:8000",
        provider="anthropic",
        log_metadata=True,  # Log metadata only (not full conversations)
    )
    async def chat_with_claude(messages, model="claude-sonnet-4-5-20250929"):
        """Your existing Anthropic chat function."""
        return await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=messages,
        )

    response = await chat_with_claude(
        messages=[{"role": "user", "content": "What is the capital of Germany?"}]
    )

    print(f"Anthropic response: {response.content[0].text}")
    print("✓ Logged to llmring-server")


# -----------------------------------------------------------------------------
# Example 3: Auto-detecting provider
# -----------------------------------------------------------------------------


async def example_auto_detection():
    """Example of auto-detecting provider from response."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @log_llm_call(
        server_url="http://localhost:8000",
        provider="auto",  # Auto-detect from response
        log_conversations=True,
    )
    async def generic_chat(messages):
        """Generic chat function - provider auto-detected."""
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

    response = await generic_chat(messages=[{"role": "user", "content": "What is 2+2?"}])

    print(f"Response: {response.choices[0].message.content}")
    print("✓ Provider auto-detected and logged")


# -----------------------------------------------------------------------------
# Example 4: Logging streaming responses
# -----------------------------------------------------------------------------


async def example_streaming_logging():
    """Example of logging streaming responses."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @log_llm_stream(
        server_url="http://localhost:8000",
        provider="openai",
        log_metadata=True,  # Log accumulated usage after stream completes
    )
    async def stream_chat(messages, model="gpt-4o-mini"):
        """Stream chat responses."""
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            yield chunk

    print("Streaming response: ", end="", flush=True)
    async for chunk in stream_chat(
        messages=[{"role": "user", "content": "Write a haiku about Python."}]
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n✓ Stream logged to llmring-server")


# -----------------------------------------------------------------------------
# Example 5: Metadata-only logging
# -----------------------------------------------------------------------------


async def example_metadata_only():
    """Example of logging only metadata (no conversation content)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @log_llm_call(
        server_url="http://localhost:8000",
        provider="openai",
        log_metadata=True,  # Log usage metadata
        log_conversations=False,  # Don't log message content
    )
    async def chat_metadata_only(messages, model="gpt-4o-mini"):
        """Log only token usage and cost, not message content."""
        return await client.chat.completions.create(
            model=model,
            messages=messages,
        )

    response = await chat_metadata_only(
        messages=[{"role": "user", "content": "This is a private message."}]
    )

    print(f"Response received (not logged): {response.choices[0].message.content}")
    print("✓ Only metadata logged (tokens, cost, model)")


# -----------------------------------------------------------------------------
# Example 6: Multiple functions with different configs
# -----------------------------------------------------------------------------


async def example_multiple_functions():
    """Example of using different logging configs for different functions."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Production function: full conversation logging
    @log_llm_call(
        server_url="http://localhost:8000",
        provider="openai",
        log_conversations=True,
        alias="production_chat",
    )
    async def production_chat(messages):
        return await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

    # Internal testing: metadata only
    @log_llm_call(
        server_url="http://localhost:8000",
        provider="openai",
        log_metadata=True,
        log_conversations=False,
        alias="testing_chat",
    )
    async def testing_chat(messages):
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

    # Use both functions
    await production_chat(messages=[{"role": "user", "content": "Production query"}])
    await testing_chat(messages=[{"role": "user", "content": "Test query"}])

    print("✓ Both functions logged with different configurations")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


async def main():
    """Run all examples (comment out ones that require API keys you don't have)."""
    print("LLMRing Decorator Logging Examples")
    print("=" * 50)

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠ Set OPENAI_API_KEY to run OpenAI examples")
        return

    print("\n1. OpenAI logging example:")
    await example_openai_logging()

    print("\n2. Auto-detection example:")
    await example_auto_detection()

    print("\n3. Streaming example:")
    await example_streaming_logging()

    print("\n4. Metadata-only example:")
    await example_metadata_only()

    print("\n5. Multiple functions example:")
    await example_multiple_functions()

    # Uncomment if you have Anthropic API key
    # if os.environ.get("ANTHROPIC_API_KEY"):
    #     print("\n2. Anthropic logging example:")
    #     await example_anthropic_logging()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nNote: Check your llmring-server logs/database to see the logged data.")


if __name__ == "__main__":
    asyncio.run(main())
