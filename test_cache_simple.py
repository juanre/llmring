#!/usr/bin/env python
"""Test Anthropic caching directly with the SDK."""

import asyncio
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

async def test_direct_caching():
    """Test caching directly with Anthropic SDK."""

    # Try with beta header even though docs say it's GA
    client = AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )

    # Create a system message that's long enough to cache (1024+ tokens)
    system_text = """You are an expert software engineer with deep knowledge of multiple programming languages and frameworks.

Your expertise includes:
- Python: Advanced features, asyncio, type hints, decorators, metaclasses
- JavaScript/TypeScript: ES6+, async/await, promises, React, Vue, Angular
- Go: Goroutines, channels, interfaces, error handling patterns
- Rust: Ownership, borrowing, lifetimes, traits, error handling
- Java: Spring Boot, microservices, JVM optimization, concurrency
- C++: Modern C++, RAII, templates, STL, memory management

You have extensive experience with:
- Web Development: REST APIs, GraphQL, WebSockets, OAuth, JWT
- Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, query optimization
- Cloud Platforms: AWS (EC2, Lambda, S3, RDS), Azure, GCP, Kubernetes, Docker
- DevOps: CI/CD pipelines, GitHub Actions, GitLab CI, Jenkins, monitoring
- Testing: Unit testing, integration testing, TDD, BDD, mocking, test coverage
- Security: OWASP Top 10, secure coding practices, encryption, authentication
- Architecture: Microservices, event-driven, serverless, clean architecture, DDD
- Performance: Profiling, optimization, caching strategies, load balancing

When answering questions:
1. Provide clear, concise, and accurate responses
2. Include code examples when appropriate
3. Explain trade-offs between different approaches
4. Consider security and performance implications
5. Follow best practices for the specific technology

You write clean, maintainable, and efficient code. You always consider edge cases and error handling. You provide practical solutions that can be implemented in real-world scenarios.

Additional context for thorough understanding:
- You understand the importance of code readability and maintainability
- You follow SOLID principles and design patterns where appropriate
- You consider scalability from the beginning of any design
- You understand the trade-offs between different architectural approaches
- You can explain complex concepts in simple terms
- You provide examples that are practical and immediately useful""" * 3  # Triple it to ensure we exceed 1024 tokens

    print("Testing Anthropic Direct Caching")
    print("=" * 50)

    # First request with cache control
    print("\n1. First request (creating cache)...")
    response1 = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ],
        max_tokens=50
    )

    print(f"Response: {response1.content[0].text[:100]}...")
    print(f"Input tokens: {response1.usage.input_tokens} (need 1024+ for Sonnet)")
    print(f"Cache creation tokens: {getattr(response1.usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read tokens: {getattr(response1.usage, 'cache_read_input_tokens', 0)}")
    if hasattr(response1.usage, 'cache_creation'):
        print(f"Cache creation detail: {response1.usage.cache_creation}")

    # Second request using cache
    print("\n2. Second request (using cache)...")
    response2 = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "What is 5+5?"
            }
        ],
        max_tokens=50
    )

    print(f"Response: {response2.content[0].text[:100]}...")
    print(f"Usage: {response2.usage}")
    print(f"Cache creation tokens: {getattr(response2.usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read tokens: {getattr(response2.usage, 'cache_read_input_tokens', 0)}")
    if hasattr(response2.usage, 'cache_creation'):
        print(f"Cache creation detail: {response2.usage.cache_creation}")

if __name__ == "__main__":
    asyncio.run(test_direct_caching())