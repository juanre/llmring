#!/usr/bin/env python
"""
Example of using Anthropic's prompt caching feature.

Prompt caching can save 90% on costs for repeated context.
This is especially useful for:
- Long system prompts
- Repeated document analysis
- Multi-turn conversations with shared context
"""

import asyncio
import time
from dotenv import load_dotenv
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

load_dotenv()


async def test_prompt_caching():
    """Test Anthropic's prompt caching feature."""
    
    service = LLMRing()
    
    # Create a long system prompt that we want to cache
    # For caching to work, you need at least 1024 tokens
    long_system_prompt = """You are an expert software engineer with deep knowledge of:

1. **Programming Languages**:
   - Python: Advanced features, asyncio, type hints, metaclasses, decorators
   - JavaScript/TypeScript: ES6+, async/await, promises, TypeScript generics
   - Go: Goroutines, channels, interfaces, error handling
   - Rust: Ownership, borrowing, lifetimes, traits

2. **Web Development**:
   - Frontend: React, Vue, Angular, Svelte, WebComponents
   - Backend: Django, FastAPI, Express, Gin, Actix
   - Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
   - APIs: REST, GraphQL, gRPC, WebSockets

3. **Cloud & DevOps**:
   - AWS: EC2, Lambda, S3, RDS, CloudFormation, CDK
   - Azure: Functions, Storage, Cosmos DB, AKS
   - GCP: Compute Engine, Cloud Functions, BigQuery
   - Kubernetes: Deployments, Services, Ingress, Helm
   - CI/CD: GitHub Actions, GitLab CI, Jenkins, CircleCI

4. **Software Architecture**:
   - Design Patterns: Singleton, Factory, Observer, Strategy, Decorator
   - Architectural Patterns: MVC, MVP, MVVM, Clean Architecture, Hexagonal
   - Microservices: Service mesh, API gateway, distributed tracing
   - Event-driven: Message queues, event sourcing, CQRS

5. **Best Practices**:
   - Clean Code principles
   - SOLID principles
   - Test-Driven Development (TDD)
   - Domain-Driven Design (DDD)
   - Security best practices
   - Performance optimization

6. **Data Engineering & Big Data**:
   - Apache Spark, Hadoop, Kafka
   - Data pipelines, ETL/ELT processes
   - Data warehousing, data lakes
   - Stream processing, batch processing
   - Data modeling, normalization, denormalization
   - Apache Airflow, Luigi, Prefect

7. **Machine Learning & AI**:
   - Python ML libraries: scikit-learn, TensorFlow, PyTorch
   - Model training, validation, deployment
   - Feature engineering, data preprocessing
   - MLOps, model monitoring, A/B testing
   - Computer vision, NLP, time series analysis

8. **Mobile Development**:
   - iOS: Swift, SwiftUI, UIKit, Core Data
   - Android: Kotlin, Jetpack Compose, Room
   - React Native, Flutter, Ionic
   - Mobile app architecture patterns
   - Push notifications, deep linking

9. **Security & Compliance**:
   - Authentication: OAuth 2.0, OpenID Connect, SAML
   - Authorization: RBAC, ABAC, ACLs
   - Encryption: TLS, AES, RSA, key management
   - Security scanning, penetration testing
   - GDPR, HIPAA, PCI DSS compliance
   - Security headers, CSP, CORS

10. **Monitoring & Observability**:
    - Metrics: Prometheus, Grafana, DataDog
    - Logging: ELK stack, Splunk, CloudWatch
    - Tracing: Jaeger, Zipkin, X-Ray
    - APM: New Relic, AppDynamics
    - Alerting strategies, SLIs/SLOs/SLAs

You provide detailed, accurate, and practical advice. You always consider:
- Scalability and maintainability
- Security implications
- Performance characteristics
- Best practices for the specific technology stack
- Trade-offs between different approaches
- Cost optimization
- Team collaboration and knowledge sharing
- Documentation and code readability
- Testing strategies and coverage
- Deployment and rollback strategies

When answering questions, you:
1. First understand the context and requirements
2. Provide clear, well-structured responses
3. Include code examples when appropriate
4. Explain the reasoning behind your recommendations
5. Mention potential pitfalls or considerations
6. Consider edge cases and error handling
7. Suggest monitoring and debugging approaches
8. Provide references to official documentation when relevant
9. Recommend appropriate tools and libraries
10. Consider internationalization and localization needs

Remember to be comprehensive in your responses. Provide detailed explanations with examples.
Your knowledge is extensive and you should demonstrate it through thorough, well-reasoned answers.
Always strive for clarity and completeness in your explanations.
"""
    
    print("=" * 60)
    print("Testing Anthropic Prompt Caching")
    print("=" * 60)
    
    # First request with cache control on system message
    print("\n1. First request (will cache the system prompt)...")
    start_time = time.time()
    
    messages = [
        Message(
            role="system",
            content=long_system_prompt,
            metadata={"cache_control": {"type": "ephemeral"}}  # Cache this message
        ),
        Message(
            role="user",
            content="What's the best way to implement a rate limiter in Python?"
        )
    ]
    
    request = LLMRequest(
        model="balanced",  # Use balanced alias for Anthropic Claude Sonnet
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    
    response = await service.chat(request)
    first_time = time.time() - start_time
    
    print(f"Response: {response.content[:200]}...")
    if response.usage:
        print(f"Input tokens: {response.usage.get('prompt_tokens', 0)}")
        print(f"Total tokens: {response.usage.get('total_tokens', 0)}")
        print(f"Cache creation tokens: {response.usage.get('cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {response.usage.get('cache_read_input_tokens', 0)}")
    print(f"Time: {first_time:.2f}s")
    
    # Second request reusing the cached system prompt
    print("\n2. Second request (using cached system prompt)...")
    start_time = time.time()
    
    messages = [
        Message(
            role="system",
            content=long_system_prompt,
            metadata={"cache_control": {"type": "ephemeral"}}  # Reuse cached
        ),
        Message(
            role="user",
            content="How do I handle database migrations in Django?"
        )
    ]
    
    request = LLMRequest(
        model="balanced",  # Use balanced alias for Anthropic Claude Sonnet
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    
    response = await service.chat(request)
    second_time = time.time() - start_time
    
    print(f"Response: {response.content[:200]}...")
    if response.usage:
        print(f"Input tokens: {response.usage.get('prompt_tokens', 0)}")
        print(f"Total tokens: {response.usage.get('total_tokens', 0)}")
        print(f"Cache creation tokens: {response.usage.get('cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {response.usage.get('cache_read_input_tokens', 0)}")
    print(f"Time: {second_time:.2f}s")
    
    # Calculate savings
    print("\n" + "=" * 60)
    print("Cache Performance Summary:")
    print(f"First request time: {first_time:.2f}s")
    print(f"Second request time: {second_time:.2f}s")
    print(f"Time saved: {first_time - second_time:.2f}s ({(1 - second_time/first_time)*100:.1f}% faster)")
    print("\nNote: Cached tokens cost 90% less than regular tokens!")
    print("=" * 60)


async def test_conversation_caching():
    """Test caching in a multi-turn conversation."""
    
    service = LLMRing()
    
    print("\n" + "=" * 60)
    print("Testing Conversation Context Caching")
    print("=" * 60)
    
    # Start a conversation with cacheable context
    messages = [
        Message(
            role="user",
            content="Let me tell you about my project. It's a web application for managing inventory. "
                   "We use Python with FastAPI for the backend, PostgreSQL for the database, "
                   "and React with TypeScript for the frontend. We have about 100,000 products "
                   "and 500 concurrent users during peak hours.",
            metadata={"cache_control": {"type": "ephemeral"}}  # Cache the project context
        ),
        Message(
            role="assistant",
            content="Thank you for sharing the details about your inventory management system. "
                   "With FastAPI, PostgreSQL, React/TypeScript, 100,000 products, and 500 concurrent users, "
                   "I can provide targeted advice for your specific setup."
        ),
        Message(
            role="user",
            content="How should I implement search functionality?"
        )
    ]
    
    request = LLMRequest(
        model="fast",  # Use fast alias for efficient Anthropic model
        messages=messages,
        max_tokens=300
    )
    
    print("\nAsking about search implementation...")
    response = await service.chat(request)
    print(f"Response: {response.content[:300]}...")
    
    # Continue conversation with cached context
    messages.append(Message(role="assistant", content=response.content))
    messages.append(Message(role="user", content="What about caching strategy?"))
    
    request = LLMRequest(
        model="fast",  # Use fast alias for efficient Anthropic model
        messages=messages,
        max_tokens=300
    )
    
    print("\nAsking about caching strategy...")
    response = await service.chat(request)
    print(f"Response: {response.content[:300]}...")
    
    if response.usage:
        print(f"\nCache read tokens in final response: {response.usage.get('cache_read_input_tokens', 0)}")


async def main():
    """Run all caching examples."""
    try:
        # Test basic prompt caching
        await test_prompt_caching()
        
        # Test conversation caching
        await test_conversation_caching()
        
        print("\n" + "=" * 60)
        print("All caching examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())