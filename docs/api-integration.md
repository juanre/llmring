# LLMRing API Integration Documentation

## Overview

The LLMRing API provides a centralized service for managing LLM model definitions, pricing information, and usage tracking. This document details how llmring-py integrates with the API to provide a seamless experience for managing LLM interactions.

## API Architecture

The LLMRing API is built with FastAPI and provides the following core services:

### 1. Model Registry Service
- **Purpose**: Centralized source of truth for LLM model information
- **Features**:
  - Current and historical model definitions
  - Pricing information (dollars per million tokens)
  - Model capabilities (vision, function calling, JSON mode, etc.)
  - Provider information and endpoints
  - Version control with immutable snapshots
  - Changelog tracking for all updates

### 2. Usage Tracking Service
- **Purpose**: Log and analyze LLM usage across applications
- **Features**:
  - Token usage logging (input, output, cached)
  - Automatic cost calculation based on registry pricing
  - Performance metrics (latency tracking)
  - Application context tracking (origin, user ID)
  - Custom metadata support
  - Aggregated statistics and reporting

### 3. Authentication Service
- **Purpose**: Secure access control and user management
- **Features**:
  - API key generation and management
  - Email/password authentication
  - OAuth integration (Google, GitHub)
  - Tiered access levels (free, hobby, pro, team)
  - Rate limiting per tier
  - Password reset functionality

### 4. Receipt Service
- **Purpose**: Cryptographically signed proof of usage
- **Features**:
  - Ed25519 signed receipts
  - Immutable audit trail
  - Billing verification
  - Compliance reporting
  - Dispute resolution support

## API Endpoints

### Public Endpoints (No Authentication Required)

#### GET /
- **Description**: API information and available endpoints
- **Response**: Basic API metadata including version and endpoint links

#### GET /health
- **Description**: Health check for API and dependencies
- **Response**: Status of API, database, and Redis connections

### Registry Endpoints

#### GET /registry/ or /registry.json
- **Description**: Get current model registry
- **Authentication**: Optional (rate limits apply without auth)
- **Query Parameters**:
  - `version`: Specific registry version (YYYY.MM.DD format)
  - `providers`: Comma-separated provider filter
  - `capabilities`: Comma-separated capability filter
- **Response**: Complete registry with models and provider information
- **Caching**: ETag support for efficient caching

#### GET /registry/v/{version}/registry.json
- **Description**: Get specific immutable registry version
- **Authentication**: Optional
- **Response**: Historical registry snapshot
- **Caching**: Immutable resource (cached forever)

#### GET /registry/changelog
- **Description**: Get registry update history
- **Authentication**: Optional
- **Query Parameters**:
  - `since`: Show changes after this version
  - `limit`: Maximum versions to return (1-100)
- **Response**: Grouped changelog entries with detailed changes

### Usage Tracking Endpoints

#### POST /api/v1/log
- **Description**: Log LLM usage for tracking and billing
- **Authentication**: Required (X-API-Key header)
- **Request Body**:
  ```json
  {
    "model": "string",
    "provider": "string",
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_input_tokens": 0,
    "cost": 0.0,  // Optional - calculated if not provided
    "latency_ms": 0,
    "origin": "string",
    "id_at_origin": "string",
    "metadata": {}
  }
  ```
- **Response**: Log ID, calculated cost, and timestamp
- **Note**: Does NOT store prompts or responses, only metrics

#### GET /api/v1/stats
- **Description**: Get usage statistics
- **Authentication**: Required
- **Query Parameters**:
  - `start_date`: Filter start date (YYYY-MM-DD)
  - `end_date`: Filter end date (YYYY-MM-DD)
  - `origin`: Filter by application
  - `model`: Filter by model name
  - `group_by`: Aggregation period (day/week/month)
- **Response**: Aggregated usage statistics with breakdowns

### Authentication Endpoints

#### POST /api/v1/register
- **Description**: Register new account with email/password
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "secure_password",
    "organization": "Optional Org Name"
  }
  ```
- **Response**: API key and tier information

#### POST /api/v1/login
- **Description**: Login with email/password
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "secure_password"
  }
  ```
- **Response**: API key and tier information

#### POST /api/v1/password-reset
- **Description**: Request password reset token
- **Request Body**: Email address

#### POST /api/v1/password-reset/confirm
- **Description**: Reset password with token
- **Request Body**: Token and new password

### OAuth Endpoints

#### GET /api/v1/auth/login/{provider}
- **Description**: Initiate OAuth flow
- **Providers**: google, github
- **Response**: Redirect to OAuth provider

#### GET /api/v1/auth/callback/{provider}
- **Description**: OAuth callback handler
- **Response**: Sets session cookie and redirects

#### GET /api/v1/auth/me
- **Description**: Get current user info
- **Authentication**: Session cookie
- **Response**: User details and API key status

#### GET /api/v1/auth/retrieve-api-key
- **Description**: One-time API key retrieval after OAuth
- **Authentication**: Session cookie
- **Response**: API key (cleared after retrieval)

#### POST /api/v1/auth/regenerate-api-key
- **Description**: Generate new API key
- **Authentication**: Session cookie
- **Response**: New API key

#### POST /api/v1/auth/logout
- **Description**: End session
- **Authentication**: Session cookie

### Receipt Endpoints

#### POST /api/v1/receipts
- **Description**: Store usage receipt
- **Authentication**: Required
- **Request Body**: Receipt with usage details
- **Response**: Receipt ID and signature

#### GET /api/v1/receipts/{receipt_id}
- **Description**: Retrieve stored receipt
- **Authentication**: Required
- **Response**: Complete receipt with signature

## Data Models

### LLMModel
Core model definition matching llmring-py schema:
```python
{
    "provider": "string",
    "model_name": "string",
    "display_name": "string",
    "description": "string",
    "max_context": 0,
    "max_output_tokens": 0,
    "supports_vision": false,
    "supports_function_calling": false,
    "supports_json_mode": false,
    "supports_parallel_tool_calls": false,
    "tool_call_format": "string",
    "dollars_per_million_tokens_input": 0.0,
    "dollars_per_million_tokens_output": 0.0,
    "inactive_from": "datetime",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### RegistryResponse
Complete registry structure:
```python
{
    "version": "YYYY.MM.DD",
    "generated_at": "datetime",
    "models": {
        "model_name": LLMModel
    },
    "providers": {
        "provider_name": {
            "name": "string",
            "base_url": "string",
            "models_endpoint": "string"
        }
    }
}
```

## Authentication & Security

### API Key Authentication
- **Header**: `X-API-Key: llmr_pk_...`
- **Format**: Prefix-based keys for easy identification
- **Storage**: SHA-256 hashed in database
- **Tiers**:
  - Free: 10,000 logs/month, 30-day retention, 60 req/min
  - Hobby: 50,000 logs/month, 90-day retention, 300 req/min
  - Pro: 500,000 logs/month, 365-day retention, 1000 req/min
  - Team: Unlimited logs, unlimited retention, 5000 req/min

### OAuth Integration
- **Providers**: Google, GitHub
- **CSRF Protection**: State parameter validation
- **Session Management**: Secure HTTP-only cookies
- **Token Storage**: One-time retrieval pattern

### Rate Limiting
- **Implementation**: Redis-backed or in-memory fallback
- **Headers**:
  - `X-RateLimit-Limit`: Maximum requests
  - `X-RateLimit-Remaining`: Requests left
  - `X-RateLimit-Reset`: Window reset time
- **Tiers**: Different limits per subscription level

## Integration with llmring-py

### Configuration

Environment variables:
```bash
export LLMRING_API_URL="https://api.llmring.ai"
export LLMRING_API_KEY="llmr_pk_your_key"
```

Or programmatic:
```python
from llmring import LLMRing

ring = LLMRing(
    api_url="https://api.llmring.ai",
    api_key="llmr_pk_your_key"
)
```

### API Service Layer

The llmring-py library includes an API service layer (`llmring.api.service`) that:

1. **Registry Synchronization**:
   - Fetches and caches model registry
   - Supports version pinning for reproducibility
   - Handles filtering by provider/capabilities
   - Implements ETag-based caching

2. **Usage Tracking**:
   - Automatic logging of all LLM interactions
   - Batching for efficiency
   - Retry logic with exponential backoff
   - Offline queue for network failures

3. **Cost Calculation**:
   - Uses registry pricing for accurate costs
   - Handles cached token discounts
   - Supports custom cost overrides

4. **Receipt Management**:
   - Generates cryptographic receipts
   - Stores for audit compliance
   - Retrieves for verification

### Example Integration Flow

```python
from llmring import LLMRing
from llmring.api import APIService

# Initialize with API backend
api_service = APIService(
    base_url="https://api.llmring.ai",
    api_key="llmr_pk_your_key"
)

ring = LLMRing(api_service=api_service)

# Registry operations (cached locally)
models = ring.list_models(provider="openai")
model = ring.get_model("gpt-4o-mini")

# Usage automatically tracked
response = ring.complete(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
# This triggers automatic usage logging to API

# Get usage statistics
stats = api_service.get_usage_stats(
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Generate and store receipt
receipt = api_service.create_receipt(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50
)
```

## Performance Considerations

### Caching Strategy
1. **Registry Caching**:
   - Local cache with TTL (default 1 hour)
   - ETag validation for efficiency
   - Immutable version caching (forever)

2. **Usage Batching**:
   - Queue logs locally
   - Batch send every N logs or T seconds
   - Retry failed batches

3. **Connection Pooling**:
   - Reuse HTTP connections
   - Configure pool size based on load

### Error Handling

```python
from llmring.api.exceptions import (
    APIAuthenticationError,
    APIRateLimitError,
    APINetworkError
)

try:
    response = api_service.log_usage(...)
except APIAuthenticationError:
    # Handle invalid API key
    pass
except APIRateLimitError as e:
    # Wait for reset time
    wait_until = e.reset_time
except APINetworkError:
    # Queue for retry
    pass
```

## Migration from Local Database

### Step 1: Export Existing Data
```python
from llmring import LLMRing

# Using local database
ring = LLMRing(use_local_db=True)

# Export usage history
usage_logs = ring.export_usage_logs(
    start_date="2023-01-01",
    format="json"
)
```

### Step 2: Import to API
```python
from llmring.api import APIService

api_service = APIService(
    base_url="https://api.llmring.ai",
    api_key="llmr_pk_your_key"
)

# Bulk import (may require special endpoint)
api_service.import_usage_logs(usage_logs)
```

### Step 3: Switch Configuration
```python
# Update configuration to use API
ring = LLMRing(
    api_url="https://api.llmring.ai",
    api_key="llmr_pk_your_key",
    use_local_db=False  # Explicitly disable local DB
)
```

## Advanced Features

### Webhook Integration (Future)
```python
# Register webhook for model updates
api_service.register_webhook(
    url="https://your-app.com/webhooks/llmring",
    events=["model.added", "price.changed", "model.deprecated"]
)
```

### Custom Metadata
```python
# Add application-specific metadata
ring.complete(
    model="gpt-4o-mini",
    messages=[...],
    metadata={
        "user_id": "user123",
        "session_id": "sess456",
        "feature": "chat",
        "experiment": "v2"
    }
)
```

### Multi-Tenant Support
```python
# Use origin field for tenant isolation
ring = LLMRing(
    api_service=api_service,
    origin="tenant-123"  # All logs tagged with this origin
)

# Query tenant-specific stats
stats = api_service.get_usage_stats(
    origin="tenant-123"
)
```

## Security Best Practices

1. **API Key Management**:
   - Never commit API keys to version control
   - Use environment variables or secrets management
   - Rotate keys regularly
   - Use different keys for dev/staging/production

2. **Network Security**:
   - Always use HTTPS in production
   - Implement certificate pinning for sensitive apps
   - Use retry with exponential backoff

3. **Data Privacy**:
   - API never stores prompts or responses
   - Only usage metrics are logged
   - All data encrypted in transit
   - Receipts provide cryptographic proof

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify API key format (starts with `llmr_pk_`)
   - Check key hasn't been revoked
   - Ensure correct header name (`X-API-Key`)

2. **Rate Limiting**:
   - Check tier limits
   - Implement backoff strategy
   - Consider upgrading tier

3. **Network Issues**:
   - Verify API URL is correct
   - Check firewall/proxy settings
   - Enable debug logging

### Debug Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# API service will log all requests/responses
api_service = APIService(
    base_url="https://api.llmring.ai",
    api_key="llmr_pk_your_key",
    debug=True
)
```

## API Versioning

The API follows semantic versioning:
- **v1**: Current stable version
- **Registry versions**: YYYY.MM.DD format
- **Breaking changes**: New major version with migration path

## Support & Resources

- **API Documentation**: https://api.llmring.ai/docs
- **OpenAPI Schema**: https://api.llmring.ai/openapi.json
- **GitHub Issues**: https://github.com/llmring/llmring-api/issues
- **Email Support**: support@llmring.ai

## Appendix: Complete API Flow Example

```python
import asyncio
from llmring import LLMRing
from llmring.api import APIService

async def main():
    # 1. Initialize API service
    api = APIService(
        base_url="https://api.llmring.ai",
        api_key="llmr_pk_your_key"
    )
    
    # 2. Fetch and cache registry
    registry = await api.get_registry()
    print(f"Registry version: {registry.version}")
    
    # 3. Initialize LLMRing with API backend
    ring = LLMRing(api_service=api)
    
    # 4. List available models with filters
    vision_models = ring.list_models(
        capabilities=["vision"],
        providers=["openai", "anthropic"]
    )
    
    # 5. Get specific model info
    model = ring.get_model("gpt-4o-mini")
    print(f"Model: {model.display_name}")
    print(f"Max context: {model.max_context}")
    print(f"Input cost: ${model.dollars_per_million_tokens_input}/M")
    
    # 6. Make completion (usage auto-tracked)
    response = await ring.complete_async(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        metadata={
            "user_id": "user123",
            "session": "sess456"
        }
    )
    
    # 7. Get usage statistics
    stats = await api.get_usage_stats(
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    print(f"Total cost: ${stats.total_cost}")
    print(f"Total requests: {stats.total_requests}")
    
    # 8. Generate receipt for audit
    receipt = await api.create_receipt(
        model="gpt-4o-mini",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=response.usage.cost
    )
    print(f"Receipt ID: {receipt.id}")
    print(f"Signature: {receipt.signature}")

if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive documentation provides everything needed to understand and integrate with the LLMRing API from llmring-py.