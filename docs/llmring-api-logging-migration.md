# LLMRing-API Migration Guide: Logging Refactoring

**Date**: October 2025
**Target**: `llmring-api` developers
**Status**: Required for compatibility with llmring v2.x

---

## Executive Summary

The `llmring` client library has undergone a major logging refactoring that introduces a **new dual-mode logging system** and **on-demand receipt generation**. This document explains the changes and how `llmring-api` needs to adapt.

### What Changed

1. **New Logging Architecture**: `LLMRing` now has a `LoggingService` that handles two modes: metadata-only and full conversation logging
2. **On-Demand Receipts**: Receipts are no longer generated automatically; they're created on-demand via explicit API calls
3. **Unified Server Communication**: Both `LLMRing` and `LLMRingSession` now use `server_url` + `api_key` for all server operations
4. **Bug Fixes**: Fixed critical bugs in session double-logging and pinned version leaking

### Impact on llmring-api

- ✅ **Low Impact**: Most changes are client-side
- ⚠️ **Medium Impact**: Receipt generation endpoints need updating
- ⚠️ **Medium Impact**: Usage logging schema should be reviewed
- ✅ **No Breaking Changes**: Existing APIs remain compatible

---

## Architecture Changes

### 1. Old Architecture (Pre-Refactor)

```
LLMRing (stateless)
  └─> No conversation tracking
  └─> No usage logging

LLMRingExtended (stateful)
  └─> ServerClient for conversations
  └─> Sent messages to /api/v1/conversations/{id}/messages/batch
  └─> Did NOT log usage to server
  └─> ReceiptManager for automatic receipt generation
```

### 2. New Architecture (Post-Refactor)

```
LLMRing (stateless with optional logging)
  ├─> LoggingService (new)
  │   ├─> Metadata logging: logs to /api/v1/log
  │   └─> Conversation logging: logs to /api/v1/log with conversation_id
  ├─> ServerClient (new)
  │   └─> Used when server_url + api_key provided
  └─> No automatic receipts (removed ReceiptManager)

LLMRingSession (extends LLMRing)
  ├─> Inherits LoggingService from parent
  ├─> Inherits ServerClient from parent
  ├─> Creates conversations on server
  ├─> Stores messages to server
  └─> Links usage logs to conversation_id
```

### 3. Key Classes Added

#### `LoggingService` (src/llmring/services/logging_service.py)

Handles all usage logging to llmring-server or llmring-api.

**Methods**:
- `log_request_response()` - Main logging entry point
- `set_conversation_id()` - Set conversation context (scoped)
- `clear_conversation_id()` - Clear conversation context

**Request Body** to `POST /api/v1/log`:
```json
{
  "model": "gpt-4o-mini",
  "provider": "openai",
  "input_tokens": 150,
  "output_tokens": 75,
  "total_tokens": 225,
  "cost": 0.00034,
  "origin": "llmring",
  "conversation_id": "uuid-here",  // Optional, if linked to conversation
  "metadata": {
    "alias": "default",
    "profile": "production",
    "finish_reason": "stop",
    "model_requested": "openai:gpt-4o-mini",
    "temperature": 0.7
  }
}
```

**Logging Modes**:
1. **Metadata-only** (`log_metadata=True`): Logs usage stats without message content
2. **Full conversation** (`log_conversations=True`): Logs stats + message content

#### `ServerClient` Enhancements (src/llmring/server_client.py)

**New Methods**:
- `generate_receipt()` - Create receipt for usage logs
- `preview_receipt()` - Preview what would be certified
- `list_receipts()` - List existing receipts

**Receipt Generation Request** to `POST /api/v1/receipts/generate`:
```json
{
  "conversation_id": "uuid-here",        // Option 1: Single conversation
  "since_last_receipt": true,            // Option 2: All uncertified logs
  "start_date": "2025-10-01",           // Option 3: Date range
  "end_date": "2025-10-31",
  "description": "Monthly certification",
  "tags": ["production", "billing"]
}
```

**Receipt Response**:
```json
{
  "receipt": {
    "receipt_id": "rcpt_xxx",
    "receipt_type": "single" | "batch",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "total_cost": 12.45,
    "total_tokens": 150000,
    "start_timestamp": "2025-10-01T00:00:00Z",
    "end_timestamp": "2025-10-31T23:59:59Z",
    "signature": "ed25519:xxxxx",
    "batch_summary": {  // Only for batch receipts
      "total_calls": 450,
      "total_tokens": 150000,
      "providers": {"openai": 300, "anthropic": 150},
      "models": {"gpt-4o-mini": 250, "claude-3.5-sonnet": 200}
    }
  },
  "certified_count": 450,
  "certified_logs": ["log_id_1", "log_id_2", ...]
}
```

---

## Database Schema Requirements

### Current Schema (llmring-api)

Based on `receipts_export.py`, llmring-api expects this schema:

```sql
-- Current receipts table
CREATE TABLE receipts (
    receipt_id TEXT PRIMARY KEY,
    registry_version TEXT,
    model TEXT,
    alias TEXT,
    profile TEXT,
    lock_digest TEXT,
    key_id TEXT,              -- Maps to api_key_id/project_id
    tokens JSONB,             -- {"prompt": 100, "completion": 50}
    cost NUMERIC,
    signature TEXT,
    receipt_timestamp TIMESTAMP,
    stored_at TIMESTAMP
);
```

### Required Schema Updates

The new logging system expects:

```sql
-- 1. Usage logs table (NEW - required for on-demand receipts)
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id TEXT NOT NULL,           -- Project ID in llmring-api
    conversation_id TEXT,                -- Optional: links to conversation

    -- LLM request info
    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Token usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER DEFAULT 0,

    -- Cost tracking
    cost NUMERIC,

    -- Receipt linking
    receipt_id TEXT,                    -- NULL until certified
    certified_at TIMESTAMP,             -- NULL until certified

    -- Request metadata
    origin TEXT DEFAULT 'llmring',
    metadata JSONB,                     -- alias, profile, finish_reason, etc.

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_logs_api_key (api_key_id),
    INDEX idx_logs_conversation (conversation_id),
    INDEX idx_logs_uncertified (api_key_id, receipt_id) WHERE receipt_id IS NULL,
    INDEX idx_logs_receipt (receipt_id) WHERE receipt_id IS NOT NULL
);

-- 2. Updated receipts table (MODIFIED)
CREATE TABLE receipts (
    receipt_id TEXT PRIMARY KEY,
    receipt_type TEXT NOT NULL,         -- NEW: 'single' or 'batch'
    api_key_id TEXT NOT NULL,          -- Maps to project_id

    -- Provider/model info
    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Cost/token aggregates
    total_cost NUMERIC NOT NULL,
    total_tokens INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,      -- NEW
    output_tokens INTEGER NOT NULL,     -- NEW

    -- Time range
    start_timestamp TIMESTAMP NOT NULL, -- NEW
    end_timestamp TIMESTAMP NOT NULL,   -- NEW

    -- Batch info (for batch receipts)
    batch_summary JSONB,               -- NEW: {total_calls, providers, models, ...}
    certified_log_count INTEGER,       -- NEW: number of logs certified

    -- Cryptographic proof
    signature TEXT NOT NULL,
    algorithm TEXT DEFAULT 'Ed25519',   -- NEW

    -- Optional metadata
    description TEXT,                   -- NEW: user-provided description
    tags TEXT[],                       -- NEW: user-provided tags
    metadata JSONB,                    -- NEW: additional context

    -- Legacy fields (kept for backward compatibility)
    registry_version TEXT,
    alias TEXT,
    profile TEXT,
    lock_digest TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_receipts_api_key (api_key_id),
    INDEX idx_receipts_created (created_at),
    INDEX idx_receipts_type (receipt_type)
);
```

---

## Required API Changes

### 1. New Endpoint: POST /api/v1/log

**Purpose**: Receive usage logs from LLMRing clients

**Request Body**:
```json
{
  "model": "gpt-4o-mini",
  "provider": "openai",
  "input_tokens": 150,
  "output_tokens": 75,
  "total_tokens": 225,
  "cached_input_tokens": 0,
  "cost": 0.00034,
  "origin": "llmring",
  "conversation_id": "uuid-here",
  "metadata": {
    "alias": "default",
    "profile": "production",
    "finish_reason": "stop",
    "model_requested": "openai:gpt-4o-mini",
    "temperature": 0.7
  }
}
```

**Response**:
```json
{
  "id": "uuid-of-log",
  "status": "logged",
  "message": "Usage logged successfully"
}
```

**Implementation**:
```python
# backend/src/llmring_api/routers/usage_logs.py
from fastapi import APIRouter, Depends, HTTPException, Request
from pgdbm import AsyncDatabaseManager
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter(prefix="/api/v1", tags=["usage"])


class UsageLogCreate(BaseModel):
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    cost: Optional[float] = None
    origin: str = "llmring"
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/log")
async def log_usage(
    body: UsageLogCreate,
    request: Request,
    db: AsyncDatabaseManager = Depends(get_server_db),
):
    """Log LLM usage from client."""
    # Get project ID from auth middleware
    api_key_id = request.headers.get("X-Project-Key")

    # Insert usage log
    log_id = await db.fetch_val(
        """
        INSERT INTO {{tables.usage_logs}}
        (api_key_id, conversation_id, provider, model,
         input_tokens, output_tokens, total_tokens,
         cached_input_tokens, cost, origin, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
        """,
        api_key_id,
        body.conversation_id,
        body.provider,
        body.model,
        body.input_tokens,
        body.output_tokens,
        body.total_tokens,
        body.cached_input_tokens,
        body.cost,
        body.origin,
        body.metadata or {},
    )

    return {
        "id": str(log_id),
        "status": "logged",
        "message": "Usage logged successfully"
    }
```

### 2. New Endpoint: POST /api/v1/receipts/generate

**Purpose**: Generate receipts on-demand from usage logs

**Request Body**:
```json
{
  "conversation_id": "uuid-here",        // Option 1
  "since_last_receipt": true,            // Option 2
  "start_date": "2025-10-01",           // Option 3
  "end_date": "2025-10-31",
  "description": "Monthly certification",
  "tags": ["production", "billing"]
}
```

**Response**: See "Receipt Response" in ServerClient section above.

**Implementation**:
```python
# backend/src/llmring_api/routers/receipts.py
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

class ReceiptGenerateRequest(BaseModel):
    conversation_id: Optional[str] = None
    since_last_receipt: bool = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@router.post("/receipts/generate")
async def generate_receipt(
    body: ReceiptGenerateRequest,
    request: Request,
    db: AsyncDatabaseManager = Depends(get_server_db),
):
    """Generate receipt from usage logs."""
    api_key_id = request.headers.get("X-Project-Key")

    # 1. Find uncertified logs matching criteria
    where_clauses = ["api_key_id = $1", "receipt_id IS NULL"]
    params = [api_key_id]

    if body.conversation_id:
        where_clauses.append(f"conversation_id = ${len(params) + 1}")
        params.append(body.conversation_id)
    elif body.since_last_receipt:
        # Already filtered by receipt_id IS NULL
        pass
    elif body.start_date and body.end_date:
        where_clauses.append(f"created_at >= ${len(params) + 1}")
        params.append(body.start_date)
        where_clauses.append(f"created_at <= ${len(params) + 1}")
        params.append(body.end_date)

    # Get logs
    logs = await db.fetch_all(
        f"""
        SELECT * FROM {{tables.usage_logs}}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY created_at ASC
        """,
        *params
    )

    if not logs:
        raise HTTPException(404, "No uncertified logs found")

    # 2. Calculate aggregates
    total_cost = sum(log["cost"] or 0 for log in logs)
    total_tokens = sum(log["total_tokens"] for log in logs)
    input_tokens = sum(log["input_tokens"] for log in logs)
    output_tokens = sum(log["output_tokens"] for log in logs)

    # 3. Determine receipt type
    receipt_type = "single" if len(logs) == 1 else "batch"

    # 4. Build batch summary (if batch)
    batch_summary = None
    if receipt_type == "batch":
        providers = {}
        models = {}
        for log in logs:
            providers[log["provider"]] = providers.get(log["provider"], 0) + 1
            models[log["model"]] = models.get(log["model"], 0) + 1

        batch_summary = {
            "total_calls": len(logs),
            "total_tokens": total_tokens,
            "providers": providers,
            "models": models,
        }

    # 5. Generate receipt ID
    receipt_id = f"rcpt_{uuid4().hex[:16]}"

    # 6. Create signature (simplified - use proper Ed25519)
    import hashlib
    receipt_data = f"{receipt_id}|{api_key_id}|{total_cost}|{total_tokens}"
    signature = f"ed25519:{hashlib.sha256(receipt_data.encode()).hexdigest()}"

    # 7. Insert receipt
    await db.execute(
        """
        INSERT INTO {{tables.receipts}}
        (receipt_id, receipt_type, api_key_id, provider, model,
         total_cost, total_tokens, input_tokens, output_tokens,
         start_timestamp, end_timestamp, batch_summary,
         certified_log_count, signature, description, tags)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """,
        receipt_id,
        receipt_type,
        api_key_id,
        logs[0]["provider"],  # Primary provider
        logs[0]["model"],     # Primary model
        total_cost,
        total_tokens,
        input_tokens,
        output_tokens,
        logs[0]["created_at"],
        logs[-1]["created_at"],
        batch_summary,
        len(logs),
        signature,
        body.description,
        body.tags or [],
    )

    # 8. Mark logs as certified
    log_ids = [log["id"] for log in logs]
    await db.execute(
        """
        UPDATE {{tables.usage_logs}}
        SET receipt_id = $1, certified_at = CURRENT_TIMESTAMP
        WHERE id = ANY($2::uuid[])
        """,
        receipt_id,
        log_ids,
    )

    # 9. Return receipt
    return {
        "receipt": {
            "receipt_id": receipt_id,
            "receipt_type": receipt_type,
            "provider": logs[0]["provider"],
            "model": logs[0]["model"],
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "start_timestamp": logs[0]["created_at"],
            "end_timestamp": logs[-1]["created_at"],
            "signature": signature,
            "batch_summary": batch_summary,
        },
        "certified_count": len(logs),
        "certified_logs": log_ids,
    }
```

### 3. New Endpoint: POST /api/v1/receipts/preview

**Purpose**: Preview what would be certified without generating a receipt

**Request Body**: Same as generate

**Response**:
```json
{
  "total_logs": 450,
  "total_cost": 12.45,
  "total_tokens": 150000,
  "receipt_type": "batch",
  "date_range": {
    "start": "2025-10-01T00:00:00Z",
    "end": "2025-10-31T23:59:59Z"
  },
  "breakdown": {
    "by_provider": {"openai": 300, "anthropic": 150},
    "by_model": {"gpt-4o-mini": 250, "claude-3.5-sonnet": 200}
  }
}
```

### 4. Updated Endpoint: GET /api/v1/receipts

**Changes**:
- Add filtering by `receipt_type`
- Add filtering by date range
- Add filtering by tags
- Return new receipt format

---

## Critical Bug Fixes (Already Completed in llmring)

### Bug 1: Session Double-Logging

**Problem**: `LLMRingSession.chat_with_conversation()` logged usage twice:
1. Once in `super().chat()` via LoggingService
2. Again in `chat_with_conversation()` with empty messages

**Fix** (src/llmring/service_extended.py:106-136):
```python
async def chat_with_conversation(
    self,
    request: LLMRequest,
    conversation_id: Optional[UUID] = None,
    ...
) -> LLMResponse:
    # Set conversation_id BEFORE calling parent to avoid duplicate logging
    if conversation_id and self.logging_service:
        self.logging_service.set_conversation_id(str(conversation_id))

    try:
        # Call parent chat method (this will log usage with conversation_id)
        response = await super().chat(request, profile)
    finally:
        # Clear conversation_id after the call
        if conversation_id and self.logging_service:
            self.logging_service.clear_conversation_id()

    # Store messages if configured...
    return response
```

**Impact on llmring-api**: ✅ None - client-side fix

### Bug 2: Pinned Version Leaking

**Problem**: When profile pins registry version, code set `provider._registry_client._pinned_version` but never cleared it. Subsequent calls without profile continued using old pinned version.

**Fix** (src/llmring/service.py:307-650):
```python
def _scoped_pinned_version(self, provider, provider_type, profile):
    """Set and save pinned registry version for scoped restoration."""
    # Save previous value, set new value
    # Return (had_previous, previous_value) for cleanup
    ...

def _restore_pinned_version(self, provider, had_previous, previous_value):
    """Restore previous pinned version state."""
    if had_previous:
        provider._registry_client._pinned_version = previous_value
    else:
        delattr(provider._registry_client, "_pinned_version")

async def chat(self, request, profile=None):
    provider = self.get_provider(provider_type)

    # Set pinned registry version (scoped to this request)
    pinned_state = self._scoped_pinned_version(provider, provider_type, profile)

    try:
        # ... make LLM call ...
        return response
    finally:
        # Restore pinned version state
        self._restore_pinned_version(provider, *pinned_state)
```

**Impact on llmring-api**: ✅ None - client-side fix

---

## Migration Checklist for llmring-api

### Phase 1: Database Schema (Required)

- [ ] Add `usage_logs` table
- [ ] Update `receipts` table with new columns
- [ ] Add indexes for performance
- [ ] Run migration on staging environment
- [ ] Test backward compatibility with old receipt queries

### Phase 2: API Endpoints (Required)

- [ ] Implement `POST /api/v1/log` endpoint
- [ ] Implement `POST /api/v1/receipts/generate` endpoint
- [ ] Implement `POST /api/v1/receipts/preview` endpoint
- [ ] Implement `POST /api/v1/receipts/verify` endpoint (public, no auth)
- [ ] Update `GET /api/v1/receipts` with new filters
- [ ] Update `GET /api/v1/receipts/export` to work with new schema

### Phase 3: Services (Recommended)

- [ ] Create `UsageLoggingService` for handling POST /api/v1/log
- [ ] Create `ReceiptGenerationService` for on-demand receipts
- [ ] Update `ConversationService` to link usage logs to conversations
- [ ] Add cryptographic signing for receipts (Ed25519)

### Phase 4: Testing (Critical)

- [ ] Test usage logging from LLMRing client
- [ ] Test receipt generation (single, batch, by date)
- [ ] Test receipt verification
- [ ] Test conversation linking
- [ ] Load test with 1000+ usage logs
- [ ] Test backward compatibility with old receipt format

### Phase 5: Documentation (Required)

- [ ] Update API documentation with new endpoints
- [ ] Document receipt generation workflow
- [ ] Provide examples for common use cases
- [ ] Migration guide for existing users

---

## Testing the Integration

### 1. Test Usage Logging

```python
from llmring import LLMRing
from llmring.schemas import LLMRequest, Message

async def test_usage_logging():
    """Test that LLMRing logs usage to llmring-api."""
    ring = LLMRing(
        server_url="https://api.example.com",
        api_key="your-project-api-key",
        log_metadata=True,  # Enable metadata logging
    )

    request = LLMRequest(
        model="openai:gpt-4o-mini",
        messages=[Message(role="user", content="Hello!")],
    )

    response = await ring.chat(request)

    # Verify log was sent to POST /api/v1/log
    # Check database: SELECT * FROM usage_logs WHERE api_key_id = 'your-project-api-key'
    assert response.usage is not None
```

### 2. Test Receipt Generation

```python
async def test_receipt_generation():
    """Test generating a receipt for usage logs."""
    ring = LLMRing(
        server_url="https://api.example.com",
        api_key="your-project-api-key",
        log_metadata=True,
    )

    # Make some LLM calls
    for i in range(5):
        await ring.chat(LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content=f"Test {i}")],
        ))

    # Generate receipt
    result = await ring.server_client.generate_receipt(
        since_last_receipt=True,
        description="Test batch"
    )

    receipt = result["receipt"]
    assert receipt["receipt_type"] == "batch"
    assert result["certified_count"] == 5
    assert receipt["signature"].startswith("ed25519:")
```

### 3. Test Conversation Linking

```python
from llmring import LLMRingSession

async def test_conversation_linking():
    """Test that usage logs are linked to conversations."""
    ring = LLMRingSession(
        server_url="https://api.example.com",
        api_key="your-project-api-key",
        enable_conversations=True,
    )

    # Create conversation
    conv_id = await ring.create_conversation(title="Test")

    # Send message (this will log usage with conversation_id)
    response = await ring.chat_with_conversation(
        request=LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Hello")],
        ),
        conversation_id=conv_id,
    )

    # Verify usage log has conversation_id
    # Check: SELECT * FROM usage_logs WHERE conversation_id = conv_id
```

---

## Common Pitfalls

### 1. Missing API Key Header

**Problem**: POST /api/v1/log fails with 401

**Solution**: Ensure auth middleware sets `X-Project-Key` header:
```python
# backend/src/llmring_api/middleware/auth.py
request.headers["X-Project-Key"] = project_id
```

### 2. Receipt ID Collisions

**Problem**: Duplicate receipt IDs when generating receipts concurrently

**Solution**: Use database sequence or UUID:
```python
receipt_id = f"rcpt_{uuid4().hex[:16]}"
```

### 3. Orphaned Usage Logs

**Problem**: Usage logs without receipts pile up

**Solution**: Implement cleanup job or enforce receipt generation:
```sql
-- Find old uncertified logs
SELECT COUNT(*) FROM usage_logs
WHERE receipt_id IS NULL
  AND created_at < NOW() - INTERVAL '30 days';
```

### 4. Cost Calculation Inconsistencies

**Problem**: Cost in usage_logs doesn't match receipt total

**Solution**: Store cost at log time, don't recalculate:
```python
# LLMRing sends calculated cost
log_data["cost"] = cost_info.total_cost

# llmring-api stores it as-is
# Receipt aggregates logged costs
total_cost = sum(log["cost"] or 0 for log in logs)
```

---

## Backward Compatibility

### Supporting Old Clients

If you have old clients still using automatic receipts:

1. **Detect client version** from User-Agent or custom header
2. **Auto-generate receipts** for old clients:

```python
@router.post("/log")
async def log_usage(body: UsageLogCreate, request: Request):
    # Log usage as normal
    log_id = await db.execute(...)

    # Check if old client (e.g., llmring < 2.0)
    if is_legacy_client(request):
        # Auto-generate receipt for this log
        receipt = await auto_generate_receipt(log_id)
        return {"id": log_id, "receipt": receipt}

    return {"id": log_id}
```

### Migration Period

1. **Support both systems** for 3-6 months
2. **Log warnings** for old clients
3. **Sunset date** for automatic receipts
4. **Final migration** to on-demand only

---

## Performance Considerations

### 1. Indexing

Critical indexes for performance:

```sql
-- Fast lookup of uncertified logs
CREATE INDEX idx_logs_uncertified
ON usage_logs (api_key_id, created_at)
WHERE receipt_id IS NULL;

-- Fast receipt lookup
CREATE INDEX idx_logs_receipt
ON usage_logs (receipt_id)
WHERE receipt_id IS NOT NULL;

-- Conversation linking
CREATE INDEX idx_logs_conversation
ON usage_logs (conversation_id, created_at);
```

### 2. Batch Receipt Generation

For large datasets (>10,000 logs), use pagination:

```python
async def generate_large_receipt(api_key_id, start_date, end_date):
    """Generate receipt for large number of logs."""
    batch_size = 1000
    offset = 0
    all_logs = []

    while True:
        logs = await db.fetch_all(
            """
            SELECT * FROM usage_logs
            WHERE api_key_id = $1
              AND receipt_id IS NULL
              AND created_at BETWEEN $2 AND $3
            ORDER BY created_at
            LIMIT $4 OFFSET $5
            """,
            api_key_id, start_date, end_date, batch_size, offset
        )

        if not logs:
            break

        all_logs.extend(logs)
        offset += batch_size

    # Generate receipt from all_logs
    ...
```

### 3. Caching

Cache receipt data for export endpoints:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_receipt_summary(api_key_id, month):
    """Cached receipt summary for dashboard."""
    ...
```

---

## Security Considerations

### 1. Receipt Verification

Implement proper Ed25519 signing:

```python
import nacl.signing
import nacl.encoding

# Server side (at startup)
signing_key = nacl.signing.SigningKey.generate()
verify_key = signing_key.verify_key

async def sign_receipt(receipt_data: dict) -> str:
    """Sign receipt with Ed25519."""
    message = json.dumps(receipt_data, sort_keys=True).encode()
    signed = signing_key.sign(message)
    return f"ed25519:{signed.signature.hex()}"

async def verify_receipt(receipt_data: dict, signature: str) -> bool:
    """Verify receipt signature."""
    if not signature.startswith("ed25519:"):
        return False

    sig_hex = signature.split(":", 1)[1]
    sig_bytes = bytes.fromhex(sig_hex)

    message = json.dumps(receipt_data, sort_keys=True).encode()

    try:
        verify_key.verify(message, sig_bytes)
        return True
    except nacl.exceptions.BadSignatureError:
        return False
```

### 2. Access Control

Ensure users can only:
- Log usage to their own projects
- Generate receipts for their own logs
- View their own receipts

```python
@router.post("/receipts/generate")
async def generate_receipt(
    body: ReceiptGenerateRequest,
    request: Request,
):
    # Get project from auth
    api_key_id = request.headers.get("X-Project-Key")

    # Ensure all logs belong to this project
    logs = await db.fetch_all(
        "SELECT * FROM usage_logs WHERE api_key_id = $1 AND ...",
        api_key_id  # Only this project's logs
    )
```

---

## Support and Questions

For questions about this migration:

1. Check the llmring documentation: `docs/receipts.md`
2. Review the migration guide: `docs/migration-to-on-demand-receipts.md`
3. See examples in `examples/decorator_logging_example.py`
4. Contact the llmring team

---

## Appendix A: Complete File Changes

### Files Modified in llmring

| File | Lines Changed | Description |
|------|--------------|-------------|
| `src/llmring/service.py` | +467/-272 | Added LoggingService integration, server_client initialization, pinned version scoping |
| `src/llmring/service_extended.py` | +81/-50 | Fixed double-logging, inheritance cleanup |
| `src/llmring/services/logging_service.py` | +256/+0 | New logging service with dual-mode logging |
| `src/llmring/server_client.py` | +251/-20 | Added receipt generation, preview, verification |
| `src/llmring/receipts.py` | -264/+50 | Removed automatic receipt generation |
| `tests/test_usage_logging.py` | +287/+0 | New tests for usage logging |
| `tests/integration/test_logging_e2e.py` | +329/+0 | E2E tests for logging workflow |

### Files to Create in llmring-api

1. `backend/src/llmring_api/routers/usage_logs.py` - POST /api/v1/log
2. `backend/src/llmring_api/routers/receipts.py` - Receipt generation endpoints
3. `backend/src/llmring_api/services/usage_logging.py` - Usage logging service
4. `backend/src/llmring_api/services/receipt_generation.py` - Receipt generation service
5. `backend/migrations/xxx_add_usage_logs_table.sql` - Database migration
6. `backend/migrations/xxx_update_receipts_table.sql` - Database migration

---

## Appendix B: Example Client Code

See `examples/decorator_logging_example.py` for complete working examples of:

- Function decorator logging
- Class decorator logging
- Conversation tracking
- Receipt generation
- Error handling

---

**End of Document**
