# Receipts and Cost Tracking

## Overview

LLMRing provides a comprehensive receipt system for tracking LLM API usage and costs. Receipts include detailed information about each API call, including token usage, costs, model selection, and cryptographic signatures for compliance.

**Important:** Receipts are generated **on-demand** via llmring-server, not automatically with every API call. This gives you control over when to generate signed receipts for compliance, billing, or audit purposes.

## Key Features

- **On-Demand Generation**: Generate receipts when you need them (single calls or batches)
- **Automatic Cost Calculation**: Tracks tokens and calculates costs based on registry pricing
- **Cryptographic Signatures**: Ed25519 signatures for receipt verification
- **Canonical JSON**: JCS (JSON Canonicalization Scheme) for deterministic serialization
- **Batch Receipts**: Certify multiple logs in a single receipt for efficiency
- **Lockfile Integration**: Includes SHA256 digest of lockfile for reproducibility
- **Profile Tracking**: Records which profile was used for each request

## Receipt Structure

Each receipt contains:

```python
{
    "receipt_id": "abc123-def456",
    "timestamp": "2024-01-15T10:30:00Z",
    "alias": "fast",
    "profile": "default",
    "lock_digest": "sha256:abc123...",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "prompt_tokens": 150,
    "completion_tokens": 300,
    "total_tokens": 450,
    "input_cost": 0.000023,
    "output_cost": 0.000180,
    "total_cost": 0.000203,
    "signature": "base64_encoded_signature"  # Optional
}
```

## Basic Usage

### Logging with LLMRing

LLMRing automatically logs usage metadata when connected to a server:

```python
from llmring import LLMRing, LLMRequest, Message

# Log usage metadata and conversations to server
service = LLMRing(
    server_url="http://localhost:8000",
    api_key="your_api_key",
    log_metadata=True,          # Log tokens, costs, model usage
    log_conversations=True,     # Log full conversations (implies log_metadata)
)

request = LLMRequest(
    model="fast",
    messages=[Message(role="user", content="Hello!")]
)

response = await service.chat(request)

# Access cost information from response
print(f"Tokens used: {response.usage['total_tokens']}")
print(f"Cost: ${response.usage['cost']:.6f}")

# Logs are stored on server, receipts generated on-demand
```

### Generating Receipts On-Demand

Generate signed receipts when you need them:

```python
from llmring.server_client import ServerClient

# Connect to server
client = ServerClient(
    server_url="http://localhost:8000",
    api_key="your_api_key"
)

# Generate receipt for all logs since last receipt (recommended)
result = await client.generate_receipt(since_last_receipt=True)
receipt = result["receipt"]
certified_count = result["certified_count"]

print(f"✅ Receipt {receipt['receipt_id']} certifies {certified_count} logs")
print(f"Total cost: ${receipt['total_cost']:.6f}")
```

## Cost Calculation

### Automatic Pricing

LLMRing automatically calculates costs based on registry pricing:

```python
async with LLMRing() as service:
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Explain quantum computing")]
    )

    response = await service.chat(request)

    # Detailed cost breakdown
    usage = response.usage
    print(f"Input tokens: {usage['prompt_tokens']} @ ${usage.get('input_cost', 0):.6f}")
    print(f"Output tokens: {usage['completion_tokens']} @ ${usage.get('output_cost', 0):.6f}")
    print(f"Total cost: ${usage['cost']:.6f}")
```

### Cost by Model

Different models have different pricing:

| Provider | Model | Input ($/M tokens) | Output ($/M tokens) |
|----------|-------|-------------------|---------------------|
| OpenAI | gpt-4o | $5.00 | $15.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | claude-sonnet-4-5-20250929 | $3.00 | $15.00 |
| Anthropic | claude-3-5-haiku | $1.00 | $5.00 |
| Google | gemini-2.5-pro | $3.50 | $10.50 |
| Google | gemini-2.5-flash | $0.075 | $0.30 |

*Note: Prices may vary. Use `llmring info <model>` for current pricing.*

## Receipt Export

### CLI Export

```bash
# Export to JSON
llmring export --output receipts.json

# Export to CSV
llmring export --format csv --output receipts.csv
```

### Programmatic Export

```python
import json
from datetime import datetime, UTC
from llmring.server_client import ServerClient

client = ServerClient(base_url="http://localhost:8000", api_key="llmr_pk_...")

page = await client.list_receipts(limit=100)

export_data = {
    "exported_at": datetime.now(UTC).isoformat(),
    "receipts": page.get("receipts", []),
}

with open("receipts.json", "w") as fh:
    json.dump(export_data, fh, indent=2)
```

## On-Demand Receipts (Phase 7.5)

### Overview

**Important**: As of Phase 7.5, receipt generation is **on-demand only**. Receipts are no longer automatically generated with conversation logging. Instead, you explicitly request receipt generation when needed. This gives you full control over:

- When receipts are generated
- What logs are certified (single conversation, date range, specific logs, or all uncertified)
- Batch receipt generation for billing periods
- Receipt descriptions and tags for categorization

### How It Works

When you use LLMRing with a server connection:

1. **Logging happens without receipts**: Conversations and usage are logged to the server
2. **Generate receipts on-demand**: Explicitly request receipts when needed for compliance, billing, or auditing

```python
from llmring import LLMRing, LLMRequest, Message
from datetime import datetime

# Connect to llmring-server with logging enabled
async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True  # Logs conversations (no automatic receipts)
) as service:
    # 1. Use LLM as normal - logs are created but no receipts yet
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Hello!")]
    )
    response = await service.chat(request)

    # 2. Generate a receipt on-demand when needed
    if service.server_client:
        result = await service.server_client.generate_receipt(
            since_last_receipt=True,  # Certify all uncertified logs
            description="Daily certification",
            tags=["daily", "compliance"]
        )
        receipt = result["receipt"]
        print(f"Generated receipt {receipt['receipt_id']}")
        print(f"Certified {result['certified_count']} logs")
```

### Four Generation Modes

Phase 7.5 supports four modes for generating receipts:

#### 1. Single Conversation Receipt

Generate a receipt for one specific conversation:

```python
result = await service.server_client.generate_receipt(
    conversation_id="550e8400-e29b-41d4-a716-446655440000",
    description="Customer support conversation #123"
)
```

**CLI:**
```bash
llmring receipts generate --conversation 550e8400-e29b-41d4-a716-446655440000
```

#### 2. Date Range Batch Receipt

Generate a single batch receipt certifying all logs within a date range:

```python
from datetime import datetime

result = await service.server_client.generate_receipt(
    start_date=datetime(2025, 10, 1),
    end_date=datetime(2025, 10, 31),
    description="October 2025 billing",
    tags=["billing", "monthly"]
)

receipt = result["receipt"]
summary = receipt["batch_summary"]
print(f"Certified {summary['total_calls']} calls")
print(f"Total cost: ${receipt['total_cost']:.4f}")
print(f"By model: {summary['by_model']}")
```

**CLI:**
```bash
llmring receipts generate --start 2025-10-01 --end 2025-10-31 \
  --description "October billing" --tags billing --tags monthly
```

#### 3. Specific Log IDs

Generate a receipt for a specific set of log IDs:

```python
result = await service.server_client.generate_receipt(
    log_ids=[
        "log_id_1",
        "log_id_2",
        "log_id_3"
    ],
    description="Specific logs for audit"
)
```

#### 4. Since Last Receipt

Generate a receipt for all logs that haven't been certified yet:

```python
result = await service.server_client.generate_receipt(
    since_last_receipt=True,
    description="Weekly certification",
    tags=["weekly", "compliance"]
)
```

**CLI:**
```bash
llmring receipts generate --since-last --description "Weekly certification"
```

### Preview Before Generating

You can preview what a receipt would certify without actually generating it:

```python
# Preview what would be certified
preview = await service.server_client.preview_receipt(
    start_date=datetime(2025, 10, 1),
    end_date=datetime(2025, 10, 31)
)

print(f"Would certify {preview['total_logs']} logs")
print(f"Total cost: ${preview['total_cost']:.4f}")
print(f"By model: {preview['by_model']}")
print(f"By alias: {preview['by_alias']}")

# If satisfied, generate the receipt
if preview['total_cost'] < 100:
    result = await service.server_client.generate_receipt(
        start_date=datetime(2025, 10, 1),
        end_date=datetime(2025, 10, 31),
        description="October billing"
    )
```

**CLI:**
```bash
# Preview first
llmring receipts preview --start 2025-10-01 --end 2025-10-31

# Then generate if satisfied
llmring receipts generate --start 2025-10-01 --end 2025-10-31
```

### Working with Uncertified Logs

Check which logs don't have receipts yet:

```python
# Get uncertified logs
uncert = await service.server_client.get_uncertified_logs(limit=10)
print(f"Found {uncert['total']} uncertified logs")

for log in uncert['logs']:
    print(f"  - {log['id']} ({log['type']}): ${log.get('cost', 0):.4f}")
```

**CLI:**
```bash
llmring receipts uncertified --limit 20
```

### Managing Receipts

List all receipts:

```python
# List receipts
receipts = await service.server_client.list_receipts(limit=10)

for receipt in receipts['receipts']:
    print(f"{receipt['receipt_id']} ({receipt['receipt_type']}): ${receipt['total_cost']:.4f}")
```

**CLI:**
```bash
llmring receipts list --limit 20
```

Get a specific receipt:

```python
# Get receipt details
receipt = await service.server_client.get_receipt("rcpt_abc123")
print(f"Receipt type: {receipt['receipt_type']}")
print(f"Total cost: ${receipt['total_cost']:.4f}")

if receipt['receipt_type'] == 'batch':
    summary = receipt['batch_summary']
    print(f"Certified {summary['total_calls']} calls")
```

**CLI:**
```bash
llmring receipts get rcpt_abc123
```

Get logs certified by a receipt:

```python
# Get logs for a receipt
logs = await service.server_client.get_receipt_logs("rcpt_abc123", limit=10)
print(f"Receipt certified {logs['total']} logs")

for log in logs['logs']:
    print(f"  - {log['id']}: ${log.get('total_cost', 0):.4f}")
```

### Single vs Batch Receipts

Phase 7.5 introduces two types of receipts:

#### Single Receipt

Traditional receipt for one conversation/call:

```json
{
    "receipt_id": "rcpt_single_123",
    "receipt_type": "single",
    "timestamp": "2025-10-02T12:00:00",
    "alias": "balanced",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300,
    "input_cost": 0.000015,
    "output_cost": 0.000120,
    "total_cost": 0.000135,
    "signature": "ed25519:ABC123xyz..."
}
```

#### Batch Receipt

Aggregated receipt certifying multiple logs:

```json
{
    "receipt_id": "rcpt_batch_456",
    "receipt_type": "batch",
    "timestamp": "2025-10-31T23:59:59",
    "total_cost": 12.5400,
    "description": "October 2025 billing",
    "tags": ["billing", "monthly"],
    "batch_summary": {
        "total_conversations": 45,
        "total_calls": 127,
        "total_tokens": 1250000,
        "start_date": "2025-10-01T00:00:00",
        "end_date": "2025-10-31T23:59:59",
        "by_model": {
            "gpt-4o-mini": {
                "calls": 85,
                "tokens": 850000,
                "cost": 8.5000
            },
            "claude-3-5-haiku": {
                "calls": 42,
                "tokens": 400000,
                "cost": 4.0400
            }
        },
        "by_alias": {
            "fast": {
                "calls": 85,
                "tokens": 850000,
                "cost": 8.5000
            },
            "balanced": {
                "calls": 42,
                "tokens": 400000,
                "cost": 4.0400
            }
        },
        "conversation_ids": ["conv_1", "conv_2", "..."],
        "log_ids": ["log_1", "log_2", "..."]
    },
    "signature": "ed25519:XYZ789abc..."
}
```

**Benefits of batch receipts:**
- Single signature for entire billing period
- Aggregate statistics for reporting
- Reduced storage overhead
- Easy monthly/weekly certification
- Breakdowns by model and alias

### Receipt Signature

All receipts (single and batch) are signed using **Ed25519** cryptography over **JCS (JSON Canonicalization Scheme)** for deterministic serialization:

- **Algorithm**: Ed25519 elliptic curve signatures
- **Canonicalization**: RFC 8785 JCS for consistent JSON encoding
- **Format**: Signatures are prefixed with `ed25519:` followed by base64url-encoded bytes

### Verifying Receipts

You can verify receipts using the server's public key:

```python
import httpx

# Get the server's public key
async with httpx.AsyncClient() as client:
    # Public key in JSON format
    response = await client.get("http://localhost:8000/api/v1/receipts/public-keys.json")
    keys_data = response.json()

    print(f"Current key ID: {keys_data['current_key_id']}")
    print(f"Public key: {keys_data['keys'][0]['public_key']}")

    # Or get PEM format
    pem_response = await client.get("http://localhost:8000/api/v1/receipts/public-key.pem")
    pem_key = pem_response.text
    print(pem_key)
```

### Verifying Receipt Signatures

Use the server's verification endpoint:

```python
import httpx

async with httpx.AsyncClient() as client:
    # Verify a receipt
    response = await client.post(
        "http://localhost:8000/api/v1/receipts/verify",
        json={
            "receipt_id": "rcpt_a1b2c3d4e5f6",
            "timestamp": "2025-10-02T12:00:00",
            "alias": "balanced",
            "provider": "openai",
            "model": "gpt-4",
            # ... all other receipt fields ...
            "signature": "ed25519:ABC123xyz..."
        }
    )

    result = response.json()
    print(f"Receipt valid: {result['valid']}")
    print(f"Algorithm: {result['algorithm']}")
```

### Offline Verification

For offline verification, use the client's `to_canonical_json()` method:

```python
from llmring.receipts import Receipt

# Load receipt (from server response, database, etc.)
receipt = Receipt(**receipt_data)

# Get canonical JSON (for manual verification)
canonical = receipt.to_canonical_json()
print(canonical)  # Deterministic, sorted JSON

# Calculate digest
digest = receipt.calculate_digest()
print(f"Receipt digest: {digest}")

# For full verification, you'd use the public key and a cryptography library
# But it's recommended to use the server's /receipts/verify endpoint
```

## Server Setup for Receipt Generation

### Configuring Receipt Signing Keys

The llmring-server requires Ed25519 keys to sign receipts. You have two options:

#### Option 1: Environment Variables

```bash
# Generate keys (will be printed to console)
python -c "from llmring_server.config import load_or_generate_keypair; private, public = load_or_generate_keypair(); print(f'LLMRING_RECEIPTS_PRIVATE_KEY_B64={private}'); print(f'LLMRING_RECEIPTS_PUBLIC_KEY_B64={public}')"

# Set environment variables
export LLMRING_RECEIPTS_PRIVATE_KEY_B64="your_private_key_here"
export LLMRING_RECEIPTS_PUBLIC_KEY_B64="your_public_key_here"

# Start the server
python -m llmring_server
```

#### Option 2: Key File

```python
from pathlib import Path
from llmring_server.config import load_or_generate_keypair

# Generate and save keys to file
key_file = Path(".keys/receipt_signing_key")
private_b64, public_b64 = load_or_generate_keypair(key_file)

print(f"Keys saved to {key_file}")
print(f"Private key: {private_b64}")
print(f"Public key: {public_b64}")
```

Then use the keys in your server startup:

```python
from pathlib import Path
from llmring_server.config import Settings, ensure_receipt_keys
from llmring_server.main import create_app

# Load settings and ensure keys
settings = Settings()
settings = ensure_receipt_keys(settings, Path(".keys/receipt_signing_key"))

# Create app with receipt signing enabled
app = create_app(settings=settings)
```

### Retrieving Receipts from the Server

```python
import httpx

async with httpx.AsyncClient() as client:
    # List all receipts (requires authentication)
    response = await client.get(
        "http://localhost:8000/api/v1/receipts/",
        headers={"X-API-Key": "your-api-key"},
        params={"limit": 10, "offset": 0}
    )

    data = response.json()
    print(f"Total receipts: {data['total']}")

    for receipt in data['receipts']:
        print(f"Receipt {receipt['receipt_id']}: ${receipt['total_cost']:.6f}")

    # Get specific receipt
    receipt_id = data['receipts'][0]['receipt_id']
    receipt_response = await client.get(
        f"http://localhost:8000/api/v1/receipts/{receipt_id}",
        headers={"X-API-Key": "your-api-key"}
    )

    receipt = receipt_response.json()
    print(f"Detailed receipt: {receipt}")
```

## Server Integration

### Using LLMRing with Conversation Logging

When using `LLMRing` with server integration and conversation logging enabled (Phase 7.5):

```python
from llmring import LLMRing, LLMRequest, Message
from datetime import datetime, timedelta

async with LLMRing(
    server_url="http://localhost:8000",  # or https://api.llmring.ai
    api_key="your-api-key",
    log_conversations=True  # Logs conversations (no automatic receipts)
) as service:
    # 1. Use LLM - conversations are logged without receipts
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Hello!")]
    )
    response = await service.chat(request)

    # Make more requests throughout the day/week/month...

    # 2. Generate receipts on-demand when needed
    if service.server_client:
        # Weekly certification
        result = await service.server_client.generate_receipt(
            since_last_receipt=True,
            description="Weekly certification",
            tags=["weekly", "compliance"]
        )
        print(f"Certified {result['certified_count']} logs")

        # Or monthly billing
        start = datetime(2025, 10, 1)
        end = datetime(2025, 10, 31)
        result = await service.server_client.generate_receipt(
            start_date=start,
            end_date=end,
            description="October billing",
            tags=["billing", "monthly"]
        )
        receipt = result["receipt"]
        print(f"Monthly cost: ${receipt['total_cost']:.2f}")
```

**Benefits:**
- Centralized conversation and usage logging
- On-demand Ed25519 signed receipts
- Flexible certification schedules (daily, weekly, monthly)
- Batch receipts with aggregate statistics
- Receipts linked to full conversation history
- Analytics and reporting
- Compliance and audit trails
- Team-wide cost tracking
- Public verification endpoints
- Control over when receipts are generated

## Usage Statistics

Usage and receipts live on the server. Inspect them via the CLI or the `ServerClient` helper.

```bash
# Aggregate usage for the current API key
llmring stats

# Export usage and receipts
llmring export --output usage.json
llmring receipts list --limit 20
llmring receipts uncertified --limit 20
```

```python
from llmring.server_client import ServerClient

client = ServerClient(base_url="http://localhost:8000", api_key="llmr_pk_...")

# Preview the next receipt before you generate it
preview = await client.preview_receipt(since_last_receipt=True)
print(
    f"Would certify {preview['total_logs']} logs "
    f"for ${preview['total_cost']:.4f}"
)

# Generate a signed receipt when you're ready
result = await client.generate_receipt(
    since_last_receipt=True,
    description="Weekly certification",
    tags=["weekly", "compliance"],
)
print(f"Issued receipt {result['receipt']['receipt_id']}")

# Inspect uncertified logs
uncertified = await client.get_uncertified_logs(limit=10)
print(f"Remaining uncertified logs: {uncertified['total']}")

# Cost breakdowns returned by cost calculator
for field, amount in result["receipt"]["cost_breakdown"].items():
    print(f"{field}: ${amount:.6f}")
```

## Compliance and Auditing

### Receipt Features for Compliance

Receipts include features for compliance and auditing:

1. **Lockfile Digest**: SHA256 hash of lockfile ensures reproducibility
2. **Profile Tracking**: Records which environment was used
3. **Timestamps**: UTC timestamps for all requests
4. **Unique IDs**: UUID for each receipt
5. **Signatures**: Cryptographic verification of receipt authenticity

### Audit Trail Example

```python
from base64 import urlsafe_b64decode
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey
from llmring.receipts import Receipt

# List of receipts returned by llmring-server (dicts or Receipt objects)
receipts = load_receipts_from_database()

# Public key published by the server (base64url encoded Ed25519 key)
public_key_b64 = load_public_key_b64()
verify_key = VerifyKey(urlsafe_b64decode(public_key_b64 + "=="))

for payload in receipts:
    receipt = Receipt.model_validate(payload)
    signature = receipt.signature
    if not signature:
        print(f"⚠️  Missing signature: {receipt.receipt_id}")
        continue

    try:
        verify_key.verify(
            receipt.to_canonical_json().encode("utf-8"),
            urlsafe_b64decode(signature + "=="),
        )
    except BadSignatureError:
        print(f"⚠️  Invalid signature: {receipt.receipt_id}")
        continue

    print(f"✅ Valid: {receipt.receipt_id}")
    print(f"   Timestamp: {receipt.timestamp}")
    print(f"   Model: {receipt.provider}:{receipt.model}")
    print(f"   Cost: ${receipt.total_cost:.6f}")
```

> Install `pynacl` to use this verification flow: `pip install pynacl`.

### Compliance Reporting

```python
from datetime import datetime, timedelta
import json

def generate_compliance_report(receipts, start_date, end_date):
    """Generate compliance report for date range."""
    filtered = [
        r for r in receipts
        if start_date <= r.timestamp <= end_date
    ]

    report = {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "summary": {
            "total_requests": len(filtered),
            "total_tokens": sum(r.total_tokens for r in filtered),
            "total_cost": sum(r.total_cost for r in filtered),
        },
        "by_model": {},
        "by_profile": {},
        "receipts": [
            {
                "id": r.receipt_id,
                "timestamp": r.timestamp.isoformat(),
                "model": f"{r.provider}:{r.model}",
                "tokens": r.total_tokens,
                "cost": r.total_cost,
                "signature_valid": r.signature is not None
            }
            for r in filtered
        ]
    }

    return report

# Fetch receipts from the server first
page = await client.list_receipts(limit=1000)
receipts = page.get("receipts", [])

# Generate monthly report
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
report = generate_compliance_report(receipts, start_date, end_date)

with open("compliance_report.json", "w") as f:
    json.dump(report, f, indent=2)
```

## Use Cases

### Monthly Billing

Generate a single batch receipt for all usage in a billing period:

```python
from datetime import datetime
from llmring import LLMRing

async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    # Use LLMs throughout the month...

    # At month end, generate billing receipt
    if ring.server_client:
        result = await ring.server_client.generate_receipt(
            start_date=datetime(2025, 10, 1),
            end_date=datetime(2025, 10, 31),
            description="October 2025 - Customer billing",
            tags=["billing", "monthly", "customer-123"]
        )

        receipt = result["receipt"]
        summary = receipt["batch_summary"]

        # Send to billing system
        invoice = {
            "customer_id": "customer-123",
            "period": "October 2025",
            "total": receipt["total_cost"],
            "breakdown": summary["by_model"],
            "receipt_id": receipt["receipt_id"],
            "signature": receipt["signature"]
        }
```

**CLI:**
```bash
# End of month
llmring receipts generate --start 2025-10-01 --end 2025-10-31 \
  --description "October 2025 billing" --tags billing --tags monthly
```

### Daily Compliance Certification

Certify all uncertified logs daily for compliance:

```python
import asyncio
from datetime import datetime
from llmring import LLMRing

async def daily_certification():
    async with LLMRing(
        server_url="http://localhost:8000",
        api_key="your-api-key",
        log_conversations=True
    ) as ring:
        if ring.server_client:
            # Check what needs certification
            uncert = await ring.server_client.get_uncertified_logs()

            if uncert["total"] > 0:
                # Generate daily receipt
                result = await ring.server_client.generate_receipt(
                    since_last_receipt=True,
                    description=f"Daily certification - {datetime.now().date()}",
                    tags=["daily", "compliance"]
                )

                print(f"✅ Certified {result['certified_count']} logs")
                print(f"   Total cost: ${result['receipt']['total_cost']:.4f}")
            else:
                print("✅ No uncertified logs")

# Run daily (e.g., via cron)
asyncio.run(daily_certification())
```

**CLI (via cron):**
```bash
# Run at 11pm daily
0 23 * * * llmring receipts generate --since-last --description "Daily certification"
```

### Audit Specific Conversation

Generate a receipt for one specific conversation for audit purposes:

```python
async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    # Make a critical conversation
    response = await ring.chat(
        "legal-assistant",
        messages=[{"role": "user", "content": "Draft contract clause..."}]
    )

    conversation_id = response.metadata.get("conversation_id")

    # Immediately certify for legal records
    if ring.server_client and conversation_id:
        result = await ring.server_client.generate_receipt(
            conversation_id=conversation_id,
            description="Legal contract drafting - Case #123",
            tags=["legal", "audit", "case-123"]
        )

        receipt = result["receipt"]
        # Store receipt with case files
        with open(f"case-123-receipt-{receipt['receipt_id']}.json", "w") as f:
            json.dump(receipt, f, indent=2)
```

**CLI:**
```bash
llmring receipts generate --conversation 550e8400-e29b-41d4-a716-446655440000 \
  --description "Legal case #123" --tags legal --tags audit
```

### Cost Preview Before Billing

Preview costs before generating final receipt:

```python
from datetime import datetime

async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    if ring.server_client:
        # Preview monthly costs
        preview = await ring.server_client.preview_receipt(
            start_date=datetime(2025, 10, 1),
            end_date=datetime(2025, 10, 31)
        )

        print(f"Monthly Preview:")
        print(f"  Total logs: {preview['total_logs']}")
        print(f"  Total cost: ${preview['total_cost']:.2f}")
        print(f"\nBy model:")
        for model, stats in preview['by_model'].items():
            print(f"  {model}: {stats['calls']} calls, ${stats['cost']:.2f}")

        # Confirm and generate
        if input("Generate receipt? (y/n): ").lower() == 'y':
            result = await ring.server_client.generate_receipt(
                start_date=datetime(2025, 10, 1),
                end_date=datetime(2025, 10, 31),
                description="October billing - confirmed",
                tags=["billing", "monthly"]
            )
            print(f"✅ Receipt generated: {result['receipt']['receipt_id']}")
```

**CLI:**
```bash
# Preview first
llmring receipts preview --start 2025-10-01 --end 2025-10-31

# If satisfied, generate
llmring receipts generate --start 2025-10-01 --end 2025-10-31
```

### Weekly Cost Reports

Generate weekly batch receipts for cost tracking:

```python
from datetime import datetime, timedelta

async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    if ring.server_client:
        # Calculate week range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Generate weekly receipt
        result = await ring.server_client.generate_receipt(
            start_date=start_date,
            end_date=end_date,
            description=f"Week of {start_date.date()}",
            tags=["weekly", "report"]
        )

        receipt = result["receipt"]
        summary = receipt["batch_summary"]

        # Send to management dashboard
        report = {
            "week": f"{start_date.date()} to {end_date.date()}",
            "total_cost": receipt["total_cost"],
            "total_calls": summary["total_calls"],
            "by_alias": summary["by_alias"],
            "receipt_id": receipt["receipt_id"]
        }

        print(f"Week: {report['week']}")
        print(f"Cost: ${report['total_cost']:.2f}")
        print(f"Calls: {report['total_calls']}")
```

## Best Practices

### Cost Management

1. **Monitor Regularly**: Check `llmring stats` daily or weekly
2. **Set Budgets**: Track costs against budgets
3. **Use Aliases**: Easier to track costs by use case
4. **Profile Strategy**: Use cheaper models in dev, production quality in prod
5. **Export Often**: Export receipts for analysis and backup

### Security

1. **Protect Keys**: Keep Ed25519 signing keys in your secret manager
2. **Verify Receipts**: Validate signatures for compliance-critical workflows
3. **Backup Receipts**: Export and archive receipts and usage logs regularly
4. **Audit Trails**: Maintain conversation logs (available when `log_conversations=True`)
5. **Server Hardening**: Run llmring-server behind TLS with restricted CORS

### Performance

1. **Server Storage**: Receipts and logs live in PostgreSQL; scale storage as needed
2. **Batch Export**: Paginate `list_receipts` / `get_receipt_logs` for large datasets
3. **Cost of Generation**: Receipt signing is lightweight; batch receipts reduce overhead
4. **Caching**: Registry and pricing data are cached client-side for 24 hours

## Troubleshooting

### Missing Cost Information

```python
# Check if usage info is present
if response.usage:
    print(f"Cost: ${response.usage.get('cost', 0):.6f}")
else:
    print("No usage information available")
```

**Causes:**
- Streaming responses (check final chunk)
- Provider errors
- Network issues

### Signature Verification Failures

```python
from base64 import urlsafe_b64decode
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

verify_key = VerifyKey(urlsafe_b64decode(public_key_b64 + "=="))

try:
    verify_key.verify(
        receipt.to_canonical_json().encode("utf-8"),
        urlsafe_b64decode(receipt.signature + "=="),
    )
except BadSignatureError:
    print("Signature invalid - receipt may be tampered")
except Exception as exc:
    print(f"Verification error: {exc}")
```

**Causes:**
- Wrong public key
- Receipt modified after signing
- Signature format issues

### Export Issues

```bash
# Check if receipts exist
llmring stats

# Try exporting to different format
llmring export --format json
llmring export --format csv
```

## Related Documentation

- [CLI Reference](cli-reference.md) - Commands for cost tracking
- [API Reference](api-reference.md) - LLMRing and LLMRingExtended
- [Lockfile Documentation](lockfile.md) - Profile and alias configuration
