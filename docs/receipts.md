# Receipts and Cost Tracking

## Overview

LLMRing provides a comprehensive receipt system for tracking LLM API usage and costs. Receipts include detailed information about each API call, including token usage, costs, model selection, and cryptographic signatures for compliance.

## Key Features

- **Automatic Cost Calculation**: Tracks tokens and calculates costs based on registry pricing
- **Cryptographic Signatures**: Ed25519 signatures for receipt verification
- **Canonical JSON**: JCS (JSON Canonicalization Scheme) for deterministic serialization
- **Lockfile Integration**: Includes SHA256 digest of lockfile for reproducibility
- **Profile Tracking**: Records which profile was used for each request
- **Local and Server Modes**: Works locally or with LLMRing server integration

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

### Automatic Receipt Generation

Receipts are generated automatically for all LLM requests:

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="fast",
        messages=[Message(role="user", content="Hello!")]
    )

    response = await service.chat(request)

    # Access cost information from response
    print(f"Tokens used: {response.usage['total_tokens']}")
    print(f"Cost: ${response.usage['cost']:.6f}")
```

### Accessing Receipts

```python
from llmring import LLMRing

service = LLMRing()

# Make some requests
# ...

# Access local receipts
for receipt in service.receipts:
    print(f"{receipt.timestamp}: {receipt.alias} → {receipt.provider}:{receipt.model}")
    print(f"  Tokens: {receipt.total_tokens}, Cost: ${receipt.total_cost:.6f}")
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
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| Anthropic | claude-3-5-haiku | $1.00 | $5.00 |
| Google | gemini-1.5-pro | $3.50 | $10.50 |
| Google | gemini-1.5-flash | $0.075 | $0.30 |

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
from llmring import LLMRing

service = LLMRing()

# Make requests...

# Export receipts
export_data = {
    "exported_at": datetime.now(UTC).isoformat(),
    "receipts": [
        {
            "receipt_id": r.receipt_id,
            "timestamp": r.timestamp.isoformat(),
            "alias": r.alias,
            "profile": r.profile,
            "provider": r.provider,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "total_cost": r.total_cost,
        }
        for r in service.receipts
    ],
}

with open("receipts.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

## Signed Receipts

### Signature System

Receipts can be cryptographically signed using Ed25519 for compliance and verification:

```python
from llmring.receipts import Receipt, ReceiptSigner, ReceiptGenerator

# Generate keypair
private_key, public_key = ReceiptSigner.generate_keypair()

# Create signer
signer = ReceiptSigner(private_key)

# Create generator with signer
generator = ReceiptGenerator(signer)

# Generate signed receipt
receipt = generator.generate_receipt(
    alias="fast",
    profile="default",
    lock_digest="sha256:abc123...",
    provider="openai",
    model="gpt-4o-mini",
    usage={
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300
    },
    costs={
        "input_cost": 0.000015,
        "output_cost": 0.000120,
        "total_cost": 0.000135
    }
)

print(f"Receipt signed: {receipt.signature}")
```

### Verifying Receipts

```python
from llmring.receipts import Receipt, ReceiptSigner

# Load receipt (from JSON, database, etc.)
receipt = Receipt(**receipt_data)

# Load public key
public_key = ReceiptSigner.load_public_key(public_key_bytes)

# Verify signature
is_valid = ReceiptSigner.verify_receipt(receipt, public_key)
print(f"Receipt valid: {is_valid}")
```

### Canonical JSON

Receipts use JCS (JSON Canonicalization Scheme) for deterministic signing:

```python
from llmring.receipts import Receipt

receipt = Receipt(...)

# Get canonical JSON (for signing/verification)
canonical = receipt.to_canonical_json()
print(canonical)  # Deterministic, sorted JSON

# Calculate digest
digest = receipt.calculate_digest()
print(f"Receipt digest: {digest}")
```

## Server Integration

### LLMRingExtended with Receipts

When using `LLMRingExtended` with server integration, receipts are automatically managed:

```python
from llmring import LLMRingExtended, LLMRequest, Message

async with LLMRingExtended(
    server_url="https://api.llmring.ai",
    api_key="your-key",
    enable_conversations=True
) as service:
    request = LLMRequest(
        model="balanced",
        messages=[Message(role="user", content="Hello!")]
    )

    response = await service.chat(request)

    # Server automatically generates and stores signed receipts
    # Receipts include signatures from LLMRing server
```

**Benefits:**
- Centralized receipt storage
- Automatic signing with server keys
- Analytics and reporting
- Compliance and audit trails
- Team-wide cost tracking

## Usage Statistics

### Local Statistics

```bash
# View local usage
llmring stats

# Detailed view
llmring stats --verbose
```

```python
from llmring import LLMRing

service = LLMRing()

# Calculate total usage
if service.receipts:
    total_cost = sum(r.total_cost for r in service.receipts)
    total_tokens = sum(r.total_tokens for r in service.receipts)

    print(f"Total requests: {len(service.receipts)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.6f}")
```

### Aggregation and Analysis

```python
from collections import defaultdict
from llmring import LLMRing

service = LLMRing()

# Group by provider
by_provider = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})

for receipt in service.receipts:
    by_provider[receipt.provider]["requests"] += 1
    by_provider[receipt.provider]["tokens"] += receipt.total_tokens
    by_provider[receipt.provider]["cost"] += receipt.total_cost

# Report
for provider, stats in by_provider.items():
    print(f"{provider}:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Tokens: {stats['tokens']:,}")
    print(f"  Cost: ${stats['cost']:.6f}")
```

### Cost by Alias

```python
from collections import defaultdict

by_alias = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})

for receipt in service.receipts:
    by_alias[receipt.alias]["requests"] += 1
    by_alias[receipt.alias]["tokens"] += receipt.total_tokens
    by_alias[receipt.alias]["cost"] += receipt.total_cost

# Find most expensive alias
most_expensive = max(by_alias.items(), key=lambda x: x[1]["cost"])
print(f"Most expensive alias: {most_expensive[0]} (${most_expensive[1]['cost']:.6f})")
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
from llmring.receipts import Receipt

# Load receipts from storage
receipts = load_receipts_from_database()

# Verify all receipts
public_key = load_public_key()

for receipt in receipts:
    if not ReceiptSigner.verify_receipt(receipt, public_key):
        print(f"⚠️  Invalid signature: {receipt.receipt_id}")
    else:
        print(f"✅ Valid: {receipt.receipt_id}")
        print(f"   Timestamp: {receipt.timestamp}")
        print(f"   Model: {receipt.provider}:{receipt.model}")
        print(f"   Cost: ${receipt.total_cost:.6f}")
```

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

# Generate monthly report
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
report = generate_compliance_report(service.receipts, start_date, end_date)

with open("compliance_report.json", "w") as f:
    json.dump(report, f, indent=2)
```

## Best Practices

### Cost Management

1. **Monitor Regularly**: Check `llmring stats` daily or weekly
2. **Set Budgets**: Track costs against budgets
3. **Use Aliases**: Easier to track costs by use case
4. **Profile Strategy**: Use cheaper models in dev, production quality in prod
5. **Export Often**: Export receipts for analysis and backup

### Security

1. **Protect Keys**: Keep signing keys secure
2. **Verify Receipts**: Validate signatures for critical applications
3. **Backup Receipts**: Export and store receipts safely
4. **Audit Trails**: Maintain logs of all API usage
5. **Server Mode**: Use LLMRing server for centralized security

### Performance

1. **Local Receipts**: Limited by memory; export periodically
2. **Server Mode**: Unlimited storage and analytics
3. **Batch Export**: Export in batches for large datasets
4. **Caching**: Receipt generation is fast; no special caching needed

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
try:
    is_valid = ReceiptSigner.verify_receipt(receipt, public_key)
    if not is_valid:
        print("Signature invalid - receipt may be tampered")
except Exception as e:
    print(f"Verification error: {e}")
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
