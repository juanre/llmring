# Migration Guide: Automatic to On-Demand Receipts

**Last Updated**: October 2025

## Overview

LLMRing has transitioned from automatic receipt generation to **on-demand receipt generation**. This guide will help you update your code to work with the new system.

## What Changed

### Before: Automatic Receipts

Previously, receipts were automatically generated and returned with every conversation log:

```python
from llmring import LLMRing

async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    response = await ring.chat(
        "balanced",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Receipt was automatically included in the response
    # This no longer happens!
```

### After: On-Demand Receipts

Now, conversations are logged **without** automatic receipts. You generate receipts explicitly when needed:

```python
from llmring import LLMRing

async with LLMRing(
    server_url="http://localhost:8000",
    api_key="your-api-key",
    log_conversations=True
) as ring:
    # 1. Use LLM as normal - conversations are logged
    response = await ring.chat(
        "balanced",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    # No receipt returned!

    # 2. Generate receipts on-demand when you need them
    if ring.server_client:
        result = await ring.server_client.generate_receipt(
            since_last_receipt=True,
            description="Daily certification"
        )
        receipt = result["receipt"]
        print(f"Generated receipt: {receipt['receipt_id']}")
```

## Why This Change?

The new on-demand system gives you:

1. **Control**: Choose when receipts are generated
2. **Batch Receipts**: Single receipt for entire billing periods
3. **Flexibility**: Daily, weekly, monthly, or custom schedules
4. **Better Performance**: Reduced overhead during normal operations
5. **Aggregate Statistics**: Breakdown by model, alias, date range
6. **Cost Efficiency**: Fewer database operations

## Breaking Changes

### 1. ConversationLogResponse No Longer Includes Receipt

**Before:**
```python
response = await ring.chat("alias", messages=[...])
# response.receipt was populated
```

**After:**
```python
response = await ring.chat("alias", messages=[...])
# response.receipt is None
# Use on-demand generation instead
```

### 2. Receipt Generation is Explicit

**Before:**
```python
# Receipts generated automatically - no action needed
```

**After:**
```python
# You must explicitly request receipt generation
result = await ring.server_client.generate_receipt(
    since_last_receipt=True
)
```

## Migration Patterns

### Pattern 1: Single Conversation Receipts

If you need a receipt for each conversation immediately:

**Before:**
```python
response = await ring.chat("alias", messages=[...])
receipt = response.receipt  # Automatic
process_receipt(receipt)
```

**After:**
```python
response = await ring.chat("alias", messages=[...])

# Generate receipt for this specific conversation
if ring.server_client:
    conversation_id = response.metadata.get("conversation_id")
    result = await ring.server_client.generate_receipt(
        conversation_id=conversation_id,
        description="Conversation receipt"
    )
    receipt = result["receipt"]
    process_receipt(receipt)
```

**CLI:**
```bash
# After making a conversation, generate its receipt
llmring receipts generate --conversation <conversation_id>
```

### Pattern 2: Periodic Batch Receipts (Recommended)

Instead of per-conversation receipts, generate batch receipts periodically:

**Before:**
```python
# Receipts generated for every conversation
for conversation in conversations:
    response = await ring.chat("alias", messages=conversation)
    receipt = response.receipt
    all_receipts.append(receipt)
```

**After:**
```python
# Use LLM normally throughout the day/week
for conversation in conversations:
    response = await ring.chat("alias", messages=conversation)
    # No receipt needed here

# At end of period, generate one batch receipt
if ring.server_client:
    result = await ring.server_client.generate_receipt(
        since_last_receipt=True,
        description="Daily batch certification",
        tags=["daily", "compliance"]
    )

    receipt = result["receipt"]
    # One receipt certifying all conversations
    print(f"Certified {result['certified_count']} conversations")
    print(f"Total cost: ${receipt['total_cost']:.2f}")

    # Receipt includes aggregated statistics
    summary = receipt["batch_summary"]
    print(f"By model: {summary['by_model']}")
```

**CLI:**
```bash
# Daily batch (via cron at end of day)
llmring receipts generate --since-last --description "Daily certification"
```

### Pattern 3: Monthly Billing

If you were collecting receipts for monthly billing:

**Before:**
```python
# Collect all individual receipts throughout month
monthly_receipts = []
for conversation in month_conversations:
    response = await ring.chat("alias", messages=conversation)
    monthly_receipts.append(response.receipt)

# Calculate total
total_cost = sum(r["total_cost"] for r in monthly_receipts)
```

**After:**
```python
from datetime import datetime

# Use LLM normally throughout the month
for conversation in month_conversations:
    response = await ring.chat("alias", messages=conversation)
    # No receipt needed during the month

# At month end, generate single batch receipt
if ring.server_client:
    result = await ring.server_client.generate_receipt(
        start_date=datetime(2025, 10, 1),
        end_date=datetime(2025, 10, 31),
        description="October 2025 billing",
        tags=["billing", "monthly"]
    )

    receipt = result["receipt"]
    summary = receipt["batch_summary"]

    # Single receipt with all the data
    total_cost = receipt["total_cost"]
    breakdown_by_model = summary["by_model"]
    breakdown_by_alias = summary["by_alias"]

    print(f"Monthly total: ${total_cost:.2f}")
    print(f"Total calls: {summary['total_calls']}")
```

**CLI:**
```bash
# End of month
llmring receipts generate --start 2025-10-01 --end 2025-10-31 \
  --description "October billing" --tags billing --tags monthly
```

### Pattern 4: Real-Time Cost Tracking

If you need to track costs in real-time:

**Before:**
```python
running_total = 0
response = await ring.chat("alias", messages=[...])
running_total += response.receipt["total_cost"]
print(f"Running total: ${running_total:.2f}")
```

**After:**
```python
# Option 1: Use response.usage (available immediately)
running_total = 0
response = await ring.chat("alias", messages=[...])
running_total += response.usage.get("cost", 0)
print(f"Running total: ${running_total:.2f}")

# Option 2: Query uncertified logs periodically
if ring.server_client:
    uncert = await ring.server_client.get_uncertified_logs()
    total_uncertified_cost = sum(log.get("cost", 0) for log in uncert["logs"])
    print(f"Uncertified cost: ${total_uncertified_cost:.2f}")
```

**CLI:**
```bash
# Check uncertified logs and their costs
llmring receipts uncertified
```

## New Features You Can Use

### 1. Preview Receipts Before Generating

Check what will be certified before committing:

```python
# Preview what would be certified
preview = await ring.server_client.preview_receipt(
    start_date=datetime(2025, 10, 1),
    end_date=datetime(2025, 10, 31)
)

print(f"Would certify: {preview['total_logs']} logs")
print(f"Total cost: ${preview['total_cost']:.2f}")
print(f"By model: {preview['by_model']}")

# If satisfied, generate the actual receipt
if preview['total_cost'] < budget:
    result = await ring.server_client.generate_receipt(
        start_date=datetime(2025, 10, 1),
        end_date=datetime(2025, 10, 31),
        description="October billing"
    )
```

**CLI:**
```bash
llmring receipts preview --start 2025-10-01 --end 2025-10-31
```

### 2. Track Uncertified Logs

See which logs don't have receipts yet:

```python
uncert = await ring.server_client.get_uncertified_logs()
print(f"Uncertified logs: {uncert['total']}")

for log in uncert['logs']:
    print(f"  {log['id']}: ${log.get('cost', 0):.4f}")
```

**CLI:**
```bash
llmring receipts uncertified --limit 20
```

### 3. Audit Receipt Coverage

Get all logs certified by a specific receipt:

```python
logs = await ring.server_client.get_receipt_logs("rcpt_abc123")
print(f"This receipt certified {logs['total']} logs")

for log in logs['logs']:
    print(f"  {log['id']}: {log.get('type')}")
```

### 4. Flexible Scheduling

Generate receipts on any schedule that fits your workflow:

```python
# Daily at end of day
result = await ring.server_client.generate_receipt(
    since_last_receipt=True,
    description=f"Daily - {datetime.now().date()}",
    tags=["daily"]
)

# Weekly on Sunday
if datetime.now().weekday() == 6:
    result = await ring.server_client.generate_receipt(
        since_last_receipt=True,
        description=f"Week ending {datetime.now().date()}",
        tags=["weekly"]
    )

# Monthly on last day
if datetime.now().day == calendar.monthrange(year, month)[1]:
    result = await ring.server_client.generate_receipt(
        since_last_receipt=True,
        description=f"{datetime.now().strftime('%B %Y')} billing",
        tags=["monthly", "billing"]
    )
```

### 5. Descriptions and Tags

Organize receipts with metadata:

```python
result = await ring.server_client.generate_receipt(
    since_last_receipt=True,
    description="Q4 2025 compliance audit",
    tags=["compliance", "audit", "q4", "2025"]
)

# Later, retrieve and filter by tags
receipts = await ring.server_client.list_receipts()
audit_receipts = [r for r in receipts["receipts"] if "audit" in r.get("tags", [])]
```

## Recommended Migration Strategy

### Step 1: Update Dependencies

Ensure you have the latest version of llmring:

```bash
uv sync  # or pip install --upgrade llmring
```

### Step 2: Review Current Code

Identify where you're currently using receipts:

```bash
# Search for receipt usage in your codebase
grep -r "\.receipt" your_project/
grep -r "response\.receipt" your_project/
```

### Step 3: Decide on Schedule

Choose when to generate receipts:
- **Immediate**: After each critical conversation (use `conversation_id`)
- **Daily**: End of day (use `since_last_receipt=True`)
- **Weekly**: End of week (use date range or `since_last_receipt=True`)
- **Monthly**: End of month (use date range for billing)

### Step 4: Update Code

Replace automatic receipt usage with on-demand generation:

```python
# Old code
response = await ring.chat("alias", messages=[...])
if response.receipt:  # Remove this
    process_receipt(response.receipt)

# New code
response = await ring.chat("alias", messages=[...])

# Later (at your chosen schedule):
if ring.server_client:
    result = await ring.server_client.generate_receipt(
        since_last_receipt=True,
        description="Your description"
    )
    process_receipt(result["receipt"])
```

### Step 5: Add Scheduled Tasks

Set up cron jobs or scheduled tasks for periodic receipts:

```bash
# crontab -e
0 23 * * * /path/to/llmring receipts generate --since-last --description "Daily certification"
0 0 1 * * /path/to/llmring receipts generate --start $(date -d "last month" +\%Y-\%m-01) --end $(date -d "last day of last month" +\%Y-\%m-\%d) --description "Monthly billing"
```

### Step 6: Test

Verify receipts are being generated correctly:

```bash
# Check uncertified logs
llmring receipts uncertified

# Generate a test receipt
llmring receipts generate --since-last --description "Migration test"

# List all receipts
llmring receipts list

# Verify a specific receipt
llmring receipts get <receipt_id>
```

## Common Questions

### Q: Can I still get a receipt for every conversation?

**A**: Yes, but it requires two calls instead of one:

```python
# Make the conversation
response = await ring.chat("alias", messages=[...])
conversation_id = response.metadata.get("conversation_id")

# Generate receipt for this specific conversation
if ring.server_client and conversation_id:
    result = await ring.server_client.generate_receipt(
        conversation_id=conversation_id,
        description="Per-conversation receipt"
    )
    receipt = result["receipt"]
```

However, we recommend using batch receipts instead for better performance.

### Q: What if I need real-time cost tracking?

**A**: Use `response.usage` instead of receipts:

```python
response = await ring.chat("alias", messages=[...])
cost = response.usage.get("cost", 0)
tokens = response.usage.get("total_tokens", 0)
```

Receipts are for **certification and compliance**, not real-time cost tracking.

### Q: Are my old receipts still valid?

**A**: Yes! Old receipts remain valid and can still be verified. The signature format and verification process haven't changed.

### Q: Can a log have multiple receipts?

**A**: Yes! If you generate overlapping receipts (e.g., a daily receipt and later a monthly receipt for the same period), logs can be certified by multiple receipts. This is useful for different reporting needs.

### Q: How do I know which logs are uncertified?

**A**: Use the uncertified logs endpoint:

```python
uncert = await ring.server_client.get_uncertified_logs()
print(f"Uncertified: {uncert['total']}")
```

**CLI:**
```bash
llmring receipts uncertified
```

### Q: What happens if I don't generate receipts?

**A**: Nothing breaks! Logs are still stored and queryable. Receipts are optional and only needed for:
- Compliance and audit requirements
- Cryptographic certification
- Billing and invoicing
- Cost reporting

If you don't need these, you don't need receipts.

## Advanced Examples

### Multi-Project Cost Allocation

Track costs separately for different projects or clients:

```python
from llmring import LLMRing
from llmring.server_client import ServerClient

# Project A work
ring_a = LLMRing(
    server_url="http://localhost:8000",
    api_key="project_a_key",  # Different API key per project
    log_conversations=True
)

for task in project_a_tasks:
    await ring_a.chat("balanced", messages=task)

# Project B work
ring_b = LLMRing(
    server_url="http://localhost:8000",
    api_key="project_b_key",  # Different API key
    log_conversations=True
)

for task in project_b_tasks:
    await ring_b.chat("balanced", messages=task)

# Generate separate invoices for each project
client_a = ServerClient("http://localhost:8000", "project_a_key")
invoice_a = await client_a.generate_receipt(
    start_date=month_start,
    end_date=month_end,
    description=f"Project A - {month_name}",
    tags=["project-a", "invoice"]
)

client_b = ServerClient("http://localhost:8000", "project_b_key")
invoice_b = await client_b.generate_receipt(
    start_date=month_start,
    end_date=month_end,
    description=f"Project B - {month_name}",
    tags=["project-b", "invoice"]
)

print(f"Project A: ${invoice_a['receipt']['total_cost']:.2f}")
print(f"Project B: ${invoice_b['receipt']['total_cost']:.2f}")
```

### Automated Daily Batch Receipts with Cron

Set up automated daily receipt generation:

**Python script (`daily_receipts.py`):**
```python
#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta
from llmring.server_client import ServerClient

async def generate_daily_receipt():
    """Generate receipt for yesterday's usage."""
    client = ServerClient(
        server_url="http://localhost:8000",
        api_key="your-api-key"
    )

    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday.replace(hour=0, minute=0, second=0)
    end_date = yesterday.replace(hour=23, minute=59, second=59)

    result = await client.generate_receipt(
        start_date=start_date,
        end_date=end_date,
        description=f"Daily receipt - {yesterday.strftime('%Y-%m-%d')}",
        tags=["automated", "daily"]
    )

    receipt = result["receipt"]
    print(f"✅ Generated receipt {receipt['receipt_id']}")
    print(f"   Certified: {result['certified_count']} logs")
    print(f"   Total cost: ${receipt['total_cost']:.2f}")

if __name__ == "__main__":
    asyncio.run(generate_daily_receipt())
```

**Cron job:**
```bash
# Run daily at 1 AM
0 1 * * * cd /path/to/project && /usr/bin/python3 daily_receipts.py >> /var/log/llmring_receipts.log 2>&1
```

### Cost Budget Monitoring with Alerts

Monitor costs and alert when approaching budget:

```python
from llmring import LLMRing
from llmring.server_client import ServerClient
from datetime import datetime

class CostMonitor:
    def __init__(self, server_url, api_key, daily_budget=10.0):
        self.client = ServerClient(server_url, api_key)
        self.daily_budget = daily_budget

    async def check_budget(self):
        """Check if daily budget is exceeded."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0)

        # Preview today's costs without generating receipt
        preview = await self.client.preview_receipt(
            start_date=today_start,
            end_date=datetime.now()
        )

        total_cost = preview["total_cost"]
        percentage = (total_cost / self.daily_budget) * 100

        if percentage >= 90:
            self.send_alert(
                f"⚠️ Daily budget at {percentage:.1f}%! "
                f"Spent ${total_cost:.2f} of ${self.daily_budget:.2f}"
            )
        elif percentage >= 75:
            self.send_warning(
                f"Daily budget at {percentage:.1f}%. "
                f"Spent ${total_cost:.2f} of ${self.daily_budget:.2f}"
            )

        return total_cost, percentage

    def send_alert(self, message):
        # Send to Slack, email, PagerDuty, etc.
        print(f"ALERT: {message}")

    def send_warning(self, message):
        print(f"WARNING: {message}")

# Use in your application
monitor = CostMonitor("http://localhost:8000", "your-api-key", daily_budget=10.0)

# Before expensive operations
current_cost, percentage = await monitor.check_budget()
if percentage < 95:
    # Safe to proceed
    response = await ring.chat("deep", messages=complex_task)
else:
    # Use cheaper model
    response = await ring.chat("fast", messages=complex_task)
```

### Exporting for Accounting Systems

Generate receipts compatible with accounting software:

```python
import csv
from datetime import datetime
from llmring.server_client import ServerClient

async def export_for_accounting(month, year):
    """Export monthly receipt data for QuickBooks/Xero."""
    client = ServerClient("http://localhost:8000", "your-api-key")

    # Generate monthly receipt
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    result = await client.generate_receipt(
        start_date=start_date,
        end_date=end_date,
        description=f"LLM API Usage - {month}/{year}",
        tags=["accounting", "monthly"]
    )

    receipt = result["receipt"]
    summary = receipt["batch_summary"]

    # Export to CSV for accounting
    with open(f"llmring_invoice_{year}_{month:02d}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Model", "Calls", "Tokens", "Cost"])

        # Write breakdown by model
        for model, stats in summary["by_model"].items():
            writer.writerow([
                f"{year}-{month:02d}",
                f"LLM API - {model}",
                model,
                stats["calls"],
                stats["tokens"],
                f"${stats['cost']:.2f}"
            ])

        # Write total
        writer.writerow([])
        writer.writerow(["", "", "TOTAL", "", "", f"${receipt['total_cost']:.2f}"])

    print(f"✅ Exported invoice to llmring_invoice_{year}_{month:02d}.csv")
    print(f"   Receipt ID: {receipt['receipt_id']}")

# Generate monthly invoice
await export_for_accounting(month=10, year=2025)
```

## Backward Compatibility

The following still work as before:

✅ **Response usage information**: `response.usage` still includes cost and token data
✅ **Conversation logging**: Conversations are still logged to the server
✅ **Usage logging**: Usage logs are still created
✅ **Receipt verification**: Old receipts can still be verified
✅ **Public key endpoints**: `/receipts/public-key.pem` still works

The only change is that **automatic receipt generation is removed**.

## Need Help?

- **Documentation**: See [receipts.md](receipts.md) for full documentation
- **CLI Reference**: Run `llmring receipts --help` for all commands
- **Examples**: Check the [Use Cases section](receipts.md#use-cases) in receipts.md
- **Issues**: Report problems at https://github.com/anthropics/llmring/issues

## Summary

**What changed**: Receipts are no longer automatic
**Why**: To give you control and enable batch receipts
**How to migrate**: Use `ring.server_client.generate_receipt()` on your schedule
**Recommended approach**: Use batch receipts (daily/weekly/monthly) instead of per-conversation
**What you gain**: Better control, aggregate statistics, flexible scheduling, reduced overhead
