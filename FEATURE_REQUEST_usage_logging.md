# Feature Request: Server-Side Usage Logging

## Overview

`LLMRingExtended` currently tracks conversations and messages on the server, but does not send usage/receipt data. This means that while conversations are visible in the dashboard, there are no usage statistics, cost tracking, or analytics available.

## Current Behavior

When using `LLMRingExtended.chat_with_conversation()`:

1. ✅ Creates conversation on server via `POST /api/v1/conversations`
2. ✅ Sends messages to server via `POST /api/v1/conversations/{id}/messages/batch`
3. ❌ **Does NOT send usage data to server**
4. ❌ **Does NOT create receipts on server**

Result: Conversations appear in dashboard, but stats show zero activity.

## Desired Behavior

After each successful LLM API call, `LLMRingExtended` should send usage data to the server's logging endpoint.

### Server Endpoint

**Endpoint**: `POST /api/v1/log`

**Request Body**:
```json
{
  "model": "gpt-4o",                    // Model name (not alias)
  "provider": "openai",                  // Provider name
  "input_tokens": 14,                    // Prompt tokens
  "output_tokens": 286,                  // Completion tokens
  "cached_input_tokens": 0,              // Optional: cached tokens
  "cost": 0.00287,                       // Optional: calculated cost
  "latency_ms": 1250,                    // Optional: request duration
  "origin": "llmring",                   // Origin identifier
  "id_at_origin": "conv-123",            // Optional: conversation ID
  "metadata": {                          // Optional: additional context
    "alias": "default",
    "conversation_id": "uuid...",
    "model_alias": "openai:gpt-4o",
    "finish_reason": "stop"
  }
}
```

**Response**: `200 OK` with receipt details

### Implementation Requirements

1. **Timing**: Send usage log immediately after receiving LLM response, before returning to caller
2. **Error Handling**: If logging fails, log warning but don't fail the request
3. **Configuration**: Should be automatic when `server_url` and `api_key` are provided
4. **Conversation Context**: If `conversation_id` is present, include it in metadata
5. **Cost Calculation**: Use existing `CostCalculator` to determine cost if not provided by LLM

### Code Location

Add logging to `LLMRingExtended.chat_with_conversation()` in `src/llmring/service_extended.py`:

```python
async def chat_with_conversation(
    self,
    request: LLMRequest,
    conversation_id: Optional[UUID] = None,
    store_messages: bool = True,
    profile: Optional[str] = None,
) -> LLMResponse:
    """Send chat request with conversation tracking."""

    # Call parent to get response
    response = await super().chat(request, profile)

    # Store messages (existing code)
    if self.server_client and conversation_id and store_messages:
        await self._store_messages(...)

    # NEW: Log usage to server
    if self.server_client and response.usage:
        await self._log_usage(
            response=response,
            request=request,
            conversation_id=conversation_id,
        )

    return response
```

### New Method to Add

```python
async def _log_usage(
    self,
    response: LLMResponse,
    request: LLMRequest,
    conversation_id: Optional[UUID] = None,
) -> None:
    """Send usage data to server logging endpoint."""
    try:
        # Parse provider and model from request
        provider, model_name = parse_model_string(request.model)

        # Calculate cost if available
        cost = None
        if response.usage:
            # Use existing cost calculation logic
            cost = self._calculate_cost(provider, model_name, response.usage)

        # Prepare log entry
        log_data = {
            "model": model_name,
            "provider": provider,
            "input_tokens": response.usage.get("prompt_tokens", 0),
            "output_tokens": response.usage.get("completion_tokens", 0),
            "origin": self.origin,  # From parent LLMRing
            "metadata": {
                "model_alias": request.model,
                "finish_reason": response.finish_reason,
            }
        }

        # Add optional fields
        if cost is not None:
            log_data["cost"] = cost

        if conversation_id:
            log_data["id_at_origin"] = str(conversation_id)
            log_data["metadata"]["conversation_id"] = str(conversation_id)

        if response.usage.get("cached_tokens"):
            log_data["cached_input_tokens"] = response.usage["cached_tokens"]

        # Send to server
        await self.server_client.post("/api/v1/log", json=log_data)

    except Exception as e:
        logger.warning(f"Failed to log usage to server: {e}")
        # Don't raise - logging failure shouldn't break the request
```

## Benefits

1. **Complete Tracking**: Users can see usage statistics, costs, and analytics in the dashboard
2. **Historical Data**: All LLM calls are logged for auditing and analysis
3. **Cost Attribution**: Costs are properly attributed to projects and conversations
4. **Compliance**: Provides audit trail for LLM usage
5. **Analytics**: Enables trend analysis, cost optimization, and usage patterns

## Backward Compatibility

- No breaking changes
- Logging happens automatically when `server_url` and `api_key` are configured
- Works with both `chat()` and `chat_with_conversation()` methods
- Graceful degradation if server is unavailable

## Testing

1. Call `chat_with_conversation()` and verify usage appears in server stats
2. Test with various models and providers
3. Test error handling when server is unavailable
4. Verify conversation_id is properly linked to usage logs
5. Confirm cost calculation matches expected values

## Related Files

- `src/llmring/service_extended.py` - Main implementation
- `src/llmring/service.py` - Parent class with cost calculation
- `src/llmring/server_client.py` - HTTP client for server communication
- Server endpoint: `llmring-server` or `llmring-api` `/api/v1/log`

## Priority

**High** - Without this, the conversation tracking feature is incomplete and users cannot see usage statistics in the dashboard.
