# OpenAI Responses API Integration - Problem & Solution

**Date:** 2025-11-06
**Priority:** HIGH
**Status:** Implemented

---

## Problem Statement

### Current Behavior

llmring's OpenAI provider previously rejected files with an error:

```python
# Current implementation in openai_api.py (lines 896-903)
if files:
    return await self._chat_with_files_responses_api(...)
```

**User experience:**
```python
from llmring import LLMRing, LLMRequest, Message

ring = LLMRing()

# Register file (works)
file_id = ring.register_file("data.csv")

# Use with Anthropic (works)
await ring.chat(LLMRequest(
    model="anthropic:claude-3-5-haiku-20241022",
    files=[file_id]
))  # ✅ Works!

# Use with OpenAI (fails)
await ring.chat(LLMRequest(
    model="openai:gpt-4o",
    files=[file_id]
))  # ❌ ValueError: Chat Completions API does not support file uploads
```

### Why This Is Better

- OpenAI’s Responses API supersedes legacy Chat Completions for multi‑modal and tools.
- Files are supported via Responses `input_file` items (pre‑uploaded `file_id`s); assistants are not required. Retrieval across large corpora can use `file_search` with vector stores if needed.
- llmring now switches to Responses when `files` are present, enabling parity with Anthropic/Google.

---

## Proposed Solution

### High-Level Approach

When `files` are present in an OpenAI request:
- Use Responses API with `input=[{"role":"user","content":[{"type":"input_text","text":...},{"type":"input_file","file_id":"file-..."}, ...]}]`
- Include user tools if provided (no `file_search` required for direct `input_file` processing)
 - Convert llmring Message[] into Responses roleful input (`system`/`developer`/`user`/`assistant`) with content parts (`input_text`, `input_image`, `input_file`) to preserve structure
- Map `temperature`, `max_tokens -> max_output_tokens`, `response_format` where applicable
- Normalize usage (prompt/completion/total) from Responses usage

### Implementation Strategy

**File:** `src/llmring/providers/openai_api.py`

**Modify chat() method:**

```python
When files are provided, we delegate to `_chat_with_files_responses_api(...)`.
```

Implemented `_chat_with_files_responses_api(...)` and `_stream_with_files_responses_api(...)`.
Both add `input_file` items for each file ID, merge user tools, handle `response_format`, and normalize usage.

---

## Notes on API Behavior (Resolved)

- Message conversion: We flatten llmring messages into a single `input` string. This works reliably with tools and attachments, and avoids edge cases between Chat vs Responses content schemas.
- Tools: `file_search` can co-exist with custom function tools. We always include `file_search` and append converted user tools. `tool_choice` is passed through.
- Structured output: `response_format` (`json_object` and `json_schema`) is supported with Responses and respected by the provider mapping.
- Streaming: Implemented via `client.responses.stream(...)`. Emits deltas as `StreamChunk`, then a final chunk with usage.
- Usage: We normalize to `prompt_tokens`, `completion_tokens`, `total_tokens`. If the SDK returns `input_tokens`/`output_tokens`, we map accordingly; otherwise we estimate conservatively.

---

## Implementation Checklist

### Implementation & Tests
- Implemented `_chat_with_files_responses_api()` and `_stream_with_files_responses_api()` in `src/llmring/providers/openai_api.py`.
- Updated `chat()` and `chat_stream()` to delegate when `files` are present.
- Added usage normalization and robust error handling.
- Updated tests to validate OpenAI file support (`tests/test_chat_with_files.py`).

---

## Expected Outcome

Usage example:

```python
from llmring import LLMRing, LLMRequest, Message

ring = LLMRing()

# Register file once
file_id = ring.register_file("sales_report.pdf")

# Use with OpenAI (should work!)
response = await ring.chat(LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Summarize Q3 performance")],
    files=[file_id]
))

print(response.content)  # Should contain summary based on PDF!
```

Behind the scenes:
- llmring detects `files` with an OpenAI model
- Uses Responses API with `input_file` items
- Flattens messages to `input`
- Normalizes usage and returns standard llmring response

---

## Code Pointers for Implementation

### Related

- We still support Responses API for o1 models and the PDF path (direct `input_file`). The new attachments path is for pre‑uploaded files via `register_file()`.

### Where to Add the Fix

**File:** `src/llmring/providers/openai_api.py`

**Method:** `chat()` around line 896

**Pattern:** Follow the existing PDF and o1 patterns - detect condition, delegate to specialized method.

---

## Status & Success Criteria

- OpenAI files now work in `chat()` and `chat_stream()`
- Tests updated: now assert success (no error on OpenAI files)
- Files uploaded once via `register_file()` and reused per provider
- No Assistants API usage; no tech debt paths left in OpenAI provider

---

## Questions for OpenAI (ChatGPT)

Please help us implement Responses API support for files in llmring:

1. **Message Conversion:** How do we convert our multi-turn message list to Responses API format?
2. **Tool Merging:** Can we use file_search + custom tools together? How?
3. **Structured Output:** Does Responses API support response_format/JSON schema?
4. **Streaming:** Does Responses API support streaming responses?
5. **Usage Tracking:** What's the format of response.usage? Any additional cost fields for file_search?

**Code Context:**
- See `src/llmring/providers/openai_api.py` lines 221-465 for existing Responses API usage
- See `src/llmring/schemas.py` for our LLMRequest and LLMResponse formats
- We use the official OpenAI Python SDK (openai>=1.99.0)

Thank you!
