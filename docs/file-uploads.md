# File Registration

## Overview

LLMRing provides a provider-agnostic file registration system that enables efficient file handling across all LLM providers. Register files once and use them anywhere with lazy uploads that happen automatically on first use per provider.

**Key benefits:**
- **Provider-agnostic** - Register once, use with any provider (Anthropic, OpenAI, Google)
- **Lazy uploads** - Files upload only when first used, not at registration
- **Automatic staleness detection** - Files are re-hashed before each use to detect changes
- **Cross-provider caching** - Upload tracking per provider prevents redundant uploads
- **Reduce token costs** - File content doesn't count against message token limits
- **Enable code execution** - Use datasets for analysis with Anthropic's code execution tool

## Quick Start

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # Register a file (doesn't upload yet)
    file_id = await service.register_file("data.csv")

    # Use with Anthropic (lazy upload happens here)
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content="Analyze this CSV")],
        files=[file_id],
        tools=[{"type": "code_execution"}]
    )

    response = await service.chat(request)
    print(response.content)

    # Use same file with Google (separate upload happens automatically)
    request = LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(role="user", content="Summarize this data")],
        files=[file_id]
    )

    response = await service.chat(request)
    print(response.content)

    # Clean up registration
    await service.deregister_file(file_id)
```

---

## API Reference

### `register_file()`

Register a file for use with any provider. Files are not uploaded until first use.

```python
async def register_file(
    file_path: Union[str, Path]
) -> str
```

**Parameters:**
- `file_path`: Path to the file to register (string or Path object)

**Returns:** `str` - A unique file ID for referencing the file in chat requests

**Raises:**
- `FileNotFoundError` - File doesn't exist
- `ValueError` - File path is invalid

**Examples:**

```python
# Register a file
file_id = await service.register_file("dataset.csv")

# Register from Path object
from pathlib import Path
file_id = await service.register_file(Path("data.json"))
```

**How it works:**
1. File path is validated but not read yet
2. A unique file ID is generated and stored
3. File metadata is tracked internally
4. On first use with a provider, the file is hashed and uploaded
5. Subsequent uses check the hash to detect changes
6. Each provider tracks its own upload separately

---

### `list_registered_files()`

List all currently registered files.

```python
async def list_registered_files() -> List[Dict[str, str]]
```

**Returns:** List of dictionaries with file information
```python
[
    {
        "file_id": "reg_abc123",
        "file_path": "/path/to/data.csv",
        "registered_at": "2024-01-15T10:30:00Z"
    },
    ...
]
```

**Example:**

```python
# List all registered files
files = await service.list_registered_files()

for file in files:
    print(f"{file['file_id']}: {file['file_path']}")
```

---

### `deregister_file()`

Remove a file registration and clean up all provider uploads.

```python
async def deregister_file(
    file_id: str
) -> bool
```

**Parameters:**
- `file_id`: File ID returned from `register_file()`

**Returns:** `True` if successful

**What it does:**
1. Removes the file registration
2. Deletes uploaded versions from all providers
3. Clears internal tracking metadata

**Example:**

```python
# Clean up when done
await service.deregister_file(file_id)
```

---

## Provider Comparison

LLMRing's file registration system abstracts provider differences while enabling cross-provider file usage. When you register a file, it can be used with any provider - each provider handles uploads independently.

| Feature | Anthropic | OpenAI | Google |
|---------|-----------|--------|--------|
| **Upload Method** | Files API | Files API | Context Caching |
| **Lazy Upload** | ✅ On first use | ✅ On first use | ✅ On first use |
| **Provider ID Format** | `file_` prefix | `file-` prefix | `cachedContents/` prefix |
| **Max File Size** | 500 MB | 512 MB | N/A (text-only) |
| **Storage Limit** | 100 GB/org | Per account | N/A |
| **TTL** | Permanent | Permanent | Configurable (default 60 min) |
| **File Types** | Any | Any | Text only (UTF-8) |
| **Use Case** | Code execution, analysis | Assistants API | Cost optimization |
| **Works with Chat** | ✅ Yes | ❌ No (Assistants only) | ✅ Yes |
| **Staleness Detection** | ✅ Hash checked | ✅ Hash checked | ✅ Hash checked |

### Anthropic Files API

**Implementation:** Discrete file upload API with persistent storage

**Features:**
- Upload any file type (CSV, JSON, TXT, PDF, etc.)
- Files persist until explicitly deleted
- Maximum 500 MB per file
- 100 GB storage per organization
- Use with code execution tool
- Reference in chat messages via `file_id`

**File ID Format:** `file_abc123def456...`

**Supported Models:** All Claude 3.5+ models

---

### OpenAI Files API

**Implementation:** Purpose-based file storage

**Features:**
- Upload files for specific purposes ("assistants", "vision", "batch")
- Files persist until deleted
- Maximum 512 MB per file
- Purpose must be specified at upload time

**File ID Format:** `file-abc123def456...`

**Important Limitation:**
- Files uploaded to OpenAI **cannot be used with Chat Completions API**
- Files only work with **Assistants API** (separate from `service.chat()`)
- Use direct SDK access for OpenAI Assistants integration

---

### Google Context Caching

**Implementation:** Content caching system (not traditional file storage)

**Features:**
- Cache large text content for reuse
- Configurable TTL (default 60 minutes)
- 90% cost savings on Gemini 2.5 models
- Minimum 1,024 tokens (2.5 Flash) or 4,096 tokens (2.5 Pro)
- Text-only (UTF-8 encoding required)
- Automatic expiration

**Cache ID Format:** `cachedContents/abc123def456...`

**Key Difference:**
- Google doesn't have discrete "files" - registering creates a cache on first use
- File content converted to text and cached
- Binary files will raise `ValueError` about UTF-8 encoding

---

## Complete Examples

### Example 1: Cross-Provider File Usage

Register a file once and use it with multiple providers:

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # Register file once (no upload yet)
    file_id = await service.register_file("sales_data.csv")

    print(f"Registered: {file_id}")

    # Use with Anthropic (lazy upload happens here)
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Analyze this sales data and calculate total revenue by quarter"
        )],
        files=[file_id],
        tools=[{"type": "code_execution"}],
        max_tokens=4000
    )

    response = await service.chat(request)
    print(f"Anthropic: {response.content}")

    # Use same file with Google (separate upload happens automatically)
    request = LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(
            role="user",
            content="Summarize the sales trends"
        )],
        files=[file_id]
    )

    response = await service.chat(request)
    print(f"Google: {response.content}")

    # Clean up (removes registration and all provider uploads)
    await service.deregister_file(file_id)
```

---

### Example 2: Multiple Files in Single Request

Register and use multiple files together:

```python
async with LLMRing() as service:
    # Register multiple files
    file1 = await service.register_file("q1_sales.csv")
    file2 = await service.register_file("q2_sales.csv")
    file3 = await service.register_file("q3_sales.csv")

    # Analyze all files together (uploads happen on first use)
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Compare quarterly sales trends across all three files"
        )],
        files=[file1, file2, file3],
        tools=[{"type": "code_execution"}],
        max_tokens=4000
    )

    response = await service.chat(request)
    print(response.content)

    # Clean up all registrations
    for file_id in [file1, file2, file3]:
        await service.deregister_file(file_id)
```

---

### Example 3: File Management Operations

List and manage registered files:

```python
from llmring import LLMRing

async with LLMRing() as service:
    # Register some files
    file1 = await service.register_file("data1.csv")
    file2 = await service.register_file("data2.csv")
    file3 = await service.register_file("data3.csv")

    # List all registered files
    files = await service.list_registered_files()
    print(f"Registered files: {len(files)}")
    for file in files:
        print(f"- {file['file_id']}: {file['file_path']}")

    # Use the files with different providers
    # ... chat operations ...

    # Clean up specific file
    await service.deregister_file(file1)

    # Verify it's gone
    remaining = await service.list_registered_files()
    print(f"Remaining files: {len(remaining)}")
```

---

### Example 4: Automatic Staleness Detection

Files are re-hashed before each use to detect changes:

```python
from llmring import LLMRing, LLMRequest, Message
from pathlib import Path

async with LLMRing() as service:
    file_path = Path("dynamic_data.csv")

    # Register file
    file_id = await service.register_file(file_path)

    # First use - uploads to Anthropic
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content="Analyze this data")],
        files=[file_id],
        tools=[{"type": "code_execution"}]
    )
    response = await service.chat(request)
    print(f"First analysis: {response.content}")

    # Simulate file update
    with open(file_path, "a") as f:
        f.write("\nnew,data,row")

    # Second use - detects change and re-uploads automatically
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content="Analyze updated data")],
        files=[file_id],
        tools=[{"type": "code_execution"}]
    )
    response = await service.chat(request)
    print(f"Updated analysis: {response.content}")

    # Clean up
    await service.deregister_file(file_id)
```

---

### Example 5: Streaming with Registered Files

Stream responses while using registered files:

```python
async with LLMRing() as service:
    # Register file
    file_id = await service.register_file("data.csv")

    # Stream response
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Briefly analyze this data"
        )],
        files=[file_id],
        tools=[{"type": "code_execution"}]
    )

    print("Response: ", end="", flush=True)
    async for chunk in service.chat_stream(request):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()

    # Clean up
    await service.deregister_file(file_id)
```

---

## Error Handling

### Common Errors and Solutions

#### FileNotFoundError

```python
try:
    file_id = await service.register_file("nonexistent.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Solution: Verify file path exists before registration
```

---

#### FileSizeError

File size validation happens during upload (on first use), not registration:

```python
from llmring.exceptions import FileSizeError

try:
    # Register large file (succeeds)
    file_id = await service.register_file("huge_file.csv")

    # Use with Anthropic (upload happens here, may fail)
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content="Analyze")],
        files=[file_id]
    )
    response = await service.chat(request)
except FileSizeError as e:
    print(f"File too large: {e.file_size} bytes")
    print(f"Maximum allowed: {e.max_size} bytes")
    print(f"Provider: {e.provider}")
    # Solution: Split file or use a different provider
```

**Provider Limits:**
- Anthropic: 500 MB
- OpenAI: 512 MB
- Google: N/A (text-only, token-based limits)

---

#### Invalid File Format

```python
try:
    # Register binary file
    file_id = await service.register_file("image.png")

    # Try to use with Google (will fail - Google requires text)
    request = LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(role="user", content="Analyze")],
        files=[file_id]
    )
    response = await service.chat(request)
except ValueError as e:
    print(f"Invalid format for Google: {e}")
    # Solution: Use text files with Google or different provider
```

---

#### OpenAI Chat Completions Not Supported

```python
# OpenAI files don't work with Chat Completions API
file_id = await service.register_file("document.pdf")

request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Analyze this")],
    files=[file_id]
)

try:
    response = await service.chat(request)
except ValueError as e:
    print(e)
    # "OpenAI Chat Completions API does not support file uploads"
```

**Solution:** Use Anthropic or Google for file-based chat interactions.

---

## Best Practices

### 1. Clean Up Registrations

Always deregister files when done to clean up provider uploads:

```python
file_id = await service.register_file("data.csv")
try:
    # Use file with various providers
    response = await service.chat(request)
finally:
    # Clean up registration and all provider uploads
    await service.deregister_file(file_id)
```

---

### 2. Register Once, Use Everywhere

The power of file registration is cross-provider reuse:

```python
# Register once
file_id = await service.register_file("analysis_data.csv")

# Use with multiple providers - uploads happen automatically
providers = ["anthropic:claude-sonnet-4-5", "google:gemini-2.5-flash"]

for model in providers:
    request = LLMRequest(
        model=model,
        messages=[Message(role="user", content="Analyze this data")],
        files=[file_id]
    )
    response = await service.chat(request)
    print(f"{model}: {response.content[:100]}...")

# Clean up once
await service.deregister_file(file_id)
```

---

### 3. Validate Files Before Registration

Check file exists and size before registering:

```python
from pathlib import Path

file_path = Path("data.csv")

# Check exists
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# Check size (Anthropic limit: 500 MB)
max_size = 500 * 1024 * 1024
if file_path.stat().st_size > max_size:
    raise ValueError(f"File too large: {file_path.stat().st_size} bytes")

# Register
file_id = await service.register_file(file_path)
```

---

### 4. Leverage Automatic Staleness Detection

File changes are detected automatically:

```python
# Register file
file_id = await service.register_file("live_data.csv")

# Use file multiple times - it's automatically re-uploaded if changed
for i in range(5):
    # File might be updated between requests
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content=f"Analysis #{i+1}")],
        files=[file_id]
    )
    response = await service.chat(request)
    # If file changed, it's re-hashed and re-uploaded automatically

await service.deregister_file(file_id)
```

---

### 5. Use Multiple Files in Single Request

Combine multiple files for complex analysis:

```python
# Register all files
file1 = await service.register_file("sales.csv")
file2 = await service.register_file("costs.csv")
file3 = await service.register_file("inventory.csv")

# Use all together
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[Message(role="user", content="Analyze profitability across all datasets")],
    files=[file1, file2, file3],
    tools=[{"type": "code_execution"}]
)
response = await service.chat(request)

# Clean up all
for file_id in [file1, file2, file3]:
    await service.deregister_file(file_id)
```

---

### 6. Handle Provider-Specific Requirements

Be aware of provider limitations:

```python
from pathlib import Path

file_path = Path("document.txt")

# Check if file is text (required for Google)
if file_path.suffix not in ['.txt', '.md', '.csv', '.json']:
    print("Warning: Google only supports text files")

# Check size for token requirements
file_size = file_path.stat().st_size
estimated_tokens = file_size // 4  # Rough estimate

if estimated_tokens < 1024:
    print("Warning: File may not meet Google's minimum token requirement")

# Register anyway - it works with other providers
file_id = await service.register_file(file_path)
```

---

## Troubleshooting

### File Registration Rejected

**Problem:** Registration fails immediately

**Solution:** Check file exists and path is valid:

```python
from pathlib import Path

file_path = Path("data.csv")

# Verify file exists
if not file_path.exists():
    print(f"File not found: {file_path}")
else:
    file_id = await service.register_file(file_path)
```

---

### Google Upload Fails with "UTF-8" Error

**Problem:** Binary file fails when used with Google

**Solution:** Google only accepts text files, but you can still use the same registered file with other providers:

```python
# Register binary file
file_id = await service.register_file("image.png")

# ✅ Works with Anthropic
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[Message(role="user", content="Analyze")],
    files=[file_id]
)
response = await service.chat(request)

# ❌ Fails with Google (requires text)
request = LLMRequest(
    model="google:gemini-2.5-flash",
    messages=[Message(role="user", content="Analyze")],
    files=[file_id]
)
# Will raise ValueError about UTF-8 encoding
```

---

### OpenAI Files Don't Work in Chat

**Problem:** File registered but OpenAI chat fails

**Solution:** OpenAI Chat Completions API doesn't support file uploads:

```python
file_id = await service.register_file("document.pdf")

# ❌ Won't work with OpenAI Chat
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Analyze")],
    files=[file_id]  # Error!
)

# ✅ Use Anthropic or Google instead
request = LLMRequest(
    model="anthropic:claude-sonnet-4-5",
    messages=[Message(role="user", content="Analyze")],
    files=[file_id]
)
response = await service.chat(request)
```

---

### File Changes Not Detected

**Problem:** Modified file not re-uploaded

**Solution:** File staleness detection is automatic via hash checking. If changes aren't detected, verify file was actually modified:

```python
import time
from pathlib import Path

file_path = Path("data.csv")
file_id = await service.register_file(file_path)

# Use file
response1 = await service.chat(request)

# Modify file (ensure enough time for filesystem)
time.sleep(0.1)
with open(file_path, "a") as f:
    f.write("\nnew,data")

# Use again - change will be detected automatically
response2 = await service.chat(request)
```

---

### Registered File Persists After Deregistration

**Problem:** File still usable after deregister

**Solution:** Deregistration is async and cleans up provider uploads. Ensure you await the call:

```python
# ✅ Correct
await service.deregister_file(file_id)

# ❌ Wrong - doesn't wait for cleanup
service.deregister_file(file_id)  # Missing await
```

---

## Related Documentation

- [API Reference](api-reference.md) - Core LLMRing API
- [Provider Guide](providers.md) - Provider-specific features
- [File Utilities](file-utilities.md) - Vision and multimodal capabilities
- [Structured Output](structured-output.md) - JSON schema support
