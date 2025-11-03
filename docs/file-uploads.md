# File Uploads

## Overview

LLMRing provides a unified file upload API that enables efficient file handling across Anthropic, OpenAI, and Google providers. Upload files once and reference them across multiple chat requests, reducing token usage and improving performance.

**Key benefits:**
- **Upload once, use many times** - Reference files across multiple requests
- **Reduce token costs** - File content doesn't count against message token limits
- **Enable code execution** - Upload datasets for analysis with Anthropic's code execution tool
- **Optimize caching** - Use Google's context caching for repeated document access
- **Build knowledge bases** - Upload documents for OpenAI Assistants API

## Quick Start

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # Upload a file
    file_resp = await service.upload_file(
        "data.csv",
        purpose="code_execution"
    )

    # Use in chat request
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content="Analyze this CSV")],
        files=[file_resp.file_id],
        tools=[{"type": "code_execution"}]
    )

    response = await service.chat(request)
    print(response.content)

    # Clean up
    await service.delete_file(file_resp.file_id)
```

---

## API Reference

### `upload_file()`

Upload a file and receive a `file_id` for reuse across requests.

```python
async def upload_file(
    file: Union[str, Path, BinaryIO],
    purpose: str = "analysis",
    provider: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    filename: Optional[str] = None,
    **kwargs
) -> FileUploadResponse
```

**Parameters:**
- `file`: File path (string/Path) or file-like object (BinaryIO)
- `purpose`: Intended use case (default: "analysis")
  - `"code_execution"` - For Anthropic code execution tool
  - `"assistant"` - For OpenAI Assistants API
  - `"analysis"` - General document analysis
  - `"cache"` - For Google context caching
- `provider`: Force specific provider (default: auto-detect from model)
- `ttl_seconds`: Cache TTL for Google (default: 3600)
- `filename`: Override filename for upload
- `**kwargs`: Provider-specific parameters

**Returns:** `FileUploadResponse`
```python
FileUploadResponse(
    file_id="file_abc123",          # Reference ID for file
    provider="anthropic",            # Provider name
    filename="data.csv",             # Original filename
    size_bytes=12345,                # File size in bytes
    created_at=datetime(...),        # Upload timestamp
    purpose="code_execution",        # File purpose
    metadata={}                      # Provider-specific metadata
)
```

**Raises:**
- `FileSizeError` - File exceeds provider limits
- `InvalidFileFormatError` - File type not supported
- `FileAccessError` - Cannot read file
- `ProviderResponseError` - Provider API error

**Examples:**

```python
# Upload from file path
file_resp = await service.upload_file(
    "dataset.csv",
    purpose="code_execution"
)

# Upload from file-like object
with open("document.txt", "rb") as f:
    file_resp = await service.upload_file(
        f,
        purpose="analysis",
        filename="document.txt"
    )

# Upload to specific provider
file_resp = await service.upload_file(
    "data.json",
    purpose="code_execution",
    provider="anthropic"
)

# Upload to Google with custom TTL
file_resp = await service.upload_file(
    "large_doc.txt",
    purpose="cache",
    provider="google",
    ttl_seconds=7200  # 2 hours
)
```

---

### `list_files()`

List uploaded files for a provider.

```python
async def list_files(
    provider: Optional[str] = None,
    purpose: Optional[str] = None,
    limit: int = 100
) -> List[FileMetadata]
```

**Parameters:**
- `provider`: Filter by provider (default: None)
- `purpose`: Filter by purpose (default: None, not all providers support)
- `limit`: Maximum files to return (default: 100)

**Returns:** List of `FileMetadata` objects

**Example:**

```python
# List all files for Anthropic
files = await service.list_files(provider="anthropic")

for file in files:
    print(f"{file.file_id}: {file.filename} ({file.size_bytes} bytes)")
```

---

### `get_file()`

Get metadata for a specific file.

```python
async def get_file(
    file_id: str,
    provider: Optional[str] = None
) -> FileMetadata
```

**Parameters:**
- `file_id`: File ID returned from `upload_file()`
- `provider`: Provider name (default: auto-detect from file_id)

**Returns:** `FileMetadata`
```python
FileMetadata(
    file_id="file_abc123",
    provider="anthropic",
    filename="data.csv",
    size_bytes=12345,
    created_at=datetime(...),
    purpose="code_execution",
    status="ready",
    metadata={}
)
```

**Example:**

```python
metadata = await service.get_file("file_abc123")
print(f"File: {metadata.filename}, Status: {metadata.status}")
```

---

### `delete_file()`

Delete an uploaded file.

```python
async def delete_file(
    file_id: str,
    provider: Optional[str] = None
) -> bool
```

**Parameters:**
- `file_id`: File ID to delete
- `provider`: Provider name (default: auto-detect from file_id)

**Returns:** `True` if successful

**Example:**

```python
deleted = await service.delete_file("file_abc123")
assert deleted
```

---

## Provider Comparison

Different providers implement file uploads differently. LLMRing abstracts these differences while preserving provider-specific capabilities.

| Feature | Anthropic | OpenAI | Google |
|---------|-----------|--------|--------|
| **Upload Method** | Files API | Files API | Context Caching |
| **Reference ID** | `file_` prefix | `file-` prefix | `cachedContents/` prefix |
| **Max File Size** | 500 MB | 512 MB | N/A (text-only) |
| **Storage Limit** | 100 GB/org | Per account | N/A |
| **TTL** | Permanent | Permanent | Configurable (default 60 min) |
| **File Types** | Any | Any | Text only (UTF-8) |
| **Use Case** | Code execution, analysis | Assistants API | Cost optimization |
| **Works with Chat** | ✅ Yes | ❌ No (Assistants only) | ✅ Yes |

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
- Google doesn't have discrete "files" - `upload_file()` creates a cache
- File content converted to text and cached
- Binary files will raise `ValueError` about UTF-8 encoding

---

## Complete Examples

### Example 1: Anthropic Code Execution with CSV

Upload a dataset and analyze it with Claude's code execution tool:

```python
from llmring import LLMRing, LLMRequest, Message

async with LLMRing() as service:
    # Upload dataset
    file_resp = await service.upload_file(
        "sales_data.csv",
        purpose="code_execution",
        provider="anthropic"
    )

    print(f"Uploaded: {file_resp.file_id} ({file_resp.size_bytes} bytes)")

    # Analyze with code execution
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Analyze this sales data and calculate total revenue by quarter"
        )],
        files=[file_resp.file_id],
        tools=[{"type": "code_execution"}],
        max_tokens=4000
    )

    response = await service.chat(request)
    print(response.content)

    # Clean up
    await service.delete_file(file_resp.file_id, provider="anthropic")
```

---

### Example 2: Multiple Files with Anthropic

Reference multiple files in a single request:

```python
async with LLMRing() as service:
    # Upload multiple files
    file1 = await service.upload_file("q1_sales.csv", purpose="code_execution")
    file2 = await service.upload_file("q2_sales.csv", purpose="code_execution")
    file3 = await service.upload_file("q3_sales.csv", purpose="code_execution")

    # Analyze all files together
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Compare quarterly sales trends across all three files"
        )],
        files=[file1.file_id, file2.file_id, file3.file_id],
        tools=[{"type": "code_execution"}],
        max_tokens=4000
    )

    response = await service.chat(request)
    print(response.content)

    # Clean up
    for file_id in [file1.file_id, file2.file_id, file3.file_id]:
        await service.delete_file(file_id)
```

---

### Example 3: OpenAI File Management

Upload and manage files for OpenAI (note: use with Assistants API, not Chat):

```python
from llmring import LLMRing

async with LLMRing() as service:
    # Upload file
    file_resp = await service.upload_file(
        "knowledge_base.txt",
        purpose="assistant",
        provider="openai"
    )

    print(f"Uploaded to OpenAI: {file_resp.file_id}")

    # List all OpenAI files
    files = await service.list_files(provider="openai")
    for file in files:
        print(f"- {file.filename}: {file.file_id} ({file.purpose})")

    # Get file metadata
    metadata = await service.get_file(file_resp.file_id, provider="openai")
    print(f"Status: {metadata.status}")

    # Delete file
    deleted = await service.delete_file(file_resp.file_id, provider="openai")
    print(f"Deleted: {deleted}")
```

**Important:** To use uploaded files with OpenAI, use the Assistants API via raw SDK access:

```python
async with LLMRing() as service:
    # Upload file
    file_resp = await service.upload_file(
        "document.pdf",
        purpose="assistant",
        provider="openai"
    )

    # Access OpenAI SDK directly
    openai_client = service.get_provider("openai").client

    # Create assistant with file
    assistant = await openai_client.beta.assistants.create(
        name="Document Analyst",
        instructions="Analyze documents",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_stores": [{
                    "file_ids": [file_resp.file_id]
                }]
            }
        }
    )

    # Use assistant (see OpenAI Assistants API docs)
```

---

### Example 4: Google Context Caching Workflow

Cache large documents for cost optimization:

```python
async with LLMRing() as service:
    # Upload large document (creates cache)
    file_resp = await service.upload_file(
        "legal_document.txt",  # Must be text file
        purpose="cache",
        provider="google",
        ttl_seconds=7200,  # 2 hour cache
        model="gemini-2.5-flash"
    )

    print(f"Cached: {file_resp.file_id}")
    print(f"Expires: {file_resp.metadata.get('expire_time')}")

    # Use cached content in multiple requests (90% cost savings!)
    for question in [
        "Summarize the main points",
        "What are the legal obligations?",
        "Who are the parties involved?"
    ]:
        request = LLMRequest(
            model="google:gemini-2.5-flash",
            messages=[Message(role="user", content=question)],
            files=[file_resp.file_id]  # Reference cached content
        )

        response = await service.chat(request)
        print(f"\nQ: {question}")
        print(f"A: {response.content}")

    # Cache auto-expires after TTL, or delete manually
    await service.delete_file(file_resp.file_id, provider="google")
```

---

### Example 5: Streaming with Files

Stream responses while using uploaded files:

```python
async with LLMRing() as service:
    # Upload file
    file_resp = await service.upload_file(
        "data.csv",
        purpose="code_execution"
    )

    # Stream response
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(
            role="user",
            content="Briefly analyze this data"
        )],
        files=[file_resp.file_id],
        tools=[{"type": "code_execution"}]
    )

    print("Response: ", end="", flush=True)
    async for chunk in service.chat_stream(request):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()

    # Clean up
    await service.delete_file(file_resp.file_id)
```

---

## Error Handling

### Common Errors and Solutions

#### FileSizeError

```python
from llmring.exceptions import FileSizeError

try:
    file_resp = await service.upload_file(
        "huge_file.csv",
        purpose="code_execution"
    )
except FileSizeError as e:
    print(f"File too large: {e.file_size} bytes")
    print(f"Maximum allowed: {e.max_size} bytes")
    print(f"Provider: {e.provider}")
    # Solution: Split file or compress it
```

**Provider Limits:**
- Anthropic: 500 MB
- OpenAI: 512 MB
- Google: N/A (text-only, token-based limits)

---

#### InvalidFileFormatError

```python
from llmring.exceptions import InvalidFileFormatError

try:
    # Google only accepts text files
    file_resp = await service.upload_file(
        "image.png",
        purpose="cache",
        provider="google"
    )
except ValueError as e:
    print(f"Invalid format: {e}")
    # Google requires UTF-8 text files
```

**Solution:** Convert binary files to text or use different provider.

---

#### FileAccessError

```python
from llmring.exceptions import FileAccessError
from pathlib import Path

file_path = "data.csv"
if not Path(file_path).exists():
    print(f"File not found: {file_path}")
else:
    try:
        file_resp = await service.upload_file(file_path)
    except FileAccessError as e:
        print(f"Cannot read file: {e}")
```

---

#### OpenAI Chat Completions Error

```python
# OpenAI files don't work with Chat Completions API
request = LLMRequest(
    model="openai:gpt-4o",
    messages=[Message(role="user", content="Analyze this")],
    files=["file-abc123"]  # Will raise ValueError
)

try:
    response = await service.chat(request)
except ValueError as e:
    print(e)
    # "Chat Completions API does not support file uploads.
    #  Use Assistants API instead."
```

**Solution:** Use Assistants API with direct SDK access (see Example 3).

---

#### ProviderResponseError

```python
from llmring.exceptions import ProviderResponseError

try:
    # Invalid file_id
    metadata = await service.get_file("file_invalid_id")
except ProviderResponseError as e:
    print(f"Provider error: {e}")
    # Handle API errors (not found, permission denied, etc.)
```

---

## Best Practices

### 1. Clean Up Files

Always delete files when done to avoid storage costs:

```python
file_resp = await service.upload_file("data.csv")
try:
    # Use file
    response = await service.chat(request)
finally:
    # Always clean up
    await service.delete_file(file_resp.file_id)
```

---

### 2. Choose the Right Provider

Match provider to your use case:

```python
# Code execution → Anthropic
file_resp = await service.upload_file(
    "dataset.csv",
    purpose="code_execution",
    provider="anthropic"
)

# Cost optimization for repeated access → Google
file_resp = await service.upload_file(
    "large_doc.txt",
    purpose="cache",
    provider="google",
    ttl_seconds=3600
)

# Knowledge base (Assistants) → OpenAI
file_resp = await service.upload_file(
    "knowledge.txt",
    purpose="assistant",
    provider="openai"
)
```

---

### 3. Validate Files Before Upload

Check file size and format:

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

# Upload
file_resp = await service.upload_file(str(file_path))
```

---

### 4. Reuse Files Across Requests

Upload once, use many times:

```python
# Upload once
file_resp = await service.upload_file("analysis_data.csv")

# Use in multiple requests
questions = [
    "What's the total revenue?",
    "What's the average transaction size?",
    "Show me the top 10 customers"
]

for question in questions:
    request = LLMRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=[Message(role="user", content=question)],
        files=[file_resp.file_id],
        tools=[{"type": "code_execution"}]
    )
    response = await service.chat(request)
    print(f"Q: {question}\nA: {response.content}\n")

# Clean up once
await service.delete_file(file_resp.file_id)
```

---

### 5. Handle Provider-Specific Quirks

Google requires text files and sufficient tokens:

```python
# Google caching requires minimum 1024 tokens (Flash) or 4096 (Pro)
import os

file_path = "document.txt"
file_size = os.path.getsize(file_path)

# Rough estimate: 1 token ≈ 4 characters
estimated_tokens = file_size // 4

if estimated_tokens < 1024:
    print("Warning: File may not meet Google's minimum token requirement")
    # Use a different approach or provider
else:
    file_resp = await service.upload_file(
        file_path,
        purpose="cache",
        provider="google"
    )
```

---

### 6. Use File-Like Objects for In-Memory Data

Upload without writing to disk:

```python
import io
import csv

# Create CSV in memory
buffer = io.BytesIO()
writer = csv.writer(io.TextIOWrapper(buffer, encoding='utf-8'))
writer.writerow(['Name', 'Value'])
writer.writerow(['Item 1', '100'])
writer.writerow(['Item 2', '200'])

# Upload from memory
buffer.seek(0)
file_resp = await service.upload_file(
    buffer,
    purpose="code_execution",
    filename="data.csv"
)
```

---

## Troubleshooting

### File Upload Fails Silently

**Problem:** Upload completes but file not usable

**Solution:** Check file_id format matches provider:

```python
file_resp = await service.upload_file("data.csv")

# Verify correct provider
if file_resp.provider == "anthropic":
    assert file_resp.file_id.startswith("file_")
elif file_resp.provider == "openai":
    assert file_resp.file_id.startswith("file-")
elif file_resp.provider == "google":
    assert file_resp.file_id.startswith("cachedContents/")
```

---

### Google Upload Fails with "UTF-8" Error

**Problem:** Binary file uploaded to Google

**Solution:** Google only accepts text files

```python
# ❌ Will fail
file_resp = await service.upload_file(
    "image.png",
    provider="google"
)

# ✅ Use text files only
file_resp = await service.upload_file(
    "document.txt",
    provider="google"
)
```

---

### OpenAI Files Don't Work in Chat

**Problem:** File uploaded but chat fails

**Solution:** OpenAI files only work with Assistants API:

```python
# ❌ Won't work
request = LLMRequest(
    model="openai:gpt-4o",
    files=["file-abc123"]  # Error!
)

# ✅ Use Assistants API
openai_client = service.get_provider("openai").client
assistant = await openai_client.beta.assistants.create(...)
```

---

### File Not Found After Upload

**Problem:** File deleted or expired

**Solution:** Check TTL for Google caches:

```python
# Google caches expire
file_resp = await service.upload_file(
    "doc.txt",
    provider="google",
    ttl_seconds=3600  # Expires after 1 hour
)

# Check expiration
print(f"Expires: {file_resp.metadata.get('expire_time')}")
```

---

## Related Documentation

- [API Reference](api-reference.md) - Core LLMRing API
- [Provider Guide](providers.md) - Provider-specific features
- [File Utilities](file-utilities.md) - Vision and multimodal capabilities
- [Structured Output](structured-output.md) - JSON schema support
