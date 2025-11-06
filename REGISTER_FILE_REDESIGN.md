# Register File Redesign - Implementation Plan

**Status:** Ready to implement
**Goal:** Make files truly provider-agnostic with lazy uploads and automatic caching

---

## Design Overview

### Current Problem
Files are bound to specific providers - can't reuse across providers.

### Solution
- `register_file()` - Remembers file, returns llmring-managed ID (no uploads)
- Lazy uploads - Upload to provider only when needed (during chat())
- Per-provider caching - Track uploads separately for each provider
- Staleness detection - Re-hash on each use, invalidate cache if changed

---

## Data Structures

### RegisteredFile Schema

```python
@dataclass
class RegisteredFile:
    """Internal representation of a registered file."""
    id: str                    # llmring-managed UUID
    file_path: str             # Path to file on disk
    content_hash: str          # SHA256 hash for staleness detection
    uploads: Dict[str, ProviderFileUpload]  # Per-provider upload info
    registered_at: datetime

@dataclass
class ProviderFileUpload:
    """Provider-specific upload information."""
    provider_file_id: str      # Provider's file ID (file_abc, file-xyz, cachedContents/...)
    uploaded_at: datetime
    metadata: Dict[str, Any]   # Provider-specific metadata (TTL, etc.)
```

### Service State

```python
class LLMRing:
    def __init__(self, ...):
        # ... existing fields ...
        self._registered_files: Dict[str, RegisteredFile] = {}
```

---

## API Design

### New Methods

```python
async def register_file(
    self,
    file: Union[str, Path],
    file_id: Optional[str] = None  # Allow custom ID for testing
) -> str:
    """
    Register a file for use across providers.

    Does NOT upload to any provider yet - uploads happen lazily
    when the file is first used in a chat request with each provider.

    Args:
        file: Path to file
        file_id: Optional custom ID (auto-generated UUID if not provided)

    Returns:
        llmring-managed file ID (e.g., "llmring-file-uuid-123")

    Example:
        file_id = await ring.register_file("data.csv")

        # Use with any provider
        await ring.chat(LLMRequest(
            model="anthropic:claude-sonnet",
            files=[file_id]  # Uploads to Anthropic first time
        ))

        await ring.chat(LLMRequest(
            model="openai:gpt-4o",
            files=[file_id]  # Uploads to OpenAI first time
        ))
    """
    ...

async def deregister_file(self, file_id: str) -> bool:
    """
    Deregister a file and clean up from all providers.

    Args:
        file_id: llmring file ID

    Returns:
        True if file was deregistered, False if not found

    Note:
        Deletes file from all providers it was uploaded to.
    """
    ...

async def list_registered_files(self) -> List[Dict[str, Any]]:
    """
    List all registered files in this service instance.

    Returns:
        List of file info dicts with:
        - id: llmring file ID
        - file_path: Path to file
        - uploads: Dict of provider uploads
        - registered_at: Registration timestamp
    """
    ...
```

### Modified Methods

```python
async def chat(self, request: LLMRequest, profile: Optional[str] = None) -> LLMResponse:
    """
    Extended to handle lazy file uploads.

    For each file in request.files:
    1. Validate file_id exists in _registered_files
    2. Re-hash file to detect changes
    3. If hash changed: clear uploads cache for that file
    4. Check if uploaded to request's provider
    5. If not: upload now and cache provider file_id
    6. Replace llmring file_id with provider file_id in request
    """
    ...
```

---

## Implementation Steps

### Step 1: Add Data Structures

**File:** `src/llmring/schemas.py`

Add internal schemas (not exported):
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class ProviderFileUpload:
    """Provider-specific file upload information."""
    provider_file_id: str
    uploaded_at: datetime
    metadata: Dict[str, Any] = None

@dataclass
class RegisteredFile:
    """Internal representation of a registered file."""
    id: str
    file_path: str
    content_hash: str
    uploads: Dict[str, ProviderFileUpload]
    registered_at: datetime
```

### Step 2: Add Helper Methods to Service

**File:** `src/llmring/service.py`

```python
def _hash_file(self, file_path: str) -> str:
    """Compute SHA256 hash of file content."""
    import hashlib
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

async def _ensure_file_uploaded(
    self,
    file_id: str,
    provider_type: str,
    model_name: str
) -> str:
    """
    Ensure file is uploaded to provider, return provider file_id.

    1. Get registered file
    2. Re-hash to detect changes
    3. If changed: clear uploads cache
    4. If not uploaded to this provider: upload now
    5. Return provider file_id
    """
    ...
```

### Step 3: Implement register_file()

```python
async def register_file(
    self,
    file: Union[str, Path],
    file_id: Optional[str] = None
) -> str:
    """Register file for cross-provider use."""
    import uuid
    from pathlib import Path

    file_path = str(Path(file).absolute())

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Generate ID
    if not file_id:
        file_id = f"llmring-file-{uuid.uuid4()}"

    # Hash file content
    content_hash = self._hash_file(file_path)

    # Store registration
    self._registered_files[file_id] = RegisteredFile(
        id=file_id,
        file_path=file_path,
        content_hash=content_hash,
        uploads={},  # Empty - uploads happen lazily
        registered_at=datetime.now()
    )

    return file_id
```

### Step 4: Implement Lazy Upload in chat()

**Location:** In `chat()` method, before calling provider.chat()

```python
# Handle file lazy uploads
if request.files:
    # Map llmring file IDs to provider file IDs
    provider_file_ids = []

    for llmring_file_id in request.files:
        provider_file_id = await self._ensure_file_uploaded(
            llmring_file_id,
            provider_type,
            model_name
        )
        provider_file_ids.append(provider_file_id)

    # Replace request.files with provider file IDs
    adapted_request.files = provider_file_ids
```

### Step 5: Implement _ensure_file_uploaded()

```python
async def _ensure_file_uploaded(
    self,
    file_id: str,
    provider_type: str,
    model_name: str
) -> str:
    """Ensure file is uploaded to provider."""

    # Get registered file
    if file_id not in self._registered_files:
        raise ValueError(
            f"File ID '{file_id}' not registered. "
            f"Use register_file() to register files before using them."
        )

    reg_file = self._registered_files[file_id]

    # Re-hash to detect changes
    current_hash = self._hash_file(reg_file.file_path)
    if current_hash != reg_file.content_hash:
        # File changed - invalidate all uploads
        logger.debug(f"File {file_id} changed on disk, clearing upload cache")
        reg_file.uploads = {}
        reg_file.content_hash = current_hash

    # Check if already uploaded to this provider
    if provider_type in reg_file.uploads:
        provider_upload = reg_file.uploads[provider_type]
        logger.debug(
            f"Using cached upload for {file_id} on {provider_type}: "
            f"{provider_upload.provider_file_id}"
        )
        return provider_upload.provider_file_id

    # Not uploaded yet - upload now
    logger.debug(f"Uploading {file_id} to {provider_type}")

    provider = self.get_provider(provider_type)

    # Determine purpose based on model (alias name if alias, else "file")
    purpose = self._get_purpose_for_model(request.model, provider_type)

    # Upload to provider
    upload_response = await provider.upload_file(
        file=reg_file.file_path,
        purpose=purpose,
        filename=None,
        ttl_seconds=3600  # Default for Google
    )

    # Cache the provider file_id
    reg_file.uploads[provider_type] = ProviderFileUpload(
        provider_file_id=upload_response.file_id,
        uploaded_at=datetime.now(),
        metadata=upload_response.metadata
    )

    return upload_response.file_id
```

### Step 6: Implement deregister_file()

```python
async def deregister_file(self, file_id: str) -> bool:
    """Deregister file and clean up from all providers."""

    if file_id not in self._registered_files:
        return False

    reg_file = self._registered_files[file_id]

    # Delete from all providers it was uploaded to
    for provider_type, upload in reg_file.uploads.items():
        try:
            provider = self.get_provider(provider_type)
            await provider.delete_file(upload.provider_file_id)
            logger.debug(f"Deleted {file_id} from {provider_type}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_id} from {provider_type}: {e}")

    # Remove from registry
    del self._registered_files[file_id]
    return True
```

### Step 7: Remove Old API

Delete:
- `upload_file()` method
- `list_files()` method
- `get_file()` method
- `delete_file()` method
- `_detect_provider_from_file_id()` helper

Keep only:
- `register_file()`
- `deregister_file()`
- `list_registered_files()`
- `_ensure_file_uploaded()` (internal)

---

## Testing Strategy

### New Tests Needed

```python
# tests/test_file_registration.py

async def test_register_file():
    """Test file registration returns llmring ID."""
    ring = LLMRing()
    file_id = await ring.register_file("tests/fixtures/sample.txt")
    assert file_id.startswith("llmring-file-")
    await ring.close()

async def test_lazy_upload_anthropic():
    """Test file uploads to Anthropic on first use."""
    ring = LLMRing()
    file_id = await ring.register_file("tests/fixtures/sample.txt")

    # First use - should upload
    response = await ring.chat(LLMRequest(
        model="anthropic:claude-3-5-haiku-20241022",
        messages=[Message(role="user", content="Summarize")],
        files=[file_id]
    ))

    # Second use - should use cache (verify via debug logs)
    response2 = await ring.chat(LLMRequest(
        model="anthropic:claude-3-5-haiku-20241022",
        messages=[Message(role="user", content="Analyze")],
        files=[file_id]
    ))

    await ring.deregister_file(file_id)
    await ring.close()

async def test_cross_provider_file_usage():
    """Test same file works with multiple providers."""
    ring = LLMRing()
    file_id = await ring.register_file("tests/fixtures/sample.txt")

    # Use with Anthropic
    r1 = await ring.chat(LLMRequest(
        model="anthropic:claude-3-5-haiku-20241022",
        files=[file_id]
    ))

    # Use with OpenAI
    r2 = await ring.chat(LLMRequest(
        model="openai:gpt-4o",
        files=[file_id]
    ))

    # Both should work!
    assert r1.content
    assert r2.content

    await ring.deregister_file(file_id)

async def test_file_staleness_detection():
    """Test that file changes are detected."""
    # Register file
    file_id = await ring.register_file("temp_file.txt")

    # Use it
    await ring.chat(LLMRequest(model="...", files=[file_id]))

    # Modify file
    with open("temp_file.txt", "w") as f:
        f.write("new content")

    # Use again - should detect change and re-upload
    # (verify via debug logs or provider file_id changes)
    await ring.chat(LLMRequest(model="...", files=[file_id]))

async def test_deregister_file():
    """Test cleanup from all providers."""
    file_id = await ring.register_file("sample.txt")

    # Upload to two providers
    await ring.chat(LLMRequest(model="anthropic:...", files=[file_id]))
    await ring.chat(LLMRequest(model="openai:...", files=[file_id]))

    # Deregister - should delete from both
    success = await ring.deregister_file(file_id)
    assert success

    # Verify file_id no longer exists
    with pytest.raises(ValueError, match="not registered"):
        await ring.chat(LLMRequest(model="...", files=[file_id]))
```

---

## Implementation Order

1. **Step 1:** Add data structures (RegisteredFile, ProviderFileUpload)
2. **Step 2:** Implement register_file()
3. **Step 3:** Implement _hash_file() and _ensure_file_uploaded()
4. **Step 4:** Modify chat() to handle lazy uploads
5. **Step 5:** Implement deregister_file() and list_registered_files()
6. **Step 6:** Remove old upload_file/list_files/get_file/delete_file methods
7. **Step 7:** Update ALL tests to use new API
8. **Step 8:** Update ALL documentation

---

## Migration from Current API

### Old API (Remove)
```python
file = await ring.upload_file("data.csv", model="anthropic:claude")
await ring.list_files()
await ring.get_file(file.file_id)
await ring.delete_file(file.file_id)
```

### New API (Implement)
```python
file_id = await ring.register_file("data.csv")
await ring.list_registered_files()  # Lists llmring registrations
# No get_file - use list_registered_files()
await ring.deregister_file(file_id)
```

---

## Provider Layer - NO CHANGES NEEDED

The provider-level methods stay as-is:
- `provider.upload_file()` - Used internally by _ensure_file_uploaded()
- `provider.list_files()` - Still useful for direct provider access
- `provider.get_file()` - Still useful for direct provider access
- `provider.delete_file()` - Used internally by deregister_file()

---

## Benefits of New Design

✅ **Truly provider-agnostic** - One file works everywhere
✅ **Automatic staleness detection** - Re-hashes on every use
✅ **Lazy uploads** - Only upload when needed
✅ **Efficient caching** - Per-provider upload tracking
✅ **Memory efficient** - No file content stored
✅ **Simple API** - register → use → deregister

---

## Testing Requirements

- All tests must use real APIs (no mocks)
- Test cross-provider usage
- Test staleness detection
- Test cleanup
- Test error cases
- All 58 existing file tests must be updated and pass

---

## Documentation Updates

All documentation must be updated:
- docs/file-uploads.md → docs/file-registration.md
- README.md examples
- Website docs
- API reference
- All code examples

---

## Decision Points (Confirmed)

1. **File storage:** Path + hash ✅
2. **Staleness:** Re-hash on every use, clear cache if changed ✅
3. **Google TTL:** Let it fail and re-upload ✅
4. **Cleanup:** Include deregister_file() ✅
5. **Persistence:** In-memory only ✅
6. **File deletion:** Error if file doesn't exist ✅

---

Ready to implement!
