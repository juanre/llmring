# Google File API Implementation Design

**Date:** 2025-11-10
**Status:** Approved
**Author:** Claude

## Problem Statement

The Google provider currently uses Context Caching for file uploads, which only supports text files (UTF-8). This is incorrect because:

1. Google has a proper File API that supports binary files (PDFs, images, audio, video) up to 2GB
2. Context Caching is a separate feature for prompt caching (90% cost savings), not file uploads
3. This creates an inconsistent user experience - binary files work with Anthropic/OpenAI but fail with Google

## Goals

1. **Provider parity**: Binary file support across all providers (Anthropic, OpenAI, Google)
2. **Proper separation**: File API for uploads, Context Caching for prompt optimization
3. **Provider-agnostic API**: Users write identical code regardless of provider
4. **Automatic expiration handling**: Transparent re-upload when Google files expire (48h)

## Architecture Overview

### Current Implementation (Incorrect)

```python
# google_api.py upload_file() - WRONG
async def upload_file(file, purpose="cache", ttl_seconds=3600):
    # Reads file as UTF-8 text only
    content = open(file_path, "r", encoding="utf-8").read()
    # Creates context cache (not a file upload!)
    cache_resp = self.client.caches.create(model=model, config=config)
    return FileUploadResponse(file_id=cache_resp.name)  # Returns "cachedContents/xxx"
```

**Problems**:
- Binary files fail with UnicodeDecodeError
- Returns cache ID not file ID
- Conflates two separate features

### New Implementation (Correct)

```python
# google_api.py upload_file() - CORRECT
async def upload_file(file, purpose="analysis", filename=None):
    # Upload binary file using File API
    uploaded_file = self.client.files.upload(file=file_path)

    # Store metadata for expiration tracking
    self._uploaded_files[uploaded_file.name] = UploadedFileInfo(
        file_name=uploaded_file.name,
        expiration_time=parse_iso(uploaded_file.expiration_time),
        local_path=str(file_path)
    )

    return FileUploadResponse(
        file_id=uploaded_file.name,  # "files/xxx"
        provider="google",
        size_bytes=uploaded_file.size_bytes,
        ...
    )
```

## Component Design

### 1. GoogleProvider Internal State

Add new dataclass and storage:

```python
@dataclass
class UploadedFileInfo:
    """Track Google uploaded files for expiration management."""
    file_name: str  # Google file name (e.g., "files/abc123")
    expiration_time: datetime  # When file expires (48h from upload)
    local_path: str  # Original file path for re-upload

class GoogleProvider:
    def __init__(self, ...):
        # Existing fields...
        self._uploaded_files: Dict[str, UploadedFileInfo] = {}  # NEW
```

### 2. File Upload Method

Replace entire `upload_file()` implementation:

```python
async def upload_file(
    self,
    file: Union[str, Path, BinaryIO],
    purpose: str = "analysis",  # Not used by Google, for API consistency
    filename: Optional[str] = None,
    **kwargs,
) -> FileUploadResponse:
    """Upload file to Google File API (supports binary files up to 2GB)."""

    # 1. Handle file path vs file-like object
    if isinstance(file, (str, Path)):
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check 2GB limit
        MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise FileSizeError(...)

        actual_filename = filename or file_path.name
        local_path = str(file_path.absolute())

        # Upload using genai SDK
        uploaded_file = self.client.files.upload(file=str(file_path))
    else:
        # File-like object
        actual_filename = filename or "uploaded_file"
        local_path = None  # Can't re-upload file-like objects
        uploaded_file = self.client.files.upload(file=file)

    # 2. Parse expiration time
    expiration_time = datetime.fromisoformat(
        uploaded_file.expiration_time.replace("Z", "+00:00")
    )

    # 3. Store metadata for expiration tracking
    self._uploaded_files[uploaded_file.name] = UploadedFileInfo(
        file_name=uploaded_file.name,
        expiration_time=expiration_time,
        local_path=local_path
    )

    # 4. Return standardized response
    return FileUploadResponse(
        file_id=uploaded_file.name,
        provider="google",
        filename=actual_filename,
        size_bytes=int(uploaded_file.size_bytes),
        created_at=datetime.fromisoformat(
            uploaded_file.create_time.replace("Z", "+00:00")
        ),
        purpose=purpose,
        metadata={
            "mime_type": uploaded_file.mime_type,
            "expiration_time": uploaded_file.expiration_time,
            "state": uploaded_file.state.name if uploaded_file.state else "ACTIVE",
        },
    )
```

### 3. Expiration Detection & Re-upload

Add helper method called by chat():

```python
async def _ensure_file_available(self, file_id: str) -> Any:
    """
    Ensure file is available, re-uploading if expired.

    Args:
        file_id: Google file ID (e.g., "files/abc123")

    Returns:
        Google file object ready for use in contents array
    """
    # Check if we have metadata for this file
    if file_id not in self._uploaded_files:
        # File uploaded elsewhere or before provider restart
        # Try to get it from Google
        try:
            return self.client.files.get(name=file_id)
        except Exception:
            raise ValueError(
                f"File '{file_id}' not found. It may have expired or been deleted."
            )

    file_info = self._uploaded_files[file_id]

    # Check expiration
    if datetime.now(timezone.utc) >= file_info.expiration_time:
        # File expired - re-upload if we have the local path
        if not file_info.local_path:
            raise ValueError(
                f"File '{file_id}' expired and cannot be re-uploaded "
                "(original was uploaded from file-like object)"
            )

        logger.info(f"Re-uploading expired file: {file_info.local_path}")

        # Re-upload
        new_upload = await self.upload_file(file=file_info.local_path)

        # Update mapping - old file_id now points to new file
        # This preserves llmring file_id → Google file_id mapping in service.py
        self._uploaded_files[file_id] = self._uploaded_files[new_upload.file_id]
        del self._uploaded_files[new_upload.file_id]

        return self.client.files.get(name=file_id)

    # File still valid
    return self.client.files.get(name=file_id)
```

### 4. Chat Integration

Modify `_chat_non_streaming()` and `_stream_chat()`:

```python
async def _chat_non_streaming(..., files: Optional[List[str]] = None, ...):
    # Remove the incorrect cached_content logic (lines 343-353 currently)

    # Convert messages to Google format
    google_messages = self._convert_messages_to_google_format(messages)

    # Handle files - get file objects and include in contents
    contents = []

    # Add messages as content
    for msg in google_messages:
        contents.append(msg)  # Simplified - actual conversion more complex

    # Add files to contents array
    if files:
        for file_id in files:
            # Ensure file is available (handles expiration)
            file_obj = await self._ensure_file_available(file_id)
            contents.append(file_obj)  # Google SDK accepts file objects directly

    # Make API call with contents array
    response = await self.client.models.generate_content(
        model=model,
        contents=contents,
        ...
    )
```

### 5. File Deletion

Update `delete_file()`:

```python
async def delete_file(self, file_id: str) -> bool:
    """Delete uploaded file."""
    try:
        self.client.files.delete(name=file_id)

        # Remove from tracking
        if file_id in self._uploaded_files:
            del self._uploaded_files[file_id]

        return True
    except Exception as e:
        await self._error_handler.handle_error(e, "files")
```

### 6. File Listing

Update `list_files()`:

```python
async def list_files(
    self, purpose: Optional[str] = None, limit: int = 100
) -> List[FileMetadata]:
    """List uploaded files."""
    try:
        # Use files API not caches API
        files_list = self.client.files.list()

        result = []
        for file_obj in files_list:
            # Parse file metadata
            result.append(
                FileMetadata(
                    file_id=file_obj.name,
                    provider="google",
                    filename=file_obj.display_name or file_obj.name.split("/")[-1],
                    size_bytes=int(file_obj.size_bytes),
                    created_at=datetime.fromisoformat(
                        file_obj.create_time.replace("Z", "+00:00")
                    ),
                    purpose="file",  # Google doesn't have purpose field
                    status=file_obj.state.name if file_obj.state else "ACTIVE",
                    metadata={
                        "mime_type": file_obj.mime_type,
                        "expiration_time": file_obj.expiration_time,
                    },
                )
            )

            if len(result) >= limit:
                break

        return result
    except Exception as e:
        await self._error_handler.handle_error(e, "files")
```

## Provider Abstraction Layer

The service layer (`service.py`) remains unchanged:

```python
# service.py - UNCHANGED
async def _ensure_file_uploaded(self, file_id: str, provider_type: str) -> str:
    """
    Ensure file is uploaded to provider, return provider file_id.

    Works identically for all providers (Anthropic, OpenAI, Google).
    """
    # Get registered file
    reg_file = self._registered_files[file_id]

    # Check if already uploaded
    if provider_type in reg_file.uploads:
        return reg_file.uploads[provider_type].provider_file_id

    # Upload to provider
    provider = self.get_provider(provider_type)
    upload_response = await provider.upload_file(
        file=reg_file.file_path,
        purpose="file",
    )

    # Cache provider file_id
    reg_file.uploads[provider_type] = ProviderFileUpload(
        provider_file_id=upload_response.file_id,
        uploaded_at=datetime.now(),
        metadata=upload_response.metadata,
    )

    return upload_response.file_id
```

## Context Caching (Separate Feature)

Context caching remains completely separate:

```python
# Users control caching via cache parameter
response = await service.chat(
    LLMRequest(
        model="google:gemini-2.5-flash",
        messages=[Message(role="user", content="Long prompt..." * 1000)],
        cache={"enable": True}  # Separate from files!
    )
)
```

Implementation in `google_api.py` uses cache control structures in message metadata (similar to Anthropic).

## Testing Strategy

### Unit Tests

1. **test_google_file_upload_binary.py**
   - Upload PDF, image, verify correct file_id format
   - Verify metadata stored in `_uploaded_files`
   - Check FileSizeError for files > 2GB

2. **test_google_file_expiration.py**
   - Mock expired file, verify re-upload triggered
   - Verify error when file-like object expires
   - Check expiration_time parsing

3. **test_google_chat_with_files.py**
   - Upload file, use in chat, verify contents array structure
   - Multiple files in one request
   - Verify files separate from cache parameter

### Integration Tests

Use existing test infrastructure (`tests/test_google_files.py`):

```python
@pytest.mark.asyncio
async def test_google_upload_pdf():
    """Test PDF upload using File API."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Upload PDF
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.pdf",
            purpose="analysis"
        )

        assert upload_response.file_id.startswith("files/")
        assert upload_response.provider == "google"
        assert upload_response.size_bytes > 0

        # Use in chat
        response = await provider.chat(
            messages=[Message(role="user", content="Summarize this PDF")],
            model="gemini-2.5-flash",
            files=[upload_response.file_id]
        )

        assert response.content

        # Cleanup
        await provider.delete_file(upload_response.file_id)
    finally:
        await provider.aclose()
```

## Migration & Backward Compatibility

**Breaking change**: Existing code using Google files via cached_content will break.

**Mitigation**:
1. Update documentation to show correct File API usage
2. Keep cache parameter for prompt caching (separate feature)
3. Version bump (minor version since it's a bug fix making things work correctly)

**Migration path**:
```python
# OLD (broken for binary files)
file_id = service.register_file("image.png")  # Would fail with UTF-8 error

# NEW (works correctly)
file_id = service.register_file("image.png")  # Works with File API
```

No code changes required for users - same API, now works correctly.

## Success Criteria

1. ✅ Binary files (PDF, PNG, etc.) upload successfully to Google
2. ✅ Files work in chat requests via contents array
3. ✅ Automatic re-upload on expiration (48h)
4. ✅ Provider-agnostic user API (same code for all providers)
5. ✅ Context caching remains separate and functional
6. ✅ All tests pass
7. ✅ Documentation updated to reflect correct behavior

## References

- [Google File API Documentation](https://ai.google.dev/gemini-api/docs/files)
- [Google Gen AI SDK](https://googleapis.github.io/python-genai/)
- Anthropic provider implementation (`anthropic_api.py:795-890`)
- Service layer abstraction (`service.py:378-454`)
