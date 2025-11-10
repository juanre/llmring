# Google File API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Google provider's Context Caching-based file upload with proper File API supporting binary files up to 2GB.

**Architecture:** Add file metadata tracking to GoogleProvider, implement upload_file() using client.files.upload(), handle 48-hour expiration with automatic re-upload, integrate file objects into chat contents array.

**Tech Stack:** google-genai SDK (genai.Client), Python 3.12, pytest for testing

---

## Task 1: Add File Metadata Tracking Structure

**Files:**
- Modify: `src/llmring/providers/google_api.py:1-50`
- Test: `tests/unit/test_google_file_metadata.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_google_file_metadata.py`:

```python
"""Unit tests for Google File API metadata tracking."""
import pytest
from datetime import datetime, timezone
from llmring.providers.google_api import UploadedFileInfo


def test_uploaded_file_info_dataclass():
    """Test UploadedFileInfo dataclass creation."""
    info = UploadedFileInfo(
        file_name="files/abc123",
        expiration_time=datetime.now(timezone.utc),
        local_path="/path/to/file.pdf"
    )

    assert info.file_name == "files/abc123"
    assert info.local_path == "/path/to/file.pdf"
    assert isinstance(info.expiration_time, datetime)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_google_file_metadata.py::test_uploaded_file_info_dataclass -v`

Expected: FAIL with "cannot import name 'UploadedFileInfo'"

**Step 3: Add UploadedFileInfo dataclass**

In `src/llmring/providers/google_api.py`, add after existing imports (around line 13):

```python
from dataclasses import dataclass
```

Then add before the GoogleProvider class definition (around line 37):

```python
@dataclass
class UploadedFileInfo:
    """Track Google uploaded files for expiration management."""
    file_name: str  # Google file name (e.g., "files/abc123")
    expiration_time: datetime  # When file expires (48h from upload)
    local_path: Optional[str]  # Original file path for re-upload (None for file-like objects)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_google_file_metadata.py::test_uploaded_file_info_dataclass -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_file_metadata.py
git commit -m "feat(google): add UploadedFileInfo dataclass for file metadata tracking"
```

---

## Task 2: Add File Storage to GoogleProvider

**Files:**
- Modify: `src/llmring/providers/google_api.py:42-92`
- Test: `tests/unit/test_google_provider_init.py`

**Step 1: Write the failing test**

Create `tests/unit/test_google_provider_init.py`:

```python
"""Unit tests for GoogleProvider initialization."""
import pytest
import os
from llmring.providers.google_api import GoogleProvider


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)
def test_google_provider_has_uploaded_files_dict():
    """Test GoogleProvider initializes with empty uploaded files dict."""
    provider = GoogleProvider()

    assert hasattr(provider, "_uploaded_files")
    assert isinstance(provider._uploaded_files, dict)
    assert len(provider._uploaded_files) == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_google_provider_init.py::test_google_provider_has_uploaded_files_dict -v`

Expected: FAIL with "AttributeError: 'GoogleProvider' object has no attribute '_uploaded_files'"

**Step 3: Add _uploaded_files to __init__**

In `src/llmring/providers/google_api.py`, in the `__init__` method (around line 88), add before the circuit breaker initialization:

```python
        # File upload tracking for expiration management
        self._uploaded_files: Dict[str, UploadedFileInfo] = {}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_google_provider_init.py::test_google_provider_has_uploaded_files_dict -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_provider_init.py
git commit -m "feat(google): initialize file upload tracking storage"
```

---

## Task 3: Implement Binary File Upload

**Files:**
- Modify: `src/llmring/providers/google_api.py:1261-1376`
- Test: `tests/unit/test_google_upload_file.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_google_upload_file.py`:

```python
"""Unit tests for Google File API upload."""
import pytest
import os
from pathlib import Path
from llmring.providers.google_api import GoogleProvider
from llmring.exceptions import FileSizeError


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)
@pytest.mark.asyncio
async def test_upload_pdf_file():
    """Test uploading a PDF file using File API."""
    provider = GoogleProvider()

    try:
        # Upload a test file
        response = await provider.upload_file(
            file="tests/fixtures/sample.pdf",
            purpose="analysis"
        )

        # Verify response structure
        assert response.file_id.startswith("files/")
        assert response.provider == "google"
        assert response.size_bytes > 0
        assert response.metadata.get("mime_type") == "application/pdf"

        # Verify file tracked internally
        assert response.file_id in provider._uploaded_files
        file_info = provider._uploaded_files[response.file_id]
        assert file_info.file_name == response.file_id
        assert file_info.local_path.endswith("sample.pdf")

        # Cleanup
        await provider.delete_file(response.file_id)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_file_size_limit():
    """Test that files over 2GB raise FileSizeError."""
    provider = GoogleProvider(api_key="fake-key")

    # Mock a file that's too large
    from unittest.mock import Mock, patch

    large_file_path = Path("/tmp/huge.bin")

    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'stat') as mock_stat:
        mock_stat.return_value = Mock(st_size=3 * 1024 * 1024 * 1024)  # 3GB

        with pytest.raises(FileSizeError) as exc_info:
            await provider.upload_file(file=large_file_path)

        assert "2 * 1024 * 1024 * 1024" in str(exc_info.value) or "2GB" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_google_upload_file.py::test_upload_pdf_file -v`

Expected: FAIL (current implementation only handles text files)

**Step 3: Replace upload_file() implementation**

In `src/llmring/providers/google_api.py`, replace the entire `upload_file()` method (lines 1261-1376) with:

```python
    async def upload_file(
        self,
        file: Union[str, Path, BinaryIO],
        purpose: str = "analysis",
        filename: Optional[str] = None,
        **kwargs,
    ) -> FileUploadResponse:
        """
        Upload file to Google File API (supports binary files up to 2GB).

        Args:
            file: File path, Path object, or file-like object
            purpose: Purpose of the file (not used by Google, for API consistency)
            filename: Optional filename (required for file-like objects)
            **kwargs: Additional provider-specific parameters

        Returns:
            FileUploadResponse with file_id, size, etc.

        Raises:
            FileSizeError: If file exceeds 2GB limit
            FileNotFoundError: If file path doesn't exist
            ProviderResponseError: If upload fails
        """
        # Google File API max size is 2GB
        MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

        # Handle file path or file-like object
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                from llmring.exceptions import FileSizeError
                raise FileSizeError(
                    f"File size {file_size} bytes exceeds Google limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="google",
                    filename=str(file_path),
                )

            actual_filename = filename or file_path.name
            local_path = str(file_path.absolute())

            # Upload using genai SDK
            try:
                # Run synchronous SDK call in thread pool
                loop = asyncio.get_event_loop()

                def _upload():
                    return self.client.files.upload(file=str(file_path))

                uploaded_file = await loop.run_in_executor(None, _upload)
            except Exception as e:
                await self._error_handler.handle_error(e, "files")

        else:
            # File-like object
            if filename is None:
                raise ValueError("filename parameter is required for file-like objects")

            # Read content to check size
            current_pos = file.tell() if hasattr(file, "tell") else 0
            file_content = file.read()
            file_size = len(file_content)

            # Reset file position if possible
            if hasattr(file, "seek"):
                file.seek(current_pos)

            # Check size
            if file_size > MAX_FILE_SIZE:
                from llmring.exceptions import FileSizeError
                raise FileSizeError(
                    f"File size {file_size} bytes exceeds Google limit of {MAX_FILE_SIZE} bytes",
                    file_size=file_size,
                    max_size=MAX_FILE_SIZE,
                    provider="google",
                    filename=filename,
                )

            actual_filename = filename
            local_path = None  # Can't re-upload file-like objects

            # Upload using genai SDK
            try:
                loop = asyncio.get_event_loop()

                def _upload():
                    # Reset position again before upload
                    if hasattr(file, "seek"):
                        file.seek(current_pos)
                    return self.client.files.upload(file=file)

                uploaded_file = await loop.run_in_executor(None, _upload)
            except Exception as e:
                await self._error_handler.handle_error(e, "files")

        # Parse expiration time
        try:
            expiration_str = uploaded_file.expiration_time
            if isinstance(expiration_str, str):
                expiration_time = datetime.fromisoformat(expiration_str.replace("Z", "+00:00"))
            else:
                # Might be a datetime object already
                expiration_time = expiration_str
        except Exception:
            # Fallback: 48 hours from now
            from datetime import timezone, timedelta
            expiration_time = datetime.now(timezone.utc) + timedelta(hours=48)

        # Store metadata for expiration tracking
        self._uploaded_files[uploaded_file.name] = UploadedFileInfo(
            file_name=uploaded_file.name,
            expiration_time=expiration_time,
            local_path=local_path,
        )

        # Parse created_at time
        try:
            create_time_str = uploaded_file.create_time
            if isinstance(create_time_str, str):
                created_at = datetime.fromisoformat(create_time_str.replace("Z", "+00:00"))
            else:
                created_at = create_time_str
        except Exception:
            created_at = datetime.now()

        # Return standardized response
        return FileUploadResponse(
            file_id=uploaded_file.name,
            provider="google",
            filename=actual_filename,
            size_bytes=int(uploaded_file.size_bytes) if hasattr(uploaded_file, "size_bytes") else file_size,
            created_at=created_at,
            purpose=purpose,
            metadata={
                "mime_type": uploaded_file.mime_type if hasattr(uploaded_file, "mime_type") else None,
                "expiration_time": str(expiration_time),
                "state": uploaded_file.state.name if hasattr(uploaded_file, "state") and uploaded_file.state else "ACTIVE",
            },
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_google_upload_file.py -v`

Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_upload_file.py
git commit -m "feat(google): implement binary file upload via File API

- Replace Context Caching approach with proper File API
- Support binary files up to 2GB
- Track file metadata for expiration handling
- Add FileSizeError for files exceeding limit"
```

---

## Task 4: Implement File Expiration Detection

**Files:**
- Modify: `src/llmring/providers/google_api.py` (add new method after upload_file)
- Test: `tests/unit/test_google_file_expiration.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_google_file_expiration.py`:

```python
"""Unit tests for Google file expiration handling."""
import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from llmring.providers.google_api import GoogleProvider, UploadedFileInfo


@pytest.mark.asyncio
async def test_ensure_file_available_not_expired():
    """Test that non-expired files are retrieved without re-upload."""
    provider = GoogleProvider(api_key="fake-key")

    # Setup: file not expired
    file_id = "files/abc123"
    provider._uploaded_files[file_id] = UploadedFileInfo(
        file_name=file_id,
        expiration_time=datetime.now(timezone.utc) + timedelta(hours=24),
        local_path="/tmp/test.pdf"
    )

    # Mock client.files.get
    mock_file_obj = Mock()
    mock_file_obj.name = file_id

    with patch.object(provider.client.files, 'get', return_value=mock_file_obj):
        result = await provider._ensure_file_available(file_id)

        assert result.name == file_id
        # Verify get was called, not upload
        provider.client.files.get.assert_called_once_with(name=file_id)


@pytest.mark.asyncio
async def test_ensure_file_available_expired_with_path():
    """Test that expired files are re-uploaded if local path available."""
    provider = GoogleProvider(api_key="fake-key")

    # Setup: expired file with local path
    file_id = "files/old123"
    provider._uploaded_files[file_id] = UploadedFileInfo(
        file_name=file_id,
        expiration_time=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
        local_path="tests/fixtures/sample.txt"
    )

    # Mock upload_file to return new file
    new_file_id = "files/new456"
    mock_upload_response = Mock()
    mock_upload_response.file_id = new_file_id

    mock_file_obj = Mock()
    mock_file_obj.name = file_id

    with patch.object(provider, 'upload_file', new_callable=AsyncMock, return_value=mock_upload_response), \
         patch.object(provider.client.files, 'get', return_value=mock_file_obj):

        result = await provider._ensure_file_available(file_id)

        # Verify upload was called with local path
        provider.upload_file.assert_called_once_with(file="tests/fixtures/sample.txt")

        # Verify old file_id now points to new metadata
        assert file_id in provider._uploaded_files


@pytest.mark.asyncio
async def test_ensure_file_available_expired_no_path():
    """Test that expired files without local path raise error."""
    provider = GoogleProvider(api_key="fake-key")

    # Setup: expired file without local path
    file_id = "files/old123"
    provider._uploaded_files[file_id] = UploadedFileInfo(
        file_name=file_id,
        expiration_time=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
        local_path=None  # No path to re-upload
    )

    with pytest.raises(ValueError, match="expired and cannot be re-uploaded"):
        await provider._ensure_file_available(file_id)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_google_file_expiration.py -v`

Expected: FAIL with "'GoogleProvider' object has no attribute '_ensure_file_available'"

**Step 3: Implement _ensure_file_available method**

In `src/llmring/providers/google_api.py`, add this method after `upload_file()` (around line 1450):

```python
    async def _ensure_file_available(self, file_id: str) -> Any:
        """
        Ensure file is available, re-uploading if expired.

        Args:
            file_id: Google file ID (e.g., "files/abc123")

        Returns:
            Google file object ready for use in contents array

        Raises:
            ValueError: If file not found or expired without re-upload path
        """
        from datetime import timezone

        # Check if we have metadata for this file
        if file_id not in self._uploaded_files:
            # File uploaded elsewhere or before provider restart
            # Try to get it from Google
            try:
                loop = asyncio.get_event_loop()

                def _get_file():
                    return self.client.files.get(name=file_id)

                return await loop.run_in_executor(None, _get_file)
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

            # Update mapping - old file_id now points to new file metadata
            # This preserves llmring file_id → Google file_id mapping in service.py
            self._uploaded_files[file_id] = self._uploaded_files[new_upload.file_id]
            del self._uploaded_files[new_upload.file_id]

            # Get the file object
            loop = asyncio.get_event_loop()

            def _get_file():
                return self.client.files.get(name=file_id)

            return await loop.run_in_executor(None, _get_file)

        # File still valid
        loop = asyncio.get_event_loop()

        def _get_file():
            return self.client.files.get(name=file_id)

        return await loop.run_in_executor(None, _get_file)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_google_file_expiration.py -v`

Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_file_expiration.py
git commit -m "feat(google): add automatic re-upload for expired files

- Detect expiration via timestamp comparison
- Re-upload from local path if available
- Raise clear error if file-like object expired"
```

---

## Task 5: Update delete_file() for File API

**Files:**
- Modify: `src/llmring/providers/google_api.py:1477-1500`
- Test: `tests/unit/test_google_delete_file.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_google_delete_file.py`:

```python
"""Unit tests for Google file deletion."""
import pytest
import os
from unittest.mock import patch
from llmring.providers.google_api import GoogleProvider, UploadedFileInfo
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_delete_file_removes_tracking():
    """Test that delete_file removes file from tracking dict."""
    provider = GoogleProvider(api_key="fake-key")

    # Setup: add file to tracking
    file_id = "files/abc123"
    provider._uploaded_files[file_id] = UploadedFileInfo(
        file_name=file_id,
        expiration_time=datetime.now(timezone.utc),
        local_path="/tmp/test.pdf"
    )

    # Mock client.files.delete
    with patch.object(provider.client.files, 'delete'):
        result = await provider.delete_file(file_id)

        assert result is True
        assert file_id not in provider._uploaded_files
        provider.client.files.delete.assert_called_once_with(name=file_id)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_google_delete_file.py::test_delete_file_removes_tracking -v`

Expected: FAIL (current implementation is for caches not files)

**Step 3: Replace delete_file() implementation**

In `src/llmring/providers/google_api.py`, replace the `delete_file()` method (around line 1477) with:

```python
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete uploaded file.

        Args:
            file_id: File ID to delete (format: "files/abc123")

        Returns:
            True on success
        """
        try:
            # Delete file using Google SDK
            loop = asyncio.get_event_loop()

            def _delete_file():
                self.client.files.delete(name=file_id)
                return True

            result = await loop.run_in_executor(None, _delete_file)

            # Remove from tracking
            if file_id in self._uploaded_files:
                del self._uploaded_files[file_id]

            return result

        except Exception as e:
            await self._error_handler.handle_error(e, "files")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_google_delete_file.py::test_delete_file_removes_tracking -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_delete_file.py
git commit -m "fix(google): update delete_file to use File API

- Use client.files.delete instead of caches.delete
- Remove file from tracking dictionary
- Handle deletion errors via error_handler"
```

---

## Task 6: Update list_files() for File API

**Files:**
- Modify: `src/llmring/providers/google_api.py:1378-1433`
- Test: `tests/unit/test_google_list_files.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_google_list_files.py`:

```python
"""Unit tests for Google file listing."""
import pytest
import os
from unittest.mock import Mock, patch
from llmring.providers.google_api import GoogleProvider
from datetime import datetime


@pytest.mark.asyncio
async def test_list_files_uses_file_api():
    """Test that list_files uses File API not Cache API."""
    provider = GoogleProvider(api_key="fake-key")

    # Mock file list response
    mock_file1 = Mock()
    mock_file1.name = "files/abc123"
    mock_file1.display_name = "test.pdf"
    mock_file1.size_bytes = "1024"
    mock_file1.create_time = "2025-11-10T12:00:00Z"
    mock_file1.mime_type = "application/pdf"
    mock_file1.expiration_time = "2025-11-12T12:00:00Z"
    mock_file1.state = Mock(name="ACTIVE")

    mock_file_list = [mock_file1]

    with patch.object(provider.client.files, 'list', return_value=mock_file_list):
        files = await provider.list_files()

        assert len(files) == 1
        assert files[0].file_id == "files/abc123"
        assert files[0].provider == "google"
        assert files[0].filename == "test.pdf"
        assert files[0].size_bytes == 1024
        assert files[0].metadata.get("mime_type") == "application/pdf"

        provider.client.files.list.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_google_list_files.py::test_list_files_uses_file_api -v`

Expected: FAIL (current implementation uses caches not files)

**Step 3: Replace list_files() implementation**

In `src/llmring/providers/google_api.py`, replace the `list_files()` method (around line 1378) with:

```python
    async def list_files(
        self, purpose: Optional[str] = None, limit: int = 100
    ) -> List[FileMetadata]:
        """
        List uploaded files.

        Args:
            purpose: Optional filter by purpose (not used by Google)
            limit: Maximum number of files to return (default 100)

        Returns:
            List of FileMetadata objects
        """
        try:
            # List files using Google SDK
            loop = asyncio.get_event_loop()

            def _list_files():
                return self.client.files.list()

            files_list = await loop.run_in_executor(None, _list_files)

            # Parse response
            result = []
            for file_obj in files_list:
                # Parse timestamps
                try:
                    create_time_str = file_obj.create_time
                    if isinstance(create_time_str, str):
                        created_at = datetime.fromisoformat(create_time_str.replace("Z", "+00:00"))
                    else:
                        created_at = create_time_str
                except Exception:
                    created_at = datetime.now()

                result.append(
                    FileMetadata(
                        file_id=file_obj.name,
                        provider="google",
                        filename=file_obj.display_name if hasattr(file_obj, "display_name") and file_obj.display_name else file_obj.name.split("/")[-1],
                        size_bytes=int(file_obj.size_bytes) if hasattr(file_obj, "size_bytes") else 0,
                        created_at=created_at,
                        purpose="file",  # Google doesn't have purpose field
                        status=file_obj.state.name if hasattr(file_obj, "state") and file_obj.state else "ACTIVE",
                        metadata={
                            "mime_type": file_obj.mime_type if hasattr(file_obj, "mime_type") else None,
                            "expiration_time": file_obj.expiration_time if hasattr(file_obj, "expiration_time") else None,
                        },
                    )
                )

                # Limit results
                if len(result) >= limit:
                    break

            return result

        except Exception as e:
            await self._error_handler.handle_error(e, "files")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_google_list_files.py::test_list_files_uses_file_api -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/unit/test_google_list_files.py
git commit -m "fix(google): update list_files to use File API

- Use client.files.list instead of caches.list
- Parse file metadata (not cache metadata)
- Include mime_type and expiration in metadata"
```

---

## Task 7: Update Chat Integration

**Files:**
- Modify: `src/llmring/providers/google_api.py:303-367, 434-600`
- Test: `tests/integration/test_google_chat_with_files.py` (new)

**Step 1: Write the failing test**

Create `tests/integration/test_google_chat_with_files.py`:

```python
"""Integration tests for Google chat with files."""
import pytest
import os
from llmring.providers.google_api import GoogleProvider
from llmring.schemas import Message


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)
@pytest.mark.asyncio
async def test_chat_with_uploaded_file():
    """Test chat request with uploaded file using File API."""
    provider = GoogleProvider()

    try:
        # Upload a file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis"
        )

        # Use file in chat
        messages = [Message(role="user", content="What is in this file?")]
        response = await provider.chat(
            messages=messages,
            model="gemini-2.5-flash",
            files=[upload_response.file_id]
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model

        # Cleanup
        await provider.delete_file(upload_response.file_id)
    finally:
        await provider.aclose()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_google_chat_with_files.py::test_chat_with_uploaded_file -v`

Expected: FAIL (current implementation treats files as cached_content)

**Step 3: Update chat() and chat_stream() file handling**

In `src/llmring/providers/google_api.py`, update the `chat()` method (around line 303):

Remove lines 343-353 (the incorrect cached_content logic):
```python
        # Google uses cached_content for files, merge files into cache if provided
        merged_cache = cache or {}
        if files:
            # Use the first file as cached_content (Google's file mechanism)
            if len(files) > 1:
                logger.warning(
                    "Google provider only supports one cached_content at a time. Using first file: %s, ignoring others: %s",
                    files[0],
                    files[1:],
                )
            merged_cache["cached_content"] = files[0]
```

Replace with:
```python
        # Files will be handled in _chat_non_streaming
```

Similarly, in `chat_stream()` (around line 408-418), replace the same cached_content logic with:
```python
        # Files will be handled in _stream_chat
```

Then in `_chat_non_streaming()` (find it around line 600), add file handling after message conversion:

Find where it builds the API request parameters and add:

```python
        # Handle files - get file objects
        file_objects = []
        if files:
            for file_id in files:
                file_obj = await self._ensure_file_available(file_id)
                file_objects.append(file_obj)
```

Then modify the `generate_content` call to include files in contents. Find the section that calls `self.client.models.generate_content()` and update it to include file objects in the contents array.

Similarly update `_stream_chat()` with the same file handling logic.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_google_chat_with_files.py::test_chat_with_uploaded_file -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/llmring/providers/google_api.py tests/integration/test_google_chat_with_files.py
git commit -m "feat(google): integrate File API with chat

- Remove incorrect cached_content file handling
- Add file object retrieval via _ensure_file_available
- Include file objects in contents array
- Support multiple files per request"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `llmring.ai/content/docs/file-uploads.md:67,123-131`

**Step 1: Update provider comparison table**

In `llmring.ai/content/docs/file-uploads.md`, find the provider comparison table (around line 67) and update the Google "Max Size" row:

Change from:
```markdown
| **Max Size** | 500 MB | 512 MB | N/A (text-only) |
```

To:
```markdown
| **Max Size** | 500 MB | 512 MB | 2 GB |
```

**Step 2: Update Google provider description**

Find the "Google - Context Caching" section (around line 123) and update it to:

```markdown
### Google - File API & Context Caching

**Best for:** Large files (up to 2GB), repeated access with 90% cost savings via caching

- Binary files supported (PDF, images, audio, video)
- Up to 2 GB per file
- 48-hour retention, automatic re-upload on expiration
- 90% cost reduction via context caching (separate feature)
- Works with Chat Completions API
```

**Step 3: Commit documentation changes**

```bash
cd ../llmring.ai
git add content/docs/file-uploads.md
git commit -m "docs: update Google File API capabilities

- Change max size from N/A to 2GB
- Clarify binary file support
- Separate File API from Context Caching features"
cd ../llmring
```

---

## Task 9: Run Full Test Suite

**Files:**
- None (verification step)

**Step 1: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`

Expected: PASS (all unit tests)

**Step 2: Run Google-specific integration tests**

Run: `uv run pytest tests/integration/test_google_chat_with_files.py tests/test_google_files.py -v`

Expected: PASS (Google file integration tests)

**Step 3: Run service-level file tests**

Run: `uv run pytest tests/test_service_files.py -v`

Expected: PASS (cross-provider file tests)

**Step 4: Commit if any test fixes needed**

If any tests needed fixes:
```bash
git add [fixed files]
git commit -m "test: fix failing tests after File API refactor"
```

---

## Task 10: Code Review

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:requesting-code-review

**Files:**
- All modified files

**Step 1: Request code review**

Use the requesting-code-review skill to review all changes against the design document.

**Step 2: Address review feedback**

Make any necessary changes based on review feedback.

**Step 3: Commit fixes**

```bash
git add [files with review fixes]
git commit -m "fix: address code review feedback"
```

---

## Success Criteria

- ✅ Binary files (PDF, PNG, etc.) upload successfully to Google
- ✅ Files work in chat requests
- ✅ Automatic re-upload on expiration works
- ✅ Provider-agnostic API maintained (same code for all providers)
- ✅ All tests pass
- ✅ Documentation updated
- ✅ Code review complete

## References

- Design doc: `docs/plans/2025-11-10-google-file-api-design.md`
- Google File API: https://ai.google.dev/gemini-api/docs/files
- Google Gen AI SDK: https://googleapis.github.io/python-genai/
