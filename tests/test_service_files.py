# ABOUTME: Integration tests for unified file upload interface in LLMRing service layer.
# ABOUTME: Tests upload, list, get, delete operations across providers without mocks.

import os

import pytest

from llmring.exceptions import ProviderNotFoundError
from llmring.schemas import FileMetadata, FileUploadResponse
from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_upload_file_with_explicit_provider():
    """Test uploading a file with explicit provider parameter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload to Anthropic explicitly
        response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Verify response
        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("file_")
        assert response.provider == "anthropic"
        assert response.size_bytes > 0

        # Cleanup
        deleted = await ring.delete_file(response.file_id)
        assert deleted is True

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_upload_file_auto_detect_provider():
    """Test uploading a file with auto-detected provider."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload using model string (provider extracted from model)
        response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Verify response
        assert isinstance(response, FileUploadResponse)
        assert response.file_id is not None
        assert response.provider in ["anthropic", "openai", "google"]
        assert response.size_bytes > 0

        # Cleanup
        deleted = await ring.delete_file(response.file_id)
        assert deleted is True

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_list_files_with_explicit_provider():
    """Test listing files with explicit provider parameter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a test file
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # List files from Anthropic
        files = await ring.list_files(provider="anthropic")
        assert isinstance(files, list)
        assert all(isinstance(f, FileMetadata) for f in files)

        # Find our uploaded file
        found = any(f.file_id == upload_response.file_id for f in files)
        assert found, "Uploaded file should appear in list"

        # Cleanup
        await ring.delete_file(upload_response.file_id, provider="anthropic")

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_list_files_auto_detect_provider():
    """Test listing files with auto-detected provider."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a test file
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # List files without specifying provider
        files = await ring.list_files()
        assert isinstance(files, list)
        assert all(isinstance(f, FileMetadata) for f in files)

        # Cleanup
        await ring.delete_file(upload_response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_get_file_with_provider_auto_detection():
    """Test getting file metadata with provider auto-detected from file_id."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a file to Anthropic
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Get file metadata without specifying provider (should auto-detect from file_id)
        metadata = await ring.get_file(upload_response.file_id)
        assert isinstance(metadata, FileMetadata)
        assert metadata.file_id == upload_response.file_id
        assert metadata.provider == "anthropic"
        assert metadata.size_bytes == upload_response.size_bytes

        # Cleanup
        await ring.delete_file(upload_response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_get_file_with_explicit_provider():
    """Test getting file metadata with explicit provider parameter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a file
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Get file metadata with explicit provider
        metadata = await ring.get_file(upload_response.file_id, provider="anthropic")
        assert isinstance(metadata, FileMetadata)
        assert metadata.file_id == upload_response.file_id
        assert metadata.provider == "anthropic"

        # Cleanup
        await ring.delete_file(upload_response.file_id, provider="anthropic")

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_delete_file_with_provider_auto_detection():
    """Test deleting a file with provider auto-detected from file_id."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a file to Anthropic
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Delete without specifying provider (should auto-detect from file_id)
        deleted = await ring.delete_file(upload_response.file_id)
        assert deleted is True

        # Verify it's gone
        files = await ring.list_files(provider="anthropic")
        found = any(f.file_id == upload_response.file_id for f in files)
        assert not found, "Deleted file should not appear in list"

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_delete_file_with_explicit_provider():
    """Test deleting a file with explicit provider parameter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a file
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Delete with explicit provider
        deleted = await ring.delete_file(upload_response.file_id, provider="anthropic")
        assert deleted is True

        # Verify it's gone
        files = await ring.list_files(provider="anthropic")
        found = any(f.file_id == upload_response.file_id for f in files)
        assert not found, "Deleted file should not appear in list"

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_provider_auto_detection_from_file_id_anthropic():
    """Test that Anthropic file_id format is correctly detected."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload to Anthropic
        response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Verify file_id format
        assert response.file_id.startswith("file_")

        # Test auto-detection
        detected_provider = ring._detect_provider_from_file_id(response.file_id)
        assert detected_provider == "anthropic"

        # Cleanup
        await ring.delete_file(response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_provider_auto_detection_from_file_id_openai():
    """Test that OpenAI file_id format is correctly detected."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload to OpenAI
        response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="openai:gpt-4o",
        )

        # Verify file_id format
        assert response.file_id.startswith("file-")

        # Test auto-detection
        detected_provider = ring._detect_provider_from_file_id(response.file_id)
        assert detected_provider == "openai"

        # Cleanup
        await ring.delete_file(response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_provider_auto_detection_from_file_id_google():
    """Test that Google cache_id format is correctly detected."""
    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    ):
        pytest.skip("GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload to Google (creates cache)
        # Google requires minimum 4096 tokens for caching
        response = await ring.upload_file(
            file="tests/fixtures/google_large_doc.txt",
            model="google:gemini-2.5-flash",
            ttl_seconds=300,
        )

        # Verify cache_id format
        assert response.file_id.startswith("cachedContents/")

        # Test auto-detection
        detected_provider = ring._detect_provider_from_file_id(response.file_id)
        assert detected_provider == "google"

        # Cleanup
        await ring.delete_file(response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_cross_provider_file_operations():
    """Test file operations across multiple providers."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key or not openai_key:
        pytest.skip("Both ANTHROPIC_API_KEY and OPENAI_API_KEY required")

    ring = LLMRing()
    try:
        # Upload to Anthropic
        anthropic_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # Upload to OpenAI
        openai_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="openai:gpt-4o",
        )

        # Verify different file_ids
        assert anthropic_response.file_id != openai_response.file_id
        assert anthropic_response.file_id.startswith("file_")
        assert openai_response.file_id.startswith("file-")

        # Verify auto-detection works for both
        detected_anthropic = ring._detect_provider_from_file_id(anthropic_response.file_id)
        detected_openai = ring._detect_provider_from_file_id(openai_response.file_id)
        assert detected_anthropic == "anthropic"
        assert detected_openai == "openai"

        # Cleanup both
        await ring.delete_file(anthropic_response.file_id)
        await ring.delete_file(openai_response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_invalid_provider_raises_error():
    """Test that specifying an invalid provider raises ProviderNotFoundError."""
    ring = LLMRing()
    try:
        with pytest.raises(ProviderNotFoundError):
            await ring.upload_file(
                file="tests/fixtures/sample.txt",
                model="invalid_provider:some-model",
            )
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_get_file_unknown_format_without_provider():
    """Test that get_file with unknown file_id format raises error if provider not specified."""
    ring = LLMRing()
    try:
        with pytest.raises(ProviderNotFoundError) as exc_info:
            await ring.get_file("unknown_format_12345")

        assert "Cannot determine provider from file_id" in str(exc_info.value)
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_delete_file_unknown_format_without_provider():
    """Test that delete_file with unknown file_id format raises error if provider not specified."""
    ring = LLMRing()
    try:
        with pytest.raises(ProviderNotFoundError) as exc_info:
            await ring.delete_file("unknown_format_12345")

        assert "Cannot determine provider from file_id" in str(exc_info.value)
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_upload_with_filelike_object():
    """Test uploading a file from a file-like object."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Open file and upload
        with open("tests/fixtures/sample.txt", "rb") as f:
            response = await ring.upload_file(
                file=f,
                model="anthropic:claude-3-5-haiku-20241022",
                filename="sample.txt",
            )

        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("file_")
        assert response.provider == "anthropic"
        assert response.size_bytes > 0

        # Cleanup
        await ring.delete_file(response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_list_files_with_purpose_filter():
    """Test listing files with purpose filter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload with specific purpose
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # List files with purpose filter
        files = await ring.list_files(provider="anthropic", purpose="code_execution")
        assert isinstance(files, list)

        # Our file should be in the list
        found = any(f.file_id == upload_response.file_id for f in files)
        assert found, "Uploaded file should appear in list"

        # Cleanup
        await ring.delete_file(upload_response.file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_list_files_with_limit():
    """Test listing files with limit parameter."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Upload a test file
        upload_response = await ring.upload_file(
            file="tests/fixtures/sample.txt",
            model="anthropic:claude-3-5-haiku-20241022",
        )

        # List files with limit
        files = await ring.list_files(provider="anthropic", limit=5)
        assert isinstance(files, list)
        assert len(files) <= 5

        # Cleanup
        await ring.delete_file(upload_response.file_id)

    finally:
        await ring.close()
