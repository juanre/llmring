# ABOUTME: Integration tests for Google Context Caching as file upload equivalent.
# ABOUTME: Tests cache creation, list, get, and delete operations without mocks.

import os

import pytest

from llmring.providers.google_api import GoogleProvider
from llmring.schemas import FileMetadata, FileUploadResponse


@pytest.mark.asyncio
async def test_upload_file_from_path():
    """Test creating a cache from a file path."""
    # Skip if no API key
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Upload test file (create cache)
        # Use large_document.txt to meet minimum token requirements
        response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            ttl_seconds=3600,
            model="gemini-2.5-flash",  # Flash model has lower min (1024 tokens)
        )

        # Verify response
        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("cachedContents/")
        assert response.provider == "google"
        assert response.size_bytes > 0
        assert response.purpose == "cache"
        assert response.filename is not None
        assert "ttl_seconds" in response.metadata
        assert response.metadata["ttl_seconds"] == 3600

        # Cleanup
        deleted = await provider.delete_file(response.file_id)
        assert deleted is True

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_file_with_filelike():
    """Test creating a cache from a file-like object."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Open file and upload (create cache)
        with open("tests/fixtures/large_document.txt", "rb") as f:
            response = await provider.upload_file(
                file=f,
                purpose="cache",
                filename="large_document.txt",
                ttl_seconds=7200,
                model="gemini-2.5-flash",
            )

        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("cachedContents/")
        assert response.provider == "google"
        assert response.size_bytes > 0
        assert response.purpose == "cache"
        assert response.metadata["ttl_seconds"] == 7200

        # Cleanup
        await provider.delete_file(response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_with_custom_ttl():
    """Test cache creation with custom TTL."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create cache with 2 hour TTL
        response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            ttl_seconds=7200,
            model="gemini-2.5-flash",
        )

        assert isinstance(response, FileUploadResponse)
        assert response.metadata["ttl_seconds"] == 7200
        assert "expire_time" in response.metadata

        # Cleanup
        await provider.delete_file(response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_list_files():
    """Test listing caches."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create a test cache
        upload_response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        # List caches
        files = await provider.list_files()
        assert isinstance(files, list)
        # May be empty or have our cache - just verify it returns a list
        assert all(isinstance(f, FileMetadata) for f in files)

        # If we found caches, verify structure
        if files:
            for f in files:
                assert f.file_id.startswith("cachedContents/")
                assert f.provider == "google"
                assert f.purpose == "cache"
                assert f.status == "ready"

        # Find our uploaded cache
        found = any(f.file_id == upload_response.file_id for f in files)
        assert found, "Uploaded cache should appear in list"

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_list_files_with_limit():
    """Test listing caches with limit."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create a test cache
        upload_response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        # List with limit
        files = await provider.list_files(limit=5)
        assert isinstance(files, list)
        assert len(files) <= 5

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_file():
    """Test getting cache metadata."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create a cache
        upload_response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        # Get cache metadata
        metadata = await provider.get_file(upload_response.file_id)
        assert isinstance(metadata, FileMetadata)
        assert metadata.file_id == upload_response.file_id
        assert metadata.provider == "google"
        assert metadata.purpose == "cache"
        assert metadata.status == "ready"
        assert "model" in metadata.metadata
        assert "expire_time" in metadata.metadata

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_delete_file():
    """Test deleting a cache."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create a cache
        upload_response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        # Delete it
        deleted = await provider.delete_file(upload_response.file_id)
        assert deleted is True

        # Verify it's gone - should raise error or not be in list
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.get_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_binary_file_fails():
    """Test that binary files are rejected (Google caching only supports text)."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Try to upload a binary file (will fail if it's not UTF-8 text)
        # Create a binary file with non-UTF-8 bytes
        import io

        binary_content = b"\x80\x81\x82\x83"  # Invalid UTF-8
        binary_file = io.BytesIO(binary_content)

        # Should raise ValueError about UTF-8 decoding
        with pytest.raises(ValueError) as exc_info:
            await provider.upload_file(
                file=binary_file,
                purpose="cache",
                filename="binary.dat",
            )

        error = exc_info.value
        assert "UTF-8" in str(error)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_file_invalid_id():
    """Test that get_file with invalid cache_id raises appropriate error."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Try to get a cache with an invalid ID
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.get_file("cachedContents/invalid_nonexistent_id")

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_delete_file_invalid_id():
    """Test that delete_file with invalid cache_id raises appropriate error."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Try to delete a cache with an invalid ID
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.delete_file("cachedContents/invalid_nonexistent_id")

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_cache_not_found():
    """Test that accessing a deleted cache raises appropriate error."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create a cache
        upload_response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        # Delete it
        await provider.delete_file(upload_response.file_id)

        # Try to get the deleted cache - should raise error
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.get_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_with_explicit_model():
    """Test cache creation with explicit model parameter."""
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Create cache with explicit model
        response = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="cache",
            model="gemini-2.5-flash",
        )

        assert isinstance(response, FileUploadResponse)
        assert response.metadata["model"] == "gemini-2.5-flash"

        # Cleanup
        await provider.delete_file(response.file_id)

    finally:
        await provider.aclose()
