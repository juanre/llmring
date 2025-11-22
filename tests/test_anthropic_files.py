"""Integration tests for Anthropic Files API support. Tests upload, list, get, and delete operations without mocks."""

import os

import pytest

from llmring.exceptions import FileSizeError
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.schemas import FileMetadata, FileUploadResponse


@pytest.mark.asyncio
async def test_upload_file_from_path():
    """Test uploading a file from a file path."""
    # Skip if no API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload test file
        response = await provider.upload_file(
            file="tests/fixtures/sample_data.csv",
            purpose="code_execution",
        )

        # Verify response
        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("file_")
        assert response.provider == "anthropic"
        assert response.size_bytes > 0
        assert response.purpose == "code_execution"
        assert response.filename is not None

        # Cleanup
        deleted = await provider.delete_file(response.file_id)
        assert deleted is True

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_file_with_filelike():
    """Test uploading a file from a file-like object."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Open file and upload
        with open("tests/fixtures/sample.txt", "rb") as f:
            response = await provider.upload_file(
                file=f,
                purpose="analysis",
                filename="sample.txt",
            )

        assert isinstance(response, FileUploadResponse)
        assert response.file_id.startswith("file_")
        assert response.provider == "anthropic"
        assert response.size_bytes > 0
        assert response.purpose == "analysis"

        # Cleanup
        await provider.delete_file(response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_list_files():
    """Test listing uploaded files."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload a test file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        # List files
        files = await provider.list_files()
        assert isinstance(files, list)
        assert len(files) > 0
        assert all(isinstance(f, FileMetadata) for f in files)

        # Find our uploaded file
        found = any(f.file_id == upload_response.file_id for f in files)
        assert found, "Uploaded file should appear in list"

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_list_files_with_purpose_filter():
    """Test listing files filtered by purpose."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload with specific purpose
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="code_execution",
        )

        # List files (Anthropic API doesn't support purpose filtering)
        # Note: The purpose parameter is kept for API consistency but may not filter
        files = await provider.list_files()
        assert isinstance(files, list)

        # Our file should be in the list
        found = any(f.file_id == upload_response.file_id for f in files)
        assert found, "Uploaded file should appear in list"

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_file():
    """Test getting file metadata."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload a file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        # Get file metadata
        metadata = await provider.get_file(upload_response.file_id)
        assert isinstance(metadata, FileMetadata)
        assert metadata.file_id == upload_response.file_id
        assert metadata.provider == "anthropic"
        assert metadata.size_bytes == upload_response.size_bytes
        # Note: Anthropic API doesn't return purpose in file metadata

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_delete_file():
    """Test deleting an uploaded file."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload a file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        # Delete it
        deleted = await provider.delete_file(upload_response.file_id)
        assert deleted is True

        # Verify it's gone (should raise error or return empty)
        files = await provider.list_files()
        found = any(f.file_id == upload_response.file_id for f in files)
        assert not found, "Deleted file should not appear in list"

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_upload_file_size_limit():
    """Test that oversized files raise FileSizeError."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        import io

        # Create a BytesIO object with actual content > 500MB
        # We'll use 501MB of data (minimum to trigger the error)
        # To avoid memory issues, we'll create a sparse representation
        size_mb = 501
        large_content = b"x" * (size_mb * 1024 * 1024)
        large_file = io.BytesIO(large_content)

        # Verify FileSizeError is raised with correct details
        with pytest.raises(FileSizeError) as exc_info:
            await provider.upload_file(
                file=large_file,
                purpose="analysis",
                filename="large_file.txt",
            )

        # Verify error details
        error = exc_info.value
        assert error.file_size == size_mb * 1024 * 1024
        assert error.max_size == 500 * 1024 * 1024
        assert "exceeds Anthropic limit" in str(error)
        assert str(error.file_size) in str(error)
        assert str(error.max_size) in str(error)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_file_invalid_id():
    """Test that get_file with invalid file_id raises appropriate error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Try to get a file with an invalid ID
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.get_file("file_invalid_nonexistent_id")

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_delete_file_invalid_id():
    """Test that delete_file with invalid file_id raises appropriate error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Try to delete a file with an invalid ID
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.delete_file("file_invalid_nonexistent_id")

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_file_not_found():
    """Test that accessing a deleted file raises appropriate error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload a file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        # Delete it
        await provider.delete_file(upload_response.file_id)

        # Try to get the deleted file - should raise error
        from llmring.exceptions import ProviderResponseError

        with pytest.raises(ProviderResponseError):
            await provider.get_file(upload_response.file_id)

    finally:
        await provider.aclose()
