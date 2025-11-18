"""Unit tests for Google File API upload."""

import os
from pathlib import Path

import pytest

from llmring.exceptions import FileSizeError
from llmring.providers.google_api import GoogleProvider


@pytest.mark.skipif(
    not (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    ),
    reason="GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not set",
)
@pytest.mark.asyncio
async def test_upload_pdf_file():
    """Test uploading a PDF file using File API."""
    provider = GoogleProvider()

    try:
        # Upload a test file
        response = await provider.upload_file(file="tests/fixtures/sample.pdf", purpose="analysis")

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

    with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value = Mock(st_size=3 * 1024 * 1024 * 1024)  # 3GB

        with pytest.raises(FileSizeError) as exc_info:
            await provider.upload_file(file=large_file_path)

        assert "exceeds Google limit" in str(exc_info.value)
