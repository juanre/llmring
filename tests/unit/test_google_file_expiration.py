"""Unit tests for Google file expiration handling. Tests file expiration detection and automatic re-upload logic."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

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
        local_path="/tmp/test.pdf",
    )

    # Mock client.files.get
    mock_file_obj = Mock()
    mock_file_obj.name = file_id

    with patch.object(provider.client.files, "get", return_value=mock_file_obj):
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
        local_path="tests/fixtures/sample.txt",
    )

    # Mock upload_file to return new file
    new_file_id = "files/new456"
    mock_upload_response = Mock()
    mock_upload_response.file_id = new_file_id

    async def mock_upload_file(file):
        # Simulate what upload_file does - add to _uploaded_files
        provider._uploaded_files[new_file_id] = UploadedFileInfo(
            file_name=new_file_id,
            expiration_time=datetime.now(timezone.utc) + timedelta(hours=48),
            local_path=file,
        )
        return mock_upload_response

    mock_file_obj = Mock()
    mock_file_obj.name = file_id

    with (
        patch.object(provider, "upload_file", side_effect=mock_upload_file),
        patch.object(provider.client.files, "get", return_value=mock_file_obj),
    ):
        result = await provider._ensure_file_available(file_id)

        # Verify upload was called with local path
        provider.upload_file.assert_called_once_with(file="tests/fixtures/sample.txt")
        assert result.name == file_id

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
        local_path=None,  # No path to re-upload
    )

    with pytest.raises(ValueError, match="expired and cannot be re-uploaded"):
        await provider._ensure_file_available(file_id)
