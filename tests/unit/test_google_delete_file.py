# ABOUTME: Unit tests for Google file deletion.
# ABOUTME: Tests that delete_file removes file from tracking dict.
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from llmring.providers.google_api import GoogleProvider, UploadedFileInfo


@pytest.mark.asyncio
async def test_delete_file_removes_tracking():
    """Test that delete_file removes file from tracking dict."""
    provider = GoogleProvider(api_key="fake-key")

    # Setup: add file to tracking
    file_id = "files/abc123"
    provider._uploaded_files[file_id] = UploadedFileInfo(
        file_name=file_id, expiration_time=datetime.now(timezone.utc), local_path="/tmp/test.pdf"
    )

    # Mock client.files.delete
    with patch.object(provider.client.files, "delete"):
        result = await provider.delete_file(file_id)

        assert result is True
        assert file_id not in provider._uploaded_files
        provider.client.files.delete.assert_called_once_with(name=file_id)
