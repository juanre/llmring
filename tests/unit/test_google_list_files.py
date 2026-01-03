"""Unit tests for Google file listing. Tests that list_files uses File API not Cache API."""

from unittest.mock import Mock, patch

import pytest

from llmring.providers.google_api import GoogleProvider


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
    # Mock state with name attribute that returns "ACTIVE"
    mock_state = Mock()
    mock_state.name = "ACTIVE"
    mock_file1.state = mock_state

    mock_file_list = [mock_file1]

    with patch.object(provider.client.files, "list", return_value=mock_file_list):
        files = await provider.list_files()

        assert len(files) == 1
        assert files[0].file_id == "files/abc123"
        assert files[0].provider == "google"
        assert files[0].filename == "test.pdf"
        assert files[0].size_bytes == 1024
        assert files[0].metadata.get("mime_type") == "application/pdf"

        provider.client.files.list.assert_called_once()
