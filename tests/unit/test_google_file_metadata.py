"""Unit tests for Google File API metadata tracking."""

from datetime import datetime, timezone

import pytest

from llmring.providers.google_api import UploadedFileInfo


def test_uploaded_file_info_dataclass():
    """Test UploadedFileInfo dataclass creation."""
    info = UploadedFileInfo(
        file_name="files/abc123",
        expiration_time=datetime.now(timezone.utc),
        local_path="/path/to/file.pdf",
    )

    assert info.file_name == "files/abc123"
    assert info.local_path == "/path/to/file.pdf"
    assert isinstance(info.expiration_time, datetime)
