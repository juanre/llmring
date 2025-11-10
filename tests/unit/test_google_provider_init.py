"""Unit tests for GoogleProvider initialization."""

import os

import pytest
from dotenv import load_dotenv

from llmring.providers.google_api import GoogleProvider

load_dotenv()


@pytest.mark.skipif(
    not (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    ),
    reason="Google API key not set",
)
def test_google_provider_has_uploaded_files_dict():
    """Test GoogleProvider initializes with empty uploaded files dict."""
    provider = GoogleProvider()

    assert hasattr(provider, "_uploaded_files")
    assert isinstance(provider._uploaded_files, dict)
    assert len(provider._uploaded_files) == 0
