"""
Unit tests for file utilities.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from llmring.file_utils import (
    _is_base64_string,
    analyze_image,
    compare_images,
    create_base64_image_content,
    create_data_url,
    create_image_content,
    create_multi_image_content,
    encode_file_to_base64,
    extract_text_from_image,
    get_file_mime_type,
    is_pdf_file,
    validate_file_for_vision_api,
    validate_image_file,
)


class TestFileMimeType:
    """Test MIME type detection."""

    def test_common_image_types(self):
        """Test MIME type detection for common image formats."""
        assert get_file_mime_type("test.png") == "image/png"
        assert get_file_mime_type("test.jpg") == "image/jpeg"
        assert get_file_mime_type("test.jpeg") == "image/jpeg"
        assert get_file_mime_type("test.gif") == "image/gif"
        assert get_file_mime_type("test.webp") == "image/webp"

    def test_document_types(self):
        """Test MIME type detection for document formats."""
        assert get_file_mime_type("test.pdf") == "application/pdf"
        assert get_file_mime_type("test.txt") == "text/plain"
        assert (
            get_file_mime_type("test.docx")
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_unknown_extension(self):
        """Test fallback for unknown extensions."""
        assert get_file_mime_type("test.unknown") == "application/octet-stream"

    def test_case_insensitive(self):
        """Test that extension matching is case insensitive."""
        assert get_file_mime_type("test.PNG") == "image/png"
        assert get_file_mime_type("test.JPG") == "image/jpeg"


class TestFileEncoding:
    """Test file encoding functions."""

    def test_encode_file_to_base64(self):
        """Test base64 encoding of files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            result = encode_file_to_base64(temp_path)
            # "Hello, World!" in base64
            expected = "SGVsbG8sIFdvcmxkIQ=="
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_encode_nonexistent_file(self):
        """Test encoding of non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            encode_file_to_base64("nonexistent.txt")

    def test_create_data_url(self):
        """Test creation of data URLs."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello")
            temp_path = f.name

        try:
            result = create_data_url(temp_path)
            assert result.startswith("data:text/plain;base64,")
            # "Hello" in base64 is "SGVsbG8="
            assert result.endswith("SGVsbG8=")
        finally:
            os.unlink(temp_path)


class TestImageValidation:
    """Test image file validation."""

    def test_validate_image_file_with_mock(self):
        """Test image validation with mocked files."""
        # Mock os.path.exists to return True
        with patch("llmring.file_utils.os.path.exists", return_value=True):
            # Mock get_file_mime_type for different types
            with patch("llmring.file_utils.get_file_mime_type") as mock_mime:
                # Test valid image types
                mock_mime.return_value = "image/png"
                assert validate_image_file("test.png") is True

                mock_mime.return_value = "image/jpeg"
                assert validate_image_file("test.jpg") is True

                # Test invalid type
                mock_mime.return_value = "application/pdf"
                assert validate_image_file("test.pdf") is False

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        assert validate_image_file("nonexistent.png") is False


class TestImageContentCreation:
    """Test image content creation functions."""

    def test_create_image_content_with_url(self):
        """Test creating image content from URL."""
        url = "https://example.com/image.png"
        os.environ["LLMRING_ALLOW_REMOTE_URLS"] = "true"
        text = "Analyze this image"

        result = create_image_content(url, text)

        expected = [
            {"type": "text", "text": "Analyze this image"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
        assert result == expected

    def test_create_image_content_without_text(self):
        """Test creating image content without text."""
        url = "https://example.com/image.png"
        os.environ["LLMRING_ALLOW_REMOTE_URLS"] = "true"

        result = create_image_content(url)

        expected = [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
        assert result == expected

    @patch("llmring.file_utils.validate_file_for_vision_api")
    @patch("llmring.file_utils.create_data_url")
    def test_create_image_content_with_file_path(self, mock_create_data_url, mock_validate):
        """Test creating image content from file path."""
        mock_create_data_url.return_value = "data:image/png;base64,test123"
        mock_validate.return_value = None  # No validation error

        result = create_image_content("test.png", "Test")

        expected = [
            {"type": "text", "text": "Test"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,test123"},
            },
        ]
        assert result == expected
        mock_create_data_url.assert_called_once_with("test.png")

    def test_create_multi_image_content(self):
        """Test creating content with multiple images."""
        urls = ["https://example.com/image1.png", "https://example.com/image2.png"]
        text = "Compare these images"
        os.environ["LLMRING_ALLOW_REMOTE_URLS"] = "true"

        result = create_multi_image_content(urls, text)

        expected = [
            {"type": "text", "text": "Compare these images"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.png"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image2.png"},
            },
        ]
        assert result == expected


class TestConvenienceFunctions:
    """Test convenience functions for common use cases."""

    def test_analyze_image(self):
        """Test analyze_image convenience function."""
        url = "https://example.com/test.jpg"
        result = analyze_image(url)

        expected = [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        assert result == expected

    def test_analyze_image_custom_prompt(self):
        """Test analyze_image with custom prompt."""
        url = "https://example.com/test.jpg"
        custom_prompt = "What objects are in this image?"
        result = analyze_image(url, custom_prompt)

        expected = [
            {"type": "text", "text": custom_prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        assert result == expected

    def test_extract_text_from_image(self):
        """Test extract_text_from_image convenience function."""
        url = "https://example.com/screenshot.png"
        result = extract_text_from_image(url)

        expected = [
            {
                "type": "text",
                "text": "Extract all text from this image. Preserve formatting and layout.",
            },
            {"type": "image_url", "image_url": {"url": url}},
        ]
        assert result == expected

    def test_compare_images(self):
        """Test compare_images convenience function."""
        image1 = "https://example.com/before.png"
        image2 = "https://example.com/after.png"
        result = compare_images(image1, image2)

        expected = [
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": image1}},
            {"type": "image_url", "image_url": {"url": image2}},
        ]
        assert result == expected

    def test_compare_images_custom_prompt(self):
        """Test compare_images with custom prompt."""
        image1 = "https://example.com/before.png"
        image2 = "https://example.com/after.png"
        custom_prompt = "What changed between these images?"
        result = compare_images(image1, image2, custom_prompt)

        expected = [
            {"type": "text", "text": custom_prompt},
            {"type": "image_url", "image_url": {"url": image1}},
            {"type": "image_url", "image_url": {"url": image2}},
        ]
        assert result == expected


class TestBase64Detection:
    """Test base64 string detection."""

    def test_is_base64_string_valid(self):
        """Test detection of valid base64 strings."""
        # Typical base64 encoded image data (long enough)
        long_base64 = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="
        assert _is_base64_string(long_base64) is True

        # Another valid base64 string
        valid_base64 = "UklGRiA" + "B" * 300 + "WEBPVP=="
        assert _is_base64_string(valid_base64) is True

    def test_is_base64_string_invalid(self):
        """Test detection of invalid/non-base64 strings."""
        # Too short
        assert _is_base64_string("short") is False

        # Looks like file path
        assert _is_base64_string("path/to/file.png") is False

        # Looks like URL
        assert _is_base64_string("https://example.com/image.jpg") is False

        # Contains invalid characters
        invalid_chars = "invalid$characters!" + "A" * 200
        assert _is_base64_string(invalid_chars) is False

        # Filename-like (short with dot)
        assert _is_base64_string("image.png") is False


class TestBase64ImageContent:
    """Test base64 image content creation."""

    def test_create_base64_image_content(self):
        """Test explicit base64 image content creation."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA"
        mime_type = "image/png"
        text = "Analyze this image"

        result = create_base64_image_content(base64_data, mime_type, text)

        expected = [
            {"type": "text", "text": "Analyze this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            },
        ]
        assert result == expected

    def test_create_base64_image_content_no_text(self):
        """Test base64 content creation without text."""
        base64_data = "UklGRiA"

        result = create_base64_image_content(base64_data)

        expected = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
            }
        ]
        assert result == expected


class TestEnhancedImageContent:
    """Test enhanced image content creation with base64 support."""

    def test_create_image_content_with_base64_auto_detect(self):
        """Test creating image content with auto-detected base64."""
        # Create a long base64 string that should be auto-detected
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="
        text = "Analyze this"

        result = create_image_content(base64_data, text, "image/png")

        expected = [
            {"type": "text", "text": "Analyze this"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"},
            },
        ]
        assert result == expected

    def test_create_image_content_with_data_url(self):
        """Test creating image content with existing data URL."""
        data_url = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA"
        text = "Test"

        result = create_image_content(data_url, text)

        expected = [
            {"type": "text", "text": "Test"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        assert result == expected

    def test_create_multi_image_content_mixed_sources(self):
        """Test multi-image content with mixed source types."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="

        images = [
            "https://example.com/image1.jpg",  # URL
            base64_data,  # Base64 (auto-detected)
            {"data": "UklGRiA", "mime_type": "image/webp"},  # Explicit base64
        ]

        result = create_multi_image_content(images, "Compare these")

        expected = [
            {"type": "text", "text": "Compare these"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.jpg"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/webp;base64,UklGRiA"},
            },
        ]
        assert result == expected


class TestEnhancedConvenienceFunctions:
    """Test enhanced convenience functions with base64 support."""

    def test_analyze_image_with_base64(self):
        """Test analyze_image with base64 data."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="
        prompt = "What's in this image?"

        result = analyze_image(base64_data, prompt, "image/png")

        expected = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"},
            },
        ]
        assert result == expected

    def test_extract_text_from_image_with_base64(self):
        """Test OCR function with base64 data."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="

        result = extract_text_from_image(base64_data, "image/png")

        expected = [
            {
                "type": "text",
                "text": "Extract all text from this image. Preserve formatting and layout.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"},
            },
        ]
        assert result == expected

    @patch("llmring.file_utils.validate_file_for_vision_api")
    @patch("llmring.file_utils.create_data_url")
    def test_compare_images_with_mixed_sources(self, mock_create_data_url, mock_validate):
        """Test compare_images with mixed source types."""
        mock_create_data_url.return_value = "data:image/png;base64,mocked_file_data"
        mock_validate.return_value = None  # No validation error

        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg=="

        result = compare_images(
            "image1.png",  # File path
            {"data": base64_data, "mime_type": "image/png"},  # Explicit base64
            "What's different?",
        )

        # The first image will be converted via create_data_url (mocked)
        # The second will use the explicit base64
        assert len(result) == 3  # text + 2 images
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What's different?"
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "data:image/png;base64,mocked_file_data"
        assert result[2]["type"] == "image_url"
        assert result[2]["image_url"]["url"] == f"data:image/png;base64,{base64_data}"

        # Verify the mock was called
        mock_create_data_url.assert_called_once_with("image1.png")


class TestPDFValidation:
    """Test PDF validation functionality."""

    @patch("llmring.file_utils.get_file_mime_type")
    @patch("os.path.exists")
    def test_is_pdf_file_true(self, mock_exists, mock_get_mime):
        """Test PDF file detection."""
        mock_exists.return_value = True
        mock_get_mime.return_value = "application/pdf"

        result = is_pdf_file("test.pdf")
        assert result is True

    @patch("llmring.file_utils.get_file_mime_type")
    @patch("os.path.exists")
    def test_is_pdf_file_false(self, mock_exists, mock_get_mime):
        """Test non-PDF file detection."""
        mock_exists.return_value = True
        mock_get_mime.return_value = "image/png"

        result = is_pdf_file("test.png")
        assert result is False

    def test_is_pdf_file_nonexistent(self):
        """Test PDF detection with non-existent file."""
        result = is_pdf_file("nonexistent.pdf")
        assert result is False


class TestVisionAPIValidation:
    """Test vision API file validation."""

    @patch("llmring.file_utils.get_file_mime_type")
    @patch("os.path.exists")
    def test_validate_pdf_file_raises_error(self, mock_exists, mock_get_mime):
        """Test that PDF files raise appropriate error."""
        mock_exists.return_value = True
        mock_get_mime.return_value = "application/pdf"

        with pytest.raises(ValueError) as exc_info:
            validate_file_for_vision_api("test.pdf")

        error_msg = str(exc_info.value)
        assert "PDF files are not supported" in error_msg
        assert "OpenAI Chat Completions API" in error_msg
        assert "Assistants API" in error_msg

    @patch("llmring.file_utils.get_file_mime_type")
    @patch("os.path.exists")
    def test_validate_unsupported_file_type(self, mock_exists, mock_get_mime):
        """Test that unsupported file types raise appropriate error."""
        mock_exists.return_value = True
        mock_get_mime.return_value = "text/plain"

        with pytest.raises(ValueError) as exc_info:
            validate_file_for_vision_api("test.txt")

        error_msg = str(exc_info.value)
        assert "Unsupported file type" in error_msg
        assert "text/plain" in error_msg
        assert "Supported image types:" in error_msg

    @patch("llmring.file_utils.get_file_mime_type")
    @patch("os.path.exists")
    def test_validate_supported_image_file(self, mock_exists, mock_get_mime):
        """Test that supported image files pass validation."""
        mock_exists.return_value = True
        mock_get_mime.return_value = "image/png"

        # Should not raise any exception
        validate_file_for_vision_api("test.png")

    def test_validate_nonexistent_file(self):
        """Test that non-existent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_for_vision_api("nonexistent.png")

        error_msg = str(exc_info.value)
        assert "File not found" in error_msg
        assert "nonexistent.png" in error_msg
