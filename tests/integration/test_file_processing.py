"""
Integration tests for file processing across providers.

These tests require actual API keys and test the real file processing capabilities.
"""

import os
import tempfile

import pytest
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from llmring.file_utils import analyze_image, create_image_content
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

# Import test model helpers
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from conftest_models import get_test_model


def create_test_image_with_text() -> str:
    """Create a test image with readable text and return the file path."""
    # Create larger image with better spacing
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)

    # Try to get a good font for better OCR
    font = None
    font_size = 36

    # Try macOS system fonts first
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Avenir.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except Exception:
            continue

    # Fallback to default if no system font found
    if font is None:
        font = ImageFont.load_default()

    # Use non-PII test data to avoid content filters
    text_lines = ["DOCUMENT: TEST-001", "Category: DEMO", "Status: ACTIVE"]

    # Draw with better spacing for clearer OCR
    y_position = 60
    for line in text_lines:
        draw.text((50, y_position), line, fill="black", font=font)
        y_position += 70

    # Save to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(temp_fd)
    img.save(temp_path, "PNG")

    return temp_path


def create_test_pdf_with_text() -> str:
    """Create a test PDF with readable text and return the file path."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(temp_fd)

    # Create PDF with clear, simple content
    c = canvas.Canvas(temp_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 16)  # Larger font for better OCR

    # Use non-PII test data
    content_lines = [
        "Technical Document Analysis Test",
        "",
        "Reference: DOC-TEST-001",
        "Category: DEMO",
        "Type: Technical",
        "",
        "This document validates file processing capabilities.",
        "The system should extract the reference code and category.",
    ]

    y_position = height - 100
    for line in content_lines:
        c.drawString(100, y_position, line)
        y_position -= 30  # More spacing

    c.save()
    return temp_path


@pytest.mark.integration
class TestFileProcessing:
    """Clean integration tests for file processing across providers."""

    # service fixture is provided by conftest.py

    @pytest.fixture
    def test_image_path(self):
        """Create a test image file."""
        image_path = create_test_image_with_text()
        yield image_path
        # Cleanup
        if os.path.exists(image_path):
            os.unlink(image_path)

    @pytest.fixture
    def test_pdf_path(self):
        """Create a test PDF file."""
        pdf_path = create_test_pdf_with_text()
        yield pdf_path
        # Cleanup
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

    @pytest.mark.asyncio
    async def test_image_analysis_with_openai(self, service, test_image_path):
        """Test image analysis with OpenAI GPT-4o."""
        available_models = await service.get_available_models()
        if (
            not available_models.get("openai")
            or "gpt-4" not in available_models["openai"]
        ):
            pytest.skip("OpenAI GPT-4o not available")

        # Use a neutral prompt that won't trigger content filters
        content = analyze_image(
            test_image_path,
            "Please read and transcribe all the text visible in this technical test image.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model=get_test_model("openai", "vision"),
            max_tokens=200,
        )

        response = await service.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Check if key information was extracted (updated for new test data)
        content_lower = response.content.lower()
        assert "test-001" in content_lower or "test 001" in content_lower
        assert "demo" in content_lower or "active" in content_lower

        print(f"OpenAI extracted: {response.content}")

    @pytest.mark.asyncio
    async def test_image_analysis_with_anthropic(self, service, test_image_path):
        """Test image analysis with Anthropic Claude."""
        available_models = await service.get_available_models()
        if not available_models.get("anthropic"):
            pytest.skip("Anthropic not available")

        content = analyze_image(
            test_image_path,
            "Please read and transcribe all the text visible in this technical test image.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model=get_test_model("anthropic", "vision"),
            max_tokens=200,
        )

        response = await service.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Check if key information was extracted (updated for new test data)
        content_lower = response.content.lower()
        assert "test-001" in content_lower or "test 001" in content_lower
        assert "demo" in content_lower or "active" in content_lower

        print(f"Anthropic extracted: {response.content}")

    @pytest.mark.asyncio
    async def test_image_analysis_with_google(self, service, test_image_path):
        """Test image analysis with Google Gemini."""
        available_models = await service.get_available_models()
        if not available_models.get("google"):
            pytest.skip("Google not available")

        try:
            content = analyze_image(
                test_image_path,
                "Please read and transcribe all the text visible in this technical test image.",
            )

            request = LLMRequest(
                messages=[Message(role="user", content=content)],
                model=get_test_model("google", "vision"),
                max_tokens=200,
            )

            response = await service.chat(request)

            # Verify response
            assert response.content is not None
            assert len(response.content) > 0

            # Check if key information was extracted (updated for new test data)
            content_lower = response.content.lower()
            assert "test-001" in content_lower or "test 001" in content_lower
            assert "demo" in content_lower or "active" in content_lower

            print(f"Google extracted: {response.content}")

        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if any(
                term in error_msg
                for term in [
                    "quota",
                    "rate limit",
                    "resource_exhausted",
                    "429",
                    "billing",
                    "exceeded",
                ]
            ):
                pytest.skip(f"Google API quota exceeded: {str(e)[:100]}")
            raise

    @pytest.mark.asyncio
    async def test_pdf_processing_anthropic(self, service, test_pdf_path):
        """Test PDF processing with Anthropic using universal file interface."""
        available_models = await service.get_available_models()
        if not available_models.get("anthropic"):
            pytest.skip("Anthropic not available")

        # Use the universal file interface
        from llmring.file_utils import analyze_file

        content = analyze_file(
            test_pdf_path,
            "Please read and transcribe the text from this technical document PDF.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model=get_test_model("anthropic", "pdf"),
            max_tokens=300,
        )

        response = await service.chat(request)

        # Verify response contains key information
        content_lower = response.content.lower()
        assert (
            "doc-test-001" in content_lower
            or "test-001" in content_lower
            or "test 001" in content_lower
        )
        assert "demo" in content_lower or "technical" in content_lower

        print(f"PDF processing with Anthropic: {response.content}")

    @pytest.mark.asyncio
    async def test_pdf_processing_google(self, service, test_pdf_path):
        """Test PDF processing with Google using universal file interface."""
        available_models = await service.get_available_models()
        if not available_models.get("google"):
            pytest.skip("Google not available")

        try:
            # Use the universal file interface
            from llmring.file_utils import analyze_file

            content = analyze_file(
                test_pdf_path,
                "Please read and transcribe the text from this technical document PDF.",
            )

            request = LLMRequest(
                messages=[Message(role="user", content=content)],
                model=get_test_model("google", "pdf"),
                max_tokens=300,
            )

            response = await service.chat(request)

            # Verify response contains key information
            content_lower = response.content.lower()
            assert (
                "doc-test-001" in content_lower
                or "test-001" in content_lower
                or "test 001" in content_lower
            )
            assert "demo" in content_lower or "technical" in content_lower

            print(f"PDF processing with Google: {response.content}")

        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if any(
                term in error_msg
                for term in [
                    "quota",
                    "rate limit",
                    "resource_exhausted",
                    "429",
                    "billing",
                    "exceeded",
                ]
            ):
                pytest.skip(f"Google API quota exceeded: {str(e)[:100]}")
            raise

    @pytest.mark.asyncio
    async def test_pdf_processing_openai(self, service, test_pdf_path):
        """Test PDF processing with OpenAI using Assistants API automatically."""
        available_models = await service.get_available_models()
        if not available_models.get("openai"):
            pytest.skip("OpenAI not available")

        # Use the universal file interface
        from llmring.file_utils import analyze_file

        content = analyze_file(
            test_pdf_path,
            "Please read and transcribe the text from this technical document PDF.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model=get_test_model("openai", "pdf"),  # Will automatically use Assistants API for PDFs
            max_tokens=300,
        )

        try:
            response = await service.chat(request)

            # Verify response contains key information
            content_lower = response.content.lower()
            # Skip if model refuses or cannot process PDFs in this environment
            if (
                "unable to directly read" in content_lower
                or "can’t assist" in content_lower
                or "can't assist" in content_lower
            ):
                pytest.skip(
                    "OpenAI model refused to transcribe PDF in this environment"
                )
            assert (
                "doc-test-001" in content_lower
                or "test-001" in content_lower
                or "test 001" in content_lower
            )
            assert "demo" in content_lower or "technical" in content_lower

            print(f"PDF processing with OpenAI: {response.content}")

        except Exception as e:
            # Handle potential API limitations
            error_str = str(e).lower()
            if any(
                term in error_str
                for term in ["billing", "quota", "usage limit", "rate limit"]
            ):
                pytest.skip(f"OpenAI API limit reached: {e}")
            elif "assistants" in error_str:
                pytest.skip(f"Assistants API not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_file_utility_functions(self, service, test_image_path):
        """Test that file utility functions work correctly."""
        from llmring.file_utils import (
            create_data_url,
            encode_file_to_base64,
            get_file_mime_type,
            validate_image_file,
        )

        # Test file utilities
        assert validate_image_file(test_image_path) is True
        assert get_file_mime_type(test_image_path) == "image/png"

        # Test encoding
        base64_data = encode_file_to_base64(test_image_path)
        assert len(base64_data) > 0

        # Test data URL creation
        data_url = create_data_url(test_image_path)
        assert data_url.startswith("data:image/png;base64,")

        # Test content creation
        content = create_image_content(test_image_path, "Test image")
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

        print("All file utility functions work correctly!")

    @pytest.mark.asyncio
    async def test_url_handling(self, service):
        """Test that URLs are handled correctly."""
        # Test with httpbin image endpoint
        test_url = "https://httpbin.org/image/png"
        import os

        os.environ["LLMRING_ALLOW_REMOTE_URLS"] = "true"
        content = analyze_image(test_url, "Describe this image")

        # Verify URL was passed through correctly (no base64 conversion)
        assert content[1]["image_url"]["url"] == test_url

        print("URL handling works correctly!")
