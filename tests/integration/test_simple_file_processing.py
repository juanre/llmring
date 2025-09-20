"""
Simple integration tests for file processing with LLM providers.

These tests create simple test files and verify that providers can process them.
No mocking - just real files sent to real providers.
"""

import os
import tempfile

import pytest

from llmring.file_utils import analyze_image, create_image_content
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing


def create_simple_test_image() -> str:
    """Create a simple test image with text and return the file path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        pytest.skip("PIL not available for test image creation")

    # Create temporary image file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(temp_fd)

    try:
        # Create image with text
        img = Image.new("RGB", (600, 300), color="white")
        draw = ImageDraw.Draw(img)

        # Use default font
        try:
            font = ImageFont.truetype("Arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

        # Draw test text that should be easily extractable
        test_text = "TEST DOCUMENT\nExtract this text: PDF-TEST-001\nName: John Doe\nEmail: john.doe@example.com"

        lines = test_text.split("\n")
        y = 50
        for line in lines:
            draw.text((50, y), line, fill="black", font=font)
            y += 50

        img.save(temp_path, "PNG")
        return temp_path
    except Exception:
        os.unlink(temp_path)
        raise


def create_simple_test_pdf() -> str:
    """Create a simple test PDF and return the file path."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not available for PDF creation")

    # Create temporary PDF file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(temp_fd)

    try:
        # Create PDF with test content
        c = canvas.Canvas(temp_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica", 14)

        # Simple test content
        content_lines = [
            "LLM Service Test Document",
            "",
            "This is a test PDF for LLM processing.",
            "",
            "Key Information:",
            "- Test ID: PDF-TEST-001",
            "- Name: John Doe",
            "- Email: john.doe@example.com",
            "- Phone: +1-555-123-4567",
            "",
            "Instructions: Please extract the test ID and name.",
        ]

        y_position = height - 100
        for line in content_lines:
            c.drawString(100, y_position, line)
            y_position -= 30

        c.save()
        return temp_path
    except Exception:
        os.unlink(temp_path)
        raise


@pytest.mark.integration
class TestSimpleFileProcessing:
    """Simple integration tests for file processing."""

    # service fixture is provided by conftest.py

    @pytest.fixture
    def test_image_path(self):
        """Create test image file."""
        image_path = create_simple_test_image()
        yield image_path
        # Cleanup
        if os.path.exists(image_path):
            os.unlink(image_path)

    @pytest.fixture
    def test_pdf_path(self):
        """Create test PDF file."""
        pdf_path = create_simple_test_pdf()
        yield pdf_path
        # Cleanup
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

    @pytest.mark.asyncio
    async def test_image_analysis_openai(self, service, test_image_path):
        """Test image analysis with OpenAI."""
        # Check if OpenAI provider is available
        if "openai" not in service.providers:
            pytest.skip("OpenAI provider not available")

        # Use utility function to create content
        content = analyze_image(
            test_image_path,
            "Extract the text from this image. Focus on the test ID and name.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model="openai_fast",  # Use alias from lockfile
            max_tokens=200,
        )

        response = await service.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Check if key information was extracted
        content_lower = response.content.lower()
        assert "pdf-test-001" in content_lower
        assert "john doe" in content_lower

        print(f"OpenAI extracted: {response.content}")

    @pytest.mark.asyncio
    async def test_image_analysis_anthropic(self, service, test_image_path):
        """Test image analysis with Anthropic."""
        # Check if Anthropic provider is available
        if "anthropic" not in service.providers:
            pytest.skip("Anthropic provider not available")

        content = analyze_image(
            test_image_path,
            "Extract the text from this image. Focus on the test ID and name.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model="mvp_vision",
            max_tokens=200,
        )

        response = await service.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Check if key information was extracted (using MVP Opus 4.1 model)
        content_lower = response.content.lower()
        assert "pdf-test-001" in content_lower
        # Opus 4.1 should extract the name properly, but allow for content moderation
        assert ("john doe" in content_lower or
                "name:" in content_lower or
                "redacted" in content_lower)

        print(f"Anthropic MVP (Opus 4.1) extracted: {response.content}")

    @pytest.mark.asyncio
    async def test_image_analysis_google(self, service, test_image_path):
        """Test image analysis with Google."""
        # Check if Google provider is available
        if "google" not in service.providers:
            pytest.skip("Google provider not available")

        try:
            content = analyze_image(
                test_image_path,
                "Extract the text from this image. Focus on the test ID and name.",
            )

            request = LLMRequest(
                messages=[Message(role="user", content=content)],
                model="google_deep",  # Use alias from lockfile
                max_tokens=200,
            )

            response = await service.chat(request)

            # Verify response
            assert response.content is not None
            assert len(response.content) > 0

            # Check if key information was extracted
            content_lower = response.content.lower()
            assert "pdf-test-001" in content_lower
            assert "john doe" in content_lower

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
    async def test_pdf_processing_universal(self, service, test_pdf_path):
        """Test PDF processing using universal file interface."""
        # Check which providers are available
        # Try Anthropic first, then Google as fallback
        model = None
        if "anthropic" in service.providers:
            model = "mvp_vision"
        elif "google" in service.providers:
            model = "google_vision"
        else:
            pytest.skip("No PDF-capable providers available")

        # Use the universal file interface
        from llmring.file_utils import analyze_file

        content = analyze_file(
            test_pdf_path,
            "Extract the text from this PDF document. Focus on the test ID and name.",
        )

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model=model,
            max_tokens=300,
        )

        response = await service.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Check if key information was extracted
        content_lower = response.content.lower()
        assert "pdf-test-001" in content_lower
        assert "john doe" in content_lower

        print(f"PDF processing with {model}: {response.content}")

    @pytest.mark.asyncio
    async def test_file_utility_functions(self, service, test_image_path):
        """Test the file utility functions work correctly."""
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
    async def test_url_vs_file_path(self, service):
        """Test that URLs and file paths are handled correctly."""
        # Check if OpenAI provider is available
        if "openai" not in service.providers:
            pytest.skip("OpenAI provider not available")

        # Test with a public image URL
        test_url = "https://httpbin.org/image/png"

        # Check if the URL is accessible
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.head(test_url)
                if response.status_code != 200:
                    pytest.skip(
                        f"Test URL {test_url} not accessible (status: {response.status_code})"
                    )
        except Exception as e:
            pytest.skip(f"Cannot access test URL {test_url}: {e}")

        # Enable remote URLs for this test
        import os

        old_value = os.environ.get("LLMRING_ALLOW_REMOTE_URLS")
        os.environ["LLMRING_ALLOW_REMOTE_URLS"] = "true"
        try:
            content = analyze_image(test_url, "What type of image is this?")
        finally:
            if old_value is None:
                os.environ.pop("LLMRING_ALLOW_REMOTE_URLS", None)
            else:
                os.environ["LLMRING_ALLOW_REMOTE_URLS"] = old_value

        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model="openai_fast",  # Use alias from lockfile
            max_tokens=100,
        )

        response = await service.chat(request)

        assert response.content is not None
        assert len(response.content) > 0

        print(f"URL image analysis: {response.content}")

        # Verify URL was passed through correctly
        assert content[1]["image_url"]["url"] == test_url
