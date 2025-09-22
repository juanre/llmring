"""
Comprehensive integration tests for Enhanced LLM file processing.

This test suite creates real files using Pillow and tests the complete
file processing pipeline with actual LLM API calls (no mocking).

Prerequisites:
- At least one LLM provider API key set in environment variables
- Internet connection for URL file testing
- PostgreSQL database (optional, for full integration testing)
"""

import base64
import io
import os

# Add project paths
import sys
import tempfile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/Users/juanre/prj/mcpp/mcp-server/mcp-client/src")
sys.path.insert(0, "/Users/juanre/prj/mcpp/mcp-server/llm_service/src")

import builtins
import contextlib

from llmring.exceptions import FileAccessError, FileProcessingError, InvalidFileFormatError
from llmring.mcp.client.enhanced_llm import create_enhanced_llm


class FileCreator:
    """Create test files dynamically for integration testing."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.created_files = []

    def create_test_image(
        self, filename: str, text: str = "TEST IMAGE", size: tuple = (400, 300)
    ) -> Path:
        """Create a test PNG image with text."""
        image = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(image)

        # Try to use a better font, fall back to default
        try:
            font = ImageFont.truetype("Arial.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                font = ImageFont.load_default()

        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        draw.text((x, y), text, fill="black", font=font)

        file_path = self.temp_dir / filename
        image.save(file_path, "PNG")
        self.created_files.append(file_path)
        return file_path

    def create_test_chart(self, filename: str) -> Path:
        """Create a test chart image with data."""
        image = Image.new("RGB", (500, 400), color="white")
        draw = ImageDraw.Draw(image)

        # Draw a simple bar chart
        data = [30, 50, 80, 40, 70]
        bar_width = 60
        max_height = 250
        start_x = 50
        start_y = 320

        # Draw bars
        for i, value in enumerate(data):
            x = start_x + i * (bar_width + 20)
            height = int((value / 100) * max_height)
            y = start_y - height

            # Draw bar
            draw.rectangle([x, y, x + bar_width, start_y], fill="blue", outline="black")

            # Draw value label
            draw.text((x + 10, y - 20), str(value), fill="black")

            # Draw x-axis label
            draw.text((x + 15, start_y + 10), f"Q{i + 1}", fill="black")

        # Draw title
        draw.text((150, 30), "Quarterly Sales Data", fill="black")

        # Draw y-axis
        draw.line(
            [start_x - 10, start_y, start_x - 10, start_y - max_height],
            fill="black",
            width=2,
        )

        file_path = self.temp_dir / filename
        image.save(file_path, "PNG")
        self.created_files.append(file_path)
        return file_path

    def create_test_pdf_content(self, filename: str) -> Path:
        """Create a simple text file that simulates PDF content."""
        content = """
QUARTERLY BUSINESS REPORT
Q4 2024

EXECUTIVE SUMMARY
This report presents the key findings from Q4 2024 operations.

KEY METRICS:
- Revenue: $2.5M (increase of 15% from Q3)
- Customer Growth: 1,200 new customers
- Product Sales: 8,500 units sold
- Customer Satisfaction: 4.7/5.0

RECOMMENDATIONS:
1. Expand marketing in high-performing regions
2. Increase inventory for popular products
3. Improve customer support response times

CONCLUSION:
Q4 2024 showed strong performance across all metrics.
The company is well-positioned for continued growth in 2025.
        """

        file_path = self.temp_dir / filename
        with open(file_path, "w") as f:
            f.write(content.strip())
        self.created_files.append(file_path)
        return file_path

    def get_base64_image(self, filename: str, text: str = "BASE64 TEST") -> str:
        """Create an image and return it as base64."""
        image = Image.new("RGB", (300, 200), color="lightblue")
        draw = ImageDraw.Draw(image)
        draw.text((50, 80), text, fill="black")

        # Convert to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    def cleanup(self):
        """Clean up created files."""
        for file_path in self.created_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                pass
        with contextlib.suppress(builtins.BaseException):
            self.temp_dir.rmdir()


@pytest.fixture
def file_creator():
    """Fixture to provide a file creator instance."""
    creator = FileCreator()
    yield creator
    creator.cleanup()


@pytest_asyncio.fixture
async def enhanced_llm():
    """Fixture to provide an Enhanced LLM instance."""
    # Use vision-capable models for file processing tests
    model = "anthropic_vision"  # Vision-capable Claude model

    # Check if we have any API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))

    if has_openai:
        model = "openai:gpt-4o-mini"  # Vision-capable OpenAI model
    elif has_google:
        model = "google:gemini-1.5-flash"  # Vision-capable Google model
    elif not has_anthropic:
        pytest.skip(
            "No LLM API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY"
        )

    llm = create_enhanced_llm(
        llm_model=model,
        llmring_server_url=None,  # No server for basic tests
        origin="integration-test",
    )

    yield llm

    # Proper async cleanup
    await llm.close()


class TestFileProcessingIntegration:
    """Integration tests for file processing with real LLM calls."""

    @pytest.mark.asyncio
    async def test_image_processing_from_upload(self, enhanced_llm, file_creator):
        """Test processing an uploaded image file with OCR."""
        # Create test image with text
        image_path = file_creator.create_test_image("test_ocr.png", "HELLO WORLD 123")

        # Process file from upload source
        attachment = await enhanced_llm.process_file_from_source(
            source_type="upload",
            source_data=str(image_path),
            filename="test_ocr.png",
            content_type="image/png",
        )

        assert attachment["type"] == "file"
        assert attachment["filename"] == "test_ocr.png"
        assert attachment["content_type"] == "image/png"
        assert isinstance(attachment["data"], bytes)

        # Test with Enhanced LLM chat (real API call)
        response = await enhanced_llm.chat(
            [
                {"role": "system", "content": "Extract text from images accurately."},
                {
                    "role": "user",
                    "content": "What text do you see in this image?",
                    "attachments": [attachment],
                },
            ]
        )

        assert response.content is not None
        assert len(response.content) > 0
        # The response should mention some of the text from the image
        content_lower = response.content.lower()
        assert "hello" in content_lower or "world" in content_lower or "123" in content_lower

        print(f"✓ OCR Test Response: {response.content[:100]}...")

    @pytest.mark.asyncio
    async def test_chart_analysis(self, enhanced_llm, file_creator):
        """Test chart analysis with a real chart image."""
        # Create test chart
        chart_path = file_creator.create_test_chart("sales_chart.png")

        # Process and analyze
        attachment = await enhanced_llm.process_file_from_source(
            source_type="upload",
            source_data=str(chart_path),
            filename="sales_chart.png",
        )

        response = await enhanced_llm.chat(
            [
                {
                    "role": "system",
                    "content": "Analyze charts and extract data accurately.",
                },
                {
                    "role": "user",
                    "content": "What data does this chart show? Describe the trends.",
                    "attachments": [attachment],
                },
            ]
        )

        assert response.content is not None
        content_lower = response.content.lower()
        # Should mention chart/data concepts
        assert any(word in content_lower for word in ["chart", "data", "bar", "sales", "quarter"])

        print(f"✓ Chart Analysis Response: {response.content[:100]}...")

    @pytest.mark.asyncio
    async def test_base64_image_processing(self, enhanced_llm, file_creator):
        """Test processing a base64 encoded image."""
        # Create base64 image
        base64_data = file_creator.get_base64_image("base64_test.png", "BASE64 IMAGE")

        # Process from base64 source
        attachment = await enhanced_llm.process_file_from_source(
            source_type="base64",
            source_data=base64_data,
            filename="base64_test.png",
            content_type="image/png",
        )

        assert attachment["content_type"] == "image/png"

        # Test with LLM
        response = await enhanced_llm.chat(
            [
                {
                    "role": "user",
                    "content": "Describe what you see in this image.",
                    "attachments": [attachment],
                }
            ]
        )

        assert response.content is not None
        print(f"✓ Base64 Image Response: {response.content[:100]}...")

    @pytest.mark.asyncio
    async def test_url_file_processing(self, enhanced_llm):
        """Test processing a file from URL."""
        # Use a reliable public image URL (GitHub's avatar service)
        test_url = "https://github.com/github.png"  # GitHub's public logo

        try:
            # Test URL accessibility first
            async with httpx.AsyncClient() as client:
                response = await client.head(test_url, timeout=10.0)
                if response.status_code != 200:
                    pytest.skip(f"Test URL not accessible: {test_url}")

            # Process from URL
            attachment = await enhanced_llm.process_file_from_source(
                source_type="url", source_data=test_url, filename="url_test.png"
            )

            assert attachment["type"] == "file"
            assert "image" in attachment["content_type"]

            # Test with LLM
            response = await enhanced_llm.chat(
                [
                    {
                        "role": "user",
                        "content": "What do you see in this image? Describe the logo or branding.",
                        "attachments": [attachment],
                    }
                ]
            )

            assert response.content is not None
            print(f"✓ URL File Response: {response.content[:100]}...")

        except (httpx.TimeoutException, httpx.ConnectError):
            pytest.skip("Network connection required for URL file test")

    @pytest.mark.asyncio
    async def test_document_processing(self, enhanced_llm, file_creator):
        """Test processing a text document (simulating document analysis)."""
        # Create test document
        doc_path = file_creator.create_test_pdf_content("quarterly_report.txt")

        # Read file as bytes
        with open(doc_path, "rb") as f:
            doc_bytes = f.read()

        # For text documents, use text/plain content type (not PDF for OpenAI)
        attachment = {
            "type": "file",
            "filename": "quarterly_report.txt",
            "content_type": "text/plain",
            "data": doc_bytes,
            "parameter_name": "report",
        }

        # Test document analysis
        response = await enhanced_llm.chat(
            [
                {
                    "role": "system",
                    "content": "Analyze business documents and extract key insights.",
                },
                {
                    "role": "user",
                    "content": "Summarize the key findings and metrics from this quarterly report.",
                    "attachments": [attachment],
                },
            ]
        )

        assert response.content is not None
        content_lower = response.content.lower()
        # Should mention business concepts from the document
        assert any(
            word in content_lower for word in ["revenue", "quarter", "customer", "growth", "report"]
        )

        print(f"✓ Document Analysis Response: {response.content[:150]}...")

    @pytest.mark.asyncio
    async def test_multiple_file_processing(self, enhanced_llm, file_creator):
        """Test processing multiple files in one conversation."""
        # Create multiple files
        chart_path = file_creator.create_test_chart("multi_chart.png")
        doc_path = file_creator.create_test_pdf_content("multi_report.txt")

        # Process both files
        chart_attachment = await enhanced_llm.process_file_from_source(
            source_type="upload", source_data=str(chart_path), filename="chart.png"
        )

        with open(doc_path, "rb") as f:
            doc_bytes = f.read()

        doc_attachment = {
            "type": "file",
            "filename": "report.txt",
            "content_type": "text/plain",
            "data": doc_bytes,
            "parameter_name": "report",
        }

        # Analyze both together
        response = await enhanced_llm.chat(
            [
                {"role": "system", "content": "You are a business analyst."},
                {
                    "role": "user",
                    "content": "Compare the chart data with the written report. Are they consistent?",
                    "attachments": [chart_attachment, doc_attachment],
                },
            ]
        )

        assert response.content is not None
        print(f"✓ Multi-file Analysis Response: {response.content[:150]}...")

    @pytest.mark.asyncio
    async def test_error_handling(self, enhanced_llm):
        """Test error handling for invalid file processing."""
        # Test invalid base64
        with pytest.raises((InvalidFileFormatError, FileProcessingError), match="Invalid base64"):
            await enhanced_llm.process_file_from_source(
                source_type="base64",
                source_data="invalid-base64-data",
                filename="test.png",
            )

        # Test invalid source type
        with pytest.raises((ValueError, FileProcessingError), match="Unsupported source type"):
            await enhanced_llm.process_file_from_source(
                source_type="invalid_source",
                source_data="some_data",
                filename="test.txt",
            )

        # Test non-existent file
        with pytest.raises((FileAccessError, FileProcessingError), match="File not found"):
            await enhanced_llm.process_file_from_source(
                source_type="upload",
                source_data="/nonexistent/path/file.txt",
                filename="test.txt",
            )

        print("✓ Error handling tests passed")

    @pytest.mark.asyncio
    async def test_content_type_detection(self, enhanced_llm, file_creator):
        """Test automatic content type detection."""
        # Create image without explicit content type
        image_path = file_creator.create_test_image("detection_test.png", "DETECTION TEST")

        attachment = await enhanced_llm.process_file_from_source(
            source_type="upload",
            source_data=str(image_path),
            filename="detection_test.png",
            # No content_type specified - should be auto-detected
        )

        assert attachment["content_type"] == "image/png"
        print("✓ Content type auto-detection works")

    @pytest.mark.asyncio
    async def test_file_size_handling(self, enhanced_llm, file_creator):
        """Test handling of different file sizes."""
        # Create a larger image
        large_image_path = file_creator.create_test_image(
            "large_test.png", "LARGE TEST IMAGE", size=(800, 600)
        )

        attachment = await enhanced_llm.process_file_from_source(
            source_type="upload",
            source_data=str(large_image_path),
            filename="large_test.png",
        )

        # Should handle larger files correctly (adjust threshold based on actual compression)
        assert len(attachment["data"]) > 5000  # Reasonable size for compressed PNG

        # Test with LLM
        response = await enhanced_llm.chat(
            [
                {
                    "role": "user",
                    "content": "What text do you see?",
                    "attachments": [attachment],
                }
            ]
        )

        assert response.content is not None
        print(f"✓ Large file handling: {len(attachment['data'])} bytes processed")


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys

    # Check for API keys
    api_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }

    available_keys = [k for k, v in api_keys.items() if v]

    if not available_keys:
        print("❌ No LLM API keys found!")
        print("Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
        print("\nExample:")
        print("export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"✓ Found API keys: {', '.join(available_keys)}")
    print("\nRunning comprehensive file processing integration tests...")
    print("=" * 60)

    # Run the tests
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"])
