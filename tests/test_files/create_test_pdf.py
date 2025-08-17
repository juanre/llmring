#!/usr/bin/env python3
"""
Create a simple test PDF file with known content for testing file extraction.
"""

import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_test_pdf(output_path: str) -> str:
    """
    Create a simple PDF file with known test content.

    Args:
        output_path: Path where to save the PDF file

    Returns:
        The test content that was written to the PDF
    """
    # Known test content
    test_content = """LLM Service Test Document

This is a test PDF file created for the LLM Service file processing tests.

Key Information:
- Document Type: Test PDF
- Creation Purpose: Automated Testing
- Expected Extraction: This exact text should be extractable
- Test ID: PDF-TEST-001
- Special Characters: áéíóú, ñ, ç, €, $, %

Test Data:
Name: John Doe
Email: john.doe@example.com
Phone: +1-555-123-4567
Date: 2024-01-15

Instructions for LLM:
Please extract all text from this document and confirm you can read:
1. The title "LLM Service Test Document"
2. The test ID "PDF-TEST-001"
3. The name "John Doe"
4. The email "john.doe@example.com"

End of test document."""

    # Create the PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Set font
    c.setFont("Helvetica", 12)

    # Split content into lines and write to PDF
    lines = test_content.split("\n")
    y_position = height - 50  # Start near top of page

    for line in lines:
        if y_position < 50:  # Start new page if near bottom
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50

        c.drawString(50, y_position, line)
        y_position -= 20

    c.save()
    return test_content


def create_simple_test_image(output_path: str) -> str:
    """
    Create a simple test image with text for OCR testing.

    Args:
        output_path: Path where to save the image file

    Returns:
        The text content that was written to the image
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create a simple image with text
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fallback to default
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()

    test_text = """LLM Service Vision Test

This is a test image for OCR extraction.

Test Information:
- Image Type: PNG Test Image
- Test ID: IMG-TEST-001
- Purpose: Vision model testing

Please extract this text accurately."""

    # Draw text on image
    lines = test_text.strip().split("\n")
    y_position = 50

    for line in lines:
        draw.text((50, y_position), line.strip(), fill="black", font=font)
        y_position += 40

    img.save(output_path, "PNG")
    return test_text


if __name__ == "__main__":
    # Create test files directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(test_dir, exist_ok=True)

    # Create test PDF
    pdf_path = os.path.join(test_dir, "test_document.pdf")
    pdf_content = create_test_pdf(pdf_path)
    print(f"Created test PDF: {pdf_path}")
    print(f"PDF contains {len(pdf_content)} characters")

    # Create test image
    img_path = os.path.join(test_dir, "test_image.png")
    img_content = create_simple_test_image(img_path)
    print(f"Created test image: {img_path}")
    print(f"Image contains text: {len(img_content)} characters")

    # Save expected content to text files for easy comparison
    with open(os.path.join(test_dir, "test_document_expected.txt"), "w") as f:
        f.write(pdf_content)

    with open(os.path.join(test_dir, "test_image_expected.txt"), "w") as f:
        f.write(img_content)

    print("\nTest files created successfully!")
    print("Expected content saved to *_expected.txt files")
