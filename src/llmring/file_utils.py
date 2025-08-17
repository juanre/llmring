"""
File handling utilities for the LLM Service.

This module provides utilities for working with files in LLM requests,
including automatic conversion to base64, file validation, and format detection.
"""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def get_file_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string (e.g., 'image/png', 'application/pdf')
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Fallback based on file extension
        ext = Path(file_path).suffix.lower()
        extension_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        return extension_map.get(ext, "application/octet-stream")
    return mime_type


def encode_file_to_base64(file_path: str) -> str:
    """
    Encode a file to base64 string.

    Args:
        file_path: Path to the file to encode

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}")


def create_data_url(file_path: str) -> str:
    """
    Create a data URL from a file.

    Args:
        file_path: Path to the file

    Returns:
        Data URL string (e.g., 'data:image/png;base64,iVBORw0KGgo...')
    """
    mime_type = get_file_mime_type(file_path)
    base64_data = encode_file_to_base64(file_path)
    return f"data:{mime_type};base64,{base64_data}"


def validate_image_file(file_path: str) -> bool:
    """
    Validate if a file is a supported image format.

    Args:
        file_path: Path to the file

    Returns:
        True if file is a supported image format
    """
    if not os.path.exists(file_path):
        return False

    mime_type = get_file_mime_type(file_path)
    supported_types = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    return mime_type in supported_types


def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF.

    Args:
        file_path: Path to the file

    Returns:
        True if file is a PDF
    """
    if not os.path.exists(file_path):
        return False

    mime_type = get_file_mime_type(file_path)
    return mime_type == "application/pdf"


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image.

    Args:
        file_path: Path to the file

    Returns:
        True if file is an image format
    """
    if not os.path.exists(file_path):
        return False

    mime_type = get_file_mime_type(file_path)
    return mime_type.startswith("image/")


def is_document_file(file_path: str) -> bool:
    """
    Check if a file is a document (non-image).

    Args:
        file_path: Path to the file

    Returns:
        True if file is a document format
    """
    if not os.path.exists(file_path):
        return False

    mime_type = get_file_mime_type(file_path)
    # Documents are non-image files
    return not mime_type.startswith("image/")


def validate_file_for_vision_api(file_path: str) -> None:
    """
    Validate that a file can be used with vision APIs (OpenAI Chat Completions).

    Args:
        file_path: Path to the file to validate

    Raises:
        ValueError: If file is not supported for vision APIs
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = get_file_mime_type(file_path)

    # Check if it's a PDF
    if mime_type == "application/pdf":
        raise ValueError(
            f"PDF files are not supported in OpenAI Chat Completions API. "
            f"To process PDFs, you need to either: "
            f"1) Use OpenAI Assistants API with file uploads, or "
            f"2) Convert PDF pages to images first. "
            f"File: {file_path}"
        )

    # Check if it's a supported image format
    supported_image_types = {"image/png", "image/jpeg", "image/gif", "image/webp"}

    if mime_type not in supported_image_types:
        raise ValueError(
            f"Unsupported file type '{mime_type}' for OpenAI vision API. "
            f"Supported image types: {', '.join(supported_image_types)}. "
            f"For PDF files, use OpenAI Assistants API instead. "
            f"File: {file_path}"
        )


def create_file_content(
    file_path_url_or_base64: str, text: str = "", mime_type: str = None
) -> List[Dict[str, Any]]:
    """
    Create file content for LLM messages from file path, URL, or base64 data.
    Works for both images and documents (PDFs, etc.).

    Args:
        file_path_url_or_base64: One of:
            - Local file path (e.g., "document.pdf", "image.png")
            - HTTP(S) URL (e.g., "https://example.com/doc.pdf")
            - Base64 string (detected if it looks like base64 data)
            - Data URL (e.g., "data:application/pdf;base64,JVBERi0...")
        text: Optional text to include with the file
        mime_type: MIME type for base64 data (auto-detected for file paths)

    Returns:
        Content list suitable for Message.content

    Examples:
        # PDF file
        content = create_file_content("document.pdf", "Analyze this document")

        # Image file
        content = create_file_content("chart.png", "What's in this chart?")

        # URL
        content = create_file_content("https://example.com/report.pdf", "Summarize this")

        # Base64 PDF
        content = create_file_content("JVBERi0xLjQ...", "Review this", "application/pdf")
    """
    content_parts = []

    # Add text if provided
    if text:
        content_parts.append({"type": "text", "text": text})

    # Determine input type and handle accordingly
    if file_path_url_or_base64.startswith("data:"):
        # It's already a data URL - extract info and use universal format
        header, data = file_path_url_or_base64.split(",", 1)
        detected_mime_type = header.split(":")[1].split(";")[0]

        if detected_mime_type.startswith("image/"):
            # Image content
            content_parts.append(
                {"type": "image_url", "image_url": {"url": file_path_url_or_base64}}
            )
        else:
            # Document content - use universal format
            content_parts.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": detected_mime_type,
                        "data": data,
                    },
                }
            )

    elif file_path_url_or_base64.startswith(("http://", "https://")):
        # It's an HTTP(S) URL - assume image for now (documents via URL are less common)
        content_parts.append(
            {"type": "image_url", "image_url": {"url": file_path_url_or_base64}}
        )

    elif _is_base64_string(file_path_url_or_base64):
        # It's base64 data - need mime_type to determine format
        if not mime_type:
            raise ValueError("mime_type is required for base64 data")

        if mime_type.startswith("image/"):
            # Image content
            url = f"data:{mime_type};base64,{file_path_url_or_base64}"
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            # Document content
            content_parts.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": file_path_url_or_base64,
                    },
                }
            )
    else:
        # Assume it's a file path
        if not os.path.exists(file_path_url_or_base64):
            raise FileNotFoundError(f"File not found: {file_path_url_or_base64}")

        detected_mime_type = get_file_mime_type(file_path_url_or_base64)

        if is_image_file(file_path_url_or_base64):
            # Image file - use existing image logic
            validate_file_for_vision_api(file_path_url_or_base64)
            url = create_data_url(file_path_url_or_base64)
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            # Document file
            base64_data = encode_file_to_base64(file_path_url_or_base64)
            content_parts.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": detected_mime_type,
                        "data": base64_data,
                    },
                }
            )

    return content_parts


def create_image_content(
    file_path_url_or_base64: str, text: str = "", mime_type: str = "image/jpeg"
) -> List[Dict[str, Any]]:
    """
    Create image content for LLM messages from file path, URL, or base64 data.

    Args:
        file_path_url_or_base64: One of:
            - Local file path (e.g., "screenshot.png")
            - HTTP(S) URL (e.g., "https://example.com/image.jpg")
            - Base64 string (detected if it looks like base64 data)
            - Data URL (e.g., "data:image/png;base64,iVBORw0KGgo...")
        text: Optional text to include with the image
        mime_type: MIME type for base64 data (default: "image/jpeg")

    Returns:
        Content list suitable for Message.content

    Examples:
        # File path
        content = create_image_content("screenshot.png", "What's in this image?")

        # URL
        content = create_image_content("https://example.com/chart.jpg", "Analyze this chart")

        # Base64 string
        content = create_image_content("iVBORw0KGgoAAAANSUhEUgAA...", "Describe this", "image/png")

        # Data URL (passes through)
        content = create_image_content("data:image/png;base64,iVBORw0KGgo...", "What's this?")
    """
    content_parts = []

    # Add text if provided
    if text:
        content_parts.append({"type": "text", "text": text})

    # Determine input type and handle accordingly
    if file_path_url_or_base64.startswith("data:"):
        # It's already a data URL - use directly
        url = file_path_url_or_base64
    elif file_path_url_or_base64.startswith(("http://", "https://")):
        # It's an HTTP(S) URL - use directly
        url = file_path_url_or_base64
    elif _is_base64_string(file_path_url_or_base64):
        # It's base64 data - create data URL
        url = f"data:{mime_type};base64,{file_path_url_or_base64}"
    else:
        # Assume it's a file path - validate and convert to data URL
        validate_file_for_vision_api(file_path_url_or_base64)
        url = create_data_url(file_path_url_or_base64)

    content_parts.append({"type": "image_url", "image_url": {"url": url}})

    return content_parts


def _is_base64_string(s: str) -> bool:
    """
    Check if a string looks like base64 data.

    Args:
        s: String to check

    Returns:
        True if string appears to be base64 data
    """
    # Basic heuristics for base64 detection:
    # - Only contains base64 characters
    # - Length is reasonable for image data (>100 chars)
    # - Doesn't look like a file path or URL

    if len(s) < 100:  # Too short for image data
        return False

    # Check if it contains only base64 characters first
    base64_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    if not all(c in base64_chars for c in s):
        return False

    # Additional heuristics to avoid false positives
    if "/" in s and len(s) < 500:  # Likely a short file path
        return False

    if "." in s and len(s) < 1000:  # Likely a filename
        return False

    # If we get here, it's probably base64
    return True


def create_multi_image_content(
    images: List[Union[str, Dict[str, str]]], text: str = ""
) -> List[Dict[str, Any]]:
    """
    Create content with multiple images for LLM messages.

    Args:
        images: List of image sources. Each item can be:
            - String: file path, URL, base64 data, or data URL
            - Dict: {"data": "base64_string", "mime_type": "image/png"} for explicit base64
        text: Optional text to include with the images

    Returns:
        Content list suitable for Message.content

    Examples:
        # Mixed sources
        content = create_multi_image_content([
            "image1.png",  # file path
            "https://example.com/image2.jpg",  # URL
            "iVBORw0KGgoAAAANSUhEUgAA...",  # base64 (auto-detected)
            {"data": "iVBORw0KGgo...", "mime_type": "image/png"}  # explicit base64
        ], "Compare these images")
    """
    content_parts = []

    # Add text if provided
    if text:
        content_parts.append({"type": "text", "text": text})

    # Add each image
    for image in images:
        if isinstance(image, dict):
            # Explicit base64 with mime type
            base64_data = image["data"]
            mime_type = image.get("mime_type", "image/jpeg")
            url = f"data:{mime_type};base64,{base64_data}"
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            # String - use create_image_content logic
            image_content = create_image_content(image)
            # Extract just the image part (skip text if any)
            for part in image_content:
                if part["type"] == "image_url":
                    content_parts.append(part)
                    break

    return content_parts


# Convenience functions for common use cases
def analyze_file(
    file_path_url_or_base64: str,
    prompt: str = "Analyze this file",
    mime_type: str = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to create file analysis content for both images and documents.

    Args:
        file_path_url_or_base64: File path, URL, base64 string, or data URL
        prompt: Analysis prompt
        mime_type: MIME type for base64 data (auto-detected for file paths)

    Examples:
        # PDF document
        content = analyze_file("document.pdf", "Extract key information from this document")

        # Image file
        content = analyze_file("chart.png", "What's in this chart?")

        # Base64 PDF data
        content = analyze_file("JVBERi0xLjQ...", "Summarize this document", "application/pdf")

        # URL
        content = analyze_file("https://example.com/report.pdf", "Extract main points")
    """
    return create_file_content(file_path_url_or_base64, prompt, mime_type)


def analyze_image(
    file_path_url_or_base64: str,
    prompt: str = "Analyze this image",
    mime_type: str = "image/jpeg",
) -> List[Dict[str, Any]]:
    """
    Convenience function to create image analysis content.

    Args:
        file_path_url_or_base64: File path, URL, base64 string, or data URL
        prompt: Analysis prompt
        mime_type: MIME type for base64 data (default: "image/jpeg")

    Examples:
        # File path
        content = analyze_image("screenshot.png", "What's in this image?")

        # Base64 data
        content = analyze_image("iVBORw0KGgoAAAANSUhEUgAA...", "Describe this", "image/png")

        # URL
        content = analyze_image("https://example.com/chart.jpg", "Extract data")
    """
    return create_image_content(file_path_url_or_base64, prompt, mime_type)


def extract_text_from_image(
    file_path_url_or_base64: str, mime_type: str = "image/jpeg"
) -> List[Dict[str, Any]]:
    """
    Convenience function to create OCR content.

    Args:
        file_path_url_or_base64: File path, URL, base64 string, or data URL
        mime_type: MIME type for base64 data (default: "image/jpeg")
    """
    return create_image_content(
        file_path_url_or_base64,
        "Extract all text from this image. Preserve formatting and layout.",
        mime_type,
    )


def compare_images(
    image1: Union[str, Dict[str, str]],
    image2: Union[str, Dict[str, str]],
    prompt: str = "Compare these images",
) -> List[Dict[str, Any]]:
    """
    Convenience function to create image comparison content.

    Args:
        image1: First image (file path, URL, base64, or dict with data/mime_type)
        image2: Second image (file path, URL, base64, or dict with data/mime_type)
        prompt: Comparison prompt

    Examples:
        # File paths
        content = compare_images("before.png", "after.png", "What changed?")

        # Mixed sources
        content = compare_images("image.png", "iVBORw0KGgo...", "Compare these")

        # Explicit base64
        content = compare_images(
            {"data": "iVBORw0KGgo...", "mime_type": "image/png"},
            "after.jpg",
            "What's different?"
        )
    """
    return create_multi_image_content([image1, image2], prompt)


def create_base64_image_content(
    base64_data: str, mime_type: str = "image/jpeg", text: str = ""
) -> List[Dict[str, Any]]:
    """
    Explicit function for creating image content from base64 data.

    Args:
        base64_data: Base64 encoded image data (without data URL prefix)
        mime_type: MIME type of the image (e.g., "image/png", "image/jpeg")
        text: Optional text to include with the image

    Returns:
        Content list suitable for Message.content

    Example:
        content = create_base64_image_content(
            "iVBORw0KGgoAAAANSUhEUgAA...",
            "image/png",
            "Analyze this image"
        )
    """
    content_parts = []

    if text:
        content_parts.append({"type": "text", "text": text})

    data_url = f"data:{mime_type};base64,{base64_data}"
    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    return content_parts
