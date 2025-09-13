"""
File processing utilities for MCP client.

This module consolidates all file handling logic including:
- Content type detection
- URL fetching
- Base64 decoding
- File processing from various sources
"""

import base64
import mimetypes
import os
from typing import Dict, Tuple, Union
from urllib.parse import urlparse

import httpx

from llmring.exceptions import (
    FileProcessingError,
    InvalidFileFormatError,
    FileAccessError,
    ValidationError,
)

# Try to import magic for better content type detection
try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


def guess_content_type_from_bytes(data: bytes) -> str:
    """
    Guess content type from file bytes.

    Args:
        data: File content as bytes

    Returns:
        MIME type string
    """
    # Try magic library if available
    if HAS_MAGIC:
        try:
            mime = magic.Magic(mime=True)
            content_type = mime.from_buffer(data)
            return content_type
        except (ImportError, OSError, RuntimeError):
            # Magic library issues - fall back to basic detection
            pass
        except Exception:
            # Unexpected error with magic library - log and fall back
            pass

        # Basic detection based on file signatures
        if data.startswith(b"\x89PNG"):
            return "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif data.startswith(b"GIF89") or data.startswith(b"GIF87"):
            return "image/gif"
        elif data.startswith(b"%PDF"):
            return "application/pdf"
        elif data.startswith(b"PK"):
            return "application/zip"
        else:
            return "application/octet-stream"


def fetch_file_from_url(url: str, timeout: float = 30.0) -> bytes:
    """
    Fetch file content from a URL.

    Args:
        url: URL to fetch from
        timeout: Request timeout in seconds

    Returns:
        File content as bytes

    Raises:
        ValueError: If remote URLs are not allowed or URL is invalid
        httpx.RequestError: If request fails
    """
    # Security check for remote URLs
    if not os.environ.get("LLMRING_ALLOW_REMOTE_URLS", "false").lower() == "true":
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.netloc:
            raise ValueError(
                "Remote URLs are not allowed. Set LLMRING_ALLOW_REMOTE_URLS=true to enable."
            )

    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def decode_base64_file(base64_string: str) -> Tuple[bytes, str]:
    """
    Decode a base64-encoded file with optional data URI scheme.

    Args:
        base64_string: Base64 encoded string, optionally with data URI prefix

    Returns:
        Tuple of (decoded bytes, content type)

    Raises:
        ValueError: If base64 string is invalid
    """
    content_type = "application/octet-stream"

    # Handle data URI scheme
    if base64_string.startswith("data:"):
        try:
            header, data = base64_string.split(",", 1)
            if ";" in header:
                content_type = header.split(":")[1].split(";")[0]
            base64_string = data
        except (IndexError, ValueError) as e:
            raise InvalidFileFormatError(
                f"Invalid data URI format: {e}",
                details={"data_uri_prefix": base64_string[:50]},
            )
        except Exception as e:
            raise FileProcessingError(
                f"Unexpected error parsing data URI: {e}",
                details={"data_uri_prefix": base64_string[:50]},
            )

    # Decode base64
    try:
        decoded = base64.b64decode(base64_string)

        # Guess content type if not provided
        if content_type == "application/octet-stream":
            content_type = guess_content_type_from_bytes(decoded)

        return decoded, content_type
    except (ValueError, TypeError) as e:
        raise InvalidFileFormatError(f"Invalid base64 encoding: {e}")
    except Exception as e:
        raise FileProcessingError(f"Unexpected error decoding base64: {e}")


def process_file_from_source(
    source: Union[str, bytes], source_type: str = "auto"
) -> Dict[str, Union[str, bytes]]:
    """
    Process a file from various sources.

    Args:
        source: File source (path, URL, base64, or bytes)
        source_type: Type of source ("path", "url", "base64", "bytes", or "auto")

    Returns:
        Dictionary with:
            - content: File content as bytes
            - content_type: MIME type
            - filename: Original filename if available

    Raises:
        ValueError: If source type cannot be determined or is invalid
        FileNotFoundError: If file path doesn't exist
    """
    result = {
        "content": b"",
        "content_type": "application/octet-stream",
        "filename": None,
    }

    # Auto-detect source type if needed
    if source_type == "auto":
        if isinstance(source, bytes):
            source_type = "bytes"
        elif isinstance(source, str):
            if source.startswith(("http://", "https://")):
                source_type = "url"
            elif source.startswith("data:") or (
                "/" not in source
                and "\\" not in source
                and len(source) > 100
                and "=" in source
            ):
                source_type = "base64"
            elif os.path.exists(source):
                source_type = "path"
            else:
                raise ValidationError(
                    "Cannot determine source type for input",
                    field="source",
                    value=str(source)[:100],
                    expected_type="path, url, base64 string, or bytes",
                )

    # Process based on source type
    if source_type == "bytes":
        result["content"] = source
        result["content_type"] = guess_content_type_from_bytes(source)

    elif source_type == "url":
        result["content"] = fetch_file_from_url(source)
        result["content_type"] = guess_content_type_from_bytes(result["content"])
        result["filename"] = os.path.basename(urlparse(source).path) or None

    elif source_type == "base64":
        content, content_type = decode_base64_file(source)
        result["content"] = content
        result["content_type"] = content_type

    elif source_type == "path":
        if not os.path.exists(source):
            raise FileAccessError(
                f"File not found: {source}", filename=os.path.basename(source)
            )

        with open(source, "rb") as f:
            result["content"] = f.read()

        # Get content type from extension or content
        content_type = mimetypes.guess_type(source)[0]
        if not content_type:
            content_type = guess_content_type_from_bytes(result["content"])
        result["content_type"] = content_type
        result["filename"] = os.path.basename(source)

    else:
        raise ValidationError(
            f"Invalid source type: {source_type}",
            field="source_type",
            value=source_type,
            expected_type="'path', 'url', 'base64', or 'bytes'",
        )

    return result


def create_file_content(
    file_data: bytes, content_type: str, as_base64: bool = True
) -> Union[str, Dict[str, str]]:
    """
    Create file content for API requests.

    Args:
        file_data: Raw file bytes
        content_type: MIME type
        as_base64: Return as base64 string (True) or dict (False)

    Returns:
        Base64 string or dict with base64 and content_type
    """
    base64_data = base64.b64encode(file_data).decode("utf-8")

    if as_base64:
        return f"data:{content_type};base64,{base64_data}"
    else:
        return {"base64": base64_data, "content_type": content_type}


def get_file_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file.

    Args:
        file_path: Path to file

    Returns:
        MIME type string
    """
    # Try from extension first
    mime_type = mimetypes.guess_type(file_path)[0]

    # Fall back to content detection
    if not mime_type and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            # Read first 8KB for detection
            data = f.read(8192)
            mime_type = guess_content_type_from_bytes(data)

    return mime_type or "application/octet-stream"


def is_text_file(content_type: str) -> bool:
    """Check if content type indicates a text file."""
    text_types = {
        "text/",
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-yaml",
    }
    return any(content_type.startswith(t) for t in text_types)


def is_image_file(content_type: str) -> bool:
    """Check if content type indicates an image file."""
    return content_type.startswith("image/")


def is_document_file(content_type: str) -> bool:
    """Check if content type indicates a document file."""
    doc_types = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument",
        "application/vnd.ms-",
        "application/rtf",
    }
    return any(content_type.startswith(t) for t in doc_types)
