# File Utilities

## Overview

LLMRing provides comprehensive file handling utilities for working with images, documents, and other files in LLM requests. These utilities support vision and multimodal capabilities across all providers.

## Core Functions

### File Encoding and Detection

#### `encode_file_to_base64(file_path: str) -> str`

Encode a file to base64 string.

```python
from llmring import encode_file_to_base64

# Encode any file
base64_data = encode_file_to_base64("document.pdf")
print(f"Encoded {len(base64_data)} characters")
```

**Parameters:**
- `file_path`: Path to the file to encode

**Returns:** Base64 encoded string

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `IOError`: If file cannot be read

---

#### `get_file_mime_type(file_path: str) -> str`

Get the MIME type of a file.

```python
from llmring import get_file_mime_type

mime_type = get_file_mime_type("screenshot.png")
print(mime_type)  # "image/png"

mime_type = get_file_mime_type("document.pdf")
print(mime_type)  # "application/pdf"
```

**Supported types:**
- Images: PNG, JPEG, GIF, WebP
- Documents: PDF, TXT, DOCX, DOC
- Fallback: `application/octet-stream`

---

#### `create_data_url(file_path: str) -> str`

Create a data URL from a file.

```python
from llmring import create_data_url

# Create data URL
data_url = create_data_url("image.png")
print(data_url)  # "data:image/png;base64,iVBORw0KGgo..."
```

**Returns:** Data URL string in format `data:{mime_type};base64,{base64_data}`

---

#### `validate_image_file(file_path: str) -> bool`

Validate if a file is a supported image format.

```python
from llmring import validate_image_file

if validate_image_file("screenshot.png"):
    print("Valid image file")
```

**Supported formats:** PNG, JPEG, GIF, WebP

---

## Content Creation Functions

### `create_image_content(file_path_url_or_base64: str, text: str = "", mime_type: str = "image/jpeg") -> List[Dict[str, Any]]`

Create image content for LLM messages from various sources.

```python
from llmring import LLMRing, LLMRequest, Message, create_image_content

async with LLMRing() as service:
    # File path
    content = create_image_content("screenshot.png", "What's in this image?")

    # URL (requires LLMRING_ALLOW_REMOTE_URLS=true)
    content = create_image_content("https://example.com/chart.jpg", "Analyze this chart")

    # Base64 string
    content = create_image_content("iVBORw0KGgoAAAANSUhEUgAA...", "Describe this", "image/png")

    # Data URL
    content = create_image_content("data:image/png;base64,iVBORw0KGgo...", "What's this?")

    # Use in message
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
```

**Parameters:**
- `file_path_url_or_base64`: File path, URL, base64 data, or data URL
- `text`: Optional text to include with the image
- `mime_type`: MIME type for base64 data (default: "image/jpeg")

**Returns:** Content list suitable for `Message.content`

**Security Note:** Remote URLs are disabled by default. Set `LLMRING_ALLOW_REMOTE_URLS=true` to enable.

---

### `create_multi_image_content(images: List[Union[str, Dict[str, str]]], text: str = "") -> List[Dict[str, Any]]`

Create content with multiple images for comparison or analysis.

```python
from llmring import create_multi_image_content, LLMRing, LLMRequest, Message

# Mixed sources
content = create_multi_image_content([
    "image1.png",  # file path
    "https://example.com/image2.jpg",  # URL
    "iVBORw0KGgoAAAANSUhEUgAA...",  # base64 (auto-detected)
    {"data": "iVBORw0KGgo...", "mime_type": "image/png"}  # explicit base64
], "Compare these images")

async with LLMRing() as service:
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
```

**Parameters:**
- `images`: List of image sources (strings or dicts with data/mime_type)
- `text`: Optional text prompt

---

### `create_base64_image_content(base64_data: str, mime_type: str = "image/jpeg", text: str = "") -> List[Dict[str, Any]]`

Explicit function for creating image content from base64 data.

```python
from llmring import create_base64_image_content

content = create_base64_image_content(
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "image/png",
    "Analyze this image"
)
```

**Use when:** You have raw base64 data and want to be explicit about the MIME type.

---

## Convenience Functions

### `analyze_image(file_path_url_or_base64: str, prompt: str = "Analyze this image", mime_type: str = "image/jpeg") -> List[Dict[str, Any]]`

Convenience function for image analysis.

```python
from llmring import analyze_image, LLMRing, LLMRequest, Message

# Analyze a screenshot
content = analyze_image("screenshot.png", "What's in this image?")

# Analyze base64 data
content = analyze_image("iVBORw0KGgoAAAANSUhEUgAA...", "Describe this", "image/png")

# Analyze from URL
content = analyze_image("https://example.com/chart.jpg", "Extract data from this chart")

async with LLMRing() as service:
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
    print(response.content)
```

---

### `extract_text_from_image(file_path_url_or_base64: str, mime_type: str = "image/jpeg") -> List[Dict[str, Any]]`

Convenience function for OCR (optical character recognition).

```python
from llmring import extract_text_from_image, LLMRing, LLMRequest, Message

# Extract text from scanned document
content = extract_text_from_image("scanned_document.png")

async with LLMRing() as service:
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
    print(response.content)  # Extracted text with formatting preserved
```

**Built-in prompt:** "Extract all text from this image. Preserve formatting and layout."

---

### `compare_images(image1: Union[str, Dict[str, str]], image2: Union[str, Dict[str, str]], prompt: str = "Compare these images") -> List[Dict[str, Any]]`

Convenience function for image comparison.

```python
from llmring import compare_images, LLMRing, LLMRequest, Message

# Compare file paths
content = compare_images("before.png", "after.png", "What changed?")

# Mixed sources
content = compare_images("image.png", "iVBORw0KGgo...", "Compare these")

# Explicit base64
content = compare_images(
    {"data": "iVBORw0KGgo...", "mime_type": "image/png"},
    "after.jpg",
    "What's different?"
)

async with LLMRing() as service:
    request = LLMRequest(
        model="vision",
        messages=[Message(role="user", content=content)]
    )
    response = await service.chat(request)
    print(response.content)
```

---

## Complete Example: Vision Analysis Pipeline

```python
import asyncio
from pathlib import Path
from llmring import (
    LLMRing,
    LLMRequest,
    Message,
    analyze_image,
    compare_images,
    extract_text_from_image,
    create_multi_image_content
)

async def analyze_screenshots():
    """Analyze multiple screenshots and extract insights."""
    async with LLMRing() as service:
        # 1. Analyze a single screenshot
        print("=== Analyzing single screenshot ===")
        content = analyze_image("screenshot1.png", "Describe what you see")
        request = LLMRequest(
            model="vision",
            messages=[Message(role="user", content=content)]
        )
        response = await service.chat(request)
        print(response.content)

        # 2. Extract text from scanned document
        print("\n=== Extracting text ===")
        content = extract_text_from_image("scanned_page.png")
        request = LLMRequest(
            model="vision",
            messages=[Message(role="user", content=content)]
        )
        response = await service.chat(request)
        print(response.content)

        # 3. Compare before/after images
        print("\n=== Comparing images ===")
        content = compare_images(
            "before_changes.png",
            "after_changes.png",
            "What are the key differences?"
        )
        request = LLMRequest(
            model="vision",
            messages=[Message(role="user", content=content)]
        )
        response = await service.chat(request)
        print(response.content)

        # 4. Analyze multiple images together
        print("\n=== Multi-image analysis ===")
        images = [
            "graph1.png",
            "graph2.png",
            "graph3.png"
        ]
        content = create_multi_image_content(
            images,
            "Compare these graphs and identify trends"
        )
        request = LLMRequest(
            model="vision",
            messages=[Message(role="user", content=content)]
        )
        response = await service.chat(request)
        print(response.content)

if __name__ == "__main__":
    asyncio.run(analyze_screenshots())
```

## Security Considerations

### Remote URLs

Remote URLs are **disabled by default** for security. To enable:

```bash
export LLMRING_ALLOW_REMOTE_URLS=true
```

**Recommendation:** Use data URLs or base64 encoding instead of remote URLs for better security and control.

### File Access

- All file paths must be absolute or relative to the current directory
- Files are read directly from the filesystem
- No network requests are made for local files
- Validate file sources before processing

## Provider Compatibility

| Function | OpenAI | Anthropic | Google | Ollama |
|----------|--------|-----------|--------|--------|
| `create_image_content` | ✅ | ✅ | ✅ | ✅ |
| `analyze_image` | ✅ | ✅ | ✅ | ✅ |
| `extract_text_from_image` | ✅ | ✅ | ✅ | ✅ |
| `compare_images` | ✅ | ✅ | ✅ | ✅ |
| `create_multi_image_content` | ✅ | ✅ | ✅ | ✅ |

**Note:** Anthropic Claude models have excellent vision capabilities. OpenAI GPT-4o models support vision. Google Gemini has native multimodal support. Ollama support depends on the specific model (e.g., llava).

## Best Practices

1. **Use semantic aliases:** Use `vision` alias instead of specific model names
2. **Validate files:** Check files exist before encoding
3. **Handle errors:** Wrap file operations in try/except blocks
4. **Optimize images:** Resize large images before sending to reduce costs
5. **Use data URLs:** Prefer data URLs over remote URLs for security
6. **Cache results:** Cache expensive vision operations when possible
7. **Choose right model:** Use vision-capable models for image tasks

## Troubleshooting

### Common Issues

**File not found:**
```python
from pathlib import Path

file_path = "image.png"
if not Path(file_path).exists():
    print(f"File not found: {file_path}")
```

**Invalid image format:**
```python
from llmring import validate_image_file

if not validate_image_file("file.png"):
    print("Unsupported image format")
```

**Remote URL disabled:**
```python
# Error: Remote URL inputs are disabled by configuration
# Solution: Use data URL instead
from llmring import create_data_url
data_url = create_data_url("local_image.png")
# Now use data_url in your content
```

**Large files:**
```python
from PIL import Image

# Resize large images before encoding
img = Image.open("large_image.png")
img.thumbnail((1024, 1024))
img.save("resized.png")

# Now use resized image
content = analyze_image("resized.png", "Analyze this")
```

## Related Documentation

- [API Reference](api-reference.md) - Core LLMRing API
- [Provider Guide](providers.md) - Provider-specific features
- [Structured Output](structured-output.md) - JSON schema support
