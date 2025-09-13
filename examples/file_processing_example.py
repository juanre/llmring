#!/usr/bin/env python3
"""
Example demonstrating file processing with the LLM Service.

This example shows how to:
1. Process images from files and URLs
2. Use convenience functions for common agents
3. Handle multiple images
4. Work with different file types

Run with: python examples/file_processing_example.py
"""

import asyncio
import os
import tempfile

from llmring import (
    LLMRing,
    analyze_image,
    compare_images,
    encode_file_to_base64,
    extract_text_from_image,
    get_file_mime_type,
)


def create_sample_text_file() -> str:
    """Create a sample text file for demonstration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "This is a sample text file for the LLM Service file processing demo.\n"
        )
        f.write("File processing made easy!")
        return f.name


async def demo_basic_file_operations():
    """Demonstrate basic file operations."""
    print("=== Basic File Operations ===")

    # Create a sample file
    sample_file = create_sample_text_file()

    try:
        # Get file information
        mime_type = get_file_mime_type(sample_file)
        print(f"File MIME type: {mime_type}")

        # Encode to base64
        encoded = encode_file_to_base64(sample_file)
        print(f"Base64 encoded length: {len(encoded)} characters")
        print(f"First 50 characters: {encoded[:50]}...")

        # Create data URL
        from llmring.file_utils import create_data_url

        data_url = create_data_url(sample_file)
        print(f"Data URL: {data_url[:70]}...")

    finally:
        os.unlink(sample_file)

    print()


async def demo_image_processing():
    """Demonstrate image processing capabilities."""
    print("=== Image Processing ===")

    # Example with URL (this won't make an actual API call)
    image_url = "https://example.com/sample-image.png"

    # Method 1: Using convenience functions
    content1 = analyze_image(image_url, "What objects are visible in this image?")
    print("Analyze image content:")
    for part in content1:
        if part["type"] == "text":
            print(f"  Text: {part['text']}")
        else:
            print(f"  Image: {part['image_url']['url'][:50]}...")

    # Method 2: OCR extraction
    content2 = extract_text_from_image(image_url)
    print("\nOCR extraction content:")
    for part in content2:
        if part["type"] == "text":
            print(f"  Text: {part['text']}")
        else:
            print(f"  Image: {part['image_url']['url'][:50]}...")

    # Method 3: Multiple images
    image_urls = ["https://example.com/image1.png", "https://example.com/image2.png"]
    content3 = compare_images(image_urls[0], image_urls[1], "What are the differences?")
    print(f"\nCompare images content has {len(content3)} parts")

    print()


async def demo_llm_integration():
    """Demonstrate integration with LLM service (requires API keys)."""
    print("=== LLM Integration Demo ===")

    # Initialize service
    service = LLMRing(enable_db_logging=False)

    # Check available models
    available_models = service.get_available_models()
    print("Available providers:")
    for provider, models in available_models.items():
        print(f"  {provider}: {len(models)} models")

    # Demo with a public image URL (if you have API keys)
    if available_models:
        print("\nTo test with real images, you would:")
        print("1. Set up API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        print("2. Use a real image URL or file path")
        print("3. Call service.chat() with the content")

        # Example code (commented out to avoid API calls):
        """
        content = analyze_image("path/to/your/image.jpg", "Describe this image")
        request = LLMRequest(
            messages=[Message(role="user", content=content)],
            model="balanced"  # Use semantic alias for vision-capable model
        )
        response = await service.chat(request)
        print(f"LLM Response: {response.content}")
        """
    else:
        print("No providers available. Set API keys to test with real LLMs.")

    print()


async def demo_advanced_usage():
    """Demonstrate advanced file processing patterns."""
    print("=== Advanced Usage Patterns ===")

    # Custom content creation
    sample_file = create_sample_text_file()

    try:
        # Build custom content manually
        content_parts = [
            {"type": "text", "text": "Please analyze this file:"},
            # Note: For text files, you'd typically just include the content directly
            # This is just to demonstrate the pattern
        ]

        print("Custom content structure:")
        for i, part in enumerate(content_parts):
            print(f"  Part {i + 1}: {part['type']}")

        # File size checking pattern
        file_size_mb = os.path.getsize(sample_file) / (1024 * 1024)
        print(f"\nFile size: {file_size_mb:.3f} MB")

        if file_size_mb > 20:  # Example limit
            print("File too large for processing")
        else:
            print("File size acceptable")

        # Batch processing pattern
        files = [sample_file]  # In practice, you'd have multiple files
        print(f"\nProcessing {len(files)} files...")

        for i, file_path in enumerate(files):
            mime_type = get_file_mime_type(file_path)
            print(f"  File {i + 1}: {mime_type}")
            # Here you'd process each file based on its type

    finally:
        os.unlink(sample_file)

    print()


async def main():
    """Run all demos."""
    print("LLM Service File Processing Demo")
    print("=" * 40)

    await demo_basic_file_operations()
    await demo_image_processing()
    await demo_llm_integration()
    await demo_advanced_usage()

    print("Demo completed! 🎉")
    print("\nNext steps:")
    print("1. Set up API keys for your preferred LLM providers")
    print("2. Try with real images and documents")
    print("3. Explore the full documentation in docs/user-guide.md")


if __name__ == "__main__":
    asyncio.run(main())
