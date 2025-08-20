#!/usr/bin/env python3
"""
Example demonstrating base64 image processing with the LLM Service.

This example shows how to:
1. Work with base64 image data
2. Use auto-detection vs explicit functions
3. Mix different input types
4. Handle MIME types correctly

Run with: python examples/base64_example.py
"""

import asyncio
import base64
import os
import tempfile

from llmring import (
    LLMRing,
    analyze_image,
    compare_images,
    create_base64_image_content,
    create_multi_image_content,
)


def create_sample_image_base64() -> tuple[str, str]:
    """Create a sample image and return its base64 data and MIME type."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Return a fake base64 string for demo
        return (
            "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 200 + "AAAABJRU5ErkJggg==",
            "image/png",
        )

    # Create a simple test image
    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 90], outline="black", width=2)
    draw.text((50, 40), "Base64 Test", fill="black")

    # Convert to base64
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        with open(tmp.name, "rb") as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
        os.unlink(tmp.name)

    return base64_data, "image/png"


async def demo_base64_auto_detection():
    """Demonstrate automatic base64 detection."""
    print("=== Base64 Auto-Detection Demo ===")

    base64_data, mime_type = create_sample_image_base64()
    print(f"Created base64 data: {len(base64_data)} characters")
    print(f"MIME type: {mime_type}")
    print(f"First 50 chars: {base64_data[:50]}...")

    # Use analyze_image with auto-detection
    content = analyze_image(
        base64_data, "What text is visible in this image?", mime_type
    )

    print("\nGenerated content structure:")
    print(f"- {len(content)} parts")
    for i, part in enumerate(content):
        if part["type"] == "text":
            print(f"  Part {i + 1}: Text - '{part['text']}'")
        else:
            url = part["image_url"]["url"]
            print(f"  Part {i + 1}: Image - {url[:50]}...")

    print()


async def demo_explicit_base64():
    """Demonstrate explicit base64 handling."""
    print("=== Explicit Base64 Demo ===")

    base64_data, mime_type = create_sample_image_base64()

    # Use explicit base64 function
    content = create_base64_image_content(
        base64_data, mime_type, "Extract any text from this base64 image"
    )

    print("Using create_base64_image_content():")
    print(f"- Generated {len(content)} content parts")
    print(f"- Text: '{content[0]['text']}'")
    print(f"- Data URL: {content[1]['image_url']['url'][:70]}...")

    print()


async def demo_mixed_sources():
    """Demonstrate mixing different image source types."""
    print("=== Mixed Sources Demo ===")

    base64_data, mime_type = create_sample_image_base64()

    # Mix file path, URL, and base64
    sources = [
        "https://httpbin.org/image/png",  # URL
        base64_data,  # Base64 (auto-detected)
        {
            "data": base64_data[:100] + "truncated",
            "mime_type": "image/png",
        },  # Explicit base64
    ]

    content = create_multi_image_content(sources, "Compare these three images")

    print("Mixed sources content:")
    print(f"- Total parts: {len(content)}")
    print(f"- Text part: '{content[0]['text']}'")

    for i in range(1, len(content)):
        url = content[i]["image_url"]["url"]
        if url.startswith("http"):
            print(f"- Image {i}: URL - {url}")
        else:
            print(f"- Image {i}: Data URL - {url[:50]}...")

    print()


async def demo_compare_with_base64():
    """Demonstrate image comparison with base64."""
    print("=== Base64 Comparison Demo ===")

    base64_data, mime_type = create_sample_image_base64()

    # Compare URL with base64
    content = compare_images(
        "https://httpbin.org/image/png",
        base64_data,
        "What are the differences between these images?",
    )

    print("Comparing URL vs Base64:")
    print(f"- Generated {len(content)} parts")
    print(f"- Prompt: '{content[0]['text']}'")
    print(f"- Image 1: {content[1]['image_url']['url']}")
    print(f"- Image 2: {content[2]['image_url']['url'][:50]}...")

    print()


async def demo_llm_integration():
    """Demonstrate actual LLM integration (if API keys available)."""
    print("=== LLM Integration Demo ===")

    service = LLMRing(enable_db_logging=False)
    available_models = service.get_available_models()

    if available_models:
        print("Available providers:")
        for provider, models in available_models.items():
            print(f"  {provider}: {len(models)} models")

        print("\nTo test with real LLMs:")
        print("1. Set API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        print("2. Use the generated content with service.chat()")

        # Show example usage (commented to avoid API calls)
        base64_data, mime_type = create_sample_image_base64()
        # Would normally call: analyze_image(base64_data, "Describe this image", mime_type)

        print("\nExample usage:")
        print(
            f"content = analyze_image(base64_data, 'Describe this image', '{mime_type}')"
        )
        print(
            "request = LLMRequest(messages=[Message(role='user', content=content)], model='gpt-4o')"
        )
        print("response = await service.chat(request)")
        print("# response.content would contain the LLM's description")

    else:
        print("No providers available. Set API keys to test with real LLMs.")

    print()


async def main():
    """Run all base64 demos."""
    print("LLM Service Base64 Support Demo")
    print("=" * 40)

    await demo_base64_auto_detection()
    await demo_explicit_base64()
    await demo_mixed_sources()
    await demo_compare_with_base64()
    await demo_llm_integration()

    print("Demo completed! 🎉")
    print("\nKey takeaways:")
    print("✅ Base64 strings are auto-detected when they're long enough")
    print("✅ Explicit functions give you full control over MIME types")
    print("✅ All utility functions support base64 alongside files and URLs")
    print("✅ Mix and match different source types in the same request")
    print("✅ Perfect for scenarios where you already have base64 data")


if __name__ == "__main__":
    asyncio.run(main())
