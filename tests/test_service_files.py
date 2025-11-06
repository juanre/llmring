# ABOUTME: Integration tests for provider-agnostic file registration in LLMRing service layer.
# ABOUTME: Tests register, deregister, list operations and cross-provider file usage without mocks.

import os

import pytest

from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_register_file():
    """Test file registration returns llmring ID."""
    ring = LLMRing()
    try:
        file_id = await ring.register_file("tests/fixtures/sample.txt")
        assert file_id.startswith("llmring-file-")

        # Verify file is in registered files
        files = await ring.list_registered_files()
        assert len(files) == 1
        assert files[0]["id"] == file_id
        assert files[0]["file_path"].endswith("sample.txt")
        assert files[0]["uploads"] == {}  # No uploads yet

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_register_file_with_custom_id():
    """Test file registration with custom ID."""
    ring = LLMRing()
    try:
        custom_id = "test-file-123"
        file_id = await ring.register_file("tests/fixtures/sample.txt", file_id=custom_id)
        assert file_id == custom_id

        # Verify file is registered
        files = await ring.list_registered_files()
        assert len(files) == 1
        assert files[0]["id"] == custom_id

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_register_file_nonexistent():
    """Test registering nonexistent file raises error."""
    ring = LLMRing()
    try:
        with pytest.raises(FileNotFoundError):
            await ring.register_file("nonexistent.txt")
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_lazy_upload_anthropic():
    """Test file uploads to Anthropic on first use."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        # Register file (no upload yet)
        file_id = await ring.register_file("tests/fixtures/sample.txt")

        # Verify no uploads yet
        files = await ring.list_registered_files()
        assert files[0]["uploads"] == {}

        # Use with Anthropic - should upload
        from llmring.schemas import LLMRequest, Message

        response = await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="What's in this file?")],
                files=[file_id],
            )
        )

        assert response.content
        assert response.model.startswith("anthropic:")

        # Verify file was uploaded to Anthropic
        files = await ring.list_registered_files()
        assert "anthropic" in files[0]["uploads"]
        assert files[0]["uploads"]["anthropic"]["provider_file_id"].startswith("file_")

        # Cleanup
        await ring.deregister_file(file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_cross_provider_file_usage():
    """Test same file works with multiple providers."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not anthropic_key or not google_key:
        pytest.skip("Both ANTHROPIC_API_KEY and GOOGLE_API_KEY required")

    ring = LLMRing()
    try:
        from llmring.schemas import LLMRequest, Message

        # Register file once
        file_id = await ring.register_file("tests/fixtures/google_large_doc.txt")

        # Use with Anthropic
        r1 = await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="Summarize this file in one sentence.")],
                files=[file_id],
            )
        )

        # Use with Google
        r2 = await ring.chat(
            LLMRequest(
                model="google:gemini-2.5-flash",
                messages=[Message(role="user", content="Analyze this file in one sentence.")],
                files=[file_id],
            )
        )

        # Both should work
        assert r1.content
        assert r2.content

        # Verify file was uploaded to both providers
        files = await ring.list_registered_files()
        assert "anthropic" in files[0]["uploads"]
        assert "google" in files[0]["uploads"]
        assert files[0]["uploads"]["anthropic"]["provider_file_id"].startswith("file_")
        assert files[0]["uploads"]["google"]["provider_file_id"].startswith("cachedContents/")

        # Cleanup
        await ring.deregister_file(file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_file_upload_caching():
    """Test that file uploads are cached per provider."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        from llmring.schemas import LLMRequest, Message

        file_id = await ring.register_file("tests/fixtures/sample.txt")

        # First use - should upload
        await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="Summarize.")],
                files=[file_id],
            )
        )

        # Get the provider file_id
        files = await ring.list_registered_files()
        first_provider_file_id = files[0]["uploads"]["anthropic"]["provider_file_id"]

        # Second use - should use cache (same provider file_id)
        await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="Analyze.")],
                files=[file_id],
            )
        )

        # Verify same provider file_id was used
        files = await ring.list_registered_files()
        second_provider_file_id = files[0]["uploads"]["anthropic"]["provider_file_id"]
        assert first_provider_file_id == second_provider_file_id

        # Cleanup
        await ring.deregister_file(file_id)

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_deregister_file():
    """Test cleanup from all providers."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not anthropic_key or not google_key:
        pytest.skip("Both ANTHROPIC_API_KEY and GOOGLE_API_KEY required")

    ring = LLMRing()
    try:
        from llmring.schemas import LLMRequest, Message

        file_id = await ring.register_file("tests/fixtures/google_large_doc.txt")

        # Upload to two providers
        await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="Hi")],
                files=[file_id],
            )
        )
        await ring.chat(
            LLMRequest(
                model="google:gemini-2.5-flash",
                messages=[Message(role="user", content="Hi")],
                files=[file_id],
            )
        )

        # Deregister - should delete from both
        success = await ring.deregister_file(file_id)
        assert success

        # Verify file_id no longer exists
        files = await ring.list_registered_files()
        assert len(files) == 0

        # Trying to use it should fail
        with pytest.raises(ValueError, match="not registered"):
            await ring.chat(
                LLMRequest(
                    model="anthropic:claude-3-5-haiku-20241022",
                    messages=[Message(role="user", content="Hi")],
                    files=[file_id],
                )
            )

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_deregister_file_not_found():
    """Test deregistering nonexistent file returns False."""
    ring = LLMRing()
    try:
        success = await ring.deregister_file("nonexistent-id")
        assert success is False
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_list_registered_files():
    """Test listing registered files."""
    ring = LLMRing()
    try:
        # No files initially
        files = await ring.list_registered_files()
        assert len(files) == 0

        # Register two files
        file_id1 = await ring.register_file("tests/fixtures/sample.txt")
        file_id2 = await ring.register_file("tests/fixtures/google_large_doc.txt")

        # List should show both
        files = await ring.list_registered_files()
        assert len(files) == 2
        file_ids = [f["id"] for f in files]
        assert file_id1 in file_ids
        assert file_id2 in file_ids

    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_file_not_registered_error():
    """Test using unregistered file_id raises error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    try:
        from llmring.schemas import LLMRequest, Message

        with pytest.raises(ValueError, match="not registered"):
            await ring.chat(
                LLMRequest(
                    model="anthropic:claude-3-5-haiku-20241022",
                    messages=[Message(role="user", content="Hi")],
                    files=["unregistered-file-id"],
                )
            )
    finally:
        await ring.close()


@pytest.mark.asyncio
async def test_file_staleness_detection():
    """Test that file changes on disk are detected and cache invalidated."""
    import tempfile

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    ring = LLMRing()
    temp_file = None
    file_id = None

    try:
        from llmring.schemas import LLMRequest, Message

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original content for staleness test")
            temp_file = f.name

        # Register file
        file_id = await ring.register_file(temp_file)

        # Use with Anthropic (first upload)
        response1 = await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="What does this say?")],
                files=[file_id],
            )
        )
        assert response1.content

        # Get registered file to check state
        files = await ring.list_registered_files()
        reg_file = next(f for f in files if f["id"] == file_id)
        assert "anthropic" in reg_file["uploads"]
        original_hash = reg_file["content_hash"]

        # Modify file content
        with open(temp_file, "w") as f:
            f.write("Modified content - this should be detected")

        # Use again - should detect change and re-upload
        response2 = await ring.chat(
            LLMRequest(
                model="anthropic:claude-3-5-haiku-20241022",
                messages=[Message(role="user", content="What does it say now?")],
                files=[file_id],
            )
        )
        assert response2.content

        # Verify hash was updated
        files_after = await ring.list_registered_files()
        reg_file_after = next(f for f in files_after if f["id"] == file_id)
        assert reg_file_after["content_hash"] != original_hash

    finally:
        # Cleanup
        if file_id:
            await ring.deregister_file(file_id)
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        await ring.close()
