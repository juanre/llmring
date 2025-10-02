"""
End-to-end integration tests for logging and receipts workflow.

Tests the complete flow:
1. Client logs conversations via LoggingService
2. Server stores logs in database
3. Client generates receipts on-demand
4. Receipts can be verified

These tests require a running llmring-server instance.
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from llmring import LLMRing
from llmring.schemas import LLMRequest, Message
from llmring.server_client import ServerClient


@pytest.fixture
def server_url():
    """Get server URL from environment or use default."""
    return os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")


@pytest.fixture
def test_api_key():
    """Generate a unique API key for each test."""
    return f"test-e2e-{uuid4()}"


class TestLoggingE2E:
    """End-to-end tests for logging and receipts."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 and running server",
    )
    async def test_full_workflow_metadata_logging(self, server_url, test_api_key):
        """
        Test complete workflow: log metadata -> generate receipt.

        This test verifies:
        1. LLMRing logs usage metadata to server
        2. Receipt can be generated on-demand
        3. Receipt contains correct data
        """
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        # Initialize LLMRing with metadata logging
        ring = LLMRing(
            server_url=server_url,
            api_key=test_api_key,
            log_metadata=True,
            log_conversations=False,
        )

        # Make a simple chat request
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Say 'hello' in one word")],
            temperature=0.0,
        )

        response = await ring.chat(request)

        # Verify response
        assert response.content is not None
        assert response.usage is not None
        assert response.usage.get("total_tokens", 0) > 0

        # Wait a moment for server to process
        import asyncio

        await asyncio.sleep(0.5)

        # Generate receipt on-demand
        client = ServerClient(server_url=server_url, api_key=test_api_key)
        result = await client.generate_receipt(since_last_receipt=True)

        # Verify receipt
        receipt = result["receipt"]
        assert receipt["receipt_id"].startswith("rcpt_")
        assert receipt["provider"] == "openai"
        assert receipt["model"] == "gpt-4o-mini"
        assert receipt["total_cost"] > 0
        assert receipt["signature"] is not None
        assert receipt["signature"].startswith("ed25519:")

        # Verify certified count
        assert result["certified_count"] >= 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 and running server",
    )
    async def test_full_workflow_conversation_logging(self, server_url, test_api_key):
        """
        Test complete workflow: log full conversation -> generate receipt.

        This test verifies:
        1. LLMRing logs full conversations to server
        2. Receipt can be generated for conversation
        3. Receipt contains correct conversation data
        """
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        # Initialize LLMRing with conversation logging
        ring = LLMRing(
            server_url=server_url,
            api_key=test_api_key,
            log_conversations=True,  # Implies log_metadata=True
        )

        # Make a chat request
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Count to 3")],
            temperature=0.0,
        )

        response = await ring.chat(request)

        # Verify response
        assert response.content is not None

        # Wait for server to process
        import asyncio

        await asyncio.sleep(0.5)

        # Get the conversation ID (stored in logging service)
        conversation_id = ring.logging_service._conversation_id if ring.logging_service else None

        # Generate receipt for this conversation
        client = ServerClient(server_url=server_url, api_key=test_api_key)

        if conversation_id:
            # Generate receipt for specific conversation
            result = await client.generate_receipt(conversation_id=conversation_id)
        else:
            # Fallback: generate for all uncertified
            result = await client.generate_receipt(since_last_receipt=True)

        # Verify receipt
        receipt = result["receipt"]
        assert receipt["receipt_id"].startswith("rcpt_")
        assert receipt["signature"] is not None
        assert result["certified_count"] >= 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 and running server",
    )
    async def test_batch_receipt_workflow(self, server_url, test_api_key):
        """
        Test batch receipt generation for multiple logs.

        This test verifies:
        1. Multiple conversations can be logged
        2. Batch receipt certifies all logs
        3. Receipt summary contains correct aggregations
        """
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        # Initialize LLMRing
        ring = LLMRing(
            server_url=server_url,
            api_key=test_api_key,
            log_conversations=True,
        )

        # Make multiple requests
        for i in range(3):
            request = LLMRequest(
                model="openai:gpt-4o-mini",
                messages=[Message(role="user", content=f"Say number {i}")],
                temperature=0.0,
            )
            await ring.chat(request)

        # Wait for server to process
        import asyncio

        await asyncio.sleep(1.0)

        # Generate batch receipt
        client = ServerClient(server_url=server_url, api_key=test_api_key)
        result = await client.generate_receipt(
            since_last_receipt=True,
            description="Test batch receipt",
            tags=["test", "e2e"],
        )

        # Verify receipt
        receipt = result["receipt"]
        assert receipt["receipt_type"] == "batch"
        assert receipt["description"] == "Test batch receipt"
        assert receipt["tags"] == ["test", "e2e"]
        assert result["certified_count"] == 3

        # Verify batch summary
        summary = receipt.get("batch_summary")
        assert summary is not None
        assert summary["total_calls"] == 3
        assert summary["total_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 and running server",
    )
    async def test_preview_and_generate_workflow(self, server_url, test_api_key):
        """
        Test preview before generating receipt.

        This test verifies:
        1. Preview shows what would be certified
        2. Actual generation matches preview
        """
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        # Initialize LLMRing and make a request
        ring = LLMRing(
            server_url=server_url,
            api_key=test_api_key,
            log_metadata=True,
        )

        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )

        await ring.chat(request)

        # Wait for server
        import asyncio

        await asyncio.sleep(0.5)

        # Preview receipt
        client = ServerClient(server_url=server_url, api_key=test_api_key)
        preview = await client.preview_receipt(since_last_receipt=True)

        # Verify preview
        assert preview["total_logs"] >= 1
        assert preview["total_cost"] > 0
        assert preview["receipt_type"] in ["single", "batch"]

        # Generate receipt
        result = await client.generate_receipt(since_last_receipt=True)

        # Verify generation matches preview
        receipt = result["receipt"]
        assert result["certified_count"] == preview["total_logs"]
        assert receipt["total_cost"] == preview["total_cost"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 and running server",
    )
    async def test_receipt_verification_workflow(self, server_url, test_api_key):
        """
        Test receipt verification.

        This test verifies:
        1. Receipt can be generated
        2. Receipt signature can be verified
        3. Tampered receipt fails verification
        """
        # Skip if no OpenAI key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        # Log and generate receipt
        ring = LLMRing(
            server_url=server_url,
            api_key=test_api_key,
            log_metadata=True,
        )

        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Test")],
            temperature=0.0,
        )

        await ring.chat(request)

        # Wait for server
        import asyncio

        await asyncio.sleep(0.5)

        # Generate receipt
        client = ServerClient(server_url=server_url, api_key=test_api_key)
        result = await client.generate_receipt(since_last_receipt=True)
        receipt = result["receipt"]

        # Verify receipt (server endpoint is public, no auth needed)
        verify_result = await client.post("/api/v1/receipts/verify", json=receipt)
        assert verify_result["valid"] is True
        assert verify_result["algorithm"] == "Ed25519"

        # Tamper with receipt
        tampered_receipt = receipt.copy()
        tampered_receipt["total_cost"] = 999.99

        # Verify tampered receipt fails
        verify_result = await client.post("/api/v1/receipts/verify", json=tampered_receipt)
        assert verify_result["valid"] is False
