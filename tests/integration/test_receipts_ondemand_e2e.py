"""
End-to-end integration tests for on-demand receipt generation (Phase 7.5).

These tests use the real llmring-server (via llmring_server_client fixture)
to verify the complete workflow from client to server.

NO MOCKS - Real server, real database, real HTTP calls.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict

import pytest

from llmring import LLMRing
from llmring.schemas import LLMRequest, Message
from llmring.server_client import ServerClient


class TestOnDemandReceiptsE2E:
    """End-to-end tests for on-demand receipt generation."""

    @pytest.mark.asyncio
    async def test_single_conversation_receipt_generation(
        self, llmring_server_client, project_headers
    ):
        """
        Test generating a receipt for a single conversation.

        Flow:
        1. Log a conversation via POST /api/v1/conversations/log
        2. Verify no automatic receipt is generated
        3. Generate receipt on-demand via POST /api/v1/receipts/generate
        4. Verify receipt contents and signature
        """
        # Step 1: Log a conversation
        log_response = await llmring_server_client.post(
            "/api/v1/conversations/log",
            json={
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "response": {
                    "content": "4",
                    "model": "gpt-4o-mini",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "alias": "fast",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost": 0.0001,
                    "input_cost": 0.00006,
                    "output_cost": 0.00004,
                },
            },
            headers=project_headers,
        )

        assert log_response.status_code == 200
        log_data = log_response.json()
        conversation_id = log_data["conversation_id"]

        # Step 2: Verify no automatic receipt
        assert log_data.get("receipt") is None, "Phase 7.5: receipts should not be auto-generated"

        # Step 3: Generate receipt on-demand
        receipt_response = await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={
                "conversation_id": conversation_id,
                "description": "Test receipt for math question",
                "tags": ["test", "math"],
            },
            headers=project_headers,
        )

        if receipt_response.status_code != 200:
            print(f"Receipt generation error: {receipt_response.text}")
        assert receipt_response.status_code == 200
        receipt_data = receipt_response.json()

        # Step 4: Verify receipt
        assert "receipt" in receipt_data
        assert "certified_count" in receipt_data
        assert receipt_data["certified_count"] == 1

        receipt = receipt_data["receipt"]
        assert receipt["receipt_type"] == "single"
        assert receipt["alias"] == "fast"
        assert receipt["provider"] == "openai"
        assert receipt["model"] == "gpt-4o-mini"
        assert receipt["prompt_tokens"] == 10
        assert receipt["completion_tokens"] == 5
        assert receipt["total_cost"] == 0.0001
        assert receipt["description"] == "Test receipt for math question"
        assert receipt["tags"] == ["test", "math"]

        # Verify signature
        assert "signature" in receipt
        assert receipt["signature"].startswith("ed25519:")
        assert "receipt_id" in receipt
        assert receipt["receipt_id"].startswith("rcpt_")

    @pytest.mark.asyncio
    async def test_batch_receipt_date_range(self, llmring_server_client, project_headers):
        """
        Test generating a batch receipt for multiple conversations in a date range.

        Flow:
        1. Log multiple conversations
        2. Generate batch receipt for date range
        3. Verify batch summary and aggregated costs
        """
        start_time = datetime.now(timezone.utc)

        # Log 3 conversations with different models
        conversations = []
        for i in range(3):
            response = await llmring_server_client.post(
                "/api/v1/conversations/log",
                json={
                    "messages": [{"role": "user", "content": f"Question {i}"}],
                    "response": {
                        "content": f"Answer {i}",
                        "model": f"test-model-{i}",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 10 * (i + 1), "completion_tokens": 5 * (i + 1)},
                    },
                    "metadata": {
                        "provider": "openai" if i % 2 == 0 else "anthropic",
                        "model": f"model-{i}",
                        "alias": f"alias_{i}",
                        "input_tokens": 10 * (i + 1),
                        "output_tokens": 5 * (i + 1),
                        "cost": 0.0001 * (i + 1),
                        "input_cost": 0.00006 * (i + 1),
                        "output_cost": 0.00004 * (i + 1),
                    },
                },
                headers=project_headers,
            )
            assert response.status_code == 200
            conversations.append(response.json()["conversation_id"])

        end_time = datetime.now(timezone.utc)

        # Generate batch receipt
        receipt_response = await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat(),
                "description": "Batch receipt for test period",
                "tags": ["batch", "test"],
            },
            headers=project_headers,
        )

        assert receipt_response.status_code == 200
        data = receipt_response.json()

        assert data["certified_count"] == 3

        receipt = data["receipt"]
        assert receipt["receipt_type"] == "batch"
        assert "batch_summary" in receipt

        summary = receipt["batch_summary"]
        assert summary["total_conversations"] >= 3
        assert summary["total_tokens"] == (10 + 20 + 30) + (5 + 10 + 15)  # input + output
        assert abs(summary["total_cost"] - 0.0006) < 0.0001  # Floating point comparison

        # Verify signature on batch receipt
        assert receipt["signature"].startswith("ed25519:")

    @pytest.mark.asyncio
    async def test_preview_receipt_before_generation(self, llmring_server_client, project_headers):
        """
        Test previewing a receipt before actually generating it.

        Flow:
        1. Log some conversations
        2. Preview what a receipt would contain
        3. Verify preview doesn't create a receipt
        4. Actually generate the receipt
        """
        # Log 2 conversations
        for i in range(2):
            await llmring_server_client.post(
                "/api/v1/conversations/log",
                json={
                    "messages": [{"role": "user", "content": f"Test {i}"}],
                    "response": {
                        "content": f"Response {i}",
                        "model": "test-model",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 15, "completion_tokens": 10},
                    },
                    "metadata": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "input_tokens": 15,
                        "output_tokens": 10,
                        "cost": 0.00015,
                    },
                },
                headers=project_headers,
            )

        # Preview the receipt
        preview_response = await llmring_server_client.post(
            "/api/v1/receipts/preview",
            json={"since_last_receipt": True},
            headers=project_headers,
        )

        assert preview_response.status_code == 200
        preview_data = preview_response.json()

        # Preview response is the preview data directly, no wrapper
        assert "total_conversations" in preview_data
        assert preview_data["total_conversations"] == 2
        assert preview_data["total_cost"] == 0.0003

        # Verify preview didn't create a receipt
        receipts_response = await llmring_server_client.get(
            "/api/v1/receipts/",
            headers=project_headers,
        )
        receipts_before = receipts_response.json()

        # Now actually generate
        generate_response = await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={"since_last_receipt": True},
            headers=project_headers,
        )

        assert generate_response.status_code == 200

        # Verify receipt was created
        receipts_response = await llmring_server_client.get(
            "/api/v1/receipts/",
            headers=project_headers,
        )
        receipts_after = receipts_response.json()
        assert len(receipts_after["receipts"]) == len(receipts_before.get("receipts", [])) + 1

    @pytest.mark.asyncio
    async def test_uncertified_logs_retrieval(self, llmring_server_client, project_headers):
        """
        Test retrieving logs that haven't been certified yet.

        Flow:
        1. Log some conversations
        2. Generate receipt for some of them
        3. Query uncertified logs
        4. Verify only uncertified ones are returned
        """
        # Log 3 conversations
        conversation_ids = []
        for i in range(3):
            response = await llmring_server_client.post(
                "/api/v1/conversations/log",
                json={
                    "messages": [{"role": "user", "content": f"Msg {i}"}],
                    "response": {
                        "content": f"Resp {i}",
                        "model": "test",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                    },
                    "metadata": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "input_tokens": 5,
                        "output_tokens": 5,
                        "cost": 0.0001,
                    },
                },
                headers=project_headers,
            )
            conversation_ids.append(response.json()["conversation_id"])

        # Certify only the first conversation
        await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={"conversation_id": conversation_ids[0]},
            headers=project_headers,
        )

        # Query uncertified logs
        uncertified_response = await llmring_server_client.get(
            "/api/v1/receipts/uncertified",
            headers=project_headers,
        )

        assert uncertified_response.status_code == 200
        uncertified = uncertified_response.json()

        # Should have 2 uncertified conversations (IDs 1 and 2)
        assert "logs" in uncertified
        assert len(uncertified["logs"]) >= 2
        assert uncertified["total"] >= 2

    @pytest.mark.asyncio
    async def test_receipt_verification(self, llmring_server_client, project_headers):
        """
        Test verifying a receipt signature.

        Flow:
        1. Generate a receipt
        2. Verify its signature via POST /api/v1/receipts/verify
        3. Tamper with the receipt and verify it fails
        """
        # Generate a receipt
        await llmring_server_client.post(
            "/api/v1/conversations/log",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "response": {
                    "content": "Response",
                    "model": "test",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 5,
                    "output_tokens": 5,
                    "cost": 0.0001,
                },
            },
            headers=project_headers,
        )

        receipt_response = await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={"since_last_receipt": True},
            headers=project_headers,
        )

        receipt = receipt_response.json()["receipt"]

        # Verify valid receipt
        # The verify endpoint expects the receipt object directly, not wrapped
        verify_response = await llmring_server_client.post(
            "/api/v1/receipts/verify",
            json=receipt,
        )

        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        assert verify_data["valid"] is True

        # Tamper with receipt (change cost)
        tampered_receipt = receipt.copy()
        tampered_receipt["total_cost"] = 999.99

        # Verify tampered receipt fails
        verify_tampered_response = await llmring_server_client.post(
            "/api/v1/receipts/verify",
            json=tampered_receipt,
        )

        assert verify_tampered_response.status_code == 200
        verify_tampered_data = verify_tampered_response.json()
        assert verify_tampered_data["valid"] is False

    @pytest.mark.asyncio
    async def test_get_receipt_logs(self, llmring_server_client, project_headers):
        """
        Test retrieving all logs certified by a specific receipt.

        Flow:
        1. Log multiple conversations
        2. Generate batch receipt
        3. Retrieve logs linked to that receipt
        """
        # Log 2 conversations
        for i in range(2):
            await llmring_server_client.post(
                "/api/v1/conversations/log",
                json={
                    "messages": [{"role": "user", "content": f"Q{i}"}],
                    "response": {
                        "content": f"A{i}",
                        "model": "test",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                    },
                    "metadata": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "input_tokens": 5,
                        "output_tokens": 5,
                        "cost": 0.0001,
                    },
                },
                headers=project_headers,
            )

        # Generate batch receipt
        receipt_response = await llmring_server_client.post(
            "/api/v1/receipts/generate",
            json={"since_last_receipt": True},
            headers=project_headers,
        )

        receipt_id = receipt_response.json()["receipt"]["receipt_id"]

        # Get logs for this receipt
        logs_response = await llmring_server_client.get(
            f"/api/v1/receipts/{receipt_id}/logs",
            headers=project_headers,
        )

        assert logs_response.status_code == 200
        logs_data = logs_response.json()

        assert "logs" in logs_data
        assert len(logs_data["logs"]) == 2


class TestClientSDKReceiptMethods:
    """Test the ServerClient receipt methods work end-to-end."""

    @pytest.mark.asyncio
    async def test_server_client_generate_receipt(self, llmring_server_client, project_headers):
        """Test ServerClient.generate_receipt() method."""
        # First log a conversation directly via HTTP
        await llmring_server_client.post(
            "/api/v1/conversations/log",
            json={
                "messages": [{"role": "user", "content": "SDK test"}],
                "response": {
                    "content": "SDK response",
                    "model": "test",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost": 0.0002,
                },
            },
            headers=project_headers,
        )

        # Use ServerClient to generate receipt
        client = ServerClient(
            base_url="http://test",
            api_key=project_headers["X-API-Key"],
        )

        # Inject the test client (replace the httpx client with our test one)
        await client.close()  # Close the default client first
        client.client = llmring_server_client

        result = await client.generate_receipt(
            since_last_receipt=True,
            description="Generated via SDK",
        )

        assert "receipt" in result
        assert "certified_count" in result
        assert result["certified_count"] >= 1
        assert result["receipt"]["description"] == "Generated via SDK"

    @pytest.mark.asyncio
    async def test_server_client_preview_receipt(self, llmring_server_client, project_headers):
        """Test ServerClient.preview_receipt() method."""
        # Log a conversation
        await llmring_server_client.post(
            "/api/v1/conversations/log",
            json={
                "messages": [{"role": "user", "content": "Preview test"}],
                "response": {
                    "content": "Preview response",
                    "model": "test",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                },
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 8,
                    "output_tokens": 4,
                    "cost": 0.00012,
                },
            },
            headers=project_headers,
        )

        # Use ServerClient to preview
        client = ServerClient(
            base_url="http://test",
            api_key=project_headers["X-API-Key"],
        )
        await client.close()
        client.client = llmring_server_client

        preview = await client.preview_receipt(since_last_receipt=True)

        # Preview response is the preview data directly, no wrapper
        assert "total_cost" in preview
        assert preview["total_cost"] >= 0.00012

    @pytest.mark.asyncio
    async def test_server_client_get_uncertified_logs(self, llmring_server_client, project_headers):
        """Test ServerClient.get_uncertified_logs() method."""
        # Log an uncertified conversation
        await llmring_server_client.post(
            "/api/v1/conversations/log",
            json={
                "messages": [{"role": "user", "content": "Uncertified"}],
                "response": {
                    "content": "Response",
                    "model": "test",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cost": 0.0001,
                },
            },
            headers=project_headers,
        )

        # Use ServerClient to get uncertified
        client = ServerClient(
            base_url="http://test",
            api_key=project_headers["X-API-Key"],
        )
        await client.close()
        client.client = llmring_server_client

        uncertified = await client.get_uncertified_logs()

        assert "logs" in uncertified
        assert len(uncertified["logs"]) >= 1
        assert uncertified["total"] >= 1
