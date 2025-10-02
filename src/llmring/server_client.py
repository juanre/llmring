"""Client utilities for interacting with llmring-server.

This module provides utilities for server communication, but does NOT include
alias sync functionality as aliases are purely local per source-of-truth v3.8.

Phase 7.5: Added on-demand receipt generation methods.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from llmring.net.http_base import BaseHTTPClient

logger = logging.getLogger(__name__)


class ServerClient(BaseHTTPClient):
    """Client for communicating with llmring-server or llmring-api."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the server client.

        Args:
            base_url: Base URL of the server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    # The base class provides all the needed methods:
    # - post(path, json) -> Dict[str, Any]
    # - get(path, params) -> Dict[str, Any]
    # - put(path, json) -> Dict[str, Any]
    # - delete(path) -> Union[Dict[str, Any], bool]
    # - close() -> None
    # - __aenter__ and __aexit__ for context manager support

    # =====================================================
    # Phase 7.5: On-Demand Receipt Generation
    # =====================================================

    async def generate_receipt(
        self,
        conversation_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        log_ids: Optional[List[str]] = None,
        since_last_receipt: bool = False,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a receipt on-demand for specified logs/conversations.

        This implements Phase 7.5 on-demand receipt generation. Unlike Phase 7
        where receipts were generated automatically, this allows explicit control
        over when receipts are created for compliance/certification.

        Supports four modes:
        1. Single conversation: Provide conversation_id
        2. Date range batch: Provide start_date and end_date
        3. Specific logs: Provide log_ids
        4. Since last receipt: Set since_last_receipt=True

        Args:
            conversation_id: Generate receipt for a single conversation
            start_date: Start date for batch receipt (requires end_date)
            end_date: End date for batch receipt (requires start_date)
            log_ids: List of specific log IDs to certify
            since_last_receipt: Certify all uncertified logs since last receipt
            description: User-provided description for the receipt
            tags: User-provided tags for categorization

        Returns:
            Dict containing:
                - receipt: The signed BatchReceipt object
                - certified_count: Number of logs certified

        Example:
            >>> # Single conversation
            >>> result = await client.generate_receipt(
            ...     conversation_id="550e8400-e29b-41d4-a716-446655440000",
            ...     description="Customer support conversation"
            ... )
            >>> receipt = result["receipt"]
            >>> print(f"Receipt {receipt['receipt_id']}: ${receipt['total_cost']}")

            >>> # Batch for billing period
            >>> result = await client.generate_receipt(
            ...     start_date=datetime(2025, 10, 1),
            ...     end_date=datetime(2025, 10, 31),
            ...     description="October 2025 billing",
            ...     tags=["billing", "monthly"]
            ... )
            >>> print(f"Certified {result['certified_count']} logs")
        """
        payload: Dict[str, Any] = {}

        if conversation_id:
            payload["conversation_id"] = conversation_id
        if start_date:
            payload["start_date"] = start_date.isoformat()
        if end_date:
            payload["end_date"] = end_date.isoformat()
        if log_ids:
            payload["log_ids"] = log_ids
        if since_last_receipt:
            payload["since_last_receipt"] = True
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags

        return await self.post("/api/v1/receipts/generate", json=payload)

    async def preview_receipt(
        self,
        conversation_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        log_ids: Optional[List[str]] = None,
        since_last_receipt: bool = False,
    ) -> Dict[str, Any]:
        """
        Preview what a receipt would certify without actually generating it.

        This allows users to see aggregate statistics and verify which logs
        would be included before committing to generating a signed receipt.

        Args:
            conversation_id: Preview receipt for a single conversation
            start_date: Start date for batch preview
            end_date: End date for batch preview
            log_ids: List of specific log IDs to preview
            since_last_receipt: Preview all uncertified logs

        Returns:
            Dict containing:
                - total_logs: Number of logs that would be certified
                - total_conversations: Number of conversations
                - total_tokens: Total tokens across all logs
                - total_cost: Total cost
                - start_date: First log timestamp (ISO format)
                - end_date: Last log timestamp (ISO format)
                - by_model: Breakdown by model
                - by_alias: Breakdown by alias
                - receipt_type: 'single' or 'batch'

        Example:
            >>> preview = await client.preview_receipt(
            ...     start_date=datetime(2025, 10, 1),
            ...     end_date=datetime(2025, 10, 31)
            ... )
            >>> print(f"Would certify {preview['total_logs']} logs")
            >>> print(f"Total cost: ${preview['total_cost']}")
            >>> print(f"By model: {preview['by_model']}")
        """
        payload: Dict[str, Any] = {}

        if conversation_id:
            payload["conversation_id"] = conversation_id
        if start_date:
            payload["start_date"] = start_date.isoformat()
        if end_date:
            payload["end_date"] = end_date.isoformat()
        if log_ids:
            payload["log_ids"] = log_ids
        if since_last_receipt:
            payload["since_last_receipt"] = True

        return await self.post("/api/v1/receipts/preview", json=payload)

    async def get_uncertified_logs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get logs that haven't been certified by any receipt.

        This helps users identify which conversations/usage logs don't yet
        have a receipt, useful for periodic certification workflows.

        Args:
            limit: Maximum logs to return (1-1000, default 100)
            offset: Number of logs to skip for pagination

        Returns:
            Dict containing:
                - logs: List of uncertified log objects
                - total: Total count of uncertified logs
                - limit: Limit used
                - offset: Offset used

        Example:
            >>> uncert = await client.get_uncertified_logs()
            >>> print(f"Found {uncert['total']} uncertified logs")
            >>> for log in uncert['logs']:
            ...     print(f"  - {log['id']} ({log['type']})")
        """
        params = {"limit": limit, "offset": offset}
        return await self.get("/api/v1/receipts/uncertified", params=params)

    async def get_receipt_logs(
        self,
        receipt_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get all logs certified by a specific receipt.

        This returns full details of all conversations/usage logs that were
        certified by the specified receipt. Useful for audit trails.

        Args:
            receipt_id: The receipt ID to get logs for
            limit: Maximum logs to return (1-1000, default 100)
            offset: Number of logs to skip for pagination

        Returns:
            Dict containing:
                - receipt_id: The receipt ID
                - logs: List of certified log objects
                - total: Total count of logs certified by this receipt
                - limit: Limit used
                - offset: Offset used

        Example:
            >>> logs = await client.get_receipt_logs("rcpt_abc123")
            >>> print(f"Receipt certified {logs['total']} logs")
            >>> for log in logs['logs']:
            ...     print(f"  - {log['id']}: {log.get('total_cost', 0)}")
        """
        params = {"limit": limit, "offset": offset}
        return await self.get(f"/api/v1/receipts/{receipt_id}/logs", params=params)

    async def list_receipts(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List receipts for the authenticated API key.

        Args:
            limit: Maximum receipts to return (1-1000, default 100)
            offset: Number of receipts to skip for pagination

        Returns:
            Dict containing:
                - receipts: List of receipt objects
                - total: Total count of receipts
                - limit: Limit used
                - offset: Offset used

        Example:
            >>> receipts = await client.list_receipts()
            >>> for receipt in receipts['receipts']:
            ...     print(f"{receipt['receipt_id']}: ${receipt['total_cost']}")
        """
        params = {"limit": limit, "offset": offset}
        return await self.get("/api/v1/receipts", params=params)

    async def get_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """
        Get a specific receipt by ID.

        Args:
            receipt_id: The receipt ID to retrieve

        Returns:
            Receipt object as dict

        Example:
            >>> receipt = await client.get_receipt("rcpt_abc123")
            >>> print(f"Receipt type: {receipt['receipt_type']}")
            >>> if receipt['receipt_type'] == 'batch':
            ...     print(f"Certified {receipt['batch_summary']['total_calls']} calls")
        """
        return await self.get(f"/api/v1/receipts/{receipt_id}")
