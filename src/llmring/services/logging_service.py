"""
Logging service for LLMRing.

Handles all server-side logging of usage metadata and conversations.
"""

import logging
from typing import Any, Dict, Optional

from llmring.schemas import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LoggingService:
    """
    Handles all logging to llmring-server.

    Supports two modes:
    - Metadata-only: logs usage data (tokens, cost, model, alias)
    - Full conversations: logs messages + responses + metadata (includes usage logs)

    Note: log_conversations=True implies logging both conversation content AND usage metadata.
    This ensures usage records exist for analytics, receipts, and cost tracking.
    """

    def __init__(
        self,
        server_client: Any,
        log_metadata: bool,
        log_conversations: bool,
        origin: str,
    ):
        """
        Initialize the logging service.

        Args:
            server_client: ServerClient instance for HTTP requests
            log_metadata: Whether to log usage metadata
            log_conversations: Whether to log full conversations (also logs usage metadata)
            origin: Origin identifier for tracking

        Note: If log_conversations=True, usage metadata will be logged regardless of log_metadata value.
        """
        self.server_client = server_client
        self.log_metadata = log_metadata
        self.log_conversations = log_conversations
        self.origin = origin
        self._conversation_id: Optional[str] = None

    async def log_request_response(
        self,
        request: LLMRequest,
        response: LLMResponse,
        alias: str,
        provider: str,
        model: str,
        cost_info: Optional[Dict[str, float]],
        profile: Optional[str],
    ):
        """
        Log request/response to server based on configuration flags.

        Args:
            request: The LLM request
            response: The LLM response
            alias: Original alias or model string used
            provider: Provider name (e.g., "openai")
            model: Model name (e.g., "gpt-4o")
            cost_info: Calculated cost information
            profile: Profile name used
        """
        try:
            # Log conversations if enabled
            if self.log_conversations:
                # Log full conversation (messages + response) to conversations/messages tables
                # This provides audit trail and conversation replay capabilities
                await self._log_conversation(
                    request=request,
                    response=response,
                    alias=alias,
                    provider=provider,
                    model=model,
                    cost_info=cost_info,
                    profile=profile,
                )

                # ALSO log aggregated usage to usage_logs table for analytics
                # This creates a separate record optimized for cost tracking and reporting
                # The two tables serve different purposes:
                # - conversations/messages: normalized, message-level audit trail
                # - usage_logs: denormalized, call-level analytics (dashboards, billing)
                # No duplicate data: conversations store content; usage_logs store metrics
                await self._log_usage_only(
                    response=response,
                    alias=alias,
                    provider=provider,
                    model=model,
                    cost_info=cost_info,
                    profile=profile,
                )
            elif self.log_metadata:
                # Log usage metadata only (no conversation content)
                await self._log_usage_only(
                    response=response,
                    alias=alias,
                    provider=provider,
                    model=model,
                    cost_info=cost_info,
                    profile=profile,
                )
        except Exception as e:
            # Log warning but don't raise - logging failures shouldn't break requests
            logger.warning(f"Failed to log to server: {e}")

    async def _log_usage_only(
        self,
        response: LLMResponse,
        alias: str,
        provider: str,
        model: str,
        cost_info: Optional[Dict[str, float]],
        profile: Optional[str],
    ):
        """
        Log only usage metadata to server.

        Args:
            response: The LLM response with usage information
            alias: Original alias or model string used
            provider: Provider name
            model: Model name
            cost_info: Calculated cost information
            profile: Profile name
        """
        if not response.usage:
            return

        # Prepare log entry matching UsageLogRequest schema
        log_data = {
            "model": model,
            "provider": provider,
            "input_tokens": response.usage.get("prompt_tokens", 0),
            "output_tokens": response.usage.get("completion_tokens", 0),
            "cached_input_tokens": response.usage.get("cached_tokens", 0),
            "origin": self.origin,
        }

        # Add optional fields
        if alias and ":" not in alias:
            # It's an alias, not a direct model reference
            log_data["alias"] = alias

        if profile:
            log_data["profile"] = profile

        if cost_info and "total_cost" in cost_info:
            log_data["cost"] = cost_info["total_cost"]

        if self._conversation_id:
            log_data["id_at_origin"] = self._conversation_id

        # Add metadata
        log_data["metadata"] = {
            "model_alias": f"{provider}:{model}",
            "finish_reason": response.finish_reason,
        }

        if self._conversation_id:
            log_data["metadata"]["conversation_id"] = self._conversation_id

        # Send to server
        await self.server_client.post("/api/v1/log", json=log_data)
        logger.debug(
            f"Logged usage to server: {provider}:{model} "
            f"({log_data['input_tokens']} in, {log_data['output_tokens']} out)"
        )

    async def _log_conversation(
        self,
        request: LLMRequest,
        response: LLMResponse,
        alias: str,
        provider: str,
        model: str,
        cost_info: Optional[Dict[str, float]],
        profile: Optional[str],
    ):
        """
        Log full conversation (messages + response + metadata) to server.

        Args:
            request: The LLM request with messages
            response: The LLM response
            alias: Original alias or model string used
            provider: Provider name
            model: Model name
            cost_info: Calculated cost information
            profile: Profile name
        """
        # Convert messages to dict format
        messages = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in request.messages
        ]

        # Prepare conversation log
        conversation_data = {
            "messages": messages,
            "response": {
                "content": response.content,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": response.usage or {},
            },
            "metadata": {
                "provider": provider,
                "model": model,
                "alias": alias if ":" not in alias else None,
                "profile": profile,
                "origin": self.origin,
            },
        }

        # Add cost if available
        if cost_info:
            conversation_data["metadata"]["cost"] = cost_info.get("total_cost", 0.0)
            conversation_data["metadata"]["input_cost"] = cost_info.get("input_cost", 0.0)
            conversation_data["metadata"]["output_cost"] = cost_info.get("output_cost", 0.0)

        # Add usage tokens
        if response.usage:
            conversation_data["metadata"]["input_tokens"] = response.usage.get("prompt_tokens", 0)
            conversation_data["metadata"]["output_tokens"] = response.usage.get(
                "completion_tokens", 0
            )
            conversation_data["metadata"]["cached_tokens"] = response.usage.get("cached_tokens", 0)

        # Send to server (endpoint will be implemented in Phase 6)
        result = await self.server_client.post("/api/v1/conversations/log", json=conversation_data)

        # Extract conversation ID from response
        if isinstance(result, dict):
            self._conversation_id = result.get("conversation_id")
            logger.debug(
                f"Logged conversation to server: {provider}:{model} "
                f"(conversation_id={self._conversation_id})"
            )

            # Phase 7.5: Receipts are now generated on-demand via POST /api/v1/receipts/generate
            # No automatic receipt generation in conversation logging

    def clear_conversation_id(self):
        """Clear the current conversation ID."""
        self._conversation_id = None

    def set_conversation_id(self, conversation_id: str):
        """
        Set the conversation ID for linking multiple requests.

        Args:
            conversation_id: Conversation identifier
        """
        self._conversation_id = conversation_id
