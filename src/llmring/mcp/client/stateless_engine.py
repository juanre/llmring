"""Stateless chat engine for processing conversations."""

import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.client.mcp_client import MCPClient
from llmring.mcp.client.models.schemas import ToolCall, ToolResult
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Stateless chat request."""

    conversation_id: str | None = None
    message: str = ""
    messages: list[Message] | None = None
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    save_to_db: bool = True
    auth_context: dict[str, Any] | None = None


@dataclass
class ChatResponse:
    """Chat completion response."""

    conversation_id: str
    message: Message
    usage: dict[str, int]
    model: str
    created_at: datetime
    processing_time_ms: int
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "message": {
                "role": self.message.role,
                "content": self.message.content,
                "timestamp": (
                    self.message.timestamp.isoformat() if self.message.timestamp else None
                ),
            },
            "usage": self.usage,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "tool_calls": ([tc.dict() for tc in self.tool_calls] if self.tool_calls else None),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StreamChatChunk:
    """Streaming response chunk."""

    conversation_id: str
    delta: str
    tool_call: ToolCall | None = None
    usage: dict[str, int] | None = None
    finished: bool = False

    def dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "delta": self.delta,
            "tool_call": self.tool_call.dict() if self.tool_call else None,
            "usage": self.usage,
            "finished": self.finished,
        }


@dataclass
class ProcessingContext:
    """Internal processing context."""

    conversation_id: str
    messages: list[Message]
    system_prompt: str | None
    model: str
    temperature: float
    max_tokens: int | None
    tools: list[dict[str, Any]] | None
    auth_context: dict[str, Any]
    mcp_client: MCPClient | None = None


class StatelessChatEngine:
    """Stateless chat processing engine."""

    def __init__(
        self,
        llmring: LLMRing | None = None,
        default_model: str = "mcp_agent",  # Use alias instead of hardcoded model
        llmring_server_url: str | None = None,
        api_key: str | None = None,
        mcp_server_url: str | None = None,
        debug: bool = False,
    ):
        self.llmring = llmring or LLMRing(origin="mcp-stateless")
        self.conversation_manager = AsyncConversationManager(
            llmring_server_url=llmring_server_url,
            api_key=api_key,
        )
        self.default_model = default_model
        self.llmring_server_url = llmring_server_url
        self.api_key = api_key
        self.mcp_server_url = mcp_server_url
        self.debug = debug
        self._mcp_clients: dict[str, MCPClient] = {}

    async def process_request(
        self, request: ChatRequest, mcp_client: MCPClient | None = None
    ) -> ChatResponse:
        """Process a chat request and return response."""
        start_time = datetime.now(UTC)

        try:
            # Create processing context
            context = await self._create_context(request)
            context.mcp_client = mcp_client

            # Add user message if provided
            if request.message:
                user_message = Message(
                    role="user", content=request.message, timestamp=datetime.now(UTC)
                )
                context.messages.append(user_message)

                # Save user message to DB
                if request.save_to_db:
                    await self.conversation_manager.add_message(
                        conversation_id=context.conversation_id,
                        role=user_message.role,
                        content=user_message.content,
                    )

            # Process with LLM
            (
                response_message,
                tool_calls,
                tool_results,
                llm_response,
            ) = await self._process_with_llm(context)

            # Use actual usage from LLM response if available, otherwise calculate
            if llm_response and llm_response.usage:
                usage = llm_response.usage
            else:
                usage = self._calculate_usage(context.messages, response_message)

            # Calculate processing time
            processing_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            # Save to database if requested
            if request.save_to_db:
                metadata = {}
                if tool_calls:
                    metadata["tool_calls"] = [tc.__dict__ for tc in tool_calls]
                if tool_results:
                    metadata["tool_results"] = [tr.__dict__ for tr in tool_results]
                if processing_time_ms:
                    metadata["processing_time_ms"] = processing_time_ms
                await self.conversation_manager.add_message(
                    conversation_id=context.conversation_id,
                    role=response_message.role,
                    content=response_message.content,
                    token_count=usage.get("completion_tokens", 0),
                    metadata=metadata if metadata else None,
                )

            return ChatResponse(
                conversation_id=context.conversation_id,
                message=response_message,
                usage=usage,
                model=context.model,
                created_at=datetime.now(UTC),
                processing_time_ms=processing_time_ms,
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Error processing chat request: {e}", exc_info=True)
            raise

    async def process_request_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChatChunk, None]:
        """Process request with streaming response."""
        # Create processing context
        context = await self._create_context(request)

        # Add user message
        if request.message:
            user_message = Message(
                role="user", content=request.message, timestamp=datetime.now(UTC)
            )
            context.messages.append(user_message)

            # Save user message to DB
            if request.save_to_db:
                await self.conversation_manager.add_message(
                    conversation_id=context.conversation_id,
                    role=user_message.role,
                    content=user_message.content,
                )

        # Stream from LLM
        full_response = ""
        tool_calls = []

        async for chunk in self._stream_from_llm(context):
            if chunk.get("type") == "text":
                full_response += chunk["text"]
                yield StreamChatChunk(
                    conversation_id=context.conversation_id,
                    delta=chunk["text"],
                    finished=False,
                )
            elif chunk.get("type") == "tool_call":
                tool_call = chunk["tool_call"]
                tool_calls.append(tool_call)
                yield StreamChatChunk(
                    conversation_id=context.conversation_id,
                    delta="",
                    tool_call=tool_call,
                    finished=False,
                )

        # Save complete response
        if request.save_to_db and full_response:
            response_message = Message(
                role="assistant", content=full_response, timestamp=datetime.now(UTC)
            )

            # Calculate usage
            usage = self._calculate_usage(context.messages, response_message)

            metadata = {}
            if tool_calls:
                metadata["tool_calls"] = [tc.__dict__ for tc in tool_calls]
            await self.conversation_manager.add_message(
                conversation_id=context.conversation_id,
                role=response_message.role,
                content=response_message.content,
                token_count=usage.get("completion_tokens", 0),
                metadata=metadata if metadata else None,
            )

            # Send final chunk with usage
            yield StreamChatChunk(
                conversation_id=context.conversation_id,
                delta="",
                usage=usage,
                finished=True,
            )

    async def execute_tool(
        self, conversation_id: str, tool_call: ToolCall, auth_context: dict[str, Any]
    ) -> ToolResult:
        """Execute a tool within conversation context."""
        # Load conversation
        conversation = await self.conversation_manager.get_conversation(conversation_id)

        # Get MCP client for conversation
        mcp_client = await self._get_mcp_client(conversation)

        if not mcp_client:
            raise ValueError("No MCP client available for tool execution")

        # Execute tool
        result = await mcp_client.call_tool(tool_call.tool_name, tool_call.arguments)

        # Create tool result
        tool_result = ToolResult(
            tool_call_id=tool_call.id, result=result, timestamp=datetime.now(UTC)
        )

        # Save tool result
        # Save tool result as a message with metadata
        await self.conversation_manager.add_message(
            conversation_id=conversation_id,
            role="tool",
            content=str(tool_result.result),
            metadata={
                "tool_result": {
                    "tool_call_id": tool_result.tool_call_id,
                    "result": tool_result.result,
                }
            },
        )

        return tool_result

    async def _create_context(self, request: ChatRequest) -> ProcessingContext:
        """Create processing context from request."""
        if request.conversation_id:
            # Load existing conversation
            conversation = await self.conversation_manager.get_conversation(request.conversation_id)

            if not conversation:
                raise ValueError(f"Conversation {request.conversation_id} not found")

            return ProcessingContext(
                conversation_id=request.conversation_id,
                messages=(
                    request.messages if request.messages is not None else conversation.messages
                ),
                system_prompt=request.system_prompt or conversation.system_prompt,
                model=request.model or conversation.model,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else conversation.temperature
                ),
                max_tokens=request.max_tokens or conversation.max_tokens,
                tools=request.tools or conversation.tool_config,
                auth_context=request.auth_context or {},
            )
        else:
            # Create new conversation
            if request.save_to_db:
                # Create conversation returns the ID
                # Get user_id from auth_context if available
                user_id = (
                    request.auth_context.get("user_id", "default-user")
                    if request.auth_context
                    else "default-user"
                )
                conversation_id = await self.conversation_manager.create_conversation(
                    user_id=user_id,
                    title="New Conversation",
                    system_prompt=request.system_prompt,
                    model=request.model or self.default_model,
                    temperature=request.temperature or 0.7,
                    max_tokens=request.max_tokens,
                    tool_config={"tools": request.tools} if request.tools else None,
                )
            else:
                conversation_id = str(uuid.uuid4())

            return ProcessingContext(
                conversation_id=conversation_id,
                messages=request.messages or [],
                system_prompt=request.system_prompt,
                model=request.model or self.default_model,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens,
                tools=request.tools,
                auth_context=request.auth_context or {},
            )

    async def _process_with_llm(
        self, context: ProcessingContext
    ) -> tuple[Message, list[ToolCall] | None, list[ToolResult] | None, Any]:
        """Process messages with LLM."""
        # Prepare messages
        messages = self._prepare_messages(context)

        # Create LLM request
        llm_request = LLMRequest(
            messages=messages,
            model=context.model,
            temperature=context.temperature,
            max_tokens=context.max_tokens,
            tools=context.tools,
        )

        # Call LLM
        # Store id_at_origin in metadata if needed for tracking
        if context.auth_context and "user_id" in context.auth_context:
            llm_request.metadata = llm_request.metadata or {}
            llm_request.metadata["id_at_origin"] = context.auth_context["user_id"]

        response = await self.llmring.chat(llm_request)

        # We're not streaming, so response should be LLMResponse
        # Type assertion to satisfy type checker
        from llmring.schemas import LLMResponse

        assert isinstance(response, LLMResponse)

        # Extract response content
        content = response.content

        # Create response message
        response_message = Message(role="assistant", content=content, timestamp=datetime.now(UTC))

        # Handle tool calls if present
        tool_calls = None
        tool_results = None

        if response.tool_calls:
            tool_calls = []
            tool_results = []

            for tc in response.tool_calls:
                tool_call = ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    tool_name=tc.get("function", {}).get("name", tc.get("name", "")),
                    arguments=tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                )
                tool_calls.append(tool_call)

                # Execute tools if MCP client available
                if context.mcp_client:
                    try:
                        result = await context.mcp_client.call_tool(
                            tool_call.tool_name, tool_call.arguments
                        )
                        tool_results.append(
                            ToolResult(
                                tool_call_id=tool_call.id,
                                result=result,
                                timestamp=datetime.now(UTC),
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_call.tool_name}: {e}")
                        tool_results.append(
                            ToolResult(
                                tool_call_id=tool_call.id,
                                result={"error": str(e)},
                                timestamp=datetime.now(UTC),
                            )
                        )

        return (
            response_message,
            tool_calls,
            tool_results if tool_results else None,
            response,
        )

    async def _stream_from_llm(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream response from LLM."""
        messages = self._prepare_messages(context)

        # Create LLM request
        llm_request = LLMRequest(
            messages=messages,
            model=context.model,
            temperature=context.temperature,
            max_tokens=context.max_tokens,
            tools=context.tools,
        )

        # For now, simulate streaming by calling regular completion
        # In production, this would use actual streaming API
        # Store id_at_origin in metadata if needed for tracking
        if context.auth_context and "user_id" in context.auth_context:
            llm_request.metadata = llm_request.metadata or {}
            llm_request.metadata["id_at_origin"] = context.auth_context["user_id"]

        response = await self.llmring.chat(llm_request)

        # We're not streaming, so response should be LLMResponse
        # Type assertion to satisfy type checker
        from llmring.schemas import LLMResponse

        assert isinstance(response, LLMResponse)

        # Simulate streaming chunks
        content = response.content
        chunk_size = 20  # Characters per chunk

        for i in range(0, len(content), chunk_size):
            yield {"type": "text", "text": content[i : i + chunk_size]}

        # Yield tool calls if any
        if response.tool_calls:
            for tc in response.tool_calls:
                yield {
                    "type": "tool_call",
                    "tool_call": ToolCall(
                        id=tc.get("id", str(uuid.uuid4())),
                        tool_name=tc.get("function", {}).get("name", tc.get("name", "")),
                        arguments=tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                    ),
                }

    def _prepare_messages(self, context: ProcessingContext) -> list[Message]:
        """Prepare messages for LLM."""
        messages = []

        # Add system prompt if present
        if context.system_prompt:
            messages.append(Message(role="system", content=context.system_prompt))

        # Add conversation messages
        messages.extend(context.messages)

        return messages

    def _calculate_usage(self, messages: list[Message], response: Message) -> dict[str, int]:
        """Calculate token usage."""
        # This is a simplified version
        # In production, use proper tokenizer for the model
        prompt_tokens = int(sum(len(msg.content.split()) * 1.3 for msg in messages))
        completion_tokens = int(len(response.content.split()) * 1.3)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    async def _get_mcp_client(self, conversation) -> MCPClient | None:
        """Get or create MCP client for conversation."""
        # This is a placeholder - actual implementation would manage MCP clients
        # based on conversation configuration
        return None
