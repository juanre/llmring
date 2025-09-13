"""
Streaming handler for MCP Enhanced LLM.

Handles streaming responses with tool execution support.
"""

import json
import logging
from typing import AsyncIterator

from llmring.schemas import LLMRequest, Message, StreamChunk
from llmring.exceptions import MCPToolError, ToolExecutionError

logger = logging.getLogger(__name__)


class StreamingToolHandler:
    """Handles streaming responses that may include tool calls."""

    def __init__(self, enhanced_llm):
        """
        Initialize the streaming handler.

        Args:
            enhanced_llm: The EnhancedLLM instance for tool execution
        """
        self.enhanced_llm = enhanced_llm

    async def handle_streaming_with_tools(
        self,
        initial_request: LLMRequest,
        formatted_messages: list[Message],
    ) -> AsyncIterator[StreamChunk]:
        """
        Handle streaming with potential tool calls.

        This method:
        1. Streams the initial response
        2. If tool calls are detected, executes them
        3. Makes a follow-up request with tool results
        4. Streams the final response

        Args:
            initial_request: The initial LLM request
            formatted_messages: The formatted message history

        Yields:
            StreamChunk objects from the LLM
        """
        # First, get the streaming response
        stream = await self.enhanced_llm.llmring.chat(initial_request)

        # Accumulate the response to check for tool calls
        accumulated_content = ""
        accumulated_tool_calls = None
        accumulated_usage = None

        async for chunk in stream:
            # Always yield the chunk to the client
            yield chunk

            # Accumulate content
            if chunk.delta:
                accumulated_content += chunk.delta

            # Check for tool calls in the chunk (they come in the final chunk)
            if chunk.tool_calls:
                accumulated_tool_calls = chunk.tool_calls

            # Usage information is already in the chunk, no need to accumulate

            # If we've finished and have tool calls, handle them
            if chunk.finish_reason == "tool_calls" and accumulated_tool_calls:
                # Execute the tool calls
                tool_results = await self._execute_tool_calls(accumulated_tool_calls)

                # Add the assistant message with tool calls to history
                formatted_messages.append(
                    Message(
                        role="assistant",
                        content=accumulated_content or "",
                        tool_calls=accumulated_tool_calls,
                    )
                )

                # Add tool results to messages
                for tool_result in tool_results:
                    formatted_messages.append(
                        Message(
                            role="tool",
                            content=tool_result["content"],
                            tool_call_id=tool_result["tool_call_id"],
                        )
                    )

                # Create follow-up request for the final response
                follow_up_request = LLMRequest(
                    model=initial_request.model,
                    messages=formatted_messages,
                    temperature=initial_request.temperature,
                    max_tokens=initial_request.max_tokens,
                    stream=True,  # Continue streaming
                )

                # Stream the follow-up response
                follow_up_stream = await self.enhanced_llm.llmring.chat(
                    follow_up_request
                )
                async for follow_up_chunk in follow_up_stream:
                    yield follow_up_chunk

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """
        Execute a list of tool calls.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of tool results with tool_call_id and content
        """
        tool_results = []

        for tool_call in tool_calls:
            try:
                # Parse arguments if they're a string
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)

                # Execute the tool
                result = await self.enhanced_llm._execute_tool(
                    tool_call["function"]["name"],
                    args,
                )

                # Format the result
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                        if not isinstance(result, str)
                        else result,
                    }
                )

            except (ToolExecutionError, MCPToolError) as e:
                logger.warning(f"Tool execution failed: {e}")
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "content": f"Tool execution failed: {e}",
                    }
                )

            except Exception as e:
                logger.error(f"Unexpected error executing tool: {e}")
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "content": f"Unexpected error executing tool: {e}",
                    }
                )

        return tool_results
