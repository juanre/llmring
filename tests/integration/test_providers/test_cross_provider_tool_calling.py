"""
Cross-provider test for tool calling conversation flow.

This test ensures all providers correctly handle the complete tool calling cycle:
1. User query requiring tool use
2. Model responds with tool call
3. Tool response is added to conversation
4. Model processes tool result and provides final answer

This would have caught the Google tool loop bug where tool responses were ignored.
"""

import json
from typing import Any, Dict, List

import pytest

from llmring.schemas import LLMResponse, Message


def get_weather_tool() -> Dict[str, Any]:
    """Simple weather tool for testing."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }


@pytest.mark.integration
@pytest.mark.llm
class TestCrossProviderToolCalling:
    """Test tool calling conversation flow across all providers."""

    @pytest.mark.parametrize(
        "provider_model",
        [
            pytest.param("openai_fast", marks=pytest.mark.openai),
            pytest.param("anthropic_fast", marks=pytest.mark.anthropic),
            pytest.param("google_fast", marks=pytest.mark.google),
        ],
    )
    @pytest.mark.asyncio
    async def test_tool_conversation_flow(self, provider_model: str, service):
        """
        Test complete tool calling conversation flow.

        This test verifies that after a tool call and tool response,
        the model correctly processes the result and provides a final answer
        instead of calling the tool again.
        """
        from llmring.schemas import LLMRequest

        # Step 1: Initial query that should trigger tool use
        messages = [Message(role="user", content="What's the weather in Paris right now?")]

        tools = [get_weather_tool()]

        request = LLMRequest(
            messages=messages,
            model=provider_model,
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
            temperature=0.1,  # Low temperature for consistency
        )

        # First call - should trigger tool use
        response1 = await service.chat(request)

        assert isinstance(response1, LLMResponse)
        assert response1.tool_calls is not None, f"{provider_model} should call the weather tool"
        assert len(response1.tool_calls) == 1, f"{provider_model} should call exactly one tool"

        tool_call = response1.tool_calls[0]
        assert tool_call["function"]["name"] == "get_weather"

        # Parse the arguments to verify they're valid
        args_str = tool_call["function"].get("arguments", "{}")
        if isinstance(args_str, str):
            args = json.loads(args_str)
        else:
            args = args_str
        assert "location" in args, "Tool call should include location parameter"

        # Step 2: Add assistant response with tool call to conversation
        messages.append(
            Message(
                role="assistant", content=response1.content or "", tool_calls=response1.tool_calls
            )
        )

        # Step 3: Add tool response
        tool_result = {
            "temperature": "22°C",
            "condition": "sunny",
            "humidity": "45%",
            "location": args.get("location", "Paris"),
        }

        messages.append(
            Message(
                role="tool",
                content=json.dumps(tool_result),
                tool_call_id=tool_call.get("id", "call_1"),
            )
        )

        # Step 4: Continue conversation - model should process tool result
        request2 = LLMRequest(
            messages=messages,
            model=provider_model,
            tools=tools,  # Keep tools available to test if model calls them again
            tool_choice="auto",
            max_tokens=150,
            temperature=0.1,
        )

        response2 = await service.chat(request2)

        assert isinstance(response2, LLMResponse)
        assert response2.content is not None, f"{provider_model} should provide a final answer"

        # CRITICAL: Model should NOT call the tool again
        if response2.tool_calls:
            # This is the bug we're catching!
            pytest.fail(
                f"{provider_model} called tools again after receiving tool response. "
                f"This indicates a conversation flow bug (tool loop). "
                f"Tool calls: {response2.tool_calls}"
            )

        # Verify the response references the weather information
        response_lower = response2.content.lower()
        weather_mentioned = any(
            [
                "22" in response2.content,
                "sunny" in response_lower,
                "weather" in response_lower,
                "temperature" in response_lower,
                args.get("location", "paris").lower() in response_lower,
            ]
        )

        assert weather_mentioned, (
            f"{provider_model} should reference the weather information in final response. "
            f"Got: {response2.content[:200]}"
        )

    @pytest.mark.parametrize(
        "provider_model",
        [
            pytest.param("openai_fast", marks=pytest.mark.openai),
            pytest.param("anthropic_fast", marks=pytest.mark.anthropic),
            pytest.param("google_fast", marks=pytest.mark.google),
        ],
    )
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_conversation(self, provider_model: str, service):
        """
        Test handling multiple sequential tool calls in a conversation.

        This ensures providers correctly maintain conversation state
        through multiple tool interactions.
        """
        from llmring.schemas import LLMRequest

        # Create a second tool
        def get_time_tool() -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string", "description": "Timezone name"}
                        },
                        "required": ["timezone"],
                    },
                },
            }

        messages = [
            Message(role="user", content="What's the weather in Tokyo and what time is it there?")
        ]

        tools = [get_weather_tool(), get_time_tool()]

        request = LLMRequest(
            messages=messages,
            model=provider_model,
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
            temperature=0.1,
        )

        # First response - might call one or both tools
        response1 = await service.chat(request)

        assert response1.tool_calls is not None, f"{provider_model} should call tools"

        # Process all tool calls
        messages.append(
            Message(
                role="assistant", content=response1.content or "", tool_calls=response1.tool_calls
            )
        )

        # Add tool responses for each call
        for tool_call in response1.tool_calls:
            func_name = tool_call["function"]["name"]

            if func_name == "get_weather":
                result = {"temperature": "18°C", "condition": "cloudy", "location": "Tokyo"}
            elif func_name == "get_time":
                result = {"time": "15:30", "timezone": "Asia/Tokyo", "date": "2024-01-15"}
            else:
                result = {"error": f"Unknown tool: {func_name}"}

            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(result),
                    tool_call_id=tool_call.get("id", f"call_{func_name}"),
                )
            )

        # Continue conversation - check if model needs more tools or can answer
        max_iterations = 3  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            request = LLMRequest(
                messages=messages,
                model=provider_model,
                tools=tools,
                tool_choice="auto",
                max_tokens=150,
                temperature=0.1,
            )

            response = await service.chat(request)

            if not response.tool_calls:
                # Model provided final answer
                assert response.content is not None

                # Verify both pieces of information are mentioned
                response_lower = response.content.lower()
                weather_mentioned = "18" in response.content or "cloudy" in response_lower
                time_mentioned = "15:30" in response.content or "3:30" in response.content

                assert weather_mentioned or time_mentioned, (
                    f"{provider_model} should mention the requested information. "
                    f"Got: {response.content[:200]}"
                )
                break

            # Model called more tools - process them
            messages.append(
                Message(
                    role="assistant", content=response.content or "", tool_calls=response.tool_calls
                )
            )

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]

                # Provide appropriate response based on tool
                if func_name == "get_weather":
                    result = {"temperature": "18°C", "condition": "cloudy", "location": "Tokyo"}
                elif func_name == "get_time":
                    result = {"time": "15:30", "timezone": "Asia/Tokyo"}
                else:
                    result = {"error": f"Unknown tool: {func_name}"}

                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(result),
                        tool_call_id=tool_call.get("id", f"call_{iteration}_{func_name}"),
                    )
                )

        # Ensure we didn't hit max iterations (would indicate a loop)
        assert (
            iteration < max_iterations
        ), f"{provider_model} hit maximum iterations - possible tool calling loop"
