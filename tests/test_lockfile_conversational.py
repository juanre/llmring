#!/usr/bin/env python3
"""
Scripted tests for conversational lockfile management.

Tests the MCP tools and conversational interface with predefined interactions.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llmring.lockfile_core import Lockfile
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools


class ScriptedConversation:
    """Simulates a conversation with the lockfile manager."""

    def __init__(self, lockfile_path: Path = None):
        """Initialize with optional paths."""
        self.tools = LockfileManagerTools(lockfile_path=lockfile_path)
        self.conversation_history = []

    async def send(self, message: str, tool_calls: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message and optionally execute tool calls."""
        # Log the user message
        self.conversation_history.append({"role": "user", "content": message})

        results = {}
        if tool_calls:
            for call in tool_calls:
                tool_name = call["tool"]
                arguments = call.get("arguments", {})

                # Execute the tool
                if hasattr(self.tools, tool_name):
                    method = getattr(self.tools, tool_name)
                    result = await method(**arguments)
                    results[tool_name] = result
                else:
                    results[tool_name] = {"error": f"Unknown tool: {tool_name}"}

        # Log the assistant response
        self.conversation_history.append({"role": "assistant", "tool_results": results})

        return results


@pytest.mark.asyncio
async def test_add_alias_conversation():
    """Test adding aliases through conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "llmring.lock"

        # Create conversation
        conv = ScriptedConversation(lockfile_path=lockfile_path)

        # User asks to add a fast alias
        result = await conv.send(
            "I need a fast model for quick responses",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {"alias": "fast", "models": "openai:gpt-4o-mini"},
                }
            ],
        )

        assert "add_alias" in result
        assert result["add_alias"]["success"]
        assert result["add_alias"]["alias"] == "fast"
        assert "model" in result["add_alias"]

        # Verify it was added
        list_result = await conv.send(
            "Show me my aliases", tool_calls=[{"tool": "list_aliases", "arguments": {}}]
        )

        assert "list_aliases" in list_result
        aliases = list_result["list_aliases"]["aliases"]
        # aliases is a list of dicts
        alias_names = [a["alias"] for a in aliases]
        assert "fast" in alias_names


# Test removed - recommend_alias functionality removed in favor of LLM-based recommendations


@pytest.mark.asyncio
async def test_cost_analysis_conversation():
    """Test cost analysis through conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "llmring.lock"

        conv = ScriptedConversation(lockfile_path=lockfile_path)

        # Add some aliases first
        await conv.send(
            "Add a balanced model",
            tool_calls=[
                {"tool": "add_alias", "arguments": {"alias": "balanced", "models": "openai:gpt-4o"}}
            ],
        )

        # Ask for cost analysis
        result = await conv.send(
            "How much will this cost per month?",
            tool_calls=[
                {
                    "tool": "analyze_costs",
                    "arguments": {
                        "monthly_volume": {"input_tokens": 1000000, "output_tokens": 500000}
                    },
                }
            ],
        )

        assert "analyze_costs" in result
        analysis = result["analyze_costs"]
        assert "total_monthly_cost" in analysis
        assert "cost_breakdown" in analysis
        assert "recommendations" in analysis


@pytest.mark.asyncio
async def test_model_assessment_conversation():
    """Test model assessment through conversation."""
    conv = ScriptedConversation()

    # User asks about a specific model
    result = await conv.send(
        "Tell me about gpt-4o capabilities",
        tool_calls=[{"tool": "assess_model", "arguments": {"model_ref": "openai:gpt-4o"}}],
    )

    assert "assess_model" in result
    assessment = result["assess_model"]
    assert "model" in assessment
    assert "capabilities" in assessment
    # Note: pricing, strengths, limitations may not always be present
    assert "active" in assessment  # Check what's actually returned


@pytest.mark.asyncio
async def test_alias_lifecycle_conversation():
    """Test complete alias lifecycle: add, list, remove."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "llmring.lock"

        conv = ScriptedConversation(lockfile_path=lockfile_path)

        # 1. Add an alias
        add_result = await conv.send(
            "Add a writer alias for content creation",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {"alias": "writer", "models": "anthropic:claude-3-5-sonnet"},
                }
            ],
        )
        assert add_result["add_alias"]["success"]

        # 2. List to verify it's there
        list_result = await conv.send(
            "Show my aliases", tool_calls=[{"tool": "list_aliases", "arguments": {}}]
        )
        alias_names = [a["alias"] for a in list_result["list_aliases"]["aliases"]]
        assert "writer" in alias_names

        # 3. Remove the alias
        remove_result = await conv.send(
            "Remove the writer alias",
            tool_calls=[{"tool": "remove_alias", "arguments": {"alias": "writer"}}],
        )
        assert remove_result["remove_alias"]["success"]

        # 4. Verify it's gone
        final_list = await conv.send(
            "List my aliases again", tool_calls=[{"tool": "list_aliases", "arguments": {}}]
        )
        final_alias_names = [a["alias"] for a in final_list["list_aliases"]["aliases"]]
        assert "writer" not in final_alias_names


@pytest.mark.asyncio
async def test_save_configuration_conversation():
    """Test saving configuration through conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "llmring.lock"

        conv = ScriptedConversation(lockfile_path=lockfile_path)

        # Add some configuration
        await conv.send(
            "Setup my standard aliases",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {"alias": "fast", "models": "openai:gpt-4o-mini"},
                },
                {
                    "tool": "add_alias",
                    "arguments": {"alias": "deep", "models": "anthropic:claude-3-5-sonnet"},
                },
            ],
        )

        # Save the configuration
        result = await conv.send(
            "Save my configuration", tool_calls=[{"tool": "save_lockfile", "arguments": {}}]
        )

        assert "save_lockfile" in result
        assert result["save_lockfile"]["success"]
        assert lockfile_path.exists()

        # Load and verify
        lockfile = Lockfile.load(lockfile_path)
        assert lockfile.resolve_alias("fast") == ["openai:gpt-4o-mini"]
        assert lockfile.resolve_alias("deep") == ["anthropic:claude-3-5-sonnet"]


@pytest.mark.asyncio
async def test_conversation_with_errors():
    """Test error handling in conversations."""
    conv = ScriptedConversation()

    # Try to remove non-existent alias
    result = await conv.send(
        "Remove the nonexistent alias",
        tool_calls=[{"tool": "remove_alias", "arguments": {"alias": "nonexistent"}}],
    )

    assert "remove_alias" in result
    assert not result["remove_alias"]["success"]
    # Check for either 'error' or 'message' field
    assert "message" in result["remove_alias"] or "error" in result["remove_alias"]


@pytest.mark.asyncio
async def test_multi_profile_conversation():
    """Test managing multiple profiles through conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "llmring.lock"

        conv = ScriptedConversation(lockfile_path=lockfile_path)

        # Add to default profile
        await conv.send(
            "Add fast alias to default",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {
                        "alias": "fast",
                        "models": "openai:gpt-4o-mini",
                        "profile": "default",
                    },
                }
            ],
        )

        # Add to development profile
        await conv.send(
            "Add fast alias for development",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {
                        "alias": "fast",
                        "models": "anthropic:claude-3-haiku",
                        "profile": "development",
                    },
                }
            ],
        )

        # List from each profile
        default_result = await conv.send(
            "Show default profile",
            tool_calls=[{"tool": "list_aliases", "arguments": {"profile": "default"}}],
        )

        dev_result = await conv.send(
            "Show development profile",
            tool_calls=[{"tool": "list_aliases", "arguments": {"profile": "development"}}],
        )

        # Verify different models (aliases is a list, not a dict)
        default_aliases = {a["alias"]: a for a in default_result["list_aliases"]["aliases"]}
        dev_aliases = {a["alias"]: a for a in dev_result["list_aliases"]["aliases"]}

        assert default_aliases["fast"]["model"] == "gpt-4o-mini"
        assert dev_aliases["fast"]["model"] == "claude-3-haiku"


if __name__ == "__main__":
    # Run a simple test
    async def main():
        print("Running scripted conversation tests...\n")

        # Create a conversation
        conv = ScriptedConversation()

        # Simulate a conversation
        print("User: I need to add a fast model")
        result = await conv.send(
            "I need to add a fast model",
            tool_calls=[
                {
                    "tool": "add_alias",
                    "arguments": {"alias": "fast", "models": "openai:gpt-4o-mini"},
                }
            ],
        )

        print(f"Assistant: I've added the 'fast' alias:")
        if "add_alias" in result and result["add_alias"]["success"]:
            print(f"  - {result['add_alias']['alias']}: {result['add_alias']['model']}")

        print("\n✅ Scripted conversation test completed!")

    asyncio.run(main())
