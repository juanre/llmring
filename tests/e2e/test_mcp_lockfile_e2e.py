#!/usr/bin/env python3
"""
End-to-end tests for MCP lockfile management.

Tests the complete flow from user input through chat app, server, tools, and back.
No mocks - real execution only.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llmring.lockfile_core import Lockfile
from llmring.mcp.client.chat.app import MCPChatApp
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools


async def simulate_tool_execution(
    app: MCPChatApp, tool_calls: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Simulate the tool execution flow that would happen in the chat app."""
    results = []

    for call in tool_calls:
        tool_name = call.get("name") or call.get("tool")
        arguments = call.get("arguments", {})

        # Find the tool
        if tool_name in app.available_tools:
            app.available_tools[tool_name]

            # Execute via MCP client
            try:
                result = await app.mcp_client.call_tool(name=tool_name, arguments=arguments)
                results.append({"tool": tool_name, "result": result})
            except Exception as e:
                results.append({"tool": tool_name, "error": str(e)})

    return results


@pytest.mark.asyncio
async def test_e2e_complete_workflow():
    """Test complete E2E workflow: setup aliases, get recommendations, analyze costs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "e2e.lock"

        # Create initial lockfile with advisor
        lockfile = Lockfile()
        lockfile.set_binding("advisor", "anthropic:claude-opus-4-1-20250805", profile="default")
        lockfile.save(lockfile_path)

        # Set environment
        os.environ["LLMRING_LOCKFILE_PATH"] = str(lockfile_path)

        try:
            # Initialize chat app without MCP server URL (direct tool access)
            app = MCPChatApp(mcp_server_url=None, llm_model="advisor", enable_telemetry=False)

            # Since we don't have a real MCP server running, we'll test tools directly
            tools = LockfileManagerTools(lockfile_path=lockfile_path)

            # Step 1: Add test aliases directly (no more recommend_alias)
            print("\n✅ Step 1: Adding test aliases...")

            test_aliases = [
                {"alias": "fast", "models": "openai:gpt-4o-mini"},
                {"alias": "deep", "models": "anthropic:claude-3-5-sonnet"},
            ]

            for alias_config in test_aliases:
                add_result = await tools.add_alias(
                    alias=alias_config["alias"], models=alias_config["models"]
                )
                assert add_result["success"]
                print(f"   Added {alias_config['alias']} -> {alias_config['models']}")

            # Step 2: List current configuration
            print("\n📋 Step 2: Listing current aliases...")

            list_result = await tools.list_aliases()
            aliases = list_result["aliases"]

            print(f"   Current aliases ({len(aliases)}):")
            for alias_info in aliases:
                print(f"   - {alias_info['alias']}: {alias_info['model']}")

            # Step 3: Analyze costs
            print("\n💰 Step 3: Analyzing monthly costs...")

            cost_result = await tools.analyze_costs(
                monthly_volume={"input_tokens": 5000000, "output_tokens": 2000000}
            )

            print(f"   Total monthly cost: ${cost_result['total_monthly_cost']:.2f}")
            print("   Breakdown by alias:")
            for alias, details in cost_result["cost_breakdown"].items():
                print(f"   - {alias}: ${details['total_cost']:.2f}")

            # Step 4: Save configuration
            print("\n💾 Step 4: Saving configuration...")

            save_result = await tools.save_lockfile()
            assert save_result["success"]
            print(f"   Saved to {lockfile_path}")

            # Step 5: Verify persistence
            print("\n🔍 Step 5: Verifying persistence...")

            loaded_lockfile = Lockfile.load(lockfile_path)
            for alias_info in aliases:
                resolved = loaded_lockfile.resolve_alias(alias_info["alias"])
                expected = f"{alias_info['provider']}:{alias_info['model']}"
                assert resolved == [
                    expected
                ], f"Alias {alias_info['alias']} not persisted correctly"

            print("   All aliases persisted correctly!")

            # Clean up
            if app:
                await app.cleanup()

            print("\n✅ E2E test completed successfully!")

        finally:
            if "LLMRING_LOCKFILE_PATH" in os.environ:
                del os.environ["LLMRING_LOCKFILE_PATH"]


@pytest.mark.asyncio
async def test_e2e_conversational_flow():
    """Test E2E conversational flow simulating natural user interaction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "conversation.lock"

        # Initialize tools directly (simulating server backend)
        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        print("\n🗣️ Simulating conversational flow...")

        # Conversation 1: User wants fast responses
        print("\n👤 User: I need a really fast model for quick responses")

        result = await tools.add_alias(alias="fast", models="openai:gpt-4o-mini")

        print(f"🤖 Assistant: I've added the 'fast' alias with {result['model']}")
        print(f"   Recommendation: {result.get('recommendation', 'Optimized for speed')}")

        # Conversation 2: User wants to know costs
        print("\n👤 User: How much will this cost me per month if I use it heavily?")

        cost_result = await tools.analyze_costs(
            monthly_volume={"input_tokens": 10000000, "output_tokens": 5000000}
        )

        print("🤖 Assistant: With heavy usage (15M tokens/month):")
        print(f"   Total cost: ${cost_result['total_monthly_cost']:.2f}")

        # Conversation 3: User wants coding model
        print("\n👤 User: I also need something for coding.")

        # Add a coding-optimized model directly
        print("🤖 Assistant: I'll add a coding-optimized model:")
        print("   coder: openai:gpt-4o")
        print("   This model has excellent code generation capabilities")

        # Add the alias
        await tools.add_alias(alias="coder", models="openai:gpt-4o")

        # Conversation 4: User wants to see everything
        print("\n👤 User: Show me all my aliases")

        list_result = await tools.list_aliases()

        print("🤖 Assistant: Here are your configured aliases:")
        for alias in list_result["aliases"]:
            print(f"   • {alias['alias']}: {alias['model']}")

        # Conversation 5: Save configuration
        print("\n👤 User: Save this configuration")

        await tools.save_lockfile()

        print(f"🤖 Assistant: Configuration saved to {lockfile_path.name}")
        print("   You can now use these aliases with 'llmring chat -m <alias>'")

        # Verify it actually saved
        assert lockfile_path.exists()
        loaded = Lockfile.load(lockfile_path)
        assert loaded.resolve_alias("fast") is not None

        print("\n✅ Conversational E2E test completed!")


@pytest.mark.asyncio
async def test_e2e_error_recovery():
    """Test E2E error handling and recovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "errors.lock"

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        print("\n🔧 Testing error handling and recovery...")

        # Error 1: Invalid model format
        print("\n❌ Attempting invalid model format...")

        result = await tools.add_alias(alias="broken", models="this:is:not:valid:format")

        if not result["success"]:
            print(f"   Handled gracefully: {result.get('message', 'Invalid format detected')}")
        else:
            # If it accepts it, try to use it
            assess_result = await tools.assess_model("broken")
            print(f"   Assessment: {assess_result}")

        # Error 2: Remove non-existent alias
        print("\n❌ Attempting to remove non-existent alias...")

        remove_result = await tools.remove_alias("does_not_exist")

        assert not remove_result["success"]
        print(f"   Handled gracefully: {remove_result.get('message', 'Alias not found')}")

        # Error 3: Add duplicate alias (should update)
        print("\n⚠️ Adding duplicate alias (should update)...")

        await tools.add_alias("test", "openai:gpt-4o-mini")
        result2 = await tools.add_alias("test", "anthropic:claude-3-haiku")

        assert result2["success"]
        print("   Successfully updated existing alias")

        # Verify it was updated
        list_result = await tools.list_aliases()
        test_alias = next((a for a in list_result["aliases"] if a["alias"] == "test"), None)
        assert test_alias["model"] == "claude-3-haiku"

        # Recovery: Continue working after errors
        print("\n✅ Continuing normal operations after errors...")

        final_result = await tools.add_alias(alias="working", models="openai:gpt-4o-mini")

        assert final_result["success"]
        print(f"   Successfully added 'working' alias: {final_result['model']}")

        print("\n✅ Error recovery E2E test completed!")


@pytest.mark.asyncio
async def test_e2e_multi_profile():
    """Test E2E multi-profile management."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "profiles.lock"

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        print("\n👥 Testing multi-profile management...")

        # Setup profiles
        profiles = {
            "default": {"fast": "openai:gpt-4o-mini", "deep": "anthropic:claude-3-5-sonnet"},
            "development": {"fast": "ollama:llama3", "deep": "anthropic:claude-3-haiku"},
            "production": {"fast": "openai:gpt-4o", "deep": "anthropic:claude-3-opus"},
        }

        # Add aliases to each profile
        for profile_name, aliases in profiles.items():
            print(f"\n📁 Setting up profile: {profile_name}")

            for alias, model in aliases.items():
                result = await tools.add_alias(alias=alias, models=model, profile=profile_name)
                assert result["success"]
                print(f"   Added {alias} -> {model}")

        # List each profile
        for profile_name in profiles.keys():
            print(f"\n📋 Profile '{profile_name}':")

            list_result = await tools.list_aliases(profile=profile_name)
            for alias in list_result["aliases"]:
                print(f"   • {alias['alias']}: {alias['model']}")

        # Analyze costs per profile
        print("\n💰 Cost analysis per profile:")

        volume = {"input_tokens": 1000000, "output_tokens": 500000}

        for profile_name in profiles.keys():
            cost_result = await tools.analyze_costs(profile=profile_name, monthly_volume=volume)

            print(f"   {profile_name}: ${cost_result['total_monthly_cost']:.2f}/month")

        # Save and verify
        save_result = await tools.save_lockfile()
        assert save_result["success"]

        # Load and verify each profile
        loaded = Lockfile.load(lockfile_path)

        for profile_name, aliases in profiles.items():
            for alias, expected_model in aliases.items():
                resolved = loaded.resolve_alias(alias, profile=profile_name)
                assert resolved == [expected_model]

        print("\n✅ Multi-profile E2E test completed!")


@pytest.mark.asyncio
async def test_e2e_model_assessment():
    """Test E2E model assessment and comparison."""
    tools = LockfileManagerTools()

    print("\n🔍 Testing model assessment and comparison...")

    # Models to assess
    models = [
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "anthropic:claude-3-5-sonnet",
        "anthropic:claude-3-haiku",
        "google:gemini-1.5-pro",
    ]

    assessments = {}

    print("\n📊 Assessing models:")
    for model_ref in models:
        result = await tools.assess_model(model_ref)

        if "error" not in result:
            assessments[model_ref] = result
            print(f"\n   {model_ref}:")
            print(f"     Active: {result.get('active', 'Unknown')}")
            print(f"     Capabilities: {', '.join(result.get('capabilities', []))[:50]}...")

    # Test is complete - recommendation functionality was removed
    print("\n✅ Model assessment E2E test completed!")


if __name__ == "__main__":
    # Run the E2E tests
    async def main():
        print("=" * 70)
        print("🚀 Running MCP Lockfile E2E Tests (No Mocks)")
        print("=" * 70)

        try:
            await test_e2e_complete_workflow()
            await test_e2e_conversational_flow()
            await test_e2e_error_recovery()
            await test_e2e_multi_profile()
            await test_e2e_model_assessment()

            print("\n" + "=" * 70)
            print("✅ All E2E tests passed!")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ E2E test failed: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(main())
