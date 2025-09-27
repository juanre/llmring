#!/usr/bin/env python3
"""
End-to-end test for conversational lockfile management.
This test actually runs the tools and verifies real functionality.
No mocks - real registry access, real lockfile operations.
"""

import asyncio
import tempfile
from pathlib import Path

from llmring.lockfile_core import Lockfile
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools


async def test_real_lockfile_operations():
    """Test real lockfile operations without any mocks."""
    print("🧪 Running end-to-end lockfile test...")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        # Create lockfile manager
        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        print("\n1️⃣ Testing add_alias...")
        # Add a real alias
        result = await tools.add_alias(alias="test_fast", model="openai:gpt-4o-mini")
        assert result["success"], f"Failed to add alias: {result}"
        print(f"   ✓ Added alias: {result['alias']} → {result['model']}")

        print("\n2️⃣ Testing list_aliases...")
        # List aliases
        result = await tools.list_aliases()
        assert result["count"] > 0, "No aliases found"
        aliases = result["aliases"]
        alias_names = [a["alias"] for a in aliases]
        assert "test_fast" in alias_names, "Added alias not in list"
        print(f"   ✓ Found {result['count']} aliases: {alias_names}")

        print("\n3️⃣ Testing assess_model...")
        # Assess a model
        result = await tools.assess_model("openai:gpt-4o-mini")
        assert "model" in result, f"Model assessment failed: {result}"
        assert "capabilities" in result, "No capabilities in assessment"
        print(f"   ✓ Assessed {result['model']}")
        print(f"     Capabilities: {result['capabilities']}")

        # Skip recommend_alias test - functionality removed in favor of LLM-based recommendations

        print("\n4️⃣ Testing save_lockfile...")
        # Save the lockfile
        result = await tools.save_lockfile()
        assert result["success"], f"Failed to save: {result}"
        assert lockfile_path.exists(), "Lockfile not created"
        print(f"   ✓ Saved to {result['path']}")

        print("\n5️⃣ Testing load and verify...")
        # Load and verify the lockfile
        loaded = Lockfile.load(lockfile_path)
        assert loaded.resolve_alias("test_fast") == "openai:gpt-4o-mini"
        print(f"   ✓ Loaded lockfile, alias resolves correctly")

        print("\n6️⃣ Testing remove_alias...")
        # Remove the alias
        result = await tools.remove_alias("test_fast")
        assert result["success"], f"Failed to remove: {result}"
        print(f"   ✓ Removed alias: test_fast")

        # Verify it's gone
        result = await tools.list_aliases()
        alias_names = [a["alias"] for a in result["aliases"]]
        assert "test_fast" not in alias_names, "Alias still exists after removal"
        print(f"   ✓ Verified removal")

        print("\n7️⃣ Testing analyze_costs...")
        # Add an alias for cost analysis
        await tools.add_alias("cost_test", "openai:gpt-4o")
        result = await tools.analyze_costs(
            monthly_volume={"input_tokens": 1000000, "output_tokens": 500000}
        )
        assert "total_monthly_cost" in result, f"No cost analysis: {result}"
        assert "cost_breakdown" in result, "No cost breakdown"
        print(f"   ✓ Estimated monthly cost: ${result['total_monthly_cost']}")

        print("\n8️⃣ Testing get_current_configuration...")
        # Get full configuration
        result = await tools.get_current_configuration()
        assert "version" in result, "No version in config"
        assert "profiles" in result, "No profiles in config"
        assert "default" in result["profiles"], "No default profile"
        print(f"   ✓ Got configuration with {len(result['profiles'])} profiles")

        print("\n✅ All end-to-end tests passed!")
        return True


async def test_advisor_alias():
    """Test that the test lockfile aliases resolve correctly."""
    print("\n🧪 Testing test lockfile alias resolution...")
    print("=" * 50)

    # Load the test lockfile
    test_lockfile_path = Path(__file__).parent / "llmring.lock.json"
    lockfile = Lockfile.load(test_lockfile_path)

    # Check some standard aliases from test lockfile
    fast = lockfile.resolve_alias("fast")
    deep = lockfile.resolve_alias("deep")
    balanced = lockfile.resolve_alias("balanced")

    print(f"Fast alias: {fast}")
    print(f"Deep alias: {deep}")
    print(f"Balanced alias: {balanced}")

    assert fast == "openai:gpt-4o-mini", f"Fast should be gpt-4o-mini, got: {fast}"
    assert (
        deep == "anthropic:claude-3-5-haiku-20241022"
    ), f"Deep should be claude-3-5-haiku, got: {deep}"
    assert balanced == "openai:gpt-4o-mini", f"Balanced should be gpt-4o-mini, got: {balanced}"

    print("\n✅ All test lockfile aliases are properly configured!")

    return True


async def main():
    """Run all integration tests."""
    try:
        # Test real lockfile operations
        await test_real_lockfile_operations()

        # Test advisor alias
        await test_advisor_alias()

        print("\n🎉 All integration tests passed successfully!")
        print("The conversational lockfile system is working correctly.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
