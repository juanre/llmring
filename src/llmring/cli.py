"""Command-line interface for LLM service."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llmring import LLMRequest, LLMRing, Message
from llmring.cli_utils import (
    format_error,
    format_info,
    format_success,
    format_warning,
    load_lockfile_or_exit,
    print_aliases,
    print_packaging_guidance,
)
from llmring.constants import LOCKFILE_NAME, PROJECT_ROOT_INDICATORS
from llmring.lockfile_core import Lockfile
from llmring.registry import RegistryClient

# Alias sync removed per source-of-truth v3.8 - aliases are purely local

# Load environment variables from .env file
load_dotenv()


async def cmd_lock_init(args):
    """Initialize a new lockfile with basic defaults from registry."""
    # Find package directory if not explicitly specified
    if args.file:
        path = Path(args.file)
        package_dir = path.parent
    else:
        # Try to find package directory where lockfile should be placed
        package_dir = Lockfile.find_package_directory()
        if package_dir:
            path = package_dir / LOCKFILE_NAME
            print(f"Found package directory: {package_dir}")
        else:
            # Fall back to current directory
            path = Path(LOCKFILE_NAME)
            package_dir = Path.cwd()
            print(f"No package directory found")
            print(f"Creating lockfile in current directory: {path.resolve()}")

    if path.exists() and not args.force:
        print(format_error(f"{path} already exists. Use --force to overwrite"))
        return 1

    print("Creating lockfile with registry-based defaults...")
    print()

    # Try to create with registry data
    try:
        from llmring.registry import RegistryClient

        registry_client = RegistryClient()
        lockfile = await Lockfile.create_default_async(registry_client)
        print("✅ Created lockfile with registry data")
    except Exception as e:
        # Fallback to basic if registry unavailable
        print(f"⚠️  Could not fetch registry data: {e}")
        print("   Creating minimal lockfile")
        lockfile = Lockfile.create_default()

    lockfile.save(path)

    print(f"✅ Created lockfile: {path}")

    # Show default bindings
    default_profile = lockfile.get_profile("default")
    if default_profile.bindings:
        print("\nDefault aliases:")
        for binding in default_profile.bindings:
            print(f"  {binding.alias} → {binding.model_ref}")
    else:
        print("\nNo default aliases configured.")

    # Check if we're in a Python project and provide packaging guidance
    pyproject_path = package_dir / "pyproject.toml"
    if pyproject_path.exists():
        print("\n⚠️  To include this lockfile in your package distribution:")
        print("\nAdd to your pyproject.toml:")
        print("")
        print("  [tool.hatch.build]  # or similar for your build system")
        print("  include = [")
        print('      "src/yourpackage/**/*.py",  # your existing patterns')
        print('      "src/yourpackage/**/*.lock",  # add this line')
        print("  ]")
        print("")
        print("Or if using setuptools with setup.py, add to MANIFEST.in:")
        print("  include src/yourpackage/*.lock")

    print("\n💡 Use 'llmring lock chat' for conversational lockfile management")

    return 0


def cmd_bind(args):
    """Bind an alias to one or more models (with fallback support)."""
    # Load or create lockfile
    # Find the package directory where lockfile should be
    package_dir = Lockfile.find_package_directory()
    if package_dir:
        lockfile_path = package_dir / LOCKFILE_NAME
    else:
        # Fall back to current directory if no package found
        lockfile_path = Path(LOCKFILE_NAME)

    if lockfile_path.exists():
        lockfile = Lockfile.load(lockfile_path)
    else:
        print(f"No lockfile found at {lockfile_path}")
        print("Creating a new lockfile...")
        lockfile = Lockfile.create_default()

    # Parse model(s) - can be comma-separated for fallbacks
    models = args.model  # Already a string, can be comma-separated

    # Set binding
    lockfile.set_binding(args.alias, models, profile=args.profile)

    # Save
    lockfile.save(lockfile_path)

    profile_name = args.profile or lockfile.default_profile

    # Display what was bound
    if "," in models:
        model_list = [m.strip() for m in models.split(",")]
        print(f"✅ Bound '{args.alias}' → {model_list[0]} in profile '{profile_name}'")
        print(f"   Fallbacks: {', '.join(model_list[1:])}")
    else:
        print(f"✅ Bound '{args.alias}' → '{models}' in profile '{profile_name}'")

    return 0


def cmd_aliases(args):
    """List aliases from lockfile."""
    from .cli_utils import load_lockfile_or_exit, print_aliases

    # Load lockfile with consistent error handling
    lockfile_path, lockfile = load_lockfile_or_exit(require_exists=True)
    if not lockfile:
        return 1

    # Use the utility function to print aliases
    print_aliases(lockfile, args.profile)
    return 0


async def cmd_lock_validate(args):
    """Validate lockfile against registry."""
    # Find the package directory where lockfile should be
    package_dir = Lockfile.find_package_directory()
    if package_dir:
        lockfile_path = package_dir / LOCKFILE_NAME
    else:
        # Fall back to current directory if no package found
        lockfile_path = Path(LOCKFILE_NAME)

    if not lockfile_path.exists():
        print(f"Error: No llmring.lock found at {lockfile_path}")
        print("Run 'llmring lock init' to create one.")
        return 1

    lockfile = Lockfile.load(lockfile_path)
    registry = RegistryClient()

    print("Validating lockfile bindings...")

    valid = True
    for profile_name, profile in lockfile.profiles.items():
        if profile.bindings:
            print(f"\nProfile '{profile_name}':")
            for binding in profile.bindings:
                # Validate model exists in registry
                try:
                    is_valid = await registry.validate_model(binding.provider, binding.model)
                    status = "✅" if is_valid else "❌"
                    print(f"  {status} {binding.alias} → {binding.model_ref}")
                    if not is_valid:
                        valid = False
                except Exception as e:
                    print(f"  ⚠️  {binding.alias} → {binding.model_ref} (couldn't validate: {e})")

    if valid:
        print("\n✅ All bindings are valid")
        return 0
    else:
        print("\n❌ Some bindings are invalid")
        return 1


async def cmd_lock_bump_registry(args):
    """Update pinned registry versions to latest."""
    # Find the package directory where lockfile should be
    package_dir = Lockfile.find_package_directory()
    if package_dir:
        lockfile_path = package_dir / LOCKFILE_NAME
    else:
        # Fall back to current directory if no package found
        lockfile_path = Path(LOCKFILE_NAME)

    if not lockfile_path.exists():
        print(f"Error: No llmring.lock found at {lockfile_path}")
        print("Run 'llmring lock init' to create one.")
        return 1

    lockfile = Lockfile.load(lockfile_path)
    registry = RegistryClient()

    print("Updating registry versions...")

    for profile_name, profile in lockfile.profiles.items():
        # Get unique providers from bindings
        providers = set(b.provider for b in profile.bindings)

        for provider in providers:
            try:
                current_version = await registry.get_current_version(provider)
                old_version = profile.registry_versions.get(provider, 0)

                if current_version > old_version:
                    profile.registry_versions[provider] = current_version
                    print(f"  {provider}: v{old_version} → v{current_version}")
                else:
                    print(f"  {provider}: v{current_version} (unchanged)")

            except Exception as e:
                print(f"  {provider}: Failed to get version ({e})")

    # Save updated lockfile
    lockfile.save(lockfile_path)
    print(f"\n✅ Updated {lockfile_path}")

    return 0


def format_models_for_prompt(models_data: dict, providers_data: dict) -> str:
    """
    Format models from registry into a concise summary for system prompt.

    Note: Expects fields from LockfileManagerTools.list_models() output format
    including aliases: input_cost, output_cost, supports_functions, supports_vision.
    These are added by list_models() to simplify access to nested pricing fields.

    Args:
        models_data: Dict from LockfileManagerTools.list_models()
        providers_data: Dict from LockfileManagerTools.get_available_providers()

    Returns:
        Formatted string with model information
    """
    configured_providers = set(providers_data.get("configured", []))
    models = models_data.get("models", [])

    if not models:
        return "(No models available from registry)"

    # Group models by provider
    by_provider = {}
    for m in models:
        provider = m.get("provider", "unknown")
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(m)

    lines = []
    for provider in sorted(by_provider.keys()):
        has_key = provider in configured_providers
        status = "✓ CONFIGURED" if has_key else "✗ no API key"
        lines.append(f"\n{provider.upper()} ({status}):")

        # Show top 5 models per provider (most relevant)
        provider_models = by_provider[provider][:5]
        for m in provider_models:
            # Format: model_name | context | cost/1M in/out | capabilities
            ctx = (
                f"{m['context_window'] // 1000}K"
                if m.get("context_window") and m["context_window"] > 0
                else "?"
            )
            cost_in = f"${m['input_cost']:.2f}" if m.get("input_cost") else "?"
            cost_out = f"${m['output_cost']:.2f}" if m.get("output_cost") else "?"
            caps = []
            if m.get("supports_vision"):
                caps.append("vision")
            if m.get("supports_functions"):
                caps.append("tools")
            caps_str = f" [{','.join(caps)}]" if caps else ""

            lines.append(
                f"  • {m['model_name']:<25} {ctx:>5} ctx | "
                f"{cost_in:>6}/{cost_out:>6} per 1M{caps_str}"
            )

        if len(by_provider[provider]) > 5:
            remaining = len(by_provider[provider]) - 5
            lines.append(f"  ... and {remaining} more models")

    return "\n".join(lines)


async def cmd_lock_chat(args):
    """Conversational lockfile management using MCP chat interface."""
    import subprocess
    import tempfile
    from pathlib import Path

    # Import MCP chat app
    from llmring.mcp.client.chat.app import MCPChatApp

    print("🤖 LLMRing Conversational Lockfile Manager")
    print("=" * 50)

    # Find the user's package lockfile path (inside their package for distribution)
    package_dir = Lockfile.find_package_directory()
    if package_dir:
        user_lockfile_path = package_dir / LOCKFILE_NAME
        print(f"📦 Package directory: {package_dir}")
    else:
        # Fall back to current directory if no package found
        user_lockfile_path = Path.cwd() / LOCKFILE_NAME
        print(f"📁 No package found, using current directory")
    print(f"📄 Managing lockfile: {user_lockfile_path}")

    # If no server URL provided, we'll use embedded server
    if not args.server_url:
        # The stdio transport will be handled by the chat app directly
        # Pass the lockfile path as an argument to the server
        # This will override any LLMRING_LOCKFILE_PATH environment variable
        server_url = (
            f"stdio://python -m llmring.mcp.server.lockfile_server --lockfile {user_lockfile_path}"
        )
        server_process = None  # stdio client will manage the process
        print("Will use embedded lockfile MCP server via stdio")
    else:
        server_url = args.server_url
        server_process = None

    # Set the bundled lockfile path AFTER configuring the server URL
    # This ensures the 'advisor' alias works but doesn't affect the server
    os.environ["LLMRING_LOCKFILE_PATH"] = str(Lockfile.get_package_lockfile_path())

    # Fetch registry models for system prompt prepopulation
    from llmring.mcp.tools.lockfile_manager import LockfileManagerTools

    print("📊 Loading available models from registry...")
    try:
        # Create tools instance to fetch models
        tools_instance = LockfileManagerTools(lockfile_path=user_lockfile_path)

        # Fetch models and providers in parallel for faster startup
        models_data, providers_data = await asyncio.gather(
            tools_instance.list_models(), tools_instance.get_available_providers()
        )

        # Format for system prompt
        models_summary = format_models_for_prompt(models_data, providers_data)
        configured_count = len(providers_data.get("configured", []))
        total_count = models_data.get("total_count", 0)
        print(f"✅ Loaded {total_count} models from {configured_count} configured provider(s)\n")
    except Exception as e:
        import logging

        logging.warning(f"Could not fetch models for system prompt: {e}")
        print("⚠️  Could not fetch models from registry")
        models_summary = "(Registry unavailable - use list_models tool to fetch current models)"

    # System prompt that explains multi-model aliases and profiles clearly
    system_prompt = f"""You are the LLMRing Lockfile Manager assistant. You help users manage their LLM aliases and model configurations.

AVAILABLE MODELS FROM REGISTRY (snapshot at startup):
{models_summary}

The models listed above are currently available from the registry. Use this information to make informed recommendations.
If a provider shows "✗ no API key", models from that provider won't work until the user configures their API key.

NOTE: This is a snapshot taken at startup. You can call the list_models tool to fetch fresh or complete model information if needed.

IMPORTANT: Understanding Model Pools in LLMRing
================================================
LLMRing aliases can be bound to a POOL OF MODELS - multiple prioritized alternatives that ensure your code always works regardless of which API keys you have configured.

IMPORTANT: Understanding Profiles
=================================
LLMRing supports PROFILES for environment-specific configurations (dev, staging, prod, test, etc.).
The same alias can have different models in different profiles:
- 'default' profile: Standard configuration
- 'dev' profile: Cheaper/faster models for development
- 'prod' profile: High-quality models for production
- 'test' profile: Minimal models for automated testing

Users can specify profiles when adding aliases: add_alias(alias="fast", models="...", profile="dev")

How Model Pools Work:
1. You provide multiple models in priority order: "anthropic:claude-3-haiku,openai:gpt-4o-mini"
2. LLMRing intelligently selects the first available model based on configured API keys
3. This ensures seamless operation across different environments and teams

Why Use Model Pools?
- **Flexibility**: Same code works whether you have OpenAI, Anthropic, or both API keys
- **Resilience**: Automatic provider switching if one service is unavailable
- **Collaboration**: Team members can use different providers without code changes
- **Cost Optimization**: Prioritize cheaper models while maintaining alternatives

Example Model Pool Configurations:
- "fast": "anthropic:claude-3-haiku,openai:gpt-4o-mini" → Budget-friendly, fast responses
- "advisor": "anthropic:claude-3-5-sonnet,openai:gpt-4o" → High-quality reasoning
- "coder": "anthropic:claude-3-5-sonnet,openai:gpt-4o,google:gemini-pro" → Maximum availability

Example Profile Usage:
- add_alias(alias="fast", models="openai:gpt-3.5-turbo", profile="test") → Test profile
- add_alias(alias="fast", models="openai:gpt-4o-mini", profile="dev") → Dev profile
- add_alias(alias="fast", models="anthropic:claude-3-5-sonnet", profile="prod") → Prod profile
- list_aliases(profile="dev") → Show dev profile aliases

Smart Selection Examples:
- Only OpenAI key available: Uses openai models from the pool
- Both keys available: Uses first model in priority order
- Only Anthropic key available: Uses anthropic models from the pool

When helping users:
- Remember that the 'models' parameter (accepts single or multiple models)
- DO suggest model pools for better flexibility and resilience
- DO help users understand priority ordering in their model pools
- DON'T suggest single models when users want flexibility
- DO NOT

You have access to tools that help manage the lockfile. Use them to provide accurate information and help users configure their model pools effectively.

Your task is to guide users in creating or updating their lockfile, where they define their profiles and their alias pools.

VERY IMPORTANT: you need to understand the use they want to give to the models and suggest the best option, but DO NOT add or delete anything until you are told to do so.
    """

    try:
        # Create and run MCP chat app with system prompt
        app = MCPChatApp(
            mcp_server_url=server_url, llm_model=args.model, system_prompt=system_prompt
        )

        # Minimal initialization
        await app.initialize_async()

        # Run the chat interface
        await app.run()

    finally:
        # Clean up server process if we started it
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("\n✅ Stopped lockfile MCP server")

    return 0


async def cmd_list_models(args):
    """List available models."""
    async with LLMRing() as ring:
        models = await ring.get_available_models()

        if args.provider:
            # Filter by provider
            models = {k: v for k, v in models.items() if k == args.provider}

        print(format_model_table(models, show_all=True))


async def cmd_chat(args):
    """Send a chat message to an LLM."""
    # Check if we should use an alias
    if ":" not in args.model:
        # Try to resolve as alias from package lockfile
        package_dir = Lockfile.find_package_directory()
        if package_dir:
            lockfile_path = package_dir / LOCKFILE_NAME
        else:
            # Fall back to current directory if no package found
            lockfile_path = Path(LOCKFILE_NAME)

        if lockfile_path.exists():
            lockfile = Lockfile.load(lockfile_path)

            # Get profile from environment or use default
            profile_name = os.environ.get("LLMRING_PROFILE", args.profile)

            # Resolve alias
            model_ref = lockfile.resolve_alias(args.model, profile_name)
            if model_ref:
                print(f"[Using alias '{args.model}' → '{model_ref}']")
                args.model = model_ref

    async with LLMRing() as ring:
        # Create message
        messages = [Message(role="user", content=args.message)]
        if args.system:
            messages.insert(0, Message(role="system", content=args.system))

        # Create request
        request = LLMRequest(
            messages=messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=args.stream if hasattr(args, "stream") else False,
        )

        try:
            # Send request
            response = await ring.chat(request)

            # Handle streaming response
            if args.stream if hasattr(args, "stream") else False:
                # Stream response chunks
                import sys

                full_content = ""
                accumulated_usage = None

                async for chunk in response:
                    if chunk.delta:
                        if not args.json:
                            # Print chunks as they arrive
                            sys.stdout.write(chunk.delta)
                            sys.stdout.flush()
                        full_content += chunk.delta

                    # Capture final usage stats
                    if chunk.usage:
                        accumulated_usage = chunk.usage

                if args.json:
                    # For JSON output, collect all chunks first
                    print(
                        json.dumps(
                            {
                                "content": full_content,
                                "model": (chunk.model if chunk and chunk.model else args.model),
                                "usage": accumulated_usage,
                                "finish_reason": chunk.finish_reason if chunk else None,
                            },
                            indent=2,
                        )
                    )
                else:
                    # Print newline after streaming
                    print()

                    if args.verbose and accumulated_usage:
                        print(f"\n[Model: {chunk.model if chunk and chunk.model else args.model}]")
                        print(
                            f"[Tokens: {accumulated_usage.get('prompt_tokens', 0)} in, {accumulated_usage.get('completion_tokens', 0)} out]"
                        )
                        if "cost" in accumulated_usage:
                            print(f"[Cost: ${accumulated_usage['cost']:.6f}]")
            else:
                # Non-streaming response (existing code)
                # Display response
                if args.json:
                    print(
                        json.dumps(
                            {
                                "content": response.content,
                                "model": response.model,
                                "usage": response.usage,
                                "finish_reason": response.finish_reason,
                            },
                            indent=2,
                        )
                    )
                else:
                    print(response.content)

                    if args.verbose and response.usage:
                        print(f"\n[Model: {response.model}]")
                        print(
                            f"[Tokens: {response.usage.get('prompt_tokens', 0)} in, {response.usage.get('completion_tokens', 0)} out]"
                        )
                        if "cost" in response.usage:
                            print(f"[Cost: ${response.usage['cost']:.6f}]")

        except Exception as e:
            print(f"Error: {e}")
            return 1

    return 0


async def cmd_info(args):
    """Show information about a specific model."""
    async with LLMRing() as ring:
        try:
            # Get enhanced info including registry data
            info = await ring.get_enhanced_model_info(args.model)

            if args.json:
                print(json.dumps(info, indent=2, default=str))
            else:
                print(f"Model: {info['model']}")
                print(f"Provider: {info['provider']}")
                print(f"Supported: {info['supported']}")

                # Show additional info if available
                if "display_name" in info:
                    print(f"Display Name: {info['display_name']}")
                if "description" in info:
                    print(f"Description: {info['description']}")
                if "max_input_tokens" in info:
                    print(f"Max Input: {info['max_input_tokens']:,} tokens")
                if "max_output_tokens" in info:
                    print(f"Max Output: {info['max_output_tokens']:,} tokens")
                if "dollars_per_million_tokens_input" in info:
                    print(f"Input Cost: ${info['dollars_per_million_tokens_input']:.2f}/M tokens")
                if "dollars_per_million_tokens_output" in info:
                    print(f"Output Cost: ${info['dollars_per_million_tokens_output']:.2f}/M tokens")
                if "supports_vision" in info and info["supports_vision"]:
                    print("Supports: Vision")
                if "supports_function_calling" in info and info["supports_function_calling"]:
                    print("Supports: Function Calling")
                if "supports_json_mode" in info and info["supports_json_mode"]:
                    print("Supports: JSON Mode")
                if "is_default" in info:
                    print(f"Default: {info['is_default']}")

        except Exception as e:
            print(f"Error: {e}")
            return 1

        return 0


# Push/pull commands removed per source-of-truth v3.8
# Aliases are managed entirely locally in each codebase's lockfile


async def cmd_stats(args):
    """Show usage statistics from server."""
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required for stats.")
        print("\nUsage statistics require connection to llmring-server.")
        print("Set LLMRING_API_KEY to your API key and optionally LLMRING_SERVER_URL.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Fetch receipts from server
        result = await client.list_receipts(limit=1000)
        receipts = result.get("receipts", [])
        total = result.get("total", 0)

        if not receipts:
            print("No usage statistics available.")
            print("\nGenerate receipts with: llmring receipts generate --since-last")
            return 0

        print(f"Usage statistics ({total} receipts):")
        print("-" * 40)

        total_cost = sum(r.get("total_cost", 0) for r in receipts)
        total_tokens = sum(r.get("total_tokens", 0) for r in receipts)

        print(f"Total receipts: {total}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.6f}")

        if args.verbose and receipts:
            print("\nRecent receipts:")
            for r in receipts[:10]:
                timestamp = r.get("timestamp", "unknown")
                alias = r.get("alias", "unknown")
                provider = r.get("provider", "unknown")
                model = r.get("model", "unknown")
                cost = r.get("total_cost", 0)
                print(f"  {timestamp}: {alias} → {provider}:{model} (${cost:.6f})")

        return 0

    except Exception as e:
        print(f"❌ Error fetching statistics: {e}")
        print("\nEnsure llmring-server is running and LLMRING_API_KEY is valid.")
        return 1


async def cmd_export(args):
    """Export receipts from server to JSON file."""
    import json
    import os
    from datetime import UTC, datetime

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required for export.")
        print("\nReceipt export requires connection to llmring-server.")
        print("Set LLMRING_API_KEY to your API key and optionally LLMRING_SERVER_URL.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Fetch all receipts from server
        result = await client.list_receipts(limit=10000)
        receipts = result.get("receipts", [])

        if not receipts:
            print("No receipts to export.")
            print("\nGenerate receipts with: llmring receipts generate --since-last")
            return 0

        # Export receipts as JSON
        export_data = {
            "exported_at": datetime.now(UTC).isoformat(),
            "server_url": server_url,
            "total_receipts": len(receipts),
            "receipts": receipts,
        }

        output_file = args.output or "llmring_receipts.json"
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Exported {len(receipts)} receipts to {output_file}")
        return 0

    except Exception as e:
        print(f"❌ Error exporting receipts: {e}")
        print("\nEnsure llmring-server is running and LLMRING_API_KEY is valid.")
        return 1


async def cmd_cache_clear(args):
    """Clear the registry cache."""
    from llmring.registry import RegistryClient

    registry = RegistryClient()
    registry.clear_cache()
    print("✅ Registry cache cleared successfully")
    print("Next model lookups will fetch fresh data from the registry")
    return 0


async def cmd_cache_info(args):
    """Show cache information."""
    from datetime import datetime, timezone
    from pathlib import Path

    from llmring.registry import RegistryClient

    registry = RegistryClient()
    cache_dir = registry.cache_dir

    print(f"📁 Cache directory: {cache_dir}")
    print(f"⏱️  Cache TTL: {registry.CACHE_DURATION_HOURS} hours")

    # List cache files and their ages
    cache_files = list(cache_dir.glob("*.json"))
    if cache_files:
        print(f"\n📄 Cached files ({len(cache_files)} total):")
        for cache_file in sorted(cache_files):
            # Get file age
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600

            # Check if still valid
            is_valid = age_hours < registry.CACHE_DURATION_HOURS
            status = "✅ valid" if is_valid else "❌ stale"

            print(f"  • {cache_file.name}: {age_hours:.1f}h old ({status})")
    else:
        print("\n📄 No cached files")

    # Show cache size
    total_size = sum(f.stat().st_size for f in cache_files) if cache_files else 0
    print(f"\n💾 Total cache size: {total_size / 1024:.1f} KB")

    return 0


async def cmd_register(args):
    """Register with LLMRing server (placeholder)."""
    print("⚠️  The 'register' command requires a server connection.")
    print("This feature is not yet available in the local-only version.")
    print("\nLLMRing SaaS features coming soon:")
    print("  • Central binding management")
    print("  • Usage analytics and cost tracking")
    print("  • Team collaboration")
    print("  • Signed receipts for compliance")
    return 0


# ============================================================================
# Receipts CLI Commands (Phase 7.5)
# ============================================================================


async def cmd_receipts_generate(args):
    """Generate a receipt on-demand."""
    import os
    from uuid import UUID

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required.")
        print("Set LLMRING_API_KEY to generate receipts.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Build request based on arguments
        request_data = {}

        if args.conversation:
            request_data["conversation_id"] = args.conversation
        elif args.start and args.end:
            request_data["start_date"] = args.start
            request_data["end_date"] = args.end
        elif args.log_ids:
            request_data["log_ids"] = [UUID(lid) for lid in args.log_ids]
        elif args.since_last:
            request_data["since_last_receipt"] = True
        else:
            print(
                "❌ Error: Must specify one of: --conversation, --start/--end, --log-ids, --since-last"
            )
            return 1

        if args.description:
            request_data["description"] = args.description
        if args.tags:
            request_data["tags"] = args.tags

        # Generate receipt
        result = await client.generate_receipt(**request_data)

        receipt = result.get("receipt", {})
        certified_count = result.get("certified_count", 0)

        print(f"✅ Receipt generated successfully!")
        print(f"Receipt ID: {receipt.get('receipt_id')}")
        print(f"Certified logs: {certified_count}")
        print(f"Total cost: ${receipt.get('total_cost', 0):.6f}")

        if receipt.get("receipt_type") == "batch":
            summary = receipt.get("batch_summary", {})
            print(f"\nBatch Summary:")
            print(f"  Total calls: {summary.get('total_calls', 0)}")
            print(f"  Total tokens: {summary.get('total_tokens', 0):,}")
            print(f"  Date range: {summary.get('start_date')} to {summary.get('end_date')}")

        return 0

    except Exception as e:
        print(f"❌ Error generating receipt: {e}")
        return 1


async def cmd_receipts_list(args):
    """List all receipts."""
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        result = await client.list_receipts(limit=args.limit, offset=args.offset)
        receipts = result.get("receipts", [])
        total = result.get("total", 0)

        if not receipts:
            print("No receipts found.")
            print("\nGenerate receipts with: llmring receipts generate --since-last")
            return 0

        print(f"Receipts ({len(receipts)} of {total} total):")
        print("-" * 80)

        for r in receipts:
            receipt_id = r.get("receipt_id", "unknown")
            timestamp = r.get("timestamp", "unknown")
            receipt_type = r.get("receipt_type", "single")
            total_cost = r.get("total_cost", 0)

            print(f"\n{receipt_id}")
            print(f"  Type: {receipt_type}")
            print(f"  Date: {timestamp}")
            print(f"  Cost: ${total_cost:.6f}")

            if receipt_type == "batch":
                summary = r.get("batch_summary", {})
                print(f"  Calls: {summary.get('total_calls', 0)}")

        return 0

    except Exception as e:
        print(f"❌ Error listing receipts: {e}")
        return 1


async def cmd_receipts_get(args):
    """Get a specific receipt by ID."""
    import json
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        receipt = await client.get_receipt(args.receipt_id)

        if args.json:
            print(json.dumps(receipt, indent=2))
        else:
            print(f"Receipt: {receipt.get('receipt_id')}")
            print(f"Type: {receipt.get('receipt_type', 'single')}")
            print(f"Timestamp: {receipt.get('timestamp')}")
            print(f"Provider: {receipt.get('provider')}")
            print(f"Model: {receipt.get('model')}")
            print(f"Total cost: ${receipt.get('total_cost', 0):.6f}")
            print(f"Signature: {receipt.get('signature')}")

        return 0

    except Exception as e:
        print(f"❌ Error fetching receipt: {e}")
        return 1


async def cmd_receipts_preview(args):
    """Preview what a receipt would certify without generating it."""
    import json
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Build request based on arguments
        request_data = {}

        if args.start and args.end:
            request_data["start_date"] = args.start
            request_data["end_date"] = args.end
        elif args.since_last:
            request_data["since_last_receipt"] = True
        else:
            print("❌ Error: Must specify --start/--end or --since-last")
            return 1

        # Preview receipt
        preview = await client.preview_receipt(**request_data)

        print("📋 Receipt Preview:")
        print(f"  Total logs: {preview.get('total_logs', 0)}")
        print(f"  Total cost: ${preview.get('total_cost', 0):.6f}")
        print(f"  Total tokens: {preview.get('total_tokens', 0):,}")
        print(f"  Date range: {preview.get('start_date')} to {preview.get('end_date')}")
        print(f"  Receipt type: {preview.get('receipt_type', 'unknown')}")

        if args.verbose:
            print("\nBreakdown by model:")
            for model, stats in preview.get("by_model", {}).items():
                print(f"  {model}: {stats.get('calls')} calls, ${stats.get('cost', 0):.6f}")

        return 0

    except Exception as e:
        print(f"❌ Error previewing receipt: {e}")
        return 1


async def cmd_receipts_uncertified(args):
    """List logs that haven't been certified by any receipt."""
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    if not api_key:
        print("❌ Error: LLMRING_API_KEY environment variable required.")
        return 1

    try:
        client = ServerClient(server_url=server_url, api_key=api_key)

        result = await client.get_uncertified_logs(limit=args.limit, offset=args.offset)
        logs = result.get("logs", [])
        total = result.get("total", 0)

        if not logs:
            print("✅ All logs have been certified!")
            return 0

        print(f"Uncertified logs ({len(logs)} of {total} total):")
        print("-" * 80)

        for log in logs:
            log_id = log.get("id", "unknown")
            log_type = log.get("type", "unknown")
            timestamp = log.get("timestamp", "unknown")
            cost = log.get("cost", 0)

            print(f"{log_id} ({log_type}) - {timestamp} - ${cost:.6f}")

        print(f"\nGenerate receipt with: llmring receipts generate --since-last")

        return 0

    except Exception as e:
        print(f"❌ Error fetching uncertified logs: {e}")
        return 1


async def cmd_receipts_verify(args):
    """Verify a receipt from a JSON file."""
    import json
    import os

    from llmring.server_client import ServerClient

    server_url = os.getenv("LLMRING_SERVER_URL", "http://localhost:8000")
    api_key = os.getenv("LLMRING_API_KEY")

    try:
        # Load receipt from file
        with open(args.receipt_file, "r") as f:
            receipt_data = json.load(f)

        # If the file contains an export with multiple receipts, verify the first one
        if "receipts" in receipt_data and isinstance(receipt_data["receipts"], list):
            if not receipt_data["receipts"]:
                print("❌ No receipts found in file.")
                return 1
            receipt_data = receipt_data["receipts"][0]

        # Verify via server endpoint (public, no auth required)
        client = ServerClient(server_url=server_url, api_key=api_key)

        # The verify endpoint is public, so we can call it without auth
        result = await client.post("/api/v1/receipts/verify", json=receipt_data)

        is_valid = result.get("valid", False)
        receipt_id = result.get("receipt_id", "unknown")

        if is_valid:
            print(f"✅ Receipt {receipt_id} is VALID")
            print(f"   Algorithm: {result.get('algorithm', 'unknown')}")
            return 0
        else:
            print(f"❌ Receipt {receipt_id} is INVALID")
            return 1

    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.receipt_file}")
        return 1
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in file: {args.receipt_file}")
        return 1
    except Exception as e:
        print(f"❌ Error verifying receipt: {e}")
        return 1


def cmd_providers(args):
    """List configured providers."""
    ring = LLMRing()

    providers = []
    for provider_name in ["openai", "anthropic", "google", "ollama"]:
        try:
            provider = ring.get_provider(provider_name)
            has_key = provider is not None
        except Exception:
            has_key = False

        providers.append(
            {
                "provider": provider_name,
                "configured": has_key,
                "api_key_env": {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_API_KEY or GEMINI_API_KEY",
                    "ollama": "(not required)",
                }.get(provider_name, ""),
            }
        )

    if args.json:
        print(json.dumps(providers, indent=2))
    else:
        print("Configured Providers:")
        print("-" * 40)
        for p in providers:
            status = "✓" if p["configured"] else "✗"
            print(f"{status} {p['provider']:<12} {p['api_key_env']}")


def format_model_table(models: dict, show_all: bool = False):
    """Format models as a readable table."""
    if not models:
        return "No models found."

    lines = []
    lines.append("Available Models:")
    lines.append("-" * 40)

    for provider, model_list in models.items():
        if model_list or show_all:
            lines.append(f"\n{provider.upper()}:")
            if model_list:
                for model in model_list:
                    lines.append(f"  - {model}")
            else:
                lines.append("  (No models available)")

    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLMRing - Unified LLM Service CLI with Profile Support\n\nProfiles allow environment-specific configurations (dev, prod, test).\nUse --profile flag or set LLMRING_PROFILE environment variable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Lock commands
    lock_parser = subparsers.add_parser("lock", help="Lockfile management")
    lock_subparsers = lock_parser.add_subparsers(dest="lock_command", help="Lock commands")

    # lock init
    init_parser = lock_subparsers.add_parser("init", help="Initialize lockfile with basic defaults")
    init_parser.add_argument("--file", help="Lockfile path (default: llmring.lock)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing file")

    # lock validate
    lock_subparsers.add_parser("validate", help="Validate lockfile against registry")

    # lock bump-registry
    lock_subparsers.add_parser("bump-registry", help="Update registry versions")

    # lock chat - conversational lockfile management
    chat_parser = lock_subparsers.add_parser(
        "chat", help="Conversational lockfile management with natural language"
    )
    chat_parser.add_argument(
        "--server-url", help="URL of lockfile MCP server (default: starts embedded server)"
    )
    chat_parser.add_argument(
        "--model",
        default="advisor",
        help="LLM model to use for conversation (default: advisor for intelligent recommendations)",
    )

    # Bind command
    bind_parser = subparsers.add_parser(
        "bind", help="Bind an alias to model(s) with fallback support"
    )
    bind_parser.add_argument("alias", help="Alias name")
    bind_parser.add_argument(
        "model",
        help="Model reference(s) - single or comma-separated for fallbacks (e.g., 'anthropic:claude-3-opus,openai:gpt-4')",
    )
    bind_parser.add_argument("--profile", help="Profile to use (default: default)")

    # Aliases command
    aliases_parser = subparsers.add_parser("aliases", help="List aliases from lockfile")
    aliases_parser.add_argument("--profile", help="Profile to use")

    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--provider", help="Filter by provider (openai, anthropic, google, ollama)"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a chat message")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument(
        "--model",
        default="fast",
        help="Model alias (fast, balanced, deep) or provider:model",
    )
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0.0-2.0)")
    chat_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    chat_parser.add_argument("--json", action="store_true", help="Output as JSON")
    chat_parser.add_argument("--verbose", action="store_true", help="Show additional information")
    chat_parser.add_argument("--profile", help="Profile to use for alias resolution")
    chat_parser.add_argument("--stream", action="store_true", help="Stream response in real-time")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "model",
        help="Model alias (fast, balanced, deep) or provider:model (e.g., openai:gpt-4)",
    )
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Providers command
    providers_parser = subparsers.add_parser("providers", help="List configured providers")
    providers_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Push/pull commands removed per source-of-truth v3.8
    # Aliases are managed entirely locally in each codebase's lockfile

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show usage statistics")
    stats_parser.add_argument("--verbose", action="store_true", help="Show detailed statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export receipts to file")
    export_parser.add_argument("--output", help="Output file (default: llmring_receipts.json)")
    export_parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Export format"
    )

    # Cache management commands
    cache_parser = subparsers.add_parser("cache", help="Registry cache management")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache commands")
    cache_subparsers.add_parser("clear", help="Clear the registry cache")
    cache_info_parser = cache_subparsers.add_parser("info", help="Show cache information")

    # Receipts management commands (Phase 7.5)
    receipts_parser = subparsers.add_parser(
        "receipts", help="On-demand receipt generation and management"
    )
    receipts_subparsers = receipts_parser.add_subparsers(
        dest="receipts_command", help="Receipts commands"
    )

    # receipts generate
    generate_parser = receipts_subparsers.add_parser(
        "generate", help="Generate a receipt on-demand"
    )
    generate_parser.add_argument(
        "--conversation", help="Generate receipt for a specific conversation ID"
    )
    generate_parser.add_argument("--start", help="Start date for batch receipt (YYYY-MM-DD)")
    generate_parser.add_argument("--end", help="End date for batch receipt (YYYY-MM-DD)")
    generate_parser.add_argument("--log-ids", nargs="+", help="List of log IDs to certify")
    generate_parser.add_argument(
        "--since-last", action="store_true", help="Certify all logs since last receipt"
    )
    generate_parser.add_argument("--description", help="User-provided description for the receipt")
    generate_parser.add_argument("--tags", nargs="+", help="Tags for categorization")

    # receipts list
    list_receipts_parser = receipts_subparsers.add_parser("list", help="List all receipts")
    list_receipts_parser.add_argument(
        "--limit", type=int, default=100, help="Number of receipts to fetch"
    )
    list_receipts_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    # receipts get
    get_receipt_parser = receipts_subparsers.add_parser("get", help="Get a specific receipt by ID")
    get_receipt_parser.add_argument("receipt_id", help="Receipt ID to fetch")
    get_receipt_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # receipts preview
    preview_parser = receipts_subparsers.add_parser(
        "preview", help="Preview what a receipt would certify"
    )
    preview_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    preview_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    preview_parser.add_argument(
        "--since-last", action="store_true", help="Preview logs since last receipt"
    )
    preview_parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")

    # receipts uncertified
    uncertified_parser = receipts_subparsers.add_parser(
        "uncertified", help="List logs without receipts"
    )
    uncertified_parser.add_argument(
        "--limit", type=int, default=100, help="Number of logs to fetch"
    )
    uncertified_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    # receipts verify
    verify_parser = receipts_subparsers.add_parser("verify", help="Verify a receipt from JSON file")
    verify_parser.add_argument("receipt_file", help="Path to receipt JSON file")

    # Register command
    register_parser = subparsers.add_parser(
        "register", help="Register with LLMRing server (for SaaS features)"
    )
    register_parser.add_argument("--email", help="Email address for registration")
    register_parser.add_argument("--org", help="Organization name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Handle lock subcommands
    if args.command == "lock":
        if not args.lock_command:
            lock_parser.print_help()
            return 1

        lock_commands = {
            "init": cmd_lock_init,
            "validate": cmd_lock_validate,
            "bump-registry": cmd_lock_bump_registry,
            "chat": cmd_lock_chat,
        }

        if args.lock_command in lock_commands:
            return asyncio.run(lock_commands[args.lock_command](args))

    # Handle cache subcommands
    if args.command == "cache":
        if not args.cache_command:
            cache_parser.print_help()
            return 1

        cache_commands = {
            "clear": cmd_cache_clear,
            "info": cmd_cache_info,
        }

        if args.cache_command in cache_commands:
            return asyncio.run(cache_commands[args.cache_command](args))

    # Handle receipts subcommands (Phase 7.5)
    if args.command == "receipts":
        if not args.receipts_command:
            receipts_parser.print_help()
            return 1

        receipts_commands = {
            "generate": cmd_receipts_generate,
            "list": cmd_receipts_list,
            "get": cmd_receipts_get,
            "preview": cmd_receipts_preview,
            "uncertified": cmd_receipts_uncertified,
            "verify": cmd_receipts_verify,
        }

        if args.receipts_command in receipts_commands:
            return asyncio.run(receipts_commands[args.receipts_command](args))

    # Run the appropriate command
    async_commands = {
        "list": cmd_list_models,
        "chat": cmd_chat,
        "info": cmd_info,
        "stats": cmd_stats,
        "export": cmd_export,
        "register": cmd_register,
    }

    sync_commands = {
        "bind": cmd_bind,
        "aliases": cmd_aliases,
        "providers": cmd_providers,
    }

    if args.command in async_commands:
        return asyncio.run(async_commands[args.command](args))
    elif args.command in sync_commands:
        return sync_commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
