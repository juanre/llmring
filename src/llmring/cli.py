"""Command-line interface for LLM service."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llmring import LLMRequest, LLMRing, Message
from llmring.lockfile import Lockfile
from llmring.registry import RegistryClient
# Alias sync removed per source-of-truth v3.8 - aliases are purely local

# Load environment variables from .env file
load_dotenv()


async def cmd_lock_init(args):
    """Initialize a new lockfile with defaults."""
    path = Path(args.file) if args.file else Path("llmring.lock")

    if path.exists() and not args.force:
        print(f"Error: {path} already exists. Use --force to overwrite.")
        return 1

    # Create default lockfile
    lockfile = Lockfile.create_default()
    lockfile.save(path)

    print(f"✅ Created lockfile: {path}")

    # Show default bindings
    default_profile = lockfile.get_profile("default")
    if default_profile.bindings:
        print("\nDefault bindings:")
        for binding in default_profile.bindings:
            print(f"  {binding.alias} → {binding.model_ref}")

    return 0


async def cmd_bind(args):
    """Bind an alias to a model."""
    # Load or create lockfile
    lockfile_path = Lockfile.find_lockfile() or Path("llmring.lock")

    if lockfile_path and lockfile_path.exists():
        lockfile = Lockfile.load(lockfile_path)
    else:
        lockfile = Lockfile.create_default()
        lockfile_path = Path("llmring.lock")

    # Set binding
    lockfile.set_binding(args.alias, args.model, profile=args.profile)

    # Save
    lockfile.save(lockfile_path)

    profile_name = args.profile or lockfile.default_profile
    print(f"✅ Bound '{args.alias}' → '{args.model}' in profile '{profile_name}'")

    return 0


async def cmd_aliases(args):
    """List aliases from lockfile."""
    # Find lockfile
    lockfile_path = Lockfile.find_lockfile()

    if not lockfile_path:
        print("Error: No llmring.lock found in current or parent directories.")
        return 1

    lockfile = Lockfile.load(lockfile_path)
    profile = lockfile.get_profile(args.profile)

    print(f"Aliases in profile '{profile.name}':")
    print("-" * 40)

    if not profile.bindings:
        print("(no aliases defined)")
    else:
        for binding in profile.bindings:
            print(f"{binding.alias:<20} → {binding.model_ref}")
            if binding.constraints:
                print(f"  Constraints: {binding.constraints}")

    return 0


async def cmd_lock_validate(args):
    """Validate lockfile against registry."""
    # Find lockfile
    lockfile_path = Lockfile.find_lockfile()

    if not lockfile_path:
        print("Error: No llmring.lock found.")
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
                    is_valid = await registry.validate_model(
                        binding.provider, binding.model
                    )
                    status = "✅" if is_valid else "❌"
                    print(f"  {status} {binding.alias} → {binding.model_ref}")
                    if not is_valid:
                        valid = False
                except Exception as e:
                    print(
                        f"  ⚠️  {binding.alias} → {binding.model_ref} (couldn't validate: {e})"
                    )

    if valid:
        print("\n✅ All bindings are valid")
        return 0
    else:
        print("\n❌ Some bindings are invalid")
        return 1


async def cmd_lock_bump_registry(args):
    """Update pinned registry versions to latest."""
    # Find lockfile
    lockfile_path = Lockfile.find_lockfile()

    if not lockfile_path:
        print("Error: No llmring.lock found.")
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


async def cmd_list_models(args):
    """List available models."""
    ring = LLMRing()
    models = ring.get_available_models()

    if args.provider:
        # Filter by provider
        models = {k: v for k, v in models.items() if k == args.provider}

    print(format_model_table(models, show_all=True))


async def cmd_chat(args):
    """Send a chat message to an LLM."""
    # Check if we should use an alias
    if ":" not in args.model:
        # Try to resolve as alias
        lockfile_path = Lockfile.find_lockfile()
        if lockfile_path:
            lockfile = Lockfile.load(lockfile_path)

            # Get profile from environment or use default
            profile_name = os.environ.get("LLMRING_PROFILE", args.profile)

            # Resolve alias
            model_ref = lockfile.resolve_alias(args.model, profile_name)
            if model_ref:
                print(f"[Using alias '{args.model}' → '{model_ref}']")
                args.model = model_ref

    ring = LLMRing()

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
                            "model": chunk.model
                            if chunk and chunk.model
                            else args.model,
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
                    print(
                        f"\n[Model: {chunk.model if chunk and chunk.model else args.model}]"
                    )
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
    ring = LLMRing()

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
                print(
                    f"Input Cost: ${info['dollars_per_million_tokens_input']:.2f}/M tokens"
                )
            if "dollars_per_million_tokens_output" in info:
                print(
                    f"Output Cost: ${info['dollars_per_million_tokens_output']:.2f}/M tokens"
                )
            if "supports_vision" in info and info["supports_vision"]:
                print("Supports: Vision")
            if (
                "supports_function_calling" in info
                and info["supports_function_calling"]
            ):
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
    """Show usage statistics (placeholder)."""
    # For now, show local receipts if available
    ring = LLMRing()

    if not ring.receipts:
        print("No usage statistics available.")
        print("\nNote: Full statistics require server connection.")
        return 0

    print(f"Local usage statistics ({len(ring.receipts)} requests):")
    print("-" * 40)

    total_cost = sum(r.total_cost for r in ring.receipts)
    total_tokens = sum(r.total_tokens for r in ring.receipts)

    print(f"Total requests: {len(ring.receipts)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.6f}")

    if args.verbose:
        print("\nRecent requests:")
        for receipt in ring.receipts[-10:]:
            print(
                f"  {receipt.timestamp}: {receipt.alias} → {receipt.provider}:{receipt.model} (${receipt.total_cost:.6f})"
            )

    return 0


async def cmd_export(args):
    """Export receipts (placeholder)."""
    ring = LLMRing()

    if not ring.receipts:
        print("No receipts to export.")
        return 0

    # Export local receipts as JSON
    import json
    from datetime import datetime, UTC

    export_data = {
        "exported_at": datetime.now(UTC).isoformat(),
        "receipts": [
            {
                "receipt_id": r.receipt_id,
                "timestamp": r.timestamp.isoformat(),
                "alias": r.alias,
                "profile": r.profile,
                "provider": r.provider,
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "total_cost": r.total_cost,
            }
            for r in ring.receipts
        ],
    }

    output_file = args.output or "llmring_receipts.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Exported {len(ring.receipts)} receipts to {output_file}")
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


async def cmd_providers(args):
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
        description="LLMRing - Unified LLM Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Lock commands
    lock_parser = subparsers.add_parser("lock", help="Lockfile management")
    lock_subparsers = lock_parser.add_subparsers(
        dest="lock_command", help="Lock commands"
    )

    # lock init
    init_parser = lock_subparsers.add_parser(
        "init", help="Initialize lockfile with defaults"
    )
    init_parser.add_argument("--file", help="Lockfile path (default: llmring.lock)")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing file"
    )

    # lock validate
    lock_subparsers.add_parser("validate", help="Validate lockfile against registry")

    # lock bump-registry
    lock_subparsers.add_parser("bump-registry", help="Update registry versions")

    # Bind command
    bind_parser = subparsers.add_parser("bind", help="Bind an alias to a model")
    bind_parser.add_argument("alias", help="Alias name")
    bind_parser.add_argument("model", help="Model reference (provider:model)")
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
        "--model", default="openai:gpt-3.5-turbo", help="Model or alias to use"
    )
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (0.0-2.0)"
    )
    chat_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate"
    )
    chat_parser.add_argument("--json", action="store_true", help="Output as JSON")
    chat_parser.add_argument(
        "--verbose", action="store_true", help="Show additional information"
    )
    chat_parser.add_argument("--profile", help="Profile to use for alias resolution")
    chat_parser.add_argument(
        "--stream", action="store_true", help="Stream response in real-time"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Model identifier (e.g., openai:gpt-4)")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Providers command
    providers_parser = subparsers.add_parser(
        "providers", help="List configured providers"
    )
    providers_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Push/pull commands removed per source-of-truth v3.8
    # Aliases are managed entirely locally in each codebase's lockfile

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show usage statistics")
    stats_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed statistics"
    )
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export receipts to file")
    export_parser.add_argument(
        "--output", help="Output file (default: llmring_receipts.json)"
    )
    export_parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Export format"
    )

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
        }

        if args.lock_command in lock_commands:
            return asyncio.run(lock_commands[args.lock_command](args))

    # Run the appropriate command
    command_map = {
        "bind": cmd_bind,
        "aliases": cmd_aliases,
        "list": cmd_list_models,
        "chat": cmd_chat,
        "info": cmd_info,
        "providers": cmd_providers,
        # Push/pull removed per source-of-truth v3.8
        "stats": cmd_stats,
        "export": cmd_export,
        "register": cmd_register,
    }

    if args.command in command_map:
        return asyncio.run(command_map[args.command](args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
