"""Command-line interface for LLM service."""

import argparse
import asyncio
import json
import os
from typing import List, Optional

from llmring import LLMRequest, LLMRing, Message


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
    )
    
    try:
        # Send request
        response = await ring.chat(request)
        
        # Display response
        if args.json:
            print(json.dumps({
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
            }, indent=2))
        else:
            print(response.content)
            
            if args.verbose and response.usage:
                print(f"\n[Model: {response.model}]")
                print(f"[Tokens: {response.usage}]")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


async def cmd_info(args):
    """Show information about a specific model."""
    ring = LLMRing()
    
    try:
        info = ring.get_model_info(args.model)
        
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"Model: {info['model']}")
            print(f"Provider: {info['provider']}")
            print(f"Supported: {info['supported']}")
            if 'is_default' in info:
                print(f"Default: {info['is_default']}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


async def cmd_providers(args):
    """List configured providers."""
    ring = LLMRing()
    
    providers = []
    for provider_name in ["openai", "anthropic", "google", "ollama"]:
        try:
            provider = ring.get_provider(provider_name)
            has_key = provider is not None
        except:
            has_key = False
        
        providers.append({
            "provider": provider_name,
            "configured": has_key,
            "api_key_env": {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY or GEMINI_API_KEY",
                "ollama": "(not required)",
            }.get(provider_name, ""),
        })
    
    if args.json:
        print(json.dumps(providers, indent=2))
    else:
        print("Configured Providers:")
        print("-" * 40)
        for p in providers:
            status = "✓" if p["configured"] else "✗"
            print(f"{status} {p['provider']:<12} {p['api_key_env']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLMRing - Unified LLM Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--provider", help="Filter by provider (openai, anthropic, google, ollama)"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a chat message")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument(
        "--model", default="openai:gpt-3.5-turbo", help="Model to use"
    )
    chat_parser.add_argument(
        "--system", help="System prompt"
    )
    chat_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (0.0-2.0)"
    )
    chat_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate"
    )
    chat_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    chat_parser.add_argument(
        "--verbose", action="store_true", help="Show additional information"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Model identifier (e.g., openai:gpt-4)")
    info_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    
    # Providers command
    providers_parser = subparsers.add_parser("providers", help="List configured providers")
    providers_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    command_map = {
        "list": cmd_list_models,
        "chat": cmd_chat,
        "info": cmd_info,
        "providers": cmd_providers,
    }
    
    if args.command in command_map:
        return asyncio.run(command_map[args.command](args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())