#!/usr/bin/env python3
"""
MCP Client LLM Models List Script

This script lists all supported LLM models in the database and allows updating the models
with the latest supported versions.
"""

import argparse
import asyncio

from dotenv import load_dotenv

from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
# Database model removed - now using HTTP-based architecture
# from llmring.mcp.client.models.db import MCPClientDB


async def list_models(
    db_path: str = None, provider_filter: str = None, verbose: bool = False
) -> None:
    """List all LLM models in the database."""
    print("Listing LLM models in database...\n")

    # Initialize database connection
    db = MCPClientDB(db_path)

    try:
        await db.initialize()

        # Get models from database
        db_models = await db.get_llm_models(provider=provider_filter)

        if not db_models:
            print(
                "No models found in database."
                + (
                    f" (with provider filter: {provider_filter})"
                    if provider_filter
                    else ""
                )
            )
            return

        # Print models
        print(f"Found {len(db_models)} models:")

        # Create a table format
        format_str = "{:<20} {:<40} {:<30} {:<10}"
        print(format_str.format("PROVIDER", "MODEL ID", "DISPLAY NAME", "ENABLED"))
        print("-" * 100)

        for model in db_models:
            print(
                format_str.format(
                    model.get("provider", "unknown"),
                    model.get("model_key", model.get("model", "unknown")),
                    model.get("display_name", "unknown"),
                    "✅" if model.get("enabled", False) else "❌",
                )
            )

            if verbose:
                print(f"  Description: {model['description']}")
                print(f"  Context window: {model['context_length']}")
                print(f"  Cost per input token: ${model['cost_per_input_token']:.6f}")
                print(f"  Cost per output token: ${model['cost_per_output_token']:.6f}")
                print()
    finally:
        await db.close()


def get_all_provider_models() -> dict:
    """Get all available models from all providers."""
    models = {}

    # Get Anthropic models
    try:
        anthropic = AnthropicProvider(api_key="dummy-key")
        models["anthropic"] = [
            (name, f"anthropic:{name}") for name in anthropic.get_supported_models()
        ]
    except Exception as e:
        print(f"Error getting Anthropic models: {e}")
        models["anthropic"] = []

    # Get OpenAI models
    try:
        openai = OpenAIProvider(api_key="dummy-key")
        models["openai"] = [
            (name, f"openai:{name}") for name in openai.get_supported_models()
        ]
    except Exception as e:
        print(f"Error getting OpenAI models: {e}")
        models["openai"] = []

    # Get Google models
    try:
        google = GoogleProvider(api_key="dummy-key")
        models["google"] = [
            (name, f"google:{name}") for name in google.get_supported_models()
        ]
    except Exception as e:
        print(f"Error getting Google models: {e}")
        models["google"] = []

    # Get Ollama models
    try:
        ollama = OllamaProvider()
        models["ollama"] = [
            (name, f"ollama:{name}") for name in ollama.get_supported_models()
        ]
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        models["ollama"] = []

    return models


async def check_missing_models(db_path: str = None, update: bool = False) -> None:
    """Check for missing models in the database."""
    print("Checking for missing models...\n")

    # Initialize database connection
    db = MCPClientDB(db_path)

    try:
        await db.initialize()

        # Get all models from providers
        all_provider_models = get_all_provider_models()

        # Get models from database
        db_models = await db.get_llm_models()
        db_model_keys = {model["model_key"] for model in db_models}

        # Check for missing models
        missing_models = {}
        for provider, models in all_provider_models.items():
            missing = []
            for model_id, model_key in models:
                if model_key not in db_model_keys:
                    missing.append((model_id, model_key))

            if missing:
                missing_models[provider] = missing

        # Print results
        if not missing_models:
            print("No missing models found!")
            return

        print("Missing models:")
        for provider, models in missing_models.items():
            print(f"\n{provider.capitalize()} ({len(models)}):")
            for model_id, model_key in models:
                print(f"  - {model_key} (ID: {model_id})")

        # Update database if requested
        if update:
            print("\nUpdating database with missing models...")

            # Default model info by provider
            default_model_info = {
                "anthropic": {
                    "description": "Anthropic's Claude model",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                    "enabled": True,
                },
                "openai": {
                    "description": "OpenAI's GPT model",
                    "context_length": 128000,
                    "cost_per_input_token": 0.000005,
                    "cost_per_output_token": 0.000015,
                    "enabled": True,
                },
                "google": {
                    "description": "Google's Gemini model",
                    "context_length": 1000000,
                    "cost_per_input_token": 0.000005,
                    "cost_per_output_token": 0.000015,
                    "enabled": True,
                },
                "ollama": {
                    "description": "Local Ollama model",
                    "context_length": 8192,
                    "cost_per_input_token": 0,
                    "cost_per_output_token": 0,
                    "enabled": True,
                },
            }

            # Model specific overrides
            model_overrides = {
                # Claude 3.7 models
                "claude-3-7-sonnet-20250219": {
                    "display_name": "Claude 3.7 Sonnet (Feb 2025)",
                    "description": "Anthropic's Claude 3.7 Sonnet model - hybrid reasoning with advanced capabilities",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                },
                "claude-3-7-sonnet": {
                    "display_name": "Claude 3.7 Sonnet (Latest)",
                    "description": "Anthropic's Claude 3.7 Sonnet model - latest version with hybrid reasoning capabilities",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                },
                # Claude 3.5 models
                "claude-3-5-sonnet-20241022-v2": {
                    "display_name": "Claude 3.5 Sonnet V2",
                    "description": "Anthropic's Claude 3.5 Sonnet model - improved Oct 2024 version",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                },
                "claude-3-5-sonnet-20241022": {
                    "display_name": "Claude 3.5 Sonnet (Oct 2024)",
                    "description": "Anthropic's Claude 3.5 Sonnet model - Oct 2024 version",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                },
                "claude-3-5-sonnet": {
                    "display_name": "Claude 3.5 Sonnet (Latest)",
                    "description": "Anthropic's Claude 3.5 Sonnet model - latest version",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000003,
                    "cost_per_output_token": 0.000015,
                },
                "claude-3-5-haiku-20241022": {
                    "display_name": "Claude 3.5 Haiku (Oct 2024)",
                    "description": "Anthropic's Claude 3.5 Haiku model - faster, more efficient",
                    "context_length": 200000,
                    "cost_per_input_token": 0.00000125,
                    "cost_per_output_token": 0.00000625,
                },
                "claude-3-5-haiku": {
                    "display_name": "Claude 3.5 Haiku (Latest)",
                    "description": "Anthropic's Claude 3.5 Haiku model - latest version",
                    "context_length": 200000,
                    "cost_per_input_token": 0.00000125,
                    "cost_per_output_token": 0.00000625,
                },
                # Claude 3 models
                "claude-3-opus": {
                    "display_name": "Claude 3 Opus",
                    "description": "Anthropic's Claude 3 Opus model - most capable Claude 3 model",
                    "context_length": 200000,
                    "cost_per_input_token": 0.000015,
                    "cost_per_output_token": 0.000075,
                },
                # OpenAI models
                "gpt-4o": {
                    "display_name": "GPT-4o",
                    "description": "OpenAI's GPT-4o model - multimodal capabilities",
                    "context_length": 128000,
                    "cost_per_input_token": 0.000005,
                    "cost_per_output_token": 0.000015,
                },
                # Google models
                "gemini-1.5-pro": {
                    "display_name": "Gemini 1.5 Pro",
                    "description": "Google's Gemini 1.5 Pro model - high performance with multimodal capabilities",
                    "context_length": 1000000,
                    "cost_per_input_token": 0.000005,
                    "cost_per_output_token": 0.000015,
                },
            }

            # Add missing models to database
            added_count = 0
            for provider, models in missing_models.items():
                for model_id, model_key in models:
                    # Get provider default info
                    model_info = default_model_info.get(provider, {}).copy()

                    # Check for specific model overrides
                    if model_id in model_overrides:
                        model_info.update(model_overrides[model_id])

                    # Generate a display name if not in overrides
                    if "display_name" not in model_info:
                        display_name = model_id.replace("-", " ").title()
                        model_info["display_name"] = display_name

                    # Add model to database
                    try:
                        await db.add_llm_model(
                            provider=provider,
                            model=model_id,
                            model_key=model_key,
                            display_name=model_info.get("display_name", model_id),
                            description=model_info.get(
                                "description",
                                f"{provider.capitalize()}'s {model_id} model",
                            ),
                            context_window=model_info.get("context_length", 4096),
                            cost_per_input_token=model_info.get(
                                "cost_per_input_token", 0
                            ),
                            cost_per_output_token=model_info.get(
                                "cost_per_output_token", 0
                            ),
                            enabled=model_info.get("enabled", True),
                        )
                        print(f"Added model: {model_key}")
                        added_count += 1
                    except Exception as e:
                        print(f"Error adding model {model_key}: {e}")

            print(f"\nAdded {added_count} models to database")
    finally:
        await db.close()


async def main():
    parser = argparse.ArgumentParser(description="MCP Client LLM Models List Script")
    parser.add_argument(
        "--db-path",
        help="Database connection string (uses DATABASE_URL env var if not specified)",
    )
    parser.add_argument("--env-file", help="Path to the .env file", default=".env")
    parser.add_argument(
        "--provider",
        help="Filter models by provider (e.g., anthropic, openai, google, ollama)",
        choices=["anthropic", "openai", "google", "ollama"],
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed model information"
    )
    parser.add_argument(
        "--check", "-c", action="store_true", help="Check for missing models"
    )
    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Update database with missing models (implies --check)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    if args.update:
        # Update implies check
        await check_missing_models(args.db_path, update=True)
    elif args.check:
        await check_missing_models(args.db_path)
    else:
        # Default: list models
        await list_models(args.db_path, args.provider, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
