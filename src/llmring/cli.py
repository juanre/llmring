"""Command-line interface for LLM model management.

Adds optional SQLite support via --sqlite PATH or LLMRING_SQLITE_DB env var
for local workflows (list/search/info/status/json-refresh/clean/suggest).
PostgreSQL remains required for provider API discovery refresh.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from llmring import LLMRequest, LLMRing, Message
from llmring.config import ModelRefreshConfig
from llmring.file_utils import create_file_content
from llmring.model_refresh.complete_refresh_manager import CompleteModelRefreshManager
from llmring.model_refresh.json_refresh_manager import JSONModelRefreshManager
from llmring.model_refresh.refresh_manager import ModelRefreshManager
from llmring.model_refresh.json_model_loader import JSONModelLoader
from llmring.model_refresh.models import ModelInfo
from llmring.db_sqlite import SQLiteDatabase


def format_model_table(models, show_pricing=True, show_capabilities=True):
    """Format models as a readable table."""
    if not models:
        return "No models found."

    # Calculate column widths
    provider_width = max(len(m.provider) for m in models) + 2
    name_width = max(len(m.model_name) for m in models) + 2
    display_width = max(len(m.display_name or "") for m in models) + 2

    # Header
    lines = []
    header = f"{'Provider':<{provider_width}} {'Model Name':<{name_width}} {'Display Name':<{display_width}}"

    if show_pricing:
        header += f" {'Input Cost':<12} {'Output Cost':<12}"

    if show_capabilities:
        header += f" {'Context':<10} {'Features':<20}"

    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for model in models:
        row = f"{model.provider:<{provider_width}} {model.model_name:<{name_width}} {model.display_name or '':<{display_width}}"

        if show_pricing:
            input_cost = (
                f"${float(model.dollars_per_million_tokens_input or 0):.2f}"
                if model.dollars_per_million_tokens_input
                else "Free"
            )
            output_cost = (
                f"${float(model.dollars_per_million_tokens_output or 0):.2f}"
                if model.dollars_per_million_tokens_output
                else "Free"
            )
            row += f" {input_cost:<12} {output_cost:<12}"

        if show_capabilities:
            context = f"{model.max_context:,}" if model.max_context else "Unknown"
            features = []
            if model.supports_vision:
                features.append("Vision")
            if model.supports_function_calling:
                features.append("Functions")
            if model.supports_json_mode:
                features.append("JSON")
            feature_str = ", ".join(features) if features else "None"
            row += f" {context:<10} {feature_str:<20}"

        lines.append(row)

    return "\n".join(lines)


def _is_sqlite_mode(args) -> Tuple[bool, Optional[str]]:
    """Determine if CLI should operate in SQLite mode and return db path if so."""
    db_path = getattr(args, "sqlite", None) or os.environ.get("LLMRING_SQLITE_DB")
    return (bool(db_path), db_path)


async def _sqlite_list_models(
    db_path: str, provider: Optional[str], active_only: bool
) -> List:
    db = SQLiteDatabase(db_path)
    await db.initialize()
    try:
        return await db.list_models(provider=provider, active_only=active_only)
    finally:
        await db.close()


async def cmd_list_models(args):
    """List models in the database."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        models = await _sqlite_list_models(db_path, args.provider, args.active_only)
    else:
        config = ModelRefreshConfig.from_environment()
        refresh_manager = ModelRefreshManager(config.get_database_connection_params())
        models = refresh_manager.db_updater.get_current_models()

    # Filter by provider if specified
    if args.provider:
        models = [m for m in models if m.provider == args.provider]

    # Filter by active status
    if args.active_only:
        models = [m for m in models if m.inactive_from is None]

    # Sort models
    if args.sort_by == "provider":
        models.sort(key=lambda x: (x.provider, x.model_name))
    elif args.sort_by == "name":
        models.sort(key=lambda x: x.model_name)
    elif args.sort_by == "cost":
        models.sort(key=lambda x: x.dollars_per_million_tokens_input or 0)

    # Output format
    if args.format == "json":
        model_data = []
        for model in models:
            data = {
                "provider": model.provider,
                "model_name": model.model_name,
                "display_name": model.display_name,
                "description": model.description,
                "max_context": model.max_context,
                "max_output_tokens": model.max_output_tokens,
                "supports_vision": model.supports_vision,
                "supports_function_calling": model.supports_function_calling,
                "supports_json_mode": model.supports_json_mode,
                "dollars_per_million_tokens_input": (
                    float(model.dollars_per_million_tokens_input)
                    if model.dollars_per_million_tokens_input
                    else None
                ),
                "dollars_per_million_tokens_output": (
                    float(model.dollars_per_million_tokens_output)
                    if model.dollars_per_million_tokens_output
                    else None
                ),
                "is_active": model.inactive_from is None,
            }
            model_data.append(data)
        print(json.dumps(model_data, indent=2))
    else:
        print(
            format_model_table(
                models,
                show_pricing=not args.no_pricing,
                show_capabilities=not args.no_capabilities,
            )
        )
        print(f"\nTotal: {len(models)} models")


async def cmd_search_models(args):
    """Search for models by name or capabilities."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        models = await _sqlite_list_models(db_path, None, True)
    else:
        config = ModelRefreshConfig.from_environment()
        refresh_manager = ModelRefreshManager(config.get_database_connection_params())
        models = refresh_manager.db_updater.get_current_models()

    # Apply search filters
    if args.name:
        models = [m for m in models if args.name.lower() in m.model_name.lower()]

    if args.provider:
        models = [m for m in models if m.provider == args.provider]

    if args.vision:
        models = [m for m in models if m.supports_vision]

    if args.functions:
        models = [m for m in models if m.supports_function_calling]

    if args.max_cost:
        models = [
            m
            for m in models
            if m.dollars_per_million_tokens_input
            and float(m.dollars_per_million_tokens_input) <= args.max_cost
        ]

    if args.min_context:
        models = [
            m for m in models if m.max_context and m.max_context >= args.min_context
        ]

    print(format_model_table(models))
    print(f"\nFound: {len(models)} models matching criteria")


async def cmd_model_info(args):
    """Get detailed information about a specific model."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        models = await _sqlite_list_models(db_path, None, False)
    else:
        config = ModelRefreshConfig.from_environment()
        refresh_manager = ModelRefreshManager(config.get_database_connection_params())
        models = refresh_manager.db_updater.get_current_models()

    # Find the model
    model = None
    for m in models:
        if (
            m.model_name == args.model_name
            or f"{m.provider}:{m.model_name}" == args.model_name
        ):
            model = m
            break

    if not model:
        print(f"Model '{args.model_name}' not found.")
        return

    # Display detailed info
    print(f"Model: {model.provider}:{model.model_name}")
    print(f"Display Name: {model.display_name}")
    print(f"Description: {model.description}")
    print(f"Active: {model.inactive_from is None}")
    print()

    print("Capabilities:")
    print(
        f"  Max Context: {model.max_context:,} tokens"
        if model.max_context
        else "  Max Context: Unknown"
    )
    print(
        f"  Max Output: {model.max_output_tokens:,} tokens"
        if model.max_output_tokens
        else "  Max Output: Unknown"
    )
    print(f"  Vision Support: {'Yes' if model.supports_vision else 'No'}")
    print(f"  Function Calling: {'Yes' if model.supports_function_calling else 'No'}")
    print(f"  JSON Mode: {'Yes' if model.supports_json_mode else 'No'}")
    print(f"  Parallel Tools: {'Yes' if model.supports_parallel_tool_calls else 'No'}")
    print(f"  Tool Format: {model.tool_call_format or 'None'}")
    print()

    print("Pricing:")
    if (
        model.dollars_per_million_tokens_input
        and model.dollars_per_million_tokens_output
    ):
        input_cost_per_million = float(model.dollars_per_million_tokens_input)
        output_cost_per_million = float(model.dollars_per_million_tokens_output)
        print(f"  Input: ${input_cost_per_million:.2f} per 1M tokens")
        print(f"  Output: ${output_cost_per_million:.2f} per 1M tokens")
    else:
        print("  Free (local) or pricing not available")


async def cmd_status(args):
    """Show system status and statistics."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        # Build a lightweight status from SQLite
        try:
            db = SQLiteDatabase(db_path)
            await db.initialize()
            models = await db.list_models(active_only=False)
            await db.close()
            total = len(models)
            active = sum(1 for m in models if m.inactive_from is None)
            by_provider = {}
            active_by_provider = {}
            for m in models:
                by_provider[m.provider] = by_provider.get(m.provider, 0) + 1
                if m.inactive_from is None:
                    active_by_provider[m.provider] = (
                        active_by_provider.get(m.provider, 0) + 1
                    )
            status = {
                "database_status": "connected",
                "system_ready": total > 0,
                "total_models": total,
                "active_models": active,
                "provider_breakdown": by_provider,
                "active_by_provider": active_by_provider,
                "model_discovery_enabled": False,
                "api_discovery_enabled": False,
                "price_scraping_enabled": False,
                "provider_credentials": {},
            }
        except Exception:
            status = {
                "database_status": "error",
                "system_ready": False,
                "total_models": 0,
                "active_models": 0,
                "provider_breakdown": {},
                "active_by_provider": {},
            }
    else:
        config = ModelRefreshConfig.from_environment()
        complete_manager = CompleteModelRefreshManager(config)
        status = complete_manager.get_status_report()

    print("=== LLM Model Management Status ===")
    print(f"Database: {status['database_status']}")
    print(f"System Ready: {status['system_ready']}")
    print()

    print("Model Statistics:")
    print(f"  Total Models: {status['total_models']}")
    print(f"  Active Models: {status['active_models']}")
    print()

    print("By Provider:")
    for provider, count in status["provider_breakdown"].items():
        active_count = status["active_by_provider"].get(provider, 0)
        print(f"  {provider}: {active_count}/{count} active")
    print()

    print("Configuration:")
    print(
        f"  Model Refresh: {'Enabled' if status.get('model_discovery_enabled') else 'Disabled'}"
    )
    print(
        f"  API Discovery: {'Enabled' if status.get('api_discovery_enabled') else 'Disabled'}"
    )
    print(
        f"  Price Scraping: {'Enabled' if status.get('price_scraping_enabled') else 'Disabled'}"
    )
    print()

    print("Provider Credentials:")
    for provider, has_creds in status.get("provider_credentials", {}).items():
        print(f"  {provider}: {'✓' if has_creds else '✗'}")

    if status.get("latest_backup"):
        backup = status["latest_backup"]
        print("\nLatest Backup:")
        print(f"  ID: {backup['backup_id']}")
        print(f"  Created: {backup['created_at']}")
        print(f"  Models: {backup['model_count']}")


async def cmd_init_db(args):
    """Initialize database schema and seed default models.

    - SQLite: creates tables and inserts default models.
    - Postgres: applies migrations (schema creation and seed models).
    """
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        db = SQLiteDatabase(db_path)
        await db.initialize()
        await db.close()
        print(f"✓ Initialized SQLite database at {db_path}")
        return

    # Postgres path
    # Use DATABASE_URL if provided; otherwise use ModelRefreshConfig pieces
    from llmring.db import LLMDatabase
    from llmring.config import ModelRefreshConfig

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        cfg = ModelRefreshConfig.from_environment()
        params = cfg.get_database_connection_params()
        host = params["host"]
        port = params["port"]
        database = params["database"]
        user = params["user"]
        password = params["password"]
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = LLMDatabase(connection_string=dsn, schema="llmring")
    await db.initialize()
    await db.apply_migrations()
    await db.close()
    print("✓ Initialized PostgreSQL database and applied migrations (seeded models)")


async def cmd_refresh(args):
    """Refresh models from providers."""
    sqlite_mode, _ = _is_sqlite_mode(args)
    if sqlite_mode:
        print(
            "SQLite mode: provider API discovery refresh is not supported. Use 'json-refresh'."
        )
        return

    config = ModelRefreshConfig.from_environment()

    # Override pricing configuration if flag is provided
    if hasattr(args, "enable_pricing") and args.enable_pricing:
        config.enable_price_scraping = True

    complete_manager = CompleteModelRefreshManager(config)

    print("Starting model refresh...")

    result = await complete_manager.perform_complete_refresh(
        dry_run=args.dry_run,
        discover_models=not args.skip_discovery,
        update_pricing=not args.skip_pricing,
        filter_models=not args.no_filter,
    )

    if result.success:
        print(f"✓ Refresh successful: {result.message}")
        if result.diff:
            print(f"Changes: {result.diff.summary}")
        if result.backup_id:
            print(f"Backup created: {result.backup_id}")
        print(f"Duration: {result.duration_seconds:.2f}s")
    else:
        print(f"✗ Refresh failed: {result.message}")
        for error in result.errors:
            print(f"  Error: {error}")


async def cmd_clean(args):
    """Clean database operations."""
    if args.clean_action == "free-models":
        await cmd_clean_free_models(args)
    elif args.clean_action == "wipe-all":
        await cmd_wipe_all(args)
    else:
        print("Usage: llm-models clean {free-models|wipe-all}")


async def cmd_clean_free_models(args):
    """Remove non-Ollama models without pricing."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            print("=== Cleaning Models Without Pricing (SQLite) ===")
            affected = await db.clean_free_models()
            print(f"✓ Deactivated {affected} models without pricing")
        finally:
            await db.close()
        return

    config = ModelRefreshConfig.from_environment()
    refresh_manager = ModelRefreshManager(config.get_database_connection_params())

    print("=== Cleaning Models Without Pricing ===")

    # Get all current models
    all_models = refresh_manager.db_updater.get_current_models()
    print(f"Found {len(all_models)} total models in database")

    # Find models without pricing (excluding Ollama)
    models_to_deactivate = []
    for model in all_models:
        if model.provider == "ollama":
            continue  # Ollama is always free

        if (
            model.dollars_per_million_tokens_input is None
            or model.dollars_per_million_tokens_output is None
        ):
            models_to_deactivate.append(model)

    if not models_to_deactivate:
        print("✓ All models have proper pricing!")
        return

    print(f"Found {len(models_to_deactivate)} non-Ollama models without pricing")

    # Group by provider
    by_provider = {}
    for model in models_to_deactivate:
        if model.provider not in by_provider:
            by_provider[model.provider] = []
        by_provider[model.provider].append(model)

    for provider, models in by_provider.items():
        print(f"\n{provider.upper()} models without pricing ({len(models)}):")
        for model in models:
            print(f"  - {model.model_name}")

    print(f"\nDeactivating {len(models_to_deactivate)} models...")

    try:
        with refresh_manager.db_updater.get_connection() as conn:
            with conn.cursor() as cursor:
                for model in models_to_deactivate:
                    cursor.execute(
                        """
                        UPDATE llmring.llm_models
                        SET inactive_from = CURRENT_TIMESTAMP
                        WHERE provider = %s AND model_name = %s
                    """,
                        (model.provider, model.model_name),
                    )

                conn.commit()
                print(
                    f"✓ Deactivated {len(models_to_deactivate)} models without pricing"
                )

    except Exception as e:
        print(f"✗ Failed to deactivate models: {e}")


async def cmd_wipe_all(args):
    """Wipe all models from database."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        print("=== WIPING ALL MODELS FROM SQLite DATABASE ===")
        if not args.force:
            import sys

            if sys.stdin.isatty():
                print("⚠️  This will DELETE ALL models - are you sure? (y/N): ", end="")
                response = input().strip().lower()
                if response != "y":
                    print("Aborted.")
                    return
            else:
                print("Use --force flag for non-interactive mode")
                return
        else:
            print("⚠️  Force mode - proceeding with deletion")

        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            deleted_calls, deleted_models = await db.wipe_all()
            print(f"✓ Deleted {deleted_calls} API call records")
            print(f"✓ Deleted {deleted_models} models from database")
        finally:
            await db.close()
        print("✓ Clean slate ready!")
        return

    config = ModelRefreshConfig.from_environment()
    refresh_manager = ModelRefreshManager(config.get_database_connection_params())

    print("=== WIPING ALL MODELS FROM DATABASE ===")

    if not args.force:
        import sys

        if sys.stdin.isatty():
            print("⚠️  This will DELETE ALL models - are you sure? (y/N): ", end="")
            response = input().strip().lower()
            if response != "y":
                print("Aborted.")
                return
        else:
            print("Use --force flag for non-interactive mode")
            return
    else:
        print("⚠️  Force mode - proceeding with deletion")

    all_models = refresh_manager.db_updater.get_current_models()
    print(f"Found {len(all_models)} models to delete")

    try:
        with refresh_manager.db_updater.get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete related records first
                cursor.execute("DELETE FROM llmring.llm_api_calls")
                api_calls_deleted = cursor.rowcount
                print(f"✓ Deleted {api_calls_deleted} API call records")

                # Delete usage hints (foreign key references)
                cursor.execute("DELETE FROM llmring.model_usage_hints")
                hints_deleted = cursor.rowcount
                print(f"✓ Deleted {hints_deleted} usage hints")

                # Delete all models
                cursor.execute("DELETE FROM llmring.llm_models")
                rows_deleted = cursor.rowcount
                conn.commit()
                print(f"✓ Deleted {rows_deleted} models from database")

    except Exception as e:
        print(f"✗ Failed to delete models: {e}")
        return

    print("✓ Clean slate ready!")


async def cmd_json_refresh(args):
    """Refresh models from JSON files."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        print("=== JSON MODEL REFRESH (SQLite) ===")
        # Load models from JSON
        models_dir = (
            Path(args.models_dir)
            if args.models_dir
            else Path(__file__).parent.parent.parent / "data" / "models"
        )
        loader = JSONModelLoader(models_dir)
        if args.provider:
            providers = [args.provider]
            print(f"Refreshing provider: {args.provider}")
            all_models = []
            for p in providers:
                all_models.extend(loader.load_provider_models(p))
        else:
            providers = loader.load_all_models().keys()
            print(f"Refreshing all providers: {', '.join(sorted(providers))}")
            all_models = []
            for p in providers:
                all_models.extend(loader.load_provider_models(p))

        if args.dry_run:
            print(f"Loaded {len(all_models)} models from JSON (dry-run)")
            return

        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            inserted, updated = await db.upsert_models(all_models)
            keep_keys = [(m.provider, m.model_name) for m in all_models]
            retired = await db.retire_missing_models(
                list(set([m.provider for m in all_models])), keep_keys
            )
        finally:
            await db.close()

        print(
            f"\n✓ Applied JSON models: {inserted} inserted, {updated} updated, {retired} retired"
        )
        return

    config = ModelRefreshConfig.from_environment()
    manager = JSONModelRefreshManager(config, models_dir=args.models_dir)

    print("=== JSON MODEL REFRESH ===")

    # Get list of providers to refresh
    providers = None
    if args.provider:
        providers = [args.provider]
        print(f"Refreshing provider: {args.provider}")
    else:
        available = manager.get_available_providers()
        print(f"Refreshing all providers: {', '.join(available)}")

    # Perform refresh
    result = await manager.refresh_from_json(
        providers=providers, dry_run=args.dry_run, create_backup=not args.no_backup
    )

    # Display results
    if args.dry_run:
        print("\n=== DRY RUN - No changes applied ===")

    if result.success:
        print(f"\n✓ {result.message}")

        if result.diff:
            print(f"\nModels added: {len(result.diff.new_models)}")
            for model in result.diff.new_models[:5]:
                print(f"  + {model.provider}:{model.model_name}")
            if len(result.diff.new_models) > 5:
                print(f"  ... and {len(result.diff.new_models) - 5} more")

            print(f"\nModels updated: {len(result.diff.updated_models)}")
            for old_model, new_model in result.diff.updated_models[:5]:
                print(f"  ~ {new_model.provider}:{new_model.model_name}")
            if len(result.diff.updated_models) > 5:
                print(f"  ... and {len(result.diff.updated_models) - 5} more")

            print(f"\nModels retired: {len(result.diff.retired_models)}")
            for model in result.diff.retired_models[:5]:
                print(f"  - {model.provider}:{model.model_name}")
            if len(result.diff.retired_models) > 5:
                print(f"  ... and {len(result.diff.retired_models) - 5} more")

        if result.backup_id and not args.dry_run:
            print(f"\nBackup created: {result.backup_id}")
    else:
        print(f"\n✗ {result.message}")
        for error in result.errors:
            print(f"  - {error}")


async def cmd_debug(args):
    """Debug operations."""
    if args.debug_action == "pricing":
        await cmd_debug_pricing(args)
    elif args.debug_action == "conservative-refresh":
        await cmd_debug_conservative_refresh(args)
    else:
        print("Usage: llm-models debug {pricing|conservative-refresh}")


async def cmd_debug_pricing(args):
    """Debug pricing extraction."""
    from llmring.pricing.anthropic_pricing import AnthropicPricingScraper
    from llmring.pricing.google_pricing import GooglePricingScraper
    from llmring.pricing.openai_pricing import OpenAIPricingScraper

    config = ModelRefreshConfig.from_environment()

    # Override pricing configuration if flag is provided
    if hasattr(args, "enable_pricing") and args.enable_pricing:
        config.enable_price_scraping = True

    if not config.openai_api_key:
        print("⚠ No OpenAI API key found for LLM processing.")
        return

    providers_to_test = (
        [args.provider] if args.provider else ["anthropic", "google", "openai"]
    )

    for provider in providers_to_test:
        print(f"\n=== Debugging {provider.title()} Pricing ===")

        if provider == "anthropic":
            scraper = AnthropicPricingScraper(config.openai_api_key)
        elif provider == "google":
            scraper = GooglePricingScraper(config.openai_api_key)
        elif provider == "openai":
            scraper = OpenAIPricingScraper(config.openai_api_key)

        url = scraper.get_pricing_url()
        print(f"URL: {url}")

        try:
            result = await scraper.scrape_pricing()
            if result.success:
                print(f"✓ Found {len(result.models)} models with pricing")
                for model in result.models:
                    input_per_million = float(model.input_cost_per_token) * 1000000
                    output_per_million = float(model.output_cost_per_token) * 1000000
                    print(
                        f"  - {model.model_name}: ${input_per_million:.3f}/${output_per_million:.3f} per 1M tokens"
                    )
            else:
                print(f"✗ Scraping failed: {result.error}")
                print("Using fallback pricing:")
                fallback_models = scraper.get_fallback_pricing()
                for model in fallback_models:
                    input_per_million = float(model.input_cost_per_token) * 1000000
                    output_per_million = float(model.output_cost_per_token) * 1000000
                    print(
                        f"  - {model.model_name}: ${input_per_million:.3f}/${output_per_million:.3f} per 1M tokens"
                    )
        except Exception as e:
            print(f"✗ Exception: {e}")


async def cmd_suggest_models(args):
    """Suggest models based on use cases."""
    sqlite_mode, db_path = _is_sqlite_mode(args)
    if sqlite_mode:
        # Compute suggestions in-memory from SQLite models
        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            models = await db.list_models(active_only=True)
        finally:
            await db.close()

        def pick_by_provider(selector_fn):
            by_provider = {}
            for m in models:
                if m.provider not in by_provider:
                    by_provider[m.provider] = []
                by_provider[m.provider].append(m)
            selections = {}
            for provider, lst in by_provider.items():
                choice = selector_fn(lst)
                if choice:
                    selections[provider] = choice
            return selections

        def deepest_selector(lst):
            # Heuristic: most expensive input cost wins
            priced = [m for m in lst if m.dollars_per_million_tokens_input]
            return (
                max(priced, key=lambda m: float(m.dollars_per_million_tokens_input))
                if priced
                else (lst[0] if lst else None)
            )

        def largest_context_selector(lst):
            return max(lst, key=lambda m: (m.max_context or 0)) if lst else None

        def largest_output_selector(lst):
            return max(lst, key=lambda m: (m.max_output_tokens or 0)) if lst else None

        def best_vision_selector(lst):
            vision = [m for m in lst if m.supports_vision]
            return (
                max(vision, key=lambda m: (m.max_context or 0))
                if vision
                else (lst[0] if lst else None)
            )

        def cheapest_good_selector(lst):
            priced = [m for m in lst if m.dollars_per_million_tokens_input]
            return (
                min(priced, key=lambda m: float(m.dollars_per_million_tokens_input))
                if priced
                else (lst[0] if lst else None)
            )

        if args.all:
            print("\n=== Suggestions (SQLite heuristic) ===")
            for use_case, selector in [
                ("deepest_model", deepest_selector),
                ("largest_context", largest_context_selector),
                ("largest_output", largest_output_selector),
                ("best_vision", best_vision_selector),
                ("cheapest_good", cheapest_good_selector),
            ]:
                print(f"\n{use_case}:")
                picks = pick_by_provider(selector)
                for provider, m in picks.items():
                    print(f"  {provider}: {m.model_name} ({m.display_name or ''})")
            return

        if not args.use_case:
            print(
                "Provide a use case or --all. Use cases: deepest_model, largest_context, largest_output, best_vision, cheapest_good"
            )
            return

        selector_map = {
            "deepest_model": deepest_selector,
            "largest_context": largest_context_selector,
            "largest_output": largest_output_selector,
            "best_vision": best_vision_selector,
            "cheapest_good": cheapest_good_selector,
        }
        selector = selector_map[args.use_case]
        picks = pick_by_provider(selector)
        if args.provider:
            m = picks.get(args.provider)
            if m:
                print(f"\n{args.provider}: {m.model_name} ({m.display_name or ''})")
            else:
                print(f"No suggestion for provider {args.provider}")
        else:
            print(f"\n=== Best models for {args.use_case} ===")
            for provider, m in picks.items():
                print(f"  {provider}: {m.model_name} ({m.display_name or ''})")
        return

    # Postgres path (existing behavior)
    config = ModelRefreshConfig.from_environment()
    # Direct database connection to run SQL functions
    import psycopg2

    conn_params = config.get_database_connection_params()
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                if args.all:
                    # Show all usage hints
                    if args.provider:
                        cursor.execute(
                            "SELECT * FROM llmring.get_provider_usage_hints(%s)",
                            (args.provider,),
                        )
                        results = cursor.fetchall()

                        print(f"\n=== Usage Hints for {args.provider.upper()} ===")
                        for (
                            use_case,
                            model_id,
                            model_name,
                            display_name,
                            reasoning,
                        ) in results:
                            print(f"\n{use_case}:")
                            print(f"  Model: {model_name}")
                            print(f"  Display: {display_name}")
                            print(f"  Reasoning: {reasoning}")
                    else:
                        # Show all providers
                        providers = ["anthropic", "openai", "google"]
                        for provider in providers:
                            cursor.execute(
                                "SELECT * FROM llmring.get_provider_usage_hints(%s)",
                                (provider,),
                            )
                            results = cursor.fetchall()

                            if results:
                                print(f"\n=== {provider.upper()} ===")
                                for (
                                    use_case,
                                    model_id,
                                    model_name,
                                    display_name,
                                    reasoning,
                                ) in results:
                                    print(f"\n{use_case}:")
                                    print(f"  Model: {model_name}")
                                    print(f"  Reasoning: {reasoning[:80]}...")

                elif args.use_case:
                    # Show specific use case
                    if args.provider:
                        cursor.execute(
                            "SELECT * FROM llmring.get_model_for_use_case(%s, %s)",
                            (args.provider, args.use_case),
                        )
                        result = cursor.fetchone()

                        if result:
                            (
                                model_id,
                                model_name,
                                display_name,
                                description,
                                reasoning,
                                max_context,
                                max_output_tokens,
                                input_cost,
                                output_cost,
                            ) = result

                            print(
                                f"\n=== Best {args.provider.upper()} model for {args.use_case} ==="
                            )
                            print(f"Model: {model_name}")
                            print(f"Display: {display_name}")
                            print(f"Description: {description}")
                            print(f"Reasoning: {reasoning}")
                            print("\nCapabilities:")
                            print(f"  Context: {max_context:,} tokens")
                            print(f"  Max Output: {max_output_tokens:,} tokens")
                            print(
                                f"  Cost: ${float(input_cost)*1000000:.2f}/${float(output_cost)*1000000:.2f} per 1M tokens"
                            )
                        else:
                            print(
                                f"No {args.provider} model suggested for {args.use_case}"
                            )
                    else:
                        # All providers
                        cursor.execute(
                            "SELECT * FROM llmring.get_all_models_for_use_case(%s)",
                            (args.use_case,),
                        )
                        results = cursor.fetchall()

                        print(f"\n=== Best models for {args.use_case} ===")
                        for result in results:
                            (
                                provider,
                                model_id,
                                model_name,
                                display_name,
                                description,
                                reasoning,
                                max_context,
                                max_output_tokens,
                                input_cost,
                                output_cost,
                            ) = result

                            print(f"\n{provider.upper()}:")
                            print(f"  Model: {model_name}")
                            print(f"  Display: {display_name}")
                            print(f"  Reasoning: {reasoning}")
                            print(
                                f"  Cost: ${float(input_cost)*1000000:.2f}/${float(output_cost)*1000000:.2f} per 1M tokens"
                            )
                else:
                    # Show available use cases
                    print("\nAvailable use cases:")
                    print(
                        "  deepest_model    - Best for complex reasoning/intelligence"
                    )
                    print("  largest_context  - Model with largest context window")
                    print("  largest_output   - Model with largest output capacity")
                    print("  best_vision      - Best for vision/image understanding")
                    print("  cheapest_good    - Best price/performance ratio")
                    print("\nUse: llm-models suggest <use_case> [--provider PROVIDER]")
                    print("Or:  llm-models suggest --all [--provider PROVIDER]")

    except Exception as e:
        print(f"Error querying usage hints: {e}")


async def cmd_extract_from_pdfs(args):
    """Extract model information from PDFs."""
    if args.mode == "download-instructions":
        await show_download_instructions()
    else:
        await generate_jsons_from_pdfs(args.provider)


async def show_download_instructions():
    """Show instructions for downloading PDFs."""
    print("=== Model JSON Generator - Download Instructions ===\n")

    print("Please download these PDFs to the res/ directory:\n")

    print("ANTHROPIC:")
    print("1. Go to: https://www.anthropic.com/pricing")
    print("2. Print/Save as PDF: res/YYYY-MM-DD-anthropic-pricing.pdf")
    print("3. Go to: https://docs.anthropic.com/en/docs/about-claude/models/overview")
    print("4. Print/Save as PDF: res/YYYY-MM-DD-anthropic-models.pdf\n")

    print("OPENAI:")
    print("1. Go to: https://platform.openai.com/docs/models/compare")
    print("2. Print/Save as PDF: res/YYYY-MM-DD-openai-models.pdf")
    print("3. Go to: https://platform.openai.com/docs/pricing")
    print("4. Print/Save as PDF: res/YYYY-MM-DD-openai-pricing.pdf\n")

    print("GOOGLE:")
    print("1. Go to: https://ai.google.dev/gemini-api/docs/models")
    print("2. IMPORTANT: Click 'Gemini Pro' to expand before printing")
    print("3. Print/Save as PDF: res/YYYY-MM-DD-google-models.pdf")
    print("4. Go to: https://ai.google.dev/gemini-api/docs/pricing")
    print("5. Print/Save as PDF: res/YYYY-MM-DD-google-pricing.pdf\n")

    print("Replace YYYY-MM-DD with today's date (e.g., 2025-06-17)")
    print("\nRun 'llm-models extract-from-pdfs generate' when done.")


async def generate_jsons_from_pdfs(provider_filter=None):
    """Generate JSON files from PDFs."""
    print("=== Model JSON Generator - Generate Mode ===\n")

    # Get paths relative to the package
    package_root = Path(__file__).parent.parent.parent
    res_dir = package_root / "res"
    output_dir = package_root / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM service
    service = LLMRing(enable_db_logging=False)

    providers = ["anthropic", "openai", "google"]
    if provider_filter:
        providers = [provider_filter]

    for provider in providers:
        print(f"\nProcessing {provider.upper()}...")

        # Find PDFs intelligently
        all_pdfs = list(res_dir.glob("*.pdf"))
        provider_pdfs = [p for p in all_pdfs if provider.lower() in p.name.lower()]

        if not provider_pdfs:
            print(f"No PDFs found for {provider}")
            continue

        # Sort by filename to get most recent
        provider_pdfs.sort(key=lambda x: x.name, reverse=True)

        # Group by type
        models_pdfs = [p for p in provider_pdfs if "model" in p.name.lower()]
        pricing_pdfs = [p for p in provider_pdfs if "pricing" in p.name.lower()]

        pdf_files = []
        if models_pdfs:
            pdf_files.append(models_pdfs[0])
        if pricing_pdfs:
            pdf_files.append(pricing_pdfs[0])

        if not pdf_files:
            pdf_files = provider_pdfs[:2]

        print(f"Using PDFs: {[p.name for p in pdf_files]}")

        # Read existing JSON if available
        example_json_path = output_dir / f"{provider}.json"
        example_data = None
        if example_json_path.exists():
            with open(example_json_path) as f:
                example_data = json.load(f)

        # Create prompt and message content
        prompt_text = create_extraction_prompt(provider, pdf_files, example_data)

        # Create content with PDFs
        content_parts = [{"type": "text", "text": prompt_text}]

        # Add PDFs as documents
        for pdf_file in pdf_files:
            pdf_content = create_file_content(str(pdf_file), f"This is {pdf_file.name}")
            content_parts.extend(pdf_content)

        print("Asking Claude to analyze PDFs...")

        try:
            # Try multiple times with exponential backoff
            max_retries = 3
            response_text = None

            for attempt in range(max_retries):
                try:
                    # Create request
                    request = LLMRequest(
                        messages=[Message(role="user", content=content_parts)],
                        model="claude-3-7-sonnet-20250219",  # Latest Claude Sonnet
                        temperature=0.1,  # Low temperature for consistent extraction
                        max_tokens=8192,
                    )

                    # Get response
                    response = await service.chat(request)
                    response_text = response.content
                    break  # Success, exit retry loop

                except Exception as e:
                    if "overloaded" in str(e).lower() and attempt < max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"  API overloaded, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise  # Re-raise if not overloaded or last attempt

            # Extract JSON
            if response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1

                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)

                    # Save
                    with open(output_dir / f"{provider}.json", "w") as f:
                        json.dump(result, f, indent=2)

                    print(
                        f"✓ Generated {provider}.json with {len(result['models'])} models"
                    )
                else:
                    print(f"✗ Failed to extract JSON for {provider}")

        except Exception as e:
            print(f"✗ Error processing {provider}: {e}")

        # Small delay between providers to avoid API overload
        if provider != providers[-1]:
            await asyncio.sleep(2)

    print("\n✓ Model extraction complete!")

    # Generate model selections for each provider
    print("\nGenerating model selections for use cases...")
    all_models = {}
    for provider in providers:
        json_path = output_dir / f"{provider}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                all_models[provider] = data.get("models", [])

    if all_models:
        model_selections = await generate_model_selections(service, all_models)

        # Update each provider JSON with its selections
        for provider in providers:
            json_path = output_dir / f"{provider}.json"
            if json_path.exists() and provider in model_selections:
                with open(json_path) as f:
                    data = json.load(f)

                data["model_selection"] = model_selections[provider]

                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Updated {provider}.json with model selections")

    await service.close()

    print(f"\nFiles saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review generated JSONs: cat data/models/*.json | jq .")
    print("2. Update database: llm-models json-refresh")


async def generate_model_selections(service, all_models):
    """Generate model selections for each provider based on use cases."""

    # Flatten models with provider info for the prompt
    flattened_models = []
    for provider, models in all_models.items():
        for model in models:
            model_copy = model.copy()
            model_copy["provider"] = provider
            flattened_models.append(model_copy)

    # Create prompt for model selection
    prompt = f"""I have extracted model information from various providers. Please analyze these models and select the best one from each provider for each use case.

MODELS DATA:
{json.dumps(flattened_models, indent=2)}

USE CASES TO SELECT FOR:
1. "deepest_model": The model you would use for your most complex agents (best intelligence/reasoning)
2. "largest_context": The model with the largest context window (max_context)
3. "largest_output": The model with the largest output capacity (max_output_tokens)
4. "best_vision": The best model for vision/image understanding agents
5. "cheapest_good": The cheapest model that still provides good quality (not just the absolute cheapest)

IMPORTANT REQUIREMENTS:
- You MUST select ONE model from EACH provider (anthropic, openai, google) for EACH use case
- This means each use case will have 3 models selected (one per provider)
- A single model can appear in multiple use cases if it's best for multiple purposes
- For "largest_output", pick the model with the highest max_output_tokens value
- For "cheapest_good", don't just pick the cheapest - pick the best price/performance ratio
- Include brief reasoning for each selection

Return a JSON object with this structure:
{{
  "last_updated": "{datetime.now().strftime('%Y-%m-%d')}",
  "use_cases": {{
    "deepest_model": {{
      "anthropic": {{
        "model_id": "exact-model-id",
        "reasoning": "Why this model was selected"
      }},
      "openai": {{
        "model_id": "exact-model-id",
        "reasoning": "Why this model was selected"
      }},
      "google": {{
        "model_id": "exact-model-id",
        "reasoning": "Why this model was selected"
      }}
    }},
    "largest_context": {{
      "anthropic": {{ ... }},
      "openai": {{ ... }},
      "google": {{ ... }}
    }},
    "largest_output": {{
      "anthropic": {{ ... }},
      "openai": {{ ... }},
      "google": {{ ... }}
    }},
    "best_vision": {{
      "anthropic": {{ ... }},
      "openai": {{ ... }},
      "google": {{ ... }}
    }},
    "cheapest_good": {{
      "anthropic": {{ ... }},
      "openai": {{ ... }},
      "google": {{ ... }}
    }}
  }}
}}

Return ONLY the JSON object, no other text."""

    try:
        # Create request
        request = LLMRequest(
            messages=[Message(role="user", content=prompt)],
            model="claude-3-7-sonnet-20250219",
            temperature=0.1,
            max_tokens=4096,
        )

        # Get response
        response = await service.chat(request)
        response_text = response.content

        # Extract JSON
        start = response_text.find("{")
        end = response_text.rfind("}") + 1

        if start >= 0 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)

            # Transform the result into per-provider format
            provider_selections = {}
            for use_case, selections in result.get("use_cases", {}).items():
                for provider, selection in selections.items():
                    if provider not in provider_selections:
                        provider_selections[provider] = {}
                    provider_selections[provider][use_case] = {
                        "model_id": selection["model_id"],
                        "reasoning": selection["reasoning"],
                    }

            return provider_selections
        else:
            print("✗ Failed to extract model selection JSON")
            return {}

    except Exception as e:
        print(f"✗ Error generating model selection: {e}")
        return {}


def create_extraction_prompt(provider, pdf_files, example_data):
    """Create extraction prompt."""

    prompt = f"""I have attached {provider.upper()} documentation PDFs that contain model information and pricing. Please analyze these PDFs and extract model information.

TASK:
1. Extract ALL models that are NOT obsolete
2. A model is considered obsolete if:
   - It costs more than a newer model with equal or better capabilities
   - It has been explicitly deprecated or marked as legacy
   - There's a direct replacement that's better AND cheaper
3. Think through each model's price/performance ratio compared to others
4. Include models that offer unique capabilities even if more expensive (e.g., largest context, best vision, etc.)

REQUIREMENTS:
1. Extract EXACT model IDs as used in API calls
2. Convert all prices to dollars per million tokens (input and output separately)
3. Write comprehensive descriptions explaining each model's strengths
4. List specific use cases where each model excels
5. Note any unique capabilities or limitations

"""

    if example_data:
        prompt += f"""
Here's the current JSON to update:
```json
{json.dumps(example_data, indent=2)}
```

Update with latest information from the PDFs.
"""
    else:
        prompt += """
Return a JSON object with this structure:
{
  "provider": "provider_name",
  "last_updated": "YYYY-MM-DD",
  "source_documents": ["pdf", "names"],
  "models": [
    {
      "model_id": "exact-api-id",
      "display_name": "Human Name",
      "description": "Comprehensive description",
      "use_cases": ["list", "of", "use", "cases"],
      "max_context": 200000,
      "max_output_tokens": 4096,
      "supports_vision": true,
      "supports_function_calling": true,
      "supports_json_mode": false,
      "supports_parallel_tool_calls": false,
      "dollars_per_million_tokens_input": 15.00,
      "dollars_per_million_tokens_output": 75.00,
      "release_date": "YYYY-MM-DD",
      "deprecation_date": null,
      "notes": "Any notes"
    }
  ]
}
"""

    prompt += "\n\nReturn ONLY the JSON object, no other text."

    return prompt


async def cmd_debug_conservative_refresh(args):
    """Test conservative pricing-first refresh."""
    config = ModelRefreshConfig.from_environment()

    # Override pricing configuration if flag is provided
    if hasattr(args, "enable_pricing") and args.enable_pricing:
        config.enable_price_scraping = True

    complete_manager = CompleteModelRefreshManager(config)

    print("=== Testing Conservative Pricing-First Refresh ===")

    # Get pricing data first
    print("\n--- Getting Verified Pricing Data ---")
    pricing_data = {}

    for provider_name, scraper in complete_manager.pricing_scrapers.items():
        try:
            print(f"Getting pricing for {provider_name}...")
            result = await scraper.get_pricing_with_cache()

            if result.success:
                print(
                    f"✓ Scraped pricing for {len(result.models)} {provider_name} models"
                )
                pricing_data[provider_name] = {
                    model.model_name: model for model in result.models
                }
            else:
                print(f"⚠ Scraping failed, using fallback for {provider_name}")
                fallback_models = scraper.get_fallback_pricing()
                pricing_data[provider_name] = {
                    model.model_name: model for model in fallback_models
                }
                print(
                    f"✓ Fallback pricing for {len(fallback_models)} {provider_name} models"
                )

        except Exception as e:
            print(f"✗ Failed to get pricing for {provider_name}: {e}")

    # Create models from pricing data only
    verified_models = []
    for provider, models in pricing_data.items():
        for model_name, pricing_model in models.items():
            from llmring.model_refresh.models import ModelInfo

            model_info = ModelInfo(
                provider=provider,
                model_name=model_name,
                display_name=model_name.replace("-", " ").title(),
                description=f"{provider.title()} {model_name} model",
                cost_per_token_input=pricing_model.input_cost_per_token,
                cost_per_token_output=pricing_model.output_cost_per_token,
                source="pricing_verified",
            )
            verified_models.append(model_info)

    print(f"\n--- Applying {len(verified_models)} Verified Models ---")

    try:
        import asyncio

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            complete_manager.refresh_manager.refresh_models,
            verified_models,
            False,  # not dry run
            True,  # create backup
        )

        if result.success:
            print(f"✓ Conservative refresh successful: {result.message}")
        else:
            print(f"✗ Conservative refresh failed: {result.message}")

    except Exception as e:
        print(f"✗ Conservative refresh exception: {e}")

    print("\n=== Conservative Test Complete ===")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LLM Model Management CLI")
    # Global options
    parser.add_argument(
        "--sqlite",
        help="Path to SQLite database file. If provided (or LLMRING_SQLITE_DB is set), CLI uses SQLite backend.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    list_parser = subparsers.add_parser("list", help="List models in database")
    list_parser.add_argument("--provider", help="Filter by provider")
    list_parser.add_argument(
        "--active-only", action="store_true", help="Show only active models"
    )
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    list_parser.add_argument(
        "--sort-by",
        choices=["provider", "name", "cost"],
        default="provider",
        help="Sort by",
    )
    list_parser.add_argument(
        "--no-pricing", action="store_true", help="Hide pricing columns"
    )
    list_parser.add_argument(
        "--no-capabilities", action="store_true", help="Hide capability columns"
    )

    # Search models command
    search_parser = subparsers.add_parser("search", help="Search for models")
    search_parser.add_argument("--name", help="Search by model name")
    search_parser.add_argument("--provider", help="Filter by provider")
    search_parser.add_argument(
        "--vision", action="store_true", help="Models with vision support"
    )
    search_parser.add_argument(
        "--functions", action="store_true", help="Models with function calling"
    )
    search_parser.add_argument(
        "--max-cost", type=float, help="Maximum cost per 1M tokens"
    )
    search_parser.add_argument("--min-context", type=int, help="Minimum context window")

    # Model info command
    info_parser = subparsers.add_parser("info", help="Get detailed model information")
    info_parser.add_argument("model_name", help="Model name or provider:model_name")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    # Init DB command
    init_db_parser = subparsers.add_parser(
        "init-db", help="Initialize database schema and seed default models"
    )

    # Suggest command
    suggest_parser = subparsers.add_parser(
        "suggest", help="Suggest models for use cases"
    )
    suggest_parser.add_argument(
        "use_case",
        choices=[
            "deepest_model",
            "largest_context",
            "largest_output",
            "best_vision",
            "cheapest_good",
        ],
        nargs="?",
        help="Use case to get suggestions for",
    )
    suggest_parser.add_argument("--provider", help="Filter by specific provider")
    suggest_parser.add_argument(
        "--all", action="store_true", help="Show all usage hints for all providers"
    )

    # Extract from PDFs command
    extract_parser = subparsers.add_parser(
        "extract-from-pdfs",
        help="Extract model information from provider PDF documentation",
    )
    extract_parser.add_argument(
        "mode",
        choices=["download-instructions", "generate"],
        help="Mode: show download instructions or generate JSONs from PDFs",
    )
    extract_parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        help="Process only a specific provider",
    )

    # Refresh command
    refresh_parser = subparsers.add_parser(
        "refresh", help="Refresh models from providers"
    )
    refresh_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    refresh_parser.add_argument(
        "--skip-discovery", action="store_true", help="Skip model discovery"
    )
    refresh_parser.add_argument(
        "--skip-pricing", action="store_true", help="Skip pricing updates"
    )
    refresh_parser.add_argument(
        "--enable-pricing",
        action="store_true",
        help="Enable web scraping for pricing data",
    )
    refresh_parser.add_argument(
        "--no-filter", action="store_true", help="Don't filter to production models"
    )

    # JSON refresh command
    json_refresh_parser = subparsers.add_parser(
        "json-refresh", help="Refresh models from JSON files"
    )
    json_refresh_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    json_refresh_parser.add_argument(
        "--provider", help="Specific provider to refresh (default: all)"
    )
    json_refresh_parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup"
    )
    json_refresh_parser.add_argument(
        "--models-dir",
        help="Directory containing JSON model files (default: package data dir)",
    )

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Database maintenance operations"
    )
    clean_subparsers = clean_parser.add_subparsers(
        dest="clean_action", help="Clean operations"
    )

    # Clean free models
    clean_free_parser = clean_subparsers.add_parser(
        "free-models", help="Remove non-Ollama models without pricing"
    )

    # Wipe all models
    clean_wipe_parser = clean_subparsers.add_parser(
        "wipe-all", help="Remove ALL models from database"
    )
    clean_wipe_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Debug pricing and discovery")
    debug_subparsers = debug_parser.add_subparsers(
        dest="debug_action", help="Debug operations"
    )

    # Debug pricing
    debug_pricing_parser = debug_subparsers.add_parser(
        "pricing", help="Debug pricing extraction"
    )
    debug_pricing_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        help="Provider to debug",
    )
    debug_pricing_parser.add_argument(
        "--enable-pricing",
        action="store_true",
        help="Enable web scraping for pricing data",
    )

    # Conservative refresh
    debug_conservative_parser = debug_subparsers.add_parser(
        "conservative-refresh", help="Test conservative pricing-first refresh"
    )
    debug_conservative_parser.add_argument(
        "--enable-pricing",
        action="store_true",
        help="Enable web scraping for pricing data",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Command dispatch
    commands = {
        "list": cmd_list_models,
        "search": cmd_search_models,
        "info": cmd_model_info,
        "status": cmd_status,
        "init-db": cmd_init_db,
        "suggest": cmd_suggest_models,
        "extract-from-pdfs": cmd_extract_from_pdfs,
        "refresh": cmd_refresh,
        "json-refresh": cmd_json_refresh,
        "clean": cmd_clean,
        "debug": cmd_debug,
    }

    if args.command in commands:
        asyncio.run(commands[args.command](args))
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
