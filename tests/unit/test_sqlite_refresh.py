import tempfile
import os
import pytest
from decimal import Decimal
from llmring.db_sqlite import SQLiteDatabase
from llmring.model_refresh.models import ModelInfo
from llmring.schemas import LLMModel


@pytest.mark.asyncio
async def test_upsert_and_retire_missing_models():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            # Prepare models from two providers
            models = [
                ModelInfo(
                    provider="openai",
                    model_name="gpt-4o-mini",
                    display_name="GPT-4o Mini",
                    dollars_per_million_tokens_input=Decimal("0.15"),
                    dollars_per_million_tokens_output=Decimal("0.60"),
                ),
                ModelInfo(
                    provider="anthropic",
                    model_name="claude-3-5-sonnet-20241022",
                    display_name="Claude 3.5 Sonnet",
                    dollars_per_million_tokens_input=Decimal("3.00"),
                    dollars_per_million_tokens_output=Decimal("15.00"),
                ),
            ]

            inserted, updated = await db.upsert_models(models)
            assert inserted + updated >= 2

            # Upsert again to trigger updates
            inserted2, updated2 = await db.upsert_models(models)
            assert updated2 >= 2

            # Retire missing models for openai when only keeping anthropic
            keep_keys = [
                (m.provider, m.model_name) for m in models if m.provider == "anthropic"
            ]
            retired = await db.retire_missing_models(["openai", "anthropic"], keep_keys)
            assert retired >= 1
        finally:
            await db.close()
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_clean_free_models_and_wipe_all():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = SQLiteDatabase(db_path)
        await db.initialize()
        try:
            # Insert a non-Ollama model without pricing
            m = LLMModel(
                provider="openai",
                model_name="test-free",
                display_name="Test Free",
                description="",
                max_context=1024,
                max_output_tokens=512,
                supports_vision=False,
                supports_function_calling=False,
                dollars_per_million_tokens_input=None,
                dollars_per_million_tokens_output=None,
            )
            await db.add_model(m)

            affected = await db.clean_free_models()
            assert affected >= 1

            # Wipe all
            deleted_calls, deleted_models = await db.wipe_all()
            # There might be zero calls, but models should be > 0 (defaults + inserted)
            assert deleted_models >= 1
        finally:
            await db.close()
    finally:
        os.unlink(db_path)
