import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Import the CLI module
from llmring import cli as cli_module


@pytest.fixture
def temp_models_dir(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Minimal provider JSONs
    openai_json = {
        "provider": "openai",
        "last_updated": "2025-01-01",
        "source_documents": [],
        "models": [
            {
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini",
                "description": "",
                "max_context": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_parallel_tool_calls": True,
                "dollars_per_million_tokens_input": 0.15,
                "dollars_per_million_tokens_output": 0.60,
            }
        ],
    }

    with open(models_dir / "openai.json", "w") as f:
        json.dump(openai_json, f)

    return str(models_dir)


def test_cli_sqlite_json_refresh_and_list(temp_models_dir):
    # Create temp SQLite DB path
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Monkeypatch argv to simulate: llmring --sqlite db json-refresh --models-dir <dir>
        sys.argv = [
            "llmring",
            "--sqlite",
            db_path,
            "json-refresh",
            "--models-dir",
            temp_models_dir,
        ]
        # Run CLI main (json-refresh)
        cli_module.main()

        # Now list
        sys.argv = [
            "llmring",
            "--sqlite",
            db_path,
            "list",
            "--provider",
            "openai",
        ]
        cli_module.main()
    finally:
        os.unlink(db_path)
