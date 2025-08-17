"""SQLite database implementation for llmring - simplified local development option."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import aiosqlite

from .schemas import CallRecord, LLMModel, UsageStats

logger = logging.getLogger(__name__)


# Configure datetime adapter for Python 3.12+
def adapt_datetime(val):
    """Adapt datetime to ISO 8601 string."""
    return val.isoformat()


def convert_datetime(val):
    """Convert ISO 8601 string to datetime."""
    return datetime.fromisoformat(val.decode())


# Register the adapters
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)


class SQLiteDatabase:
    """Simplified SQLite database for local development - alternative to pgdbm-based LLMDatabase."""

    def __init__(self, db_path: str = "llmring.db"):
        """Initialize SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables if needed."""
        if self._initialized:
            return

        # Enable parse_decltypes to use our custom converters
        self.conn = await aiosqlite.connect(
            self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self.conn.execute("PRAGMA foreign_keys = ON")
        # Enable WAL for better concurrency in dev
        try:
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass

        # Create tables
        await self._create_tables()
        await self._insert_default_models()

        self._initialized = True
        logger.info(f"SQLite database initialized at {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _create_tables(self):
        """Create required tables if they don't exist."""
        # Models table
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                display_name TEXT,
                description TEXT,
                max_context INTEGER,
                max_output_tokens INTEGER,
                supports_vision BOOLEAN DEFAULT 0,
                supports_function_calling BOOLEAN DEFAULT 0,
                supports_json_mode BOOLEAN DEFAULT 0,
                supports_parallel_tool_calls BOOLEAN DEFAULT 0,
                tool_call_format TEXT,
                dollars_per_million_tokens_input REAL,
                dollars_per_million_tokens_output REAL,
                inactive_from TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, model_name)
            )
        """
        )

        # API calls table
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_calls (
                id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                id_at_origin TEXT NOT NULL,
                model_id INTEGER,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL,
                dollars_per_million_tokens_input_used REAL,
                dollars_per_million_tokens_output_used REAL,
                status TEXT DEFAULT 'success',
                error_type TEXT,
                error_message TEXT,
                called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """
        )

        # Create indices
        await self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_api_calls_called_at
            ON api_calls(called_at DESC)
        """
        )

        await self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_api_calls_origin
            ON api_calls(origin, called_at DESC)
        """
        )

        await self.conn.commit()

    async def _insert_default_models(self):
        """Insert default model configurations if table is empty."""
        cursor = await self.conn.execute("SELECT COUNT(*) FROM models")
        row = await cursor.fetchone()
        if row[0] > 0:
            return

        default_models = [
            # OpenAI
            (
                "openai",
                "gpt-4o",
                "GPT-4o",
                "Latest GPT-4 Omni model",
                128000,
                16384,
                1,
                1,
                1,
                1,
                2.50,
                10.00,
            ),
            (
                "openai",
                "gpt-4o-mini",
                "GPT-4o Mini",
                "Small, affordable GPT-4 Omni",
                128000,
                16384,
                1,
                1,
                1,
                1,
                0.15,
                0.60,
            ),
            (
                "openai",
                "gpt-4-turbo",
                "GPT-4 Turbo",
                "GPT-4 Turbo with vision",
                128000,
                4096,
                1,
                1,
                1,
                1,
                10.00,
                30.00,
            ),
            (
                "openai",
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo",
                "Fast, affordable GPT-3.5",
                16385,
                4096,
                0,
                1,
                1,
                0,
                0.50,
                1.50,
            ),
            # Anthropic
            (
                "anthropic",
                "claude-3-5-sonnet-20241022",
                "Claude 3.5 Sonnet",
                "Most intelligent Claude",
                200000,
                8192,
                1,
                1,
                0,
                0,
                3.00,
                15.00,
            ),
            (
                "anthropic",
                "claude-3-5-haiku-20241022",
                "Claude 3.5 Haiku",
                "Fast Claude model",
                200000,
                8192,
                0,
                1,
                0,
                0,
                1.00,
                5.00,
            ),
            (
                "anthropic",
                "claude-3-opus-20240229",
                "Claude 3 Opus",
                "Powerful Claude model",
                200000,
                4096,
                1,
                1,
                0,
                0,
                15.00,
                75.00,
            ),
            # Google
            (
                "google",
                "gemini-1.5-pro",
                "Gemini 1.5 Pro",
                "Google's most capable",
                2097152,
                8192,
                1,
                1,
                0,
                0,
                1.25,
                5.00,
            ),
            (
                "google",
                "gemini-1.5-flash",
                "Gemini 1.5 Flash",
                "Fast Gemini model",
                1048576,
                8192,
                1,
                1,
                0,
                0,
                0.075,
                0.30,
            ),
            (
                "google",
                "gemini-pro",
                "Gemini Pro",
                "General-purpose model",
                32768,
                8192,
                0,
                1,
                0,
                0,
                0.50,
                1.50,
            ),
        ]

        for model in default_models:
            await self.conn.execute(
                """
                INSERT OR IGNORE INTO models (
                    provider, model_name, display_name, description,
                    max_context, max_output_tokens, supports_vision,
                    supports_function_calling, supports_json_mode, supports_parallel_tool_calls,
                    dollars_per_million_tokens_input, dollars_per_million_tokens_output
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                model,
            )

        await self.conn.commit()
        logger.info("Default models inserted")

    # Model management methods

    async def get_model(self, provider: str, model_name: str) -> Optional[LLMModel]:
        """Get a specific model by provider and name."""
        cursor = await self.conn.execute(
            """
            SELECT * FROM models
            WHERE provider = ? AND model_name = ? AND inactive_from IS NULL
        """,
            (provider, model_name),
        )

        row = await cursor.fetchone()
        if row:
            return self._row_to_model(dict(row))
        return None

    async def list_models(
        self, provider: Optional[str] = None, active_only: bool = True
    ) -> List[LLMModel]:
        """List all models, optionally filtered by provider."""
        query = "SELECT * FROM models"
        params = []

        conditions = []
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if active_only:
            conditions.append("inactive_from IS NULL")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY provider, model_name"

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_model(dict(row)) for row in rows]

    async def list_all_models(self) -> List[LLMModel]:
        """List all models including inactive ones."""
        cursor = await self.conn.execute(
            "SELECT * FROM models ORDER BY provider, model_name"
        )
        rows = await cursor.fetchall()
        return [self._row_to_model(dict(row)) for row in rows]

    async def upsert_models(self, models: List["ModelInfo"]) -> Tuple[int, int]:
        """Upsert a collection of models by (provider, model_name).
        Returns (inserted_count, updated_count).
        """
        from llmring.model_refresh.models import (
            ModelInfo,
        )  # local import to avoid cycles

        inserted = 0
        updated = 0
        for m in models:
            assert isinstance(m, ModelInfo)
            # First try INSERT OR IGNORE to detect true inserts
            await self.conn.execute(
                """
                INSERT OR IGNORE INTO models (
                    provider, model_name, display_name, description,
                    max_context, max_output_tokens, supports_vision,
                    supports_function_calling, supports_json_mode, supports_parallel_tool_calls,
                    tool_call_format, dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                    inactive_from, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    m.provider,
                    m.model_name,
                    m.display_name,
                    m.description,
                    m.max_context,
                    m.max_output_tokens,
                    1 if m.supports_vision else 0,
                    1 if m.supports_function_calling else 0,
                    1 if m.supports_json_mode else 0,
                    1 if m.supports_parallel_tool_calls else 0,
                    m.tool_call_format,
                    (
                        float(m.dollars_per_million_tokens_input)
                        if m.dollars_per_million_tokens_input is not None
                        else None
                    ),
                    (
                        float(m.dollars_per_million_tokens_output)
                        if m.dollars_per_million_tokens_output is not None
                        else None
                    ),
                ),
            )
            # Check how many rows were inserted
            cur = await self.conn.execute("SELECT changes()")
            change_row = await cur.fetchone()
            changes = change_row[0] if change_row else 0
            if changes and changes > 0:
                inserted += 1
                continue

            # If no insert, perform UPDATE to refresh fields and clear inactive flag
            await self.conn.execute(
                """
                UPDATE models SET
                    display_name = ?,
                    description = ?,
                    max_context = ?,
                    max_output_tokens = ?,
                    supports_vision = ?,
                    supports_function_calling = ?,
                    supports_json_mode = ?,
                    supports_parallel_tool_calls = ?,
                    tool_call_format = ?,
                    dollars_per_million_tokens_input = ?,
                    dollars_per_million_tokens_output = ?,
                    inactive_from = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE provider = ? AND model_name = ?
                """,
                (
                    m.display_name,
                    m.description,
                    m.max_context,
                    m.max_output_tokens,
                    1 if m.supports_vision else 0,
                    1 if m.supports_function_calling else 0,
                    1 if m.supports_json_mode else 0,
                    1 if m.supports_parallel_tool_calls else 0,
                    m.tool_call_format,
                    (
                        float(m.dollars_per_million_tokens_input)
                        if m.dollars_per_million_tokens_input is not None
                        else None
                    ),
                    (
                        float(m.dollars_per_million_tokens_output)
                        if m.dollars_per_million_tokens_output is not None
                        else None
                    ),
                    m.provider,
                    m.model_name,
                ),
            )
            cur2 = await self.conn.execute("SELECT changes()")
            change_row2 = await cur2.fetchone()
            changes2 = change_row2[0] if change_row2 else 0
            if changes2 and changes2 > 0:
                updated += 1
        await self.conn.commit()
        return inserted, updated

    async def retire_missing_models(
        self, providers: List[str], keep_keys: List[Tuple[str, str]]
    ) -> int:
        """Mark models as inactive if they belong to providers but are not in keep_keys.
        Returns number of rows affected.
        """
        # Build set for quick lookup
        keep_set = set(keep_keys)
        # Fetch current models for providers
        placeholders = ",".join(["?"] * len(providers)) if providers else None
        if not providers:
            return 0
        cursor = await self.conn.execute(
            f"SELECT provider, model_name FROM models WHERE provider IN ({placeholders})",
            providers,
        )
        rows = await cursor.fetchall()
        to_retire = [
            (row[0], row[1]) for row in rows if (row[0], row[1]) not in keep_set
        ]
        if not to_retire:
            return 0
        retired = 0
        for provider, model_name in to_retire:
            res = await self.conn.execute(
                """
                UPDATE models
                SET inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE provider = ? AND model_name = ? AND inactive_from IS NULL
                """,
                (provider, model_name),
            )
            retired += res.rowcount or 0
        await self.conn.commit()
        return retired

    async def clean_free_models(self) -> int:
        """Deactivate non-Ollama models without pricing."""
        res = await self.conn.execute(
            """
            UPDATE models
            SET inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE provider != 'ollama' AND (
                dollars_per_million_tokens_input IS NULL OR dollars_per_million_tokens_output IS NULL
            ) AND inactive_from IS NULL
            """
        )
        await self.conn.commit()
        return res.rowcount or 0

    async def wipe_all(self) -> Tuple[int, int]:
        """Delete all models and API calls. Returns (deleted_calls, deleted_models)."""
        calls_res = await self.conn.execute("DELETE FROM api_calls")
        models_res = await self.conn.execute("DELETE FROM models")
        await self.conn.commit()
        return (calls_res.rowcount or 0, models_res.rowcount or 0)

    async def add_model(self, model: LLMModel) -> int:
        """Add a new model to the registry."""
        cursor = await self.conn.execute(
            """
            INSERT INTO models (
                provider, model_name, display_name, description,
                max_context, max_output_tokens, supports_vision,
                supports_function_calling, supports_json_mode, supports_parallel_tool_calls,
                tool_call_format, dollars_per_million_tokens_input, dollars_per_million_tokens_output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                model.provider,
                model.model_name,
                model.display_name,
                model.description,
                model.max_context,
                model.max_output_tokens,
                1 if model.supports_vision else 0,
                1 if model.supports_function_calling else 0,
                1 if model.supports_json_mode else 0,
                1 if model.supports_parallel_tool_calls else 0,
                model.tool_call_format,
                (
                    float(model.dollars_per_million_tokens_input)
                    if model.dollars_per_million_tokens_input
                    else None
                ),
                (
                    float(model.dollars_per_million_tokens_output)
                    if model.dollars_per_million_tokens_output
                    else None
                ),
            ),
        )

        await self.conn.commit()
        return cursor.lastrowid

    # API call tracking

    async def record_api_call(self, call_record: CallRecord) -> UUID:
        """Record an API call for tracking and billing."""
        call_id = call_record.id or uuid4()

        await self.conn.execute(
            """
            INSERT INTO api_calls (
                id, origin, id_at_origin, model_id, provider, model_name,
                prompt_tokens, completion_tokens, total_tokens,
                estimated_cost, dollars_per_million_tokens_input_used,
                dollars_per_million_tokens_output_used, status,
                error_type, error_message, called_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(call_id),
                call_record.origin,
                call_record.id_at_origin,
                call_record.model_id,
                call_record.provider,
                call_record.model_name,
                call_record.prompt_tokens,
                call_record.completion_tokens,
                call_record.total_tokens,
                float(call_record.estimated_cost),
                (
                    float(call_record.dollars_per_million_tokens_input_used)
                    if call_record.dollars_per_million_tokens_input_used
                    else None
                ),
                (
                    float(call_record.dollars_per_million_tokens_output_used)
                    if call_record.dollars_per_million_tokens_output_used
                    else None
                ),
                call_record.status or "success",
                call_record.error_type,
                call_record.error_message,
                (
                    call_record.called_at
                    if hasattr(call_record, "called_at")
                    else datetime.now(timezone.utc)
                ),
            ),
        )

        await self.conn.commit()
        return call_id

    async def get_usage_stats(
        self, origin: Optional[str] = None, days: int = 30
    ) -> UsageStats:
        """Get usage statistics for the specified period."""
        date_limit = f"datetime('now', '-{days} days')"

        query = f"""
            SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(estimated_cost), 0) as total_cost,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) as success_count
            FROM api_calls
            WHERE called_at >= {date_limit}
        """

        params = []
        if origin:
            query += " AND origin = ?"
            params.append(origin)

        cursor = await self.conn.execute(query, params)
        row = await cursor.fetchone()

        total_calls = row[0]
        total_cost = float(row[2]) if row[2] else 0.0
        success_count = row[3]

        return UsageStats(
            total_calls=total_calls,
            total_tokens=row[1],
            total_cost=Decimal(str(total_cost)),
            avg_cost_per_call=(
                Decimal(str(total_cost / total_calls))
                if total_calls > 0
                else Decimal("0")
            ),
            success_rate=(
                Decimal(str(success_count / total_calls))
                if total_calls > 0
                else Decimal("1.0")
            ),
        )

    async def get_recent_calls(
        self, limit: int = 100, offset: int = 0
    ) -> List[CallRecord]:
        """Get recent API calls."""
        cursor = await self.conn.execute(
            """
            SELECT * FROM api_calls
            ORDER BY called_at DESC
            LIMIT ? OFFSET ?
        """,
            (limit, offset),
        )

        rows = await cursor.fetchall()

        result = []
        for row in rows:
            # Convert Row to dict for easier access
            row_dict = dict(row)
            result.append(
                CallRecord(
                    id=UUID(row_dict["id"]),
                    origin=row_dict["origin"],
                    id_at_origin=row_dict["id_at_origin"],
                    model_id=row_dict["model_id"],
                    provider=row_dict["provider"],
                    model_name=row_dict["model_name"],
                    prompt_tokens=row_dict["prompt_tokens"],
                    completion_tokens=row_dict["completion_tokens"],
                    total_tokens=row_dict["total_tokens"],
                    estimated_cost=Decimal(str(row_dict["estimated_cost"])),
                    dollars_per_million_tokens_input_used=(
                        Decimal(str(row_dict["dollars_per_million_tokens_input_used"]))
                        if row_dict["dollars_per_million_tokens_input_used"]
                        else None
                    ),
                    dollars_per_million_tokens_output_used=(
                        Decimal(str(row_dict["dollars_per_million_tokens_output_used"]))
                        if row_dict["dollars_per_million_tokens_output_used"]
                        else None
                    ),
                    status=row_dict.get("status", "success"),
                    error_type=row_dict.get("error_type"),
                    error_message=row_dict.get("error_message"),
                    called_at=row_dict["called_at"],
                )
            )
        return result

    def _row_to_model(self, row: Dict) -> LLMModel:
        """Convert a database row to an LLMModel object."""
        return LLMModel(
            id=row.get("id"),
            provider=row["provider"],
            model_name=row["model_name"],
            display_name=row.get("display_name"),
            description=row.get("description"),
            max_context=row.get("max_context"),
            max_output_tokens=row.get("max_output_tokens"),
            supports_vision=bool(row.get("supports_vision")),
            supports_function_calling=bool(row.get("supports_function_calling")),
            supports_json_mode=bool(row.get("supports_json_mode")),
            supports_parallel_tool_calls=bool(row.get("supports_parallel_tool_calls")),
            tool_call_format=row.get("tool_call_format"),
            dollars_per_million_tokens_input=(
                Decimal(str(row["dollars_per_million_tokens_input"]))
                if row.get("dollars_per_million_tokens_input")
                else None
            ),
            dollars_per_million_tokens_output=(
                Decimal(str(row["dollars_per_million_tokens_output"]))
                if row.get("dollars_per_million_tokens_output")
                else None
            ),
            inactive_from=row.get("inactive_from"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    # Compatibility methods for API
    async def apply_migrations(self):
        """No-op for SQLite - tables are created in initialize()."""
        pass
