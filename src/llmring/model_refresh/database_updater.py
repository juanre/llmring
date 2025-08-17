"""Safe database operations for model refresh with rollback capability."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List

import psycopg2

from .models import ModelDiff, ModelInfo, RefreshResult

logger = logging.getLogger(__name__)


class DatabaseUpdater:
    """Handles safe database operations for model refresh."""

    def __init__(self, connection_params: Dict[str, Any]):
        """Initialize with database connection parameters."""
        self.connection_params = connection_params

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def get_current_models(self) -> List[ModelInfo]:
        """Retrieve all current models from the database."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT provider, model_name, display_name, description,
                       max_context, max_output_tokens, supports_vision,
                       supports_function_calling, supports_json_mode,
                       supports_parallel_tool_calls, tool_call_format,
                       dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                       inactive_from, created_at, updated_at
                FROM llmring.llm_models
                ORDER BY provider, model_name
            """
            )

            models = []
            for row in cur.fetchall():
                # dollars_per_million_tokens columns store cost per million tokens
                input_cost = float(row[11]) if row[11] else None
                output_cost = float(row[12]) if row[12] else None

                model = ModelInfo(
                    provider=row[0],
                    model_name=row[1],
                    display_name=row[2],
                    description=row[3],
                    max_context=row[4],
                    max_output_tokens=row[5],
                    supports_vision=row[6],
                    supports_function_calling=row[7],
                    supports_json_mode=row[8],
                    supports_parallel_tool_calls=row[9],
                    tool_call_format=row[10],
                    dollars_per_million_tokens_input=input_cost,
                    dollars_per_million_tokens_output=output_cost,
                    is_active=row[13] is None,  # inactive_from is NULL = active
                    inactive_from=row[13],
                    created_at=row[14],
                    updated_at=row[15],
                    source="database",
                )
                models.append(model)

            return models

    def apply_model_diff(self, diff: ModelDiff, dry_run: bool = False) -> RefreshResult:
        """Apply model differences to the database."""
        if dry_run:
            return self._preview_changes(diff)

        errors = []

        try:
            with self.get_connection() as conn:
                cur = conn.cursor()

                # Start transaction
                conn.autocommit = False

                try:
                    # Add new models
                    for model in diff.new_models:
                        self._insert_model(cur, model)

                    # Update existing models
                    for old_model, new_model in diff.updated_models:
                        self._update_model(cur, old_model, new_model)

                    # Retire models (mark as inactive)
                    for model in diff.retired_models:
                        self._retire_model(cur, model)

                    # Commit transaction
                    conn.commit()

                    logger.info(f"Successfully applied model diff: {diff.summary}")
                    return RefreshResult.success_result(
                        message=f"Applied changes: {diff.summary}",
                        diff=diff,
                        backup_id="",  # Will be set by backup manager
                        duration=0.0,  # Will be set by refresh manager
                    )

                except Exception as e:
                    conn.rollback()
                    error_msg = f"Database update failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    raise

        except Exception as e:
            error_msg = f"Failed to apply model diff: {str(e)}"
            logger.error(error_msg)
            return RefreshResult.error_result(error_msg, errors)

    def _insert_model(self, cursor, model: ModelInfo):
        """Insert a new model into the database."""
        cursor.execute(
            """
            INSERT INTO llmring.llm_models (
                provider, model_name, display_name, description,
                max_context, max_output_tokens, supports_vision,
                supports_function_calling, supports_json_mode,
                supports_parallel_tool_calls, tool_call_format,
                dollars_per_million_tokens_input, dollars_per_million_tokens_output,
                inactive_from
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                model.provider,
                model.model_name,
                model.display_name,
                model.description,
                model.max_context,
                model.max_output_tokens,
                model.supports_vision,
                model.supports_function_calling,
                model.supports_json_mode,
                model.supports_parallel_tool_calls,
                model.tool_call_format,
                model.dollars_per_million_tokens_input,
                model.dollars_per_million_tokens_output,
                None if model.is_active else model.inactive_from,
            ),
        )
        logger.info(f"Inserted new model: {model.provider}:{model.model_name}")

    def _update_model(self, cursor, old_model: ModelInfo, new_model: ModelInfo):
        """Update an existing model in the database."""
        cursor.execute(
            """
            UPDATE llmring.llm_models SET
                display_name = %s, description = %s,
                max_context = %s, max_output_tokens = %s, supports_vision = %s,
                supports_function_calling = %s, supports_json_mode = %s,
                supports_parallel_tool_calls = %s, tool_call_format = %s,
                dollars_per_million_tokens_input = %s, dollars_per_million_tokens_output = %s,
                inactive_from = %s, updated_at = CURRENT_TIMESTAMP
            WHERE provider = %s AND model_name = %s
        """,
            (
                new_model.display_name,
                new_model.description,
                new_model.max_context,
                new_model.max_output_tokens,
                new_model.supports_vision,
                new_model.supports_function_calling,
                new_model.supports_json_mode,
                new_model.supports_parallel_tool_calls,
                new_model.tool_call_format,
                new_model.dollars_per_million_tokens_input,
                new_model.dollars_per_million_tokens_output,
                None if new_model.is_active else new_model.inactive_from,
                new_model.provider,
                new_model.model_name,
            ),
        )
        logger.info(f"Updated model: {new_model.provider}:{new_model.model_name}")

    def _retire_model(self, cursor, model: ModelInfo):
        """Mark a model as inactive (retired)."""
        cursor.execute(
            """
            UPDATE llmring.llm_models SET
                inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE provider = %s AND model_name = %s AND inactive_from IS NULL
        """,
            (model.provider, model.model_name),
        )
        logger.info(f"Retired model: {model.provider}:{model.model_name}")

    def _preview_changes(self, diff: ModelDiff) -> RefreshResult:
        """Preview changes without applying them."""
        changes = []

        for model in diff.new_models:
            changes.append(f"ADD: {model.provider}:{model.model_name}")

        for old_model, new_model in diff.updated_models:
            changes.append(f"UPDATE: {new_model.provider}:{new_model.model_name}")

        for model in diff.retired_models:
            changes.append(f"RETIRE: {model.provider}:{model.model_name}")

        message = f"Preview: {diff.summary}\nChanges:\n" + "\n".join(changes)

        return RefreshResult.success_result(
            message=message,
            diff=diff,
            backup_id="preview",
            duration=0.0,
        )

    def validate_database_connection(self) -> bool:
        """Validate that database connection and schema are working."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM llmring.llm_models")
                count = cur.fetchone()[0]
                logger.info(f"Database connection validated. Found {count} models.")
                return True
        except Exception as e:
            logger.error(f"Database validation failed: {str(e)}")
            return False
