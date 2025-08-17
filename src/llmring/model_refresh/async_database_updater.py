"""Async database operations for model refresh with rollback capability."""

import logging
from typing import List, Optional

from ..db import LLMDatabase
from ..schemas import LLMModel
from .models import ModelDiff, ModelInfo, RefreshResult

logger = logging.getLogger(__name__)


class AsyncDatabaseUpdater:
    """Handles async database operations for model refresh."""

    def __init__(self, db: Optional[LLMDatabase] = None):
        """Initialize with database instance or create new one."""
        self.db = db or LLMDatabase(
            enable_monitoring=False
        )  # Disable monitoring for batch operations
        self._owns_db = db is None  # Track if we created the DB

    async def initialize(self):
        """Initialize database connection if needed."""
        if self._owns_db:
            await self.db.initialize()

    async def close(self):
        """Close database connection if we own it."""
        if self._owns_db:
            await self.db.close()

    async def get_current_models(self) -> List[ModelInfo]:
        """Retrieve all current models from the database."""
        models = await self.db.list_models(active_only=False)

        # Convert to ModelInfo objects
        model_infos = []
        for model in models:
            model_infos.append(
                ModelInfo(
                    provider=model.provider,
                    model_name=model.model_name,
                    display_name=model.display_name,
                    description=model.description,
                    max_context=model.max_context,
                    max_output_tokens=model.max_output_tokens,
                    supports_vision=model.supports_vision,
                    supports_function_calling=model.supports_function_calling,
                    supports_json_mode=model.supports_json_mode,
                    supports_parallel_tool_calls=model.supports_parallel_tool_calls,
                    tool_call_format=model.tool_call_format,
                    dollars_per_million_tokens_input=model.dollars_per_million_tokens_input,
                    dollars_per_million_tokens_output=model.dollars_per_million_tokens_output,
                    is_active=model.is_active,
                )
            )

        return model_infos

    async def apply_changes(
        self,
        new_models: List[ModelDiff],
        updated_models: List[ModelDiff],
        deactivated_models: List[ModelDiff],
    ) -> RefreshResult:
        """Apply model changes to the database in a transaction."""
        added_count = 0
        updated_count = 0
        deactivated_count = 0
        errors = []

        try:
            # Use a transaction for all changes
            async with self.db.db.transaction() as conn:
                # Add new models
                for diff in new_models:
                    try:
                        model = diff.new_model
                        llm_model = LLMModel(
                            provider=model.provider,
                            model_name=model.model_name,
                            display_name=model.display_name,
                            description=model.description,
                            max_context=model.max_context,
                            max_output_tokens=model.max_output_tokens,
                            supports_vision=model.supports_vision,
                            supports_function_calling=model.supports_function_calling,
                            supports_json_mode=model.supports_json_mode,
                            supports_parallel_tool_calls=model.supports_parallel_tool_calls,
                            tool_call_format=model.tool_call_format,
                            dollars_per_million_tokens_input=model.dollars_per_million_tokens_input,
                            dollars_per_million_tokens_output=model.dollars_per_million_tokens_output,
                        )

                        # Use the connection from the transaction
                        query = self.db.db._prepare_query(
                            """
                            INSERT INTO {{tables.llm_models}} (
                                provider, model_name, display_name, description,
                                max_context, max_output_tokens, supports_vision,
                                supports_function_calling, supports_json_mode,
                                supports_parallel_tool_calls, tool_call_format,
                                dollars_per_million_tokens_input, dollars_per_million_tokens_output
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                            ON CONFLICT (provider, model_name) DO NOTHING
                            RETURNING id
                            """
                        )

                        result = await conn.fetchval(
                            query,
                            llm_model.provider,
                            llm_model.model_name,
                            llm_model.display_name,
                            llm_model.description,
                            llm_model.max_context,
                            llm_model.max_output_tokens,
                            llm_model.supports_vision,
                            llm_model.supports_function_calling,
                            llm_model.supports_json_mode,
                            llm_model.supports_parallel_tool_calls,
                            llm_model.tool_call_format,
                            llm_model.dollars_per_million_tokens_input,
                            llm_model.dollars_per_million_tokens_output,
                        )

                        if result:
                            added_count += 1
                            logger.info(
                                f"Added model: {model.provider}/{model.model_name}"
                            )
                    except Exception as e:
                        error_msg = f"Failed to add {diff.new_model.provider}/{diff.new_model.model_name}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                # Update existing models
                for diff in updated_models:
                    try:
                        model = diff.new_model

                        # Build update query dynamically based on what changed
                        update_fields = []
                        params = []
                        param_count = 0

                        for field_name, change in diff.changes.items():
                            if field_name in [
                                "dollars_per_million_tokens_input",
                                "dollars_per_million_tokens_output",
                            ]:
                                # Skip cost updates if they're None
                                if change["new"] is not None:
                                    param_count += 1
                                    update_fields.append(
                                        f"{field_name} = ${param_count}"
                                    )
                                    params.append(change["new"])
                            elif field_name not in [
                                "provider",
                                "model_name",
                            ]:  # Don't update key fields
                                param_count += 1
                                update_fields.append(f"{field_name} = ${param_count}")
                                params.append(change["new"])

                        if update_fields:
                            # Add provider and model_name at the end for WHERE clause
                            param_count += 1
                            provider_param = f"${param_count}"
                            params.append(model.provider)

                            param_count += 1
                            model_param = f"${param_count}"
                            params.append(model.model_name)

                            query = self.db.db._prepare_query(
                                f"""
                                UPDATE {{{{tables.llm_models}}}}
                                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                                WHERE provider = {provider_param} AND model_name = {model_param}
                                """
                            )

                            result = await conn.execute(query, *params)
                            if result != "UPDATE 0":
                                updated_count += 1
                                logger.info(
                                    f"Updated model: {model.provider}/{model.model_name}"
                                )
                    except Exception as e:
                        error_msg = f"Failed to update {diff.new_model.provider}/{diff.new_model.model_name}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                # Deactivate removed models
                for diff in deactivated_models:
                    try:
                        model = diff.old_model
                        query = self.db.db._prepare_query(
                            """
                            UPDATE {{tables.llm_models}}
                            SET inactive_from = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                            WHERE provider = $1 AND model_name = $2 AND inactive_from IS NULL
                            """
                        )

                        result = await conn.execute(
                            query, model.provider, model.model_name
                        )
                        if result != "UPDATE 0":
                            deactivated_count += 1
                            logger.info(
                                f"Deactivated model: {model.provider}/{model.model_name}"
                            )
                    except Exception as e:
                        error_msg = f"Failed to deactivate {diff.old_model.provider}/{diff.old_model.model_name}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

        except Exception as e:
            error_msg = f"Transaction failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            # Transaction will be rolled back automatically

        return RefreshResult(
            added_count=added_count,
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            errors=errors,
        )

    async def create_snapshot(self, name: str) -> bool:
        """Create a snapshot of current model state."""
        try:
            # Create a snapshot table with timestamp
            snapshot_table = f"llm_models_snapshot_{name}"

            query = self.db.db._prepare_query(
                f"""
                CREATE TABLE {{{{schema}}}}.{snapshot_table} AS
                SELECT *, CURRENT_TIMESTAMP as snapshot_time
                FROM {{{{tables.llm_models}}}}
                """
            )

            await self.db.db.execute(query)
            logger.info(f"Created snapshot: {snapshot_table}")
            return True
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False

    async def restore_snapshot(self, name: str) -> bool:
        """Restore model state from a snapshot."""
        try:
            snapshot_table = f"llm_models_snapshot_{name}"

            async with self.db.db.transaction() as conn:
                # Check if snapshot exists
                check_query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = $1 AND table_name = $2
                    )
                """
                exists = await conn.fetchval(
                    check_query, self.db.config.schema or "public", snapshot_table
                )

                if not exists:
                    logger.error(f"Snapshot not found: {snapshot_table}")
                    return False

                # Clear current models
                await conn.execute(
                    self.db.db._prepare_query(
                        "TRUNCATE TABLE {{tables.llm_models}} CASCADE"
                    )
                )

                # Restore from snapshot
                restore_query = self.db.db._prepare_query(
                    f"""
                    INSERT INTO {{{{tables.llm_models}}}}
                    SELECT * FROM {{{{schema}}}}.{snapshot_table}
                    WHERE snapshot_time IS NOT NULL
                    """
                )
                await conn.execute(restore_query)

                logger.info(f"Restored from snapshot: {snapshot_table}")
                return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    async def get_model_by_name(
        self, provider: str, model_name: str
    ) -> Optional[ModelInfo]:
        """Get a specific model by provider and name."""
        model = await self.db.get_model(provider, model_name)
        if not model:
            return None

        return ModelInfo(
            provider=model.provider,
            model_name=model.model_name,
            display_name=model.display_name,
            description=model.description,
            max_context=model.max_context,
            max_output_tokens=model.max_output_tokens,
            supports_vision=model.supports_vision,
            supports_function_calling=model.supports_function_calling,
            supports_json_mode=model.supports_json_mode,
            supports_parallel_tool_calls=model.supports_parallel_tool_calls,
            tool_call_format=model.tool_call_format,
            dollars_per_million_tokens_input=model.dollars_per_million_tokens_input,
            dollars_per_million_tokens_output=model.dollars_per_million_tokens_output,
            is_active=model.is_active,
        )
