"""Async model refresh manager that coordinates the refresh process."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..db import LLMDatabase
from .async_database_updater import AsyncDatabaseUpdater
from .backup_manager import BackupManager
from .model_comparator import ModelComparator
from .model_filter import ModelFilter
from .models import RefreshResult, RefreshSummary

logger = logging.getLogger(__name__)


class AsyncModelRefreshManager:
    """Coordinates the async model refresh process."""

    def __init__(
        self,
        db: Optional[LLMDatabase] = None,
        backup_enabled: bool = True,
        dry_run: bool = False,
    ):
        """Initialize the refresh manager.

        Args:
            db: Database instance to use (creates new if None)
            backup_enabled: Whether to create backups before refresh
            dry_run: If True, only simulate changes without applying
        """
        self.db = db or LLMDatabase(enable_monitoring=False)
        self._owns_db = db is None
        self.db_updater = AsyncDatabaseUpdater(self.db)
        self.comparator = ModelComparator()
        self.backup_manager = BackupManager() if backup_enabled else None
        self.filter = ModelFilter()
        self.dry_run = dry_run

    async def initialize(self):
        """Initialize database connection if needed."""
        if self._owns_db:
            await self.db.initialize()
            await self.db.apply_migrations()

    async def close(self):
        """Close database connection if we own it."""
        if self._owns_db:
            await self.db.close()

    async def refresh_models(
        self,
        scraped_models: List[Dict[str, Any]],
        provider_filter: Optional[str] = None,
        skip_cost_update: bool = False,
    ) -> RefreshSummary:
        """Refresh models in the database with scraped data.

        Args:
            scraped_models: List of scraped model data
            provider_filter: Only update models from this provider
            skip_cost_update: Skip updating cost information

        Returns:
            RefreshSummary with results
        """
        start_time = datetime.now()

        try:
            # Get current models from database
            logger.info("Fetching current models from database...")
            current_models = await self.db_updater.get_current_models()
            logger.info(f"Found {len(current_models)} models in database")

            # Filter scraped models
            filtered_models = self.filter.filter_models(scraped_models, provider_filter)
            logger.info(f"Processing {len(filtered_models)} scraped models")

            # Compare models
            logger.info("Comparing models...")
            new_models, updated_models, deactivated_models = (
                self.comparator.compare_models(
                    current_models, filtered_models, skip_cost_update
                )
            )

            # Log what will be done
            logger.info(
                f"Changes to apply: {len(new_models)} new, {len(updated_models)} updated, {len(deactivated_models)} to deactivate"
            )

            if self.dry_run:
                logger.info("DRY RUN: Not applying changes")
                result = RefreshResult(
                    added_count=len(new_models),
                    updated_count=len(updated_models),
                    deactivated_count=len(deactivated_models),
                    errors=[],
                )
            else:
                # Create backup if enabled
                backup_created = False
                if self.backup_manager and (
                    new_models or updated_models or deactivated_models
                ):
                    logger.info("Creating backup before applying changes...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_created = await self.db_updater.create_snapshot(timestamp)
                    if backup_created:
                        logger.info(f"Backup created: llm_models_snapshot_{timestamp}")
                    else:
                        logger.warning("Failed to create backup, proceeding anyway...")

                # Apply changes
                logger.info("Applying changes to database...")
                result = await self.db_updater.apply_changes(
                    new_models, updated_models, deactivated_models
                )

                # Log results
                if result.added_count > 0:
                    logger.info(f"Added {result.added_count} new models")
                if result.updated_count > 0:
                    logger.info(f"Updated {result.updated_count} models")
                if result.deactivated_count > 0:
                    logger.info(f"Deactivated {result.deactivated_count} models")
                if result.errors:
                    logger.error(f"Encountered {len(result.errors)} errors")
                    for error in result.errors:
                        logger.error(f"  - {error}")

            # Create summary
            duration = (datetime.now() - start_time).total_seconds()

            return RefreshSummary(
                timestamp=start_time,
                duration_seconds=duration,
                models_before=len(current_models),
                models_after=len(current_models)
                + result.added_count
                - result.deactivated_count,
                new_models=new_models,
                updated_models=updated_models,
                deactivated_models=deactivated_models,
                result=result,
                dry_run=self.dry_run,
            )

        except Exception as e:
            logger.error(f"Model refresh failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()

            return RefreshSummary(
                timestamp=start_time,
                duration_seconds=duration,
                models_before=0,
                models_after=0,
                new_models=[],
                updated_models=[],
                deactivated_models=[],
                result=RefreshResult(errors=[str(e)]),
                dry_run=self.dry_run,
            )

    async def get_refresh_preview(
        self,
        scraped_models: List[Dict[str, Any]],
        provider_filter: Optional[str] = None,
        skip_cost_update: bool = False,
    ) -> Dict[str, Any]:
        """Get a preview of what changes would be made without applying them.

        Args:
            scraped_models: List of scraped model data
            provider_filter: Only preview models from this provider
            skip_cost_update: Skip cost update preview

        Returns:
            Dictionary with preview information
        """
        # Get current models
        current_models = await self.db_updater.get_current_models()

        # Filter scraped models
        filtered_models = self.filter.filter_models(scraped_models, provider_filter)

        # Compare models
        new_models, updated_models, deactivated_models = self.comparator.compare_models(
            current_models, filtered_models, skip_cost_update
        )

        # Build preview
        preview = {
            "current_model_count": len(current_models),
            "scraped_model_count": len(filtered_models),
            "changes": {
                "new": len(new_models),
                "updated": len(updated_models),
                "deactivated": len(deactivated_models),
            },
            "new_models": [
                {
                    "provider": m.new_model.provider,
                    "model": m.new_model.model_name,
                    "display_name": m.new_model.display_name,
                }
                for m in new_models[:10]  # Show first 10
            ],
            "updated_models": [
                {
                    "provider": m.new_model.provider,
                    "model": m.new_model.model_name,
                    "changes": list(m.changes.keys()),
                }
                for m in updated_models[:10]  # Show first 10
            ],
            "deactivated_models": [
                {
                    "provider": m.old_model.provider,
                    "model": m.old_model.model_name,
                    "reason": "Not found in scraped data",
                }
                for m in deactivated_models[:10]  # Show first 10
            ],
        }

        # Add truncation notice if needed
        if len(new_models) > 10:
            preview["new_models"].append({"...": f"and {len(new_models) - 10} more"})
        if len(updated_models) > 10:
            preview["updated_models"].append(
                {"...": f"and {len(updated_models) - 10} more"}
            )
        if len(deactivated_models) > 10:
            preview["deactivated_models"].append(
                {"...": f"and {len(deactivated_models) - 10} more"}
            )

        return preview

    async def rollback_to_snapshot(self, snapshot_name: str) -> bool:
        """Rollback to a previous snapshot.

        Args:
            snapshot_name: Name/timestamp of the snapshot

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Would rollback to snapshot {snapshot_name}")
            return True

        logger.info(f"Rolling back to snapshot: {snapshot_name}")
        success = await self.db_updater.restore_snapshot(snapshot_name)

        if success:
            logger.info("Rollback completed successfully")
        else:
            logger.error("Rollback failed")

        return success
