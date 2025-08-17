"""Main orchestration class for model refresh operations."""

import logging
import time
from typing import Any, Dict, List

from .backup_manager import BackupManager
from .database_updater import DatabaseUpdater
from .model_comparator import ModelComparator
from .models import ModelInfo, RefreshResult

logger = logging.getLogger(__name__)


class ModelRefreshManager:
    """Orchestrates the complete model refresh process."""

    def __init__(
        self,
        connection_params: Dict[str, Any],
        backup_dir: str = "/tmp/llm_model_backups",
    ):
        """
        Initialize refresh manager.

        Args:
            connection_params: Database connection parameters
            backup_dir: Directory for backup files
        """
        self.db_updater = DatabaseUpdater(connection_params)
        self.comparator = ModelComparator()
        self.backup_manager = BackupManager(backup_dir)

    def refresh_models(
        self,
        discovered_models: List[ModelInfo],
        dry_run: bool = False,
        create_backup: bool = True,
    ) -> RefreshResult:
        """
        Perform a complete model refresh operation.

        Args:
            discovered_models: Models discovered from providers
            dry_run: If True, preview changes without applying
            create_backup: If True, create backup before changes

        Returns:
            RefreshResult with operation details
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting model refresh with {len(discovered_models)} discovered models"
            )

            # Validate database connection
            if not self.db_updater.validate_database_connection():
                return RefreshResult.error_result(
                    "Database connection validation failed",
                    ["Cannot connect to database or schema is invalid"],
                )

            # Get current models from database
            current_models = self.db_updater.get_current_models()
            logger.info(f"Retrieved {len(current_models)} current models from database")

            # Compare models to find differences
            diff = self.comparator.compare_models(discovered_models, current_models)

            # If no changes and not dry run, return early
            if not diff.has_changes and not dry_run:
                duration = time.time() - start_time
                return RefreshResult.success_result(
                    message="No model changes detected",
                    diff=diff,
                    backup_id="",
                    duration=duration,
                )

            # Create backup if requested and not dry run
            backup_id = ""
            if create_backup and not dry_run and diff.has_changes:
                backup_id = self.backup_manager.create_backup(
                    current_models, reason="refresh"
                )
                logger.info(f"Created backup: {backup_id}")

            # Apply changes
            result = self.db_updater.apply_model_diff(diff, dry_run=dry_run)

            # Update result with timing and backup info
            result.duration_seconds = time.time() - start_time
            if backup_id:
                result.backup_id = backup_id

            logger.info(
                f"Model refresh completed in {result.duration_seconds:.2f}s: {result.message}"
            )
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Model refresh failed after {duration:.2f}s: {str(e)}"
            logger.error(error_msg)
            return RefreshResult.error_result(error_msg, [str(e)])

    def preview_changes(self, discovered_models: List[ModelInfo]) -> RefreshResult:
        """
        Preview what changes would be made without applying them.

        Args:
            discovered_models: Models discovered from providers

        Returns:
            RefreshResult with preview information
        """
        return self.refresh_models(discovered_models, dry_run=True, create_backup=False)

    def rollback_to_backup(self, backup_id: str) -> RefreshResult:
        """
        Rollback to a previous backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            RefreshResult with rollback details
        """
        start_time = time.time()

        try:
            logger.info(f"Starting rollback to backup: {backup_id}")

            # Validate database connection
            if not self.db_updater.validate_database_connection():
                return RefreshResult.error_result(
                    "Database connection validation failed",
                    ["Cannot connect to database or schema is invalid"],
                )

            # Create backup of current state before rollback
            current_models = self.db_updater.get_current_models()
            rollback_backup_id = self.backup_manager.create_backup(
                current_models, reason=f"pre_rollback_{backup_id}"
            )

            # Restore models from backup
            backup_models = self.backup_manager.restore_backup(backup_id)

            # Get current models and compare
            diff = self.comparator.compare_models(backup_models, current_models)

            # Apply the rollback
            result = self.db_updater.apply_model_diff(diff, dry_run=False)
            result.duration_seconds = time.time() - start_time
            result.backup_id = rollback_backup_id
            result.message = f"Rolled back to backup {backup_id}. Pre-rollback backup: {rollback_backup_id}"

            logger.info(f"Rollback completed in {result.duration_seconds:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Rollback failed after {duration:.2f}s: {str(e)}"
            logger.error(error_msg)
            return RefreshResult.error_result(error_msg, [str(e)])

    def generate_status_report(self) -> Dict[str, Any]:
        """
        Generate a status report of the current system state.

        Returns:
            Dictionary with system status information
        """
        try:
            # Database status
            db_valid = self.db_updater.validate_database_connection()
            current_models = self.db_updater.get_current_models() if db_valid else []

            # Provider breakdown
            provider_counts = {}
            active_counts = {}

            for model in current_models:
                provider_counts[model.provider] = (
                    provider_counts.get(model.provider, 0) + 1
                )
                if model.is_active:
                    active_counts[model.provider] = (
                        active_counts.get(model.provider, 0) + 1
                    )

            # Backup status
            backups = self.backup_manager.list_backups()

            return {
                "database_status": "connected" if db_valid else "disconnected",
                "total_models": len(current_models),
                "active_models": sum(1 for m in current_models if m.is_active),
                "provider_breakdown": provider_counts,
                "active_by_provider": active_counts,
                "backup_count": len(backups),
                "latest_backup": backups[0] if backups else None,
                "system_ready": db_valid and len(current_models) > 0,
            }

        except Exception as e:
            logger.error(f"Failed to generate status report: {str(e)}")
            return {
                "database_status": "error",
                "error": str(e),
                "system_ready": False,
            }

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Clean up old backup files.

        Args:
            keep_count: Number of recent backups to keep

        Returns:
            Number of backups deleted
        """
        return self.backup_manager.cleanup_old_backups(keep_count)
