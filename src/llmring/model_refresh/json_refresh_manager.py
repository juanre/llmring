"""Simplified model refresh manager using JSON files."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmring.config import ModelRefreshConfig
from llmring.model_refresh.json_model_loader import JSONModelLoader
from llmring.model_refresh.models import RefreshResult
from llmring.model_refresh.refresh_manager import ModelRefreshManager

logger = logging.getLogger(__name__)


class JSONModelRefreshManager:
    """Refresh models from JSON files instead of API discovery."""

    def __init__(self, config: ModelRefreshConfig, models_dir: Optional[str] = None):
        """
        Initialize JSON refresh manager.

        Args:
            config: Model refresh configuration
            models_dir: Optional directory containing JSON model files
        """
        self.config = config
        self.refresh_manager = ModelRefreshManager(
            config.get_database_connection_params(), config.backup_directory
        )

        # Determine models directory
        if models_dir:
            models_dir = Path(models_dir)
        else:
            models_dir = Path(__file__).parent.parent.parent.parent / "data" / "models"
        self.loader = JSONModelLoader(models_dir)

    async def refresh_from_json(
        self,
        providers: Optional[List[str]] = None,
        dry_run: bool = False,
        create_backup: bool = True,
    ) -> RefreshResult:
        """
        Refresh models from JSON files.

        Args:
            providers: List of providers to refresh (None = all)
            dry_run: If True, preview changes without applying
            create_backup: If True, create backup before changes

        Returns:
            RefreshResult with operation details
        """
        logger.info("Starting JSON-based model refresh")

        try:
            # Load models from JSON
            if providers:
                all_models = []
                for provider in providers:
                    models = self.loader.load_provider_models(provider)
                    all_models.extend(models)
            else:
                provider_models = self.loader.load_all_models()
                all_models = []
                for models in provider_models.values():
                    all_models.extend(models)

            logger.info(f"Loaded {len(all_models)} models from JSON files")

            if not all_models:
                return RefreshResult.error_result(
                    "No models found in JSON files",
                    ["Check that JSON files exist in data/models/"],
                )

            # Convert dollars per million back to per token for database
            # (JSON stores in dollars per million, database expects per token)
            db_models = []
            for model in all_models:
                # The loader already does this conversion, but let's ensure it
                db_models.append(model)

            # Use the standard refresh manager to update database
            result = self.refresh_manager.refresh_models(
                db_models, dry_run=dry_run, create_backup=create_backup
            )

            # If successful and not dry_run, also update usage hints
            if result.success and not dry_run:
                self._update_usage_hints(providers)

            return result

        except Exception as e:
            error_msg = f"JSON model refresh failed: {str(e)}"
            logger.error(error_msg)
            return RefreshResult.error_result(error_msg, [str(e)])

    def get_available_providers(self) -> List[str]:
        """Get list of providers with JSON files."""
        providers = []
        models_dir = self.loader.models_dir

        for json_file in models_dir.glob("*.json"):
            if json_file.name != "summary.json":
                providers.append(json_file.stem)

        return sorted(providers)

    def get_refresh_status(self) -> Dict[str, Any]:
        """Get status of JSON-based refresh system."""
        status = {
            "json_models_dir": str(self.loader.models_dir),
            "available_providers": self.get_available_providers(),
            "provider_metadata": {},
        }

        # Get metadata for each provider
        for provider in status["available_providers"]:
            metadata = self.loader.get_model_metadata(provider)
            if metadata:
                status["provider_metadata"][provider] = metadata

        # Add database status
        db_status = self.refresh_manager.generate_status_report()
        status.update(db_status)

        return status

    def _update_usage_hints(self, providers: Optional[List[str]] = None):
        """
        Update usage hints in the database from JSON files.

        Args:
            providers: List of providers to update (None = all)
        """
        # Determine which providers to update
        if providers is None:
            providers = self.get_available_providers()

        logger.info(f"Updating usage hints for providers: {providers}")

        # Get database connection
        with self.refresh_manager.db_updater.get_connection() as conn:
            with conn.cursor() as cursor:
                for provider in providers:
                    try:
                        # Load usage hints from JSON
                        usage_hints = self.loader.load_provider_usage_hints(provider)

                        if not usage_hints:
                            logger.warning(f"No usage hints found for {provider}")
                            continue

                        # Convert to JSONB for the stored procedure
                        import json

                        hints_json = json.dumps(usage_hints)

                        # Call the refresh_usage_hints function
                        cursor.execute(
                            "SELECT llmring.refresh_usage_hints(%s, %s::jsonb)",
                            (provider, hints_json),
                        )

                        logger.info(f"Updated usage hints for {provider}")

                    except Exception as e:
                        logger.error(
                            f"Failed to update usage hints for {provider}: {e}"
                        )

                # Commit all updates
                conn.commit()
                logger.info("Usage hints update completed")
