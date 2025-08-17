"""Backup and restore functionality for model data."""

import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List

from .models import ModelInfo

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class BackupManager:
    """Manages backup and restore operations for model data."""

    def __init__(self, backup_dir: str = "/tmp/llm_model_backups"):
        """
        Initialize backup manager.

        Args:
            backup_dir: Directory to store backup files
        """
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, models: List[ModelInfo], reason: str = "refresh") -> str:
        """
        Create a backup of current model data.

        Args:
            models: List of models to backup
            reason: Reason for backup (e.g., "refresh", "manual")

        Returns:
            Backup ID for restoration
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{reason}_{timestamp}"
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.json")

        backup_data = {
            "backup_id": backup_id,
            "created_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "model_count": len(models),
            "models": [self._model_to_dict(model) for model in models],
        }

        try:
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2, cls=DecimalEncoder)

            logger.info(f"Created backup {backup_id} with {len(models)} models")
            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise

    def restore_backup(self, backup_id: str) -> List[ModelInfo]:
        """
        Restore models from a backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            List of models from backup
        """
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.json")

        if not os.path.exists(backup_file):
            raise ValueError(f"Backup {backup_id} not found")

        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            models = [
                self._dict_to_model(model_dict) for model_dict in backup_data["models"]
            ]
            logger.info(f"Restored {len(models)} models from backup {backup_id}")
            return models

        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {str(e)}")
            raise

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.

        Returns:
            List of backup metadata
        """
        backups = []

        for filename in os.listdir(self.backup_dir):
            if filename.endswith(".json"):
                backup_file = os.path.join(self.backup_dir, filename)
                try:
                    with open(backup_file, "r") as f:
                        backup_data = json.load(f)

                    backups.append(
                        {
                            "backup_id": backup_data["backup_id"],
                            "created_at": backup_data["created_at"],
                            "reason": backup_data["reason"],
                            "model_count": backup_data["model_count"],
                            "file_size": os.path.getsize(backup_file),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to read backup metadata from {filename}: {str(e)}"
                    )

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Clean up old backups, keeping only the most recent ones.

        Args:
            keep_count: Number of backups to keep

        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return 0

        deleted_count = 0
        for backup in backups[keep_count:]:
            backup_file = os.path.join(self.backup_dir, f"{backup['backup_id']}.json")
            try:
                os.remove(backup_file)
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup['backup_id']}")
            except Exception as e:
                logger.warning(
                    f"Failed to delete backup {backup['backup_id']}: {str(e)}"
                )

        return deleted_count

    def _model_to_dict(self, model: ModelInfo) -> Dict[str, Any]:
        """Convert ModelInfo to dictionary for JSON serialization."""
        return {
            "provider": model.provider,
            "model_name": model.model_name,
            "display_name": model.display_name,
            "description": model.description,
            "max_context": model.max_context,
            "max_output_tokens": model.max_output_tokens,
            "supports_vision": model.supports_vision,
            "supports_function_calling": model.supports_function_calling,
            "supports_json_mode": model.supports_json_mode,
            "supports_parallel_tool_calls": model.supports_parallel_tool_calls,
            "tool_call_format": model.tool_call_format,
            "dollars_per_million_tokens_input": model.dollars_per_million_tokens_input,
            "dollars_per_million_tokens_output": model.dollars_per_million_tokens_output,
            "is_active": model.is_active,
            "source": model.source,
            "discovered_at": model.discovered_at,
            "raw_data": model.raw_data,
        }

    def _dict_to_model(self, model_dict: Dict[str, Any]) -> ModelInfo:
        """Convert dictionary to ModelInfo."""
        # Handle Decimal conversion
        cost_input = model_dict.get("dollars_per_million_tokens_input")
        cost_output = model_dict.get("dollars_per_million_tokens_output")

        if cost_input is not None:
            cost_input = Decimal(str(cost_input))
        if cost_output is not None:
            cost_output = Decimal(str(cost_output))

        # Handle datetime conversion
        discovered_at = model_dict.get("discovered_at")
        if discovered_at and isinstance(discovered_at, str):
            discovered_at = datetime.fromisoformat(discovered_at)
        elif not discovered_at:
            discovered_at = datetime.utcnow()

        return ModelInfo(
            provider=model_dict["provider"],
            model_name=model_dict["model_name"],
            display_name=model_dict.get("display_name"),
            description=model_dict.get("description"),
            max_context=model_dict.get("max_context"),
            max_output_tokens=model_dict.get("max_output_tokens"),
            supports_vision=model_dict.get("supports_vision", False),
            supports_function_calling=model_dict.get(
                "supports_function_calling", False
            ),
            supports_json_mode=model_dict.get("supports_json_mode", False),
            supports_parallel_tool_calls=model_dict.get(
                "supports_parallel_tool_calls", False
            ),
            tool_call_format=model_dict.get("tool_call_format"),
            dollars_per_million_tokens_input=cost_input,
            dollars_per_million_tokens_output=cost_output,
            is_active=model_dict.get("is_active", True),
            source=model_dict.get("source", "backup"),
            discovered_at=discovered_at,
            raw_data=model_dict.get("raw_data", {}),
        )
