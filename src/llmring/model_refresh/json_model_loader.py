"""Load models from JSON files instead of API discovery."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmring.model_refresh.models import ModelInfo

logger = logging.getLogger(__name__)


class JSONModelLoader:
    """Load model information from JSON files."""

    def __init__(self, models_dir: Path):
        """
        Initialize loader with models directory.

        Args:
            models_dir: Directory containing provider JSON files
        """
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            raise ValueError(f"Models directory not found: {self.models_dir}")

    def load_provider_models(self, provider: str) -> List[ModelInfo]:
        """
        Load models for a specific provider from JSON.

        Args:
            provider: Provider name (anthropic, google, openai)

        Returns:
            List of ModelInfo objects
        """
        json_path = self.models_dir / f"{provider}.json"

        if not json_path.exists():
            logger.warning(f"No JSON file found for provider {provider}")
            return []

        try:
            with open(json_path) as f:
                data = json.load(f)

            models = []
            for model_data in data.get("models", []):
                model = self._json_to_model_info(provider, model_data)
                if model:
                    models.append(model)

            logger.info(f"Loaded {len(models)} models for {provider} from {json_path}")
            return models

        except Exception as e:
            logger.error(f"Failed to load models from {json_path}: {e}")
            return []

    def load_provider_usage_hints(
        self, provider: str
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Load usage hints (model selection) for a specific provider from JSON.

        Args:
            provider: Provider name

        Returns:
            Dict mapping use case to model selection info, or None
        """
        json_path = self.models_dir / f"{provider}.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path) as f:
                data = json.load(f)

            return data.get("model_selection")

        except Exception as e:
            logger.error(f"Failed to load usage hints from {json_path}: {e}")
            return None

    def load_all_models(self) -> Dict[str, List[ModelInfo]]:
        """
        Load models for all providers.

        Returns:
            Dict mapping provider name to list of models
        """
        all_models = {}

        # Look for all JSON files in the models directory
        for json_file in self.models_dir.glob("*.json"):
            if json_file.name == "summary.json":
                continue

            provider = json_file.stem
            models = self.load_provider_models(provider)
            if models:
                all_models[provider] = models

        total_models = sum(len(models) for models in all_models.values())
        logger.info(
            f"Loaded {total_models} total models from {len(all_models)} providers"
        )

        return all_models

    def _json_to_model_info(
        self, provider: str, data: Dict[str, Any]
    ) -> Optional[ModelInfo]:
        """
        Convert JSON model data to ModelInfo object.

        Args:
            provider: Provider name
            data: Model data from JSON

        Returns:
            ModelInfo object or None if conversion fails
        """
        try:
            # Handle tool call format based on provider
            tool_call_format = None
            if data.get("supports_function_calling"):
                tool_call_format = {
                    "openai": "openai",
                    "anthropic": "anthropic",
                    "google": "google",
                }.get(provider, "openai")

            # Get dollars per million tokens from JSON
            input_cost = data.get("dollars_per_million_tokens_input")
            output_cost = data.get("dollars_per_million_tokens_output")

            return ModelInfo(
                provider=provider,
                model_name=data["model_id"],
                display_name=data.get("display_name", data["model_id"]),
                description=data.get("description", ""),
                max_context=data.get("max_context", 0),
                max_output_tokens=data.get("max_output_tokens"),
                supports_vision=data.get("supports_vision", False),
                supports_function_calling=data.get("supports_function_calling", False),
                supports_json_mode=data.get("supports_json_mode", False),
                supports_parallel_tool_calls=data.get(
                    "supports_parallel_tool_calls", False
                ),
                tool_call_format=tool_call_format,
                dollars_per_million_tokens_input=input_cost,
                dollars_per_million_tokens_output=output_cost,
                is_active=True,  # All models in JSON are considered active
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to convert JSON to ModelInfo: {e}")
            logger.error(f"Data: {data}")
            return None

    def get_model_metadata(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the model JSON file.

        Args:
            provider: Provider name

        Returns:
            Metadata dict or None
        """
        json_path = self.models_dir / f"{provider}.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path) as f:
                data = json.load(f)

            return {
                "last_updated": data.get("last_updated"),
                "source_documents": data.get("source_documents", []),
                "model_count": len(data.get("models", [])),
                "has_model_selection": "model_selection" in data,
            }

        except Exception as e:
            logger.error(f"Failed to get metadata from {json_path}: {e}")
            return None

    def load_provider_model_selection(
        self, provider: str
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Load model selection (usage hints) for a provider.

        Args:
            provider: Provider name

        Returns:
            Dict mapping use case to model selection info, or None
        """
        json_path = self.models_dir / f"{provider}.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path) as f:
                data = json.load(f)

            return data.get("model_selection")

        except Exception as e:
            logger.error(f"Failed to load model selection from {json_path}: {e}")
            return None
