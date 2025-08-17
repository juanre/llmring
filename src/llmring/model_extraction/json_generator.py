"""JSON generator for creating provider model files."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .pdf_parser import ModelInfo

logger = logging.getLogger(__name__)


class JSONGenerator:
    """Generate JSON files for each provider with model information."""

    def __init__(self, output_dir: Path):
        """Initialize generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_provider_json(
        self, provider: str, models: List[ModelInfo], source_documents: List[str]
    ) -> Path:
        """
        Generate JSON file for a provider.

        Args:
            provider: Provider name
            models: List of curated models
            source_documents: List of source PDF filenames

        Returns:
            Path to generated JSON file
        """
        logger.info(f"Generating JSON for {provider} with {len(models)} models")

        # Create provider data structure
        provider_data = {
            "provider": provider,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "source_documents": source_documents,
            "models": [],
        }

        # Add models
        for model in models:
            model_dict = asdict(model)
            # Ensure proper formatting
            model_dict["dollars_per_million_tokens_input"] = round(
                model_dict["dollars_per_million_tokens_input"], 2
            )
            model_dict["dollars_per_million_tokens_output"] = round(
                model_dict["dollars_per_million_tokens_output"], 2
            )
            provider_data["models"].append(model_dict)

        # Sort models by a logical order
        provider_data["models"].sort(
            key=lambda x: (
                -x["dollars_per_million_tokens_input"],  # Most expensive first
                x["model_id"],  # Then alphabetically
            )
        )

        # Write JSON file
        output_path = self.output_dir / f"{provider}.json"
        with open(output_path, "w") as f:
            json.dump(provider_data, f, indent=2)

        logger.info(f"Generated {output_path}")

        # Validate the generated JSON
        self._validate_json(output_path)

        return output_path

    def _validate_json(self, json_path: Path) -> bool:
        """Validate generated JSON file."""
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Check required fields
            assert "provider" in data
            assert "models" in data
            assert isinstance(data["models"], list)

            # Validate each model
            for model in data["models"]:
                assert "model_id" in model
                assert "dollars_per_million_tokens_input" in model
                assert "dollars_per_million_tokens_output" in model
                assert model["dollars_per_million_tokens_input"] >= 0
                assert model["dollars_per_million_tokens_output"] >= 0

            logger.info(f"JSON validation passed for {json_path}")
            return True

        except Exception as e:
            logger.error(f"JSON validation failed for {json_path}: {e}")
            return False

    def generate_all_providers(
        self,
        provider_models: Dict[str, List[ModelInfo]],
        provider_sources: Dict[str, List[str]],
    ) -> Dict[str, Path]:
        """
        Generate JSON files for all providers.

        Args:
            provider_models: Dict mapping provider to list of models
            provider_sources: Dict mapping provider to source documents

        Returns:
            Dict mapping provider to generated JSON path
        """
        generated = {}

        for provider, models in provider_models.items():
            sources = provider_sources.get(provider, [])
            json_path = self.generate_provider_json(provider, models, sources)
            generated[provider] = json_path

        # Generate summary
        self._generate_summary(provider_models)

        return generated

    def _generate_summary(self, provider_models: Dict[str, List[ModelInfo]]) -> None:
        """Generate a summary of all models."""
        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": sum(len(models) for models in provider_models.values()),
            "providers": {},
        }

        for provider, models in provider_models.items():
            summary["providers"][provider] = {
                "model_count": len(models),
                "models": [m.model_id for m in models],
                "price_range": {
                    "input": {
                        "min": min(m.dollars_per_million_tokens_input for m in models),
                        "max": max(m.dollars_per_million_tokens_input for m in models),
                    },
                    "output": {
                        "min": min(m.dollars_per_million_tokens_output for m in models),
                        "max": max(m.dollars_per_million_tokens_output for m in models),
                    },
                },
            }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Generated summary at {summary_path}")
