#!/usr/bin/env python3
"""Extract model information from provider PDFs and generate JSON files."""

import logging
import os
import sys
from pathlib import Path

from llmring.model_extraction.json_generator import JSONGenerator
from llmring.model_extraction.model_curator import CurationCriteria, ModelCurator
from llmring.model_extraction.pdf_parser import PDFParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main extraction workflow."""
    # Get paths
    project_root = Path(__file__).parent.parent.parent.parent
    res_dir = project_root / "res"
    output_dir = project_root / "data" / "models"

    # Check if res directory exists
    if not res_dir.exists():
        logger.error(f"Resource directory not found: {res_dir}")
        sys.exit(1)

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize components
    parser = PDFParser(api_key=api_key)
    curator = ModelCurator(
        CurationCriteria(
            max_models_per_provider=3, exclude_deprecated=True, prefer_newer=True
        )
    )
    generator = JSONGenerator(output_dir)

    # Define provider PDF mappings
    provider_pdfs = {
        "anthropic": [
            res_dir / "2025-06-16-anthropic-models.pdf",
            res_dir / "2025-06-16-anthropic-pricing-context.pdf",
        ],
        "google": [
            res_dir / "2025-06-16-google-models.pdf",
            res_dir / "2025-06-16-google-pricing-context.pdf",
        ],
        "openai": [
            res_dir / "2025-06-16-openai-models-pricing.pdf",
            res_dir / "2025-06-16-openai-pricing-context.pdf",
        ],
    }

    # Process each provider
    all_models = {}
    all_sources = {}

    for provider, pdf_paths in provider_pdfs.items():
        logger.info(f"\nProcessing {provider}...")

        try:
            # Parse PDFs
            extracted_models = parser.parse_provider_docs(provider, pdf_paths)
            logger.info(f"Extracted {len(extracted_models)} models from {provider}")

            # Curate models
            selected_models = curator.select_best_models(extracted_models, provider)

            # Store results
            all_models[provider] = selected_models
            all_sources[provider] = [p.name for p in pdf_paths if p.exists()]

        except Exception as e:
            logger.error(f"Failed to process {provider}: {e}")
            continue

    # Generate JSON files
    logger.info("\nGenerating JSON files...")
    generated_files = generator.generate_all_providers(all_models, all_sources)

    # Summary
    logger.info("\nExtraction complete!")
    logger.info(f"Generated files in: {output_dir}")
    for provider, path in generated_files.items():
        logger.info(f"  - {provider}: {path}")


if __name__ == "__main__":
    main()
