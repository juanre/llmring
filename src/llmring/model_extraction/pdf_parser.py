"""PDF parser using Claude Code SDK to extract model information."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from claude_code_sdk import ClaudeCodeClient

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information extracted from documentation."""

    model_id: str
    display_name: str
    description: str
    use_cases: List[str]
    max_context: int
    max_output_tokens: Optional[int]
    supports_vision: bool
    supports_function_calling: bool
    supports_json_mode: bool
    supports_parallel_tool_calls: bool
    dollars_per_million_tokens_input: float
    dollars_per_million_tokens_output: float
    release_date: Optional[str]
    deprecation_date: Optional[str]
    notes: Optional[str]


class PDFParser:
    """Parse PDF documents to extract model information using Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize parser with Claude Code SDK."""
        self.client = ClaudeCodeClient(api_key=api_key)

    def parse_provider_docs(
        self, provider: str, pdf_paths: List[Path]
    ) -> List[ModelInfo]:
        """
        Parse PDF documents for a provider to extract model information.

        Args:
            provider: Provider name (anthropic, google, openai)
            pdf_paths: List of PDF file paths to analyze

        Returns:
            List of extracted model information
        """
        logger.info(f"Parsing {len(pdf_paths)} PDFs for {provider}")

        # Read all PDFs
        pdf_contents = []
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            logger.info(f"Reading {pdf_path}")

            # Option 1: Extract text from PDF
            try:
                import PyPDF2

                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    pdf_contents.append(f"=== {pdf_path.name} ===\n{text}")
            except ImportError:
                logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
                # Fallback: Use Claude's vision capabilities
                # Convert PDF to images and send to Claude
                pdf_contents.append(
                    f"[PDF: {pdf_path.name} - requires PyPDF2 or image conversion]"
                )

        # Create prompt for Claude
        prompt = self._create_extraction_prompt(provider, pdf_contents)

        # Call Claude to extract information
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text
            models_data = json.loads(content)

            # Convert to ModelInfo objects
            models = []
            for model_data in models_data:
                try:
                    models.append(ModelInfo(**model_data))
                except Exception as e:
                    logger.error(f"Failed to parse model data: {e}")
                    logger.error(f"Data: {model_data}")

            return models

        except Exception as e:
            logger.error(f"Failed to extract models for {provider}: {e}")
            return []

    def _create_extraction_prompt(self, provider: str, pdf_contents: List[str]) -> str:
        """Create extraction prompt for Claude."""
        return f"""Analyze the following PDF contents from {provider} documentation and extract detailed model information.

PDF Contents:
{chr(10).join(pdf_contents)}

Extract information for ALL models mentioned and return as a JSON array. For each model, include:

{{
    "model_id": "exact API model identifier",
    "display_name": "human-friendly name",
    "description": "comprehensive description of the model's capabilities and strengths",
    "use_cases": ["list", "of", "ideal", "use", "cases"],
    "max_context": context_window_tokens,
    "max_output_tokens": max_output_tokens_or_null,
    "supports_vision": boolean,
    "supports_function_calling": boolean,
    "supports_json_mode": boolean,
    "supports_parallel_tool_calls": boolean,
    "dollars_per_million_tokens_input": input_cost,
    "dollars_per_million_tokens_output": output_cost,
    "release_date": "YYYY-MM-DD or null",
    "deprecation_date": "YYYY-MM-DD or null",
    "notes": "any additional important notes or null"
}}

Important:
1. Extract EXACT model IDs as used in API calls
2. Convert all pricing to dollars per million tokens
3. Be comprehensive in descriptions - these help users choose models
4. Include specific use cases where each model excels
5. Note any deprecation dates or warnings

Return ONLY the JSON array, no other text."""
