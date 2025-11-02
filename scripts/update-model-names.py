#!/usr/bin/env python3
"""
Update old model names to current ones in documentation and skills.

Run with: python scripts/update-model-names.py
"""

import re
from pathlib import Path

# Model name mappings (old → new)
# Based on llmring list output
MODEL_UPDATES = {
    # Anthropic updates
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-5-20250929",
    "claude-3-5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805": "claude-opus-4-1-20250805",  # Still current
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",  # Still current
    # Google updates
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",  # Already current
    # OpenAI - most are still current
    "gpt-4o": "gpt-4o",  # Still current
    "gpt-4o-mini": "gpt-4o-mini",  # Still current
    "gpt-5-2025-08-07": "gpt-5-2025-08-07",  # Still current
}


def update_file(file_path: Path):
    """Update model names in a file."""
    if not file_path.exists():
        return

    content = file_path.read_text()
    original = content

    # Replace each old model with new
    for old_model, new_model in MODEL_UPDATES.items():
        if old_model != new_model:
            content = content.replace(old_model, new_model)

    # Write if changed
    if content != original:
        file_path.write_text(content)
        print(f"Updated: {file_path}")
        return True
    return False


def main():
    """Update all docs and skills."""
    root = Path(__file__).parent.parent

    # Files to update
    patterns = [
        root / "README.md",
        root / "skills/**/*.md",
        root / "docs/**/*.md",
        root / "examples/**/*.py",
    ]

    updated_count = 0

    # Update README
    if update_file(root / "README.md"):
        updated_count += 1

    # Update skills
    for skill_file in (root / "skills").glob("**/SKILL.md"):
        if update_file(skill_file):
            updated_count += 1

    # Update docs
    for doc_file in (root / "docs").glob("**/*.md"):
        if update_file(doc_file):
            updated_count += 1

    # Update examples
    for example_file in (root / "examples").glob("**/*.py"):
        if update_file(example_file):
            updated_count += 1

    print(f"\nUpdated {updated_count} files")


if __name__ == "__main__":
    main()
