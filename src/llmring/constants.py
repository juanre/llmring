"""
Common constants used throughout the LLMRing package.
"""

# Lockfile constants
LOCKFILE_NAME = "llmring.lock"
LOCKFILE_JSON_NAME = "llmring.lock.json"
DEFAULT_PROFILE = "default"

# Project root indicators (checked in order)
PROJECT_ROOT_INDICATORS = ["pyproject.toml", "setup.py", "setup.cfg", ".git"]

# Error message formats
ERROR_NO_LOCKFILE = "Error: No llmring.lock found in current directory."
ERROR_CREATE_LOCKFILE = "Run 'llmring lock init' to create one."

# Success message prefixes
SUCCESS_PREFIX = "✅ "
ERROR_PREFIX = "Error: "
WARNING_PREFIX = "⚠️  "
INFO_PREFIX = "💡 "
