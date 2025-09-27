"""
Common utilities for CLI commands to reduce duplication.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

from llmring.constants import (
    ERROR_CREATE_LOCKFILE,
    ERROR_NO_LOCKFILE,
    ERROR_PREFIX,
    INFO_PREFIX,
    LOCKFILE_NAME,
    SUCCESS_PREFIX,
    WARNING_PREFIX,
)
from llmring.lockfile_core import Lockfile


def format_error(message: str) -> str:
    """Format an error message with consistent style."""
    if not message.startswith(ERROR_PREFIX):
        message = f"{ERROR_PREFIX}{message}"
    # Remove trailing period for consistency
    if message.endswith("."):
        message = message[:-1]
    return message


def format_success(message: str) -> str:
    """Format a success message with consistent style."""
    if not message.startswith(SUCCESS_PREFIX):
        message = f"{SUCCESS_PREFIX}{message}"
    return message


def format_warning(message: str) -> str:
    """Format a warning message with consistent style."""
    if not message.startswith(WARNING_PREFIX):
        message = f"{WARNING_PREFIX}{message}"
    return message


def format_info(message: str) -> str:
    """Format an info message with consistent style."""
    if not message.startswith(INFO_PREFIX):
        message = f"{INFO_PREFIX}{message}"
    return message


def load_lockfile_or_exit(
    require_exists: bool = True, path: Optional[Path] = None
) -> Tuple[Path, Optional[Lockfile]]:
    """
    Load a lockfile with consistent error handling.

    Args:
        require_exists: If True, exit if lockfile doesn't exist
        path: Optional specific path to lockfile

    Returns:
        Tuple of (lockfile_path, lockfile or None)

    Note:
        This function may call sys.exit(1) if require_exists=True and no lockfile found
    """
    if path:
        lockfile_path = path
    else:
        # Find the package directory where lockfile should be
        package_dir = Lockfile.find_package_directory()
        if package_dir:
            lockfile_path = package_dir / LOCKFILE_NAME
        else:
            # Fall back to current directory if no package found
            lockfile_path = Path(LOCKFILE_NAME)

    if require_exists and not lockfile_path.exists():
        print(format_error(ERROR_NO_LOCKFILE))
        print(ERROR_CREATE_LOCKFILE)
        sys.exit(1)

    if lockfile_path.exists():
        lockfile = Lockfile.load(lockfile_path)
    else:
        # Create default lockfile if it doesn't exist and not required
        lockfile = Lockfile.create_default()

    return lockfile_path, lockfile


def print_aliases(lockfile: Lockfile, profile: Optional[str] = None) -> None:
    """
    Print aliases in a consistent format, showing model pools.

    Args:
        lockfile: The lockfile to print aliases from
        profile: Optional specific profile to use
    """
    profile_config = lockfile.get_profile(profile)

    if profile_config.bindings:
        print(f"\nAliases in profile '{profile_config.name}':")
        for binding in profile_config.bindings:
            # Check if we have multiple models (model pool)
            if binding.models and len(binding.models) > 1:
                print(f"  {binding.alias} → {binding.models[0]}")
                print(f"      alternatives: {', '.join(binding.models[1:])}")
            else:
                # Single model or legacy format
                print(f"  {binding.alias} → {binding.model_ref}")
    else:
        print(f"\nNo aliases configured in profile '{profile_config.name}'")


def print_packaging_guidance(project_root: Path) -> None:
    """
    Print guidance for including lockfile in package distribution.

    Args:
        project_root: Path to the project root
    """
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        print("\n⚠️  To include this lockfile in your package distribution:")
        print("\nAdd to your pyproject.toml:")
        print("")
        print("  [tool.hatch.build]  # or similar for your build system")
        print("  include = [")
        print('      "src/yourpackage/**/*.py",  # your existing patterns')
        print('      "src/yourpackage/**/*.lock",  # add this line')
        print("  ]")
        print("")
        print("Or if using setuptools with setup.py, add to MANIFEST.in:")
        print("  include src/yourpackage/*.lock")
