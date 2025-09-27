# Lockfile Packaging and Resolution Strategy

## Overview

LLMRing uses a hierarchical lockfile resolution strategy that allows packages to ship with their own lockfiles while users can override with their own configurations. This document describes how lockfiles are resolved, packaged, and distributed.

## Lockfile Resolution Order

When `LLMRing` is initialized without an explicit lockfile path, it searches for lockfiles in the following order:

1. **Explicit Path Parameter** - If `lockfile_path` is provided to `LLMRing()`, that file MUST exist
2. **Environment Variable** - `LLMRING_LOCKFILE_PATH` environment variable (file MUST exist)
3. **Current Directory** - `./llmring.lock` in the current working directory
4. **Bundled Fallback** - Package's bundled `src/llmring/llmring.lock` (ships with llmring)

The first lockfile found is used. If none are found, the service continues without a lockfile (some operations may fail).

## Key Principles

### Explicit Lockfile Creation

**Lockfile creation is always explicit** - services never create lockfiles implicitly. This ensures:

- Clear separation between "reading" and "creating" operations
- No unexpected file creation in user directories
- Predictable behavior across environments
- Tests must be intentional about their setup

If a lockfile path is specified (via parameter or environment variable) but the file doesn't exist, `LLMRing` will raise a `FileNotFoundError`. This prevents accidental lockfile creation in unexpected locations.

## Packaging Your Lockfile

### For Library Authors

Libraries that use LLMRing can ship with their own lockfiles to provide default configurations:

1. **Create your lockfile** at the package root or in your source directory:
   ```bash
   llmring lock init
   ```

2. **Configure your aliases** for your library's needs:
   ```bash
   llmring bind fast openai:gpt-4o-mini
   llmring bind deep anthropic:claude-3-5-sonnet
   ```

3. **Include the lockfile in your package distribution** by updating `pyproject.toml`:
   ```toml
   [tool.hatch.build]
   include = [
       "src/yourpackage/**/*.py",   # Your Python files
       "src/yourpackage/**/*.lock",  # Include lockfiles
   ]
   ```

   Or if using setuptools, add to `MANIFEST.in`:
   ```
   include src/yourpackage/*.lock
   ```

4. **Load your bundled lockfile** in your code:
   ```python
   from pathlib import Path
   from llmring import LLMRing

   # Get path to your bundled lockfile
   package_lockfile = Path(__file__).parent / "llmring.lock"

   # Use it explicitly
   service = LLMRing(lockfile_path=str(package_lockfile))
   ```

### For Application Authors

Applications typically want users to create their own lockfiles:

1. **Check for user lockfile** and guide them to create one if missing:
   ```python
   from pathlib import Path
   from llmring import LLMRing

   if not Path("llmring.lock").exists():
       print("No lockfile found. Create one with: llmring lock init")
       exit(1)

   service = LLMRing()  # Will use ./llmring.lock
   ```

2. **Use project root discovery** to find the best location:
   ```python
   from llmring.lockfile_core import Lockfile

   project_root = Lockfile.find_project_root()
   if project_root:
       lockfile_path = project_root / "llmring.lock"
   else:
       lockfile_path = Path("llmring.lock")
   ```

## LLMRing's Bundled Lockfile

LLMRing itself ships with a minimal bundled lockfile (`src/llmring/llmring.lock`) that contains:

- The `advisor` alias pointing to `anthropic:claude-opus-4-1-20250805`
- Used by `llmring lock chat` for intelligent lockfile creation

This ensures that `llmring lock chat` works out of the box for users who haven't created their own lockfile yet.

## Lockfile Formats

LLMRing supports both TOML and JSON formats:

- **TOML** (`.lock`) - Default format, human-readable
- **JSON** (`.lock.json`) - Machine-readable, useful for programmatic generation

The format is detected by file extension. Both formats contain the same data structure.

## Environment-Specific Configuration

Use environment variables to override lockfile location per environment:

```bash
# Development
export LLMRING_LOCKFILE_PATH=./dev.lock

# Production
export LLMRING_LOCKFILE_PATH=/etc/myapp/llmring.lock

# Testing
export LLMRING_LOCKFILE_PATH=/tmp/test.lock
```

## Testing Considerations

Tests should always create lockfiles explicitly:

```python
def test_with_lockfile(tmp_path):
    # Create lockfile explicitly
    lockfile_path = tmp_path / "llmring.lock"
    lockfile = Lockfile.create_default()
    lockfile.set_binding("test", "openai:gpt-4o-mini")
    lockfile.save(lockfile_path)

    # Now use it
    service = LLMRing(lockfile_path=str(lockfile_path))
    # ... test code ...
```

A test fixture is available in `tests/conftest.py`:

```python
def test_something(create_test_lockfile):
    lockfile_path = create_test_lockfile(
        bindings={"fast": "openai:gpt-4o-mini"}
    )
    service = LLMRing(lockfile_path=str(lockfile_path))
    # ... test code ...
```

## Project Root Discovery

The `find_project_root()` function searches for project indicators in parent directories:

1. `pyproject.toml`
2. `setup.py`
3. `setup.cfg`
4. `.git`

This helps tools like `llmring lock chat` create lockfiles in the appropriate location for packaging.

## Best Practices

1. **Libraries** should ship with minimal lockfiles containing only essential aliases
2. **Applications** should guide users to create their own lockfiles
3. **Tests** should always create lockfiles explicitly, never rely on implicit creation
4. **CI/CD** should use environment variables to specify lockfile paths
5. **Docker** images should include lockfiles at known paths (e.g., `/app/llmring.lock`)

## Migration Guide

If you're updating from an older version where lockfiles were created implicitly:

1. Ensure all lockfile paths are created explicitly before use
2. Update tests to create lockfiles in setup/fixtures
3. Add error handling for missing lockfiles
4. Consider shipping a default lockfile with your package

## Example: Complete Setup

Here's a complete example for a package that ships with its own lockfile:

```python
# src/mypackage/__init__.py
from pathlib import Path
from llmring import LLMRing
from llmring.lockfile_core import Lockfile

class MyService:
    def __init__(self, lockfile_path=None):
        if lockfile_path:
            # User-provided lockfile
            self.service = LLMRing(lockfile_path=lockfile_path)
        else:
            # Try to find user's lockfile first
            if Path("llmring.lock").exists():
                self.service = LLMRing()
            else:
                # Fall back to our bundled lockfile
                bundled = Path(__file__).parent / "llmring.lock"
                self.service = LLMRing(lockfile_path=str(bundled))
```

This pattern gives users flexibility while ensuring your package works out of the box.
