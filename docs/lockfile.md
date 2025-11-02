# LLMRing Lockfile Guide

## Purpose

Each codebase that uses LLMRing owns an `llmring.lock`. This file is the authoritative source for:

- Semantic aliases and their ordered model pools
- Per-environment profiles (dev, prod, test, …)
- Pinned registry versions per provider for drift detection
- Optional metadata about the configuration

Aliases are **never** synced through the server or SaaS; they stay in version control next to your code, matching the source-of-truth v4.3 architecture.

## File Structure

A lockfile is a single TOML (default) or JSON document. The current schema is:

- `version` – lockfile format version (`"1.0"` today)
- `created_at` / `updated_at` – ISO timestamps
- `default_profile` – name of the profile used when none is supplied
- `metadata` – free-form dictionary (optional)
- `profiles` – map of profile name to profile configuration
  - `name` – repeated for clarity inside the profile
  - `registry_versions` – provider → pinned registry version integer
  - `bindings` – list of alias bindings
    - `alias` – alias string (e.g., `fast`)
    - `models` – ordered list of `provider:model` strings (fallbacks after index 0)
    - `constraints` – optional dictionary for future rules (rate limits, etc.)

### Example (TOML)

```toml
version = "1.0"
created_at = "2025-09-28T12:00:00+00:00"
updated_at = "2025-09-28T12:05:34.812341+00:00"
default_profile = "default"

[metadata]
project = "customer-support-bot"

[profiles.default]
name = "default"

[profiles.default.registry_versions]
openai = 27
anthropic = 14

[[profiles.default.bindings]]
alias = "assistant"
models = [
    "anthropic:claude-sonnet-4-5-20250929",
    "openai:gpt-4o",
]

[[profiles.default.bindings]]
alias = "vision"
models = ["google:gemini-2.5-pro"]

[profiles.dev]
name = "dev"

[profiles.dev.registry_versions]
openai = 27

[[profiles.dev.bindings]]
alias = "assistant"
models = ["openai:gpt-4o-mini"]

[profiles.test]
name = "test"

[[profiles.test.bindings]]
alias = "assistant"
models = ["ollama:llama3"]
```

### Example (JSON)

```json
{
  "version": "1.0",
  "created_at": "2025-09-28T12:00:00+00:00",
  "updated_at": "2025-09-28T12:05:34.812341+00:00",
  "default_profile": "default",
  "metadata": {
    "project": "customer-support-bot"
  },
  "profiles": {
    "default": {
      "name": "default",
      "registry_versions": {
        "openai": 27,
        "anthropic": 14
      },
      "bindings": [
        {
          "alias": "assistant",
          "models": [
            "anthropic:claude-sonnet-4-5-20250929",
            "openai:gpt-4o"
          ]
        },
        {
          "alias": "vision",
          "models": ["google:gemini-2.5-pro"]
        }
      ]
    },
    "dev": {
      "name": "dev",
      "registry_versions": {
        "openai": 27
      },
      "bindings": [
        {
          "alias": "assistant",
          "models": ["openai:gpt-4o-mini"]
        }
      ]
    },
    "test": {
      "name": "test",
      "bindings": [
        {
          "alias": "assistant",
          "models": ["ollama:llama3"]
        }
      ]
    }
  }
}
```

Both formats are interchangeable via `Lockfile.load()` and `Lockfile.save()`. TOML is the default when running CLI commands.

## Resolution Order

When `LLMRing()` starts, it resolves the lockfile in the following order:

1. `lockfile_path` parameter (must exist)
2. `LLMRING_LOCKFILE_PATH` environment variable (must exist)
3. `./llmring.lock` in the current working directory
4. Bundled fallback at `llmring/llmring.lock` inside the package (minimal advisor-only file)

Lockfiles are **never** created implicitly. If the selected path is missing, LLMRing raises `FileNotFoundError`.

## Creating and Managing Lockfiles

### Initialize with Registry Defaults

```bash
llmring lock init            # auto-detect package directory
llmring lock init --file ./config/llmring.lock
llmring lock init --file ./llmring.lock --force  # overwrite existing file
```

The command queries the registry, inspects available API keys, and seeds the default profile with useful aliases. If the registry is unreachable, it falls back to an empty skeleton.

### Conversational Editing (Recommended)

```bash
llmring lock chat
llmring lock chat --model advisor --server-url "http://localhost:8080"
```

This starts an MCP chat session that:
- Reads your existing lockfile
- Fetches provider capabilities and pricing from the registry
- Proposes alias pools and profiles based on your prompts
- Applies changes only after you confirm them

### Direct CLI Helpers

```bash
llmring aliases --profile prod
llmring bind assistant "anthropic:claude-sonnet-4-5-20250929,openai:gpt-4o" --profile prod
llmring lock validate
llmring lock bump-registry
```

- `llmring aliases` – list bindings for a profile
- `llmring bind` – add or update an alias (supports comma-separated fallbacks)
- `llmring lock validate` – confirm that every referenced model exists in the registry
- `llmring lock bump-registry` – update pinned registry versions to the latest release

### Programmatic Access

```python
from llmring.lockfile_core import Lockfile

lockfile = Lockfile.load("llmring.lock")
prod = lockfile.get_profile("prod")
prod.set_binding("assistant", ["anthropic:claude-sonnet-4-5-20250929", "openai:gpt-4o"])
lockfile.save()
```

`Lockfile.calculate_digest()` returns the SHA256 hash used in receipt signatures. `Lockfile.find_package_directory()` helps installers place lockfiles inside `src/<package>/`.

## Best Practices

1. **Keep lockfiles in version control.** Review changes the same way you review code.
2. **Use semantic aliases everywhere.** Only fall back to direct `provider:model` strings for one-off experiments.
3. **Provide fallbacks.** Ordered pools protect you from outages and missing API keys.
4. **Pin registry versions.** `llmring lock bump-registry` helps you detect pricing/model drift intentionally.
5. **Per-profile discipline.** Use dev/staging/prod/test profiles instead of separate lockfiles.
6. **No secrets inside.** Lockfiles do not store API keys; use environment variables or secret managers.

By following these guidelines, contributors and automated agents always know which models are in play and can audit changes via Git history. This keeps the alias-first workflow aligned with the LLMRing source of truth.
