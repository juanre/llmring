# LLMRing Lockfile Documentation

## Overview

The lockfile (`llmring.lock`) is the central configuration mechanism in LLMRing that maps semantic aliases to specific LLM models. It provides version control for your AI model choices, enables reproducible deployments, and supports environment-specific configurations through profiles.

## Key Concepts

### Aliases
Instead of hardcoding model IDs like `"openai:gpt-4o"` throughout your code, you use semantic aliases like `"fast"`, `"balanced"`, or `"deep"`. This allows you to:
- Change models without modifying code
- Optimize costs by switching providers
- Use different models in different environments
- Share consistent configurations across teams

### Fallback Models
Aliases can specify multiple models in priority order. If the primary model fails (rate limit, availability, etc.), LLMRing automatically tries the fallbacks:

```toml
[[bindings]]
alias = "assistant"
models = [
    "anthropic:claude-3-5-sonnet",  # Primary choice
    "openai:gpt-4o",                 # First fallback
    "google:gemini-1.5-pro"          # Second fallback
]
```

## Lockfile Format

LLMRing supports both TOML (default) and JSON formats. The format is detected by file extension.

### Basic Structure (TOML)

```toml
# Registry version pinning (optional)
[registry_versions]
openai = 12
anthropic = 8
google = 15

# Default profile bindings
[[bindings]]
alias = "fast"
models = ["openai:gpt-4o-mini"]

[[bindings]]
alias = "balanced"
models = ["anthropic:claude-3-5-sonnet", "openai:gpt-4o"]

[[bindings]]
alias = "deep"
models = ["anthropic:claude-3-5-sonnet-20241022"]

[[bindings]]
alias = "vision"
models = ["google:gemini-1.5-flash"]

# Environment-specific profiles
[profiles.dev]
[[profiles.dev.bindings]]
alias = "assistant"
models = ["openai:gpt-4o-mini"]  # Cheaper for development

[profiles.prod]
[[profiles.prod.bindings]]
alias = "assistant"
models = ["anthropic:claude-3-5-sonnet"]  # Higher quality for production
```

### JSON Format

LLMRing also supports JSON format for lockfiles. Simply use `.json` extension instead of `.lock`.

**Complete JSON Example:**

```json
{
  "registry_versions": {
    "openai": 12,
    "anthropic": 8,
    "google": 15
  },
  "bindings": [
    {
      "alias": "fast",
      "models": ["openai:gpt-4o-mini"]
    },
    {
      "alias": "balanced",
      "models": ["anthropic:claude-3-5-sonnet", "openai:gpt-4o"]
    },
    {
      "alias": "deep",
      "models": ["anthropic:claude-3-5-sonnet-20241022"]
    }
  ],
  "profiles": {
    "dev": {
      "registry_versions": {
        "openai": 12
      },
      "bindings": [
        {
          "alias": "assistant",
          "models": ["openai:gpt-4o-mini"]
        }
      ]
    },
    "prod": {
      "registry_versions": {
        "anthropic": 8,
        "openai": 12
      },
      "bindings": [
        {
          "alias": "assistant",
          "models": ["anthropic:claude-3-5-sonnet", "openai:gpt-4o"]
        }
      ]
    }
  }
}
```

**When to Use JSON vs TOML:**

| Format | Best For | Pros | Cons |
|--------|---------|------|------|
| **TOML** | Human editing, configuration | Readable, comments supported | Slightly more verbose |
| **JSON** | Programmatic generation, APIs | Machine-friendly, ubiquitous | No comments, less readable |

**Format Detection:**
- `.lock` or `.toml` → TOML format
- `.json` → JSON format
- Auto-detected by file content if extension is ambiguous

**Converting Between Formats:**

```python
from llmring.lockfile_core import Lockfile

# Load TOML
lockfile = Lockfile.load("llmring.lock")

# Save as JSON
lockfile.save("llmring.json")

# Or vice versa
lockfile = Lockfile.load("llmring.json")
lockfile.save("llmring.lock")
```

## Resolution Order

When LLMRing starts, it looks for a lockfile in this order:

1. **Explicit Path** - If `lockfile_path` parameter is provided, that file MUST exist
2. **Environment Variable** - `LLMRING_LOCKFILE_PATH` (file MUST exist if set)
3. **Current Directory** - `./llmring.lock` (if exists)
4. **Bundled Fallback** - `src/llmring/llmring.lock` (ships with packages)

**Important**: Lockfile creation is always explicit. If you specify a path (via parameter or environment variable) and the file doesn't exist, LLMRing will raise an error rather than creating it implicitly.

## Creating and Managing Lockfiles

### Basic Creation

```bash
# Create a basic lockfile with common aliases
llmring lock init

# Create at specific location
llmring lock init --path /path/to/llmring.lock
```

### Conversational Configuration (Recommended)

The most powerful way to configure your lockfile is through the MCP chat interface:

```bash
# Start conversational configuration
llmring lock chat

# Example conversation:
You: I need a configuration for a coding assistant that prioritizes accuracy
Assistant: I'll help you configure that. Based on the registry, I recommend...
```

The chat interface:
- Analyzes the current model registry for capabilities and pricing
- Recommends optimal configurations based on your needs
- Explains tradeoffs between different models
- Updates your lockfile based on the conversation

### CLI Commands

```bash
# View current bindings
llmring aliases

# Add or update an alias
llmring bind assistant "anthropic:claude-3-5-sonnet"

# Add fallback models
llmring bind assistant "anthropic:claude-3-5-sonnet,openai:gpt-4o,google:gemini-1.5-pro"

# Use different profile
llmring bind assistant "openai:gpt-4o-mini" --profile dev

# Analyze current configuration
llmring lock analyze

# Validate lockfile
llmring lock validate
```

## Profiles

Profiles allow different configurations for different environments:

### Setting the Active Profile

```python
# Via code
from llmring import LLMRing

async with LLMRing() as service:
    # Use dev profile
    response = await service.chat(request, profile="dev")
```

```bash
# Via environment variable
export LLMRING_PROFILE=dev

# Via CLI
llmring chat "Hello" --profile dev
```

### Profile Selection Priority

1. Explicit parameter (`profile="dev"` or `--profile dev`)
2. Environment variable (`LLMRING_PROFILE`)
3. Default profile (`default`)

### Common Profile Patterns

```toml
# Development: Cheaper, faster models
[profiles.dev]
[[profiles.dev.bindings]]
alias = "assistant"
models = ["openai:gpt-4o-mini"]

# Staging: Production models with fallbacks
[profiles.staging]
[[profiles.staging.bindings]]
alias = "assistant"
models = ["anthropic:claude-3-5-sonnet", "openai:gpt-4o"]

# Production: Highest quality with multiple fallbacks
[profiles.prod]
[[profiles.prod.bindings]]
alias = "assistant"
models = [
    "anthropic:claude-3-5-sonnet",
    "openai:gpt-4o",
    "google:gemini-1.5-pro"
]

# Testing: Local models
[profiles.test]
[[profiles.test.bindings]]
alias = "assistant"
models = ["ollama:llama3.2"]
```

## Packaging Lockfiles with Libraries

Libraries using LLMRing can ship with their own default lockfiles.

### Include in Package Distribution

In `pyproject.toml`:
```toml
[tool.hatch.build]
include = [
    "src/yourpackage/**/*.py",
    "src/yourpackage/**/*.lock",  # Include lockfiles
]
```

Or with setuptools in `MANIFEST.in`:
```
include src/yourpackage/*.lock
```

### Load Bundled Lockfile

```python
from pathlib import Path
from llmring import LLMRing

class YourService:
    def __init__(self, lockfile_path=None):
        if lockfile_path:
            # User-provided lockfile
            self.service = LLMRing(lockfile_path=lockfile_path)
        else:
            # Check for user's lockfile first
            if Path("llmring.lock").exists():
                self.service = LLMRing()
            else:
                # Fall back to bundled lockfile
                bundled = Path(__file__).parent / "llmring.lock"
                self.service = LLMRing(lockfile_path=str(bundled))
```

## Model Capabilities and Constraints

### Temperature Filtering

Some models don't support temperature parameters. LLMRing automatically filters out the temperature parameter for these models to prevent errors.

### Registry Integration

The lockfile can pin specific registry versions to ensure consistent model information:

```toml
[registry_versions]
openai = 12      # Use version 12 of OpenAI registry
anthropic = 8    # Use version 8 of Anthropic registry
```

## Special Aliases

LLMRing ships with a minimal bundled lockfile containing:

- **`advisor`** - Points to `anthropic:claude-opus-4-1-20250805`, used by `llmring lock chat` for intelligent configuration

## Best Practices

1. **Use Semantic Aliases**: Choose aliases that describe the use case (`coder`, `writer`, `analyzer`) rather than model properties
2. **Configure Fallbacks**: Always provide at least one fallback for production aliases
3. **Environment Profiles**: Use different profiles for dev/staging/prod
4. **Version Control**: Commit your `llmring.lock` file to ensure reproducible deployments
5. **Regular Updates**: Periodically run `llmring lock analyze` to check for better models
6. **Cost Awareness**: Use `llmring lock chat` to understand cost implications of your choices

## Troubleshooting

### Lockfile Not Found

If you see "No lockfile found", create one:
```bash
llmring lock init
```

### Invalid Alias

If an alias isn't recognized:
```bash
# Check current aliases
llmring aliases

# Add the missing alias
llmring bind myalias "provider:model"
```

### Model Not Available

If a model fails, check:
1. API key is set for that provider
2. Model name is correct (check registry)
3. Add fallback models for resilience

## Example Configurations

### Cost-Optimized Setup

```toml
[[bindings]]
alias = "default"
models = ["openai:gpt-4o-mini"]

[[bindings]]
alias = "when-needed"
models = ["anthropic:claude-3-5-haiku"]
```

### High-Reliability Setup

```toml
[[bindings]]
alias = "critical"
models = [
    "anthropic:claude-3-5-sonnet",
    "openai:gpt-4o",
    "google:gemini-1.5-pro",
    "anthropic:claude-3-5-haiku"  # Last resort
]
```

### Development Setup

```toml
[[bindings]]
alias = "local"
models = ["ollama:llama3.2"]

[[bindings]]
alias = "debug"
models = ["openai:gpt-4o-mini"]  # Cheap for testing
```
