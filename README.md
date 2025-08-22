# LLMRing

Alias-first LLM service for Python. Map tasks to models, not code to model IDs. Supports OpenAI, Anthropic, Google, and Ollama with a unified interface.

## Highlights

- **Alias-first identity**: Map semantic tasks to models via lockfile
- **Lockfile-based configuration**: Version-controlled, reproducible model bindings  
- **Multi-provider support**: OpenAI, Anthropic, Google, Ollama
- **Profile support**: Different configurations for prod/staging/dev
- **Registry integration**: Automatic model capabilities and pricing from GitHub Pages
- **Local-first**: Fully functional without backend services
- **Smart defaults**: Auto-detects API keys and suggests appropriate models
- **Cost tracking**: Automatic cost calculation based on registry pricing

## Installation

```bash
# Basic installation
pip install llmring

# Or with uv
uv add llmring

# Development installation (from source)
uv pip install -e ".[dev]"
```

### Requirements
- Python 3.10+
- API keys for the LLM providers you want to use

## Quick Start

### 1) Initialize a lockfile

```bash
# Create a lockfile with auto-detected defaults based on available API keys
llmring lock init

# This creates llmring.lock with smart defaults based on your API keys:
# If OPENAI_API_KEY is set:
#   - long_context → openai:gpt-4-turbo-preview
#   - low_cost → openai:gpt-3.5-turbo
# If ANTHROPIC_API_KEY is set:
#   - deep → anthropic:claude-3-opus-20240229
#   - balanced → anthropic:claude-3-sonnet-20240229
# Always available:
#   - default → ollama:llama3
```

### 2) Bind aliases to models

```bash
# Bind an alias to a specific model
llmring bind summarizer ollama:llama3.3

# List all aliases
llmring aliases
```

### 3) Use aliases in code

```python
import asyncio
from llmring import LLMRing, LLMRequest, Message

async def main():
    # Initialize service
    service = LLMRing()
    
    # Use an alias instead of hardcoding model names
    request = LLMRequest(
        messages=[Message(role="user", content="Summarize this text...")],
        model="summarizer"  # Uses the alias from lockfile
    )
    
    response = await service.chat(request)
    print(response.content)

asyncio.run(main())
```

### 4) Direct model usage (without aliases)

```python
# You can still use provider:model format directly
request = LLMRequest(
    messages=[Message(role="user", content="Hello!")],
    model="openai:gpt-4o-mini"  # Direct model reference
)
```

## Lockfile Configuration

The `llmring.lock` file is the authoritative configuration source:

```toml
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
provider = "ollama"
model = "llama3.3"

[[profiles.default.bindings]]
alias = "deep"
provider = "anthropic"
model = "claude-3-opus"

[profiles.prod]
name = "prod"
# Production-specific bindings...

[profiles.dev]
name = "dev"
# Development-specific bindings...
```

## Profiles

Switch between different configurations using profiles:

```bash
# Use a specific profile
llmring chat "Hello" --model summarizer --profile prod

# Or via environment variable
export LLMRING_PROFILE=prod
llmring chat "Hello" --model summarizer
```

## Registry Integration

Track model changes and detect drift:

```bash
# Validate lockfile against current registry
llmring lock validate

# Update registry versions to latest
llmring lock bump-registry
```

## CLI Reference

### Lockfile Management

```bash
# Initialize lockfile with defaults
llmring lock init [--force]

# Validate lockfile bindings against registry
llmring lock validate

# Update pinned registry versions
llmring lock bump-registry
```

### Alias Management

```bash
# Bind an alias to a model
llmring bind <alias> <provider:model> [--profile <profile>]

# Remove an alias
llmring unbind <alias> [--profile <profile>]

# List all aliases
llmring aliases [--profile <profile>]
```

### Chat & Model Usage

```bash
# Send a chat message (supports aliases)
llmring chat "Your message" --model <alias_or_model> [options]
  --system <prompt>      # System prompt
  --temperature <float>  # Temperature (0.0-2.0)
  --max-tokens <int>     # Max tokens to generate
  --profile <profile>    # Profile for alias resolution
  --json                 # Output as JSON
  --verbose              # Show usage stats

# Show model information
llmring info <provider:model> [--json]

# List available models
llmring list [--provider <provider>]

# List configured providers
llmring providers [--json]
```

## Provider Configuration

Set API keys via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...  # or GEMINI_API_KEY
# Ollama doesn't require an API key (local)
```

## Cost Tracking

Track costs automatically using registry pricing data:

```python
# After any API call
response = await service.chat("summarizer", messages)

# Calculate cost from response
cost = await service.calculate_cost(response)

if cost:
    print(f"Cost: ${cost['total_cost']:.4f}")
    print(f"Tokens: {response.total_tokens}")
```

## Advanced Usage

### Constraints in Lockfile

Apply model constraints through the lockfile:

```toml
[[profiles.default.bindings]]
alias = "creative_writer"
provider = "openai"
model = "gpt-4"
constraints = { temperature = 0.9, max_tokens = 2000 }

[[profiles.default.bindings]]
alias = "code_reviewer"  
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
constraints = { temperature = 0.2 }
```

### Convenience Method for Aliases

```python
# Use the convenience method for simpler alias-based chat
async def main():
    service = LLMRing()
    
    # Direct alias usage without creating a request object
    response = await service.chat_with_alias(
        "summarizer",  # Alias or model string
        messages=[{"role": "user", "content": "Summarize quantum computing"}],
        temperature=0.5,
        max_tokens=200,
        profile="prod"  # Optional profile
    )
    print(response.content)
```

### Programmatic Alias Management

```python
# Manage aliases from code
service = LLMRing()

# Bind an alias
service.bind_alias("translator", "openai:gpt-4o", profile="default")

# List aliases
aliases = service.list_aliases(profile="default")
print(aliases)  # {'translator': 'openai:gpt-4o', ...}

# Resolve an alias
model = service.resolve_alias("translator")
print(model)  # 'openai:gpt-4o'
```

### Working with Files and Images

```python
from llmring.file_utils import create_image_content, analyze_image

# Analyze an image
image_content = create_image_content("path/to/image.png")
messages = [
    Message(role="user", content=[
        {"type": "text", "text": "What's in this image?"},
        image_content
    ])
]

response = await analyze_image(
    service, 
    "path/to/image.png",
    "Describe this image",
    model="openai:gpt-4o"  # Or use an alias
)
```

### Custom System Prompts

```python
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!")
]

request = LLMRequest(
    messages=messages,
    model="summarizer",
    temperature=0.7,
    max_tokens=1000
)
```

## Security

### API Key Management

LLMRing never stores API keys in files. Keys are only read from environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

**Best practices:**
- Use `.env` files locally (never commit them)
- Use secrets management in production (AWS Secrets Manager, Vault, etc.)
- The lockfile is safe to commit - it contains no secrets

For technical details and security specifications, see [docs/technical.md](docs/technical.md).

## Architecture

LLMRing follows an alias-first, lockfile-based architecture:

1. **Lockfile (`llmring.lock`)**: The authoritative configuration source containing alias→model bindings, profiles, and registry versions
2. **Registry**: Public model information hosted on GitHub Pages for drift detection
3. **Service**: Lightweight routing layer that resolves aliases and forwards to providers
4. **Receipts**: Optional Ed25519-signed receipts when connected to server/SaaS

The system is designed to be:
- **Local-first**: Fully functional without backend services
- **Version-controlled**: Lockfile can be committed for reproducible deployments
- **Drift-aware**: Detects when models change between registry versions

## License

MIT

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## Support

For issues and questions, please use the GitHub issue tracker.
