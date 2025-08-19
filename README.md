# LLMRing

Alias-first LLM service for Python. Map tasks to models, not code to model IDs. Supports OpenAI, Anthropic, Google, and Ollama with a unified interface.

## Highlights

- **Alias-first identity**: Map semantic tasks to models via lockfile
- **Lockfile-based configuration**: Version-controlled, reproducible model bindings
- **Multi-provider support**: OpenAI, Anthropic, Google, Ollama
- **Profile support**: Different configurations for prod/staging/dev
- **Registry integration**: Track model changes and detect drift
- **Local-first**: Fully functional without backend services
- **Receipt generation**: Ed25519-signed receipts for usage tracking (when connected to server)

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

# This creates llmring.lock with default aliases like:
# - default → ollama:llama3
# - summarizer → anthropic:claude-3-haiku
# - deep → openai:gpt-4
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

## CLI Usage

```bash
# Chat with a model (using alias or direct reference)
llmring chat "What is the capital of France?" --model summarizer

# List available providers
llmring providers

# Show model information
llmring info openai:gpt-4

# List available models
llmring list --provider openai
```

## Provider Configuration

Set API keys via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...  # or GEMINI_API_KEY
# Ollama doesn't require an API key (local)
```

## Advanced Usage

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