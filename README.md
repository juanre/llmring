# LLMRing

Unified Python service to call multiple LLM providers (OpenAI, Anthropic, Google, Ollama) with one API. Optional DB logging and model registry. Built-in response caching.

## Highlights

- **One interface** for all providers
- **Optional DB** (SQLite or PostgreSQL) for logging/models
- **Built-in caching** (opt‑in, TTL, deterministic requests)
- **Model registry & costs** (optional)
- **Files/images** helpers

## Installation

```bash
# Basic installation (SQLite support only)
uv add llmring-py

# With PostgreSQL support
uv add "llmring-py[postgres]"

# Development installation (includes examples dependencies)
uv add --dev llmring-py
# Or from source:
uv pip install -e ".[dev]"
```

### Requirements
- Python 3.10+
- API keys for the LLM providers you want to use

## Quick Start

### 1) No database (just call a model)

```python
import asyncio
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

async def main():
    # Initialize service without database
    service = LLMRing(enable_db_logging=False)

    # Make a request
    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="gpt-4o-mini"
    )

    response = await service.chat(request)
    print(response.content)

asyncio.run(main())
```

### 2) With SQLite (local logging)

```python
import asyncio
from llmring.service_sqlite import LLMRingSQLite
from llmring.schemas import LLMRequest, Message

async def main():
    # Initialize with SQLite (default: llmring.db)
    service = LLMRingSQLite()

    # Or specify a custom SQLite file
    service = LLMRingSQLite(db_path="my_app.db")

    # Calls are logged to SQLite
    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="claude-3-5-haiku-20241022"
    )

    response = await service.chat(request)
    print(f"Response: {response.content}")
    print(f"Cost: ${response.usage.get('cost', 0):.4f} if tracked")

asyncio.run(main())
```

### 3) With PostgreSQL (production logging)

```python
import asyncio
from llmring.service import LLMRing

async def main():
    # Initialize with PostgreSQL
    service = LLMRing(
        db_connection_string="postgresql://user:pass@localhost/dbname"
    )

    # Use service.chat(...) as above

asyncio.run(main())
```

## Database setup (optional)

- SQLite: no setup, tables auto-created on first use.
- PostgreSQL: point `db_connection_string` to an existing DB; schema/tables are created on first use.

## Caching

Response cache is opt‑in per request and applies only to deterministic calls (temperature ≤ 0.1). You control the TTL.

```python
request = LLMRequest(
    messages=[Message(role="user", content="What is RAG?")],
    model="gpt-4o-mini",
    temperature=0.0,
    cache={"enabled": True, "ttl_seconds": 600},
)
response = await service.chat(request)
```

Notes:
- Cache key is provider‑agnostic (messages, model, format, tools, limits, temperature).
- If DB logging is on, a small DB‑backed cache is used; otherwise in‑memory.
- Anthropic additionally uses provider‑side prompt caching for the system prompt when `cache.enabled` is true.

## Model registry (optional)

When DB logging is enabled, you can query models and usage via the service:

```python
# List active models
models = await service.get_models_from_db()
for m in models:
    print(m.provider, m.model_name)

# Per-user usage (id_at_origin is your user/session ID)
stats = await service.get_usage_stats(id_at_origin="user-123", days=30)
```

## CLI (optional)

`llmring` manages initialization and the model registry.

```bash
# Initialize database schema and seed default models

# PostgreSQL (use DATABASE_URL)
export DATABASE_URL=postgresql://user:pass@localhost/dbname
llmring init-db

# SQLite
llmring --sqlite ./llmring.db init-db

# Load curated JSONs (PostgreSQL or SQLite)
llmring json-refresh

# With SQLite file
llmring --sqlite ./llmring.db json-refresh
```

## Configuration

Set env vars or a `.env` file:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...  # or GEMINI_API_KEY
OLLAMA_BASE_URL=http://localhost:11434  # Optional

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/dbname  # For PostgreSQL
# Or leave unset to use SQLite
```

### Provider selection

```python
# Explicitly specify provider
response = await service.chat(
    LLMRequest(
        messages=[Message(role="user", content="Hello")],
        model="anthropic:claude-3-5-sonnet-20241022"  # Provider prefix
    )
)

# Auto-detection also works
response = await service.chat(
    LLMRequest(
        messages=[Message(role="user", content="Hello")],
        model="gpt-4o"  # Automatically uses OpenAI
    )
)
```

## Files and images

```python
from llmring.file_utils import analyze_image

# Analyze an image
image_content = analyze_image(
    "path/to/image.png",
    "What's in this image?"
)

request = LLMRequest(
    messages=[Message(role="user", content=image_content)],
    model="gpt-4o"  # Use a vision-capable model
)

response = await service.chat(request)
```

Notes:
- When sending messages that include PDF documents with OpenAI models, the service automatically routes to the Assistants API for analysis. Tools and custom response formats are not supported in this PDF path.
- OpenAI reasoning models (`o1`, `o1-mini`) are routed via the Responses API. These do not support tools or custom response formats; attempting to use them will raise a validation error.

## Reference (minimal)

- `LLMRing` (service) → `await service.chat(LLMRequest(...))`
- `LLMRequest` fields: `messages`, `model`, optional `temperature`, `max_tokens`, `tools`, `response_format`, `cache={enabled: bool, ttl_seconds: int}`
- Provider string: `"provider:model"` or just `"model"` (auto-detected)
- Optional DB helpers: `await service.get_models_from_db()`, `await service.get_usage_stats(id_at_origin, days)`

## Initialization patterns

There are two production modes for database usage:

- Managed by llmring (recommended default)
  - PostgreSQL: `service = LLMRing(db_connection_string=os.environ["DATABASE_URL"])`. The service initializes the connection, applies migrations (creates schema/tables/functions), and seeds curated models on first use.
  - SQLite (local/dev): `service = LLMRingSQLite(db_path="llmring.db")`. Tables are created and default models inserted automatically on initialization.

- Managed by host app (pgdbm)
  - Create an `AsyncDatabaseManager` in your application and pass it in. llmring will apply migrations and seed models within the provided schema but will not own the pool.

```python
from pgdbm import AsyncDatabaseManager
from llmring.service import LLMRing

# Example: use an externally created manager (pool & config omitted here)
db_manager = AsyncDatabaseManager(..., schema="llmring")
service = LLMRing(db_manager=db_manager, origin="myapp")
# On first use, llmring will initialize and migrate the llmring schema
```

PostgreSQL migrations require the `pgcrypto` extension; the migrations enable it if missing and use `gen_random_uuid()` for primary keys.

## Development

```bash
# Clone and install
git clone https://github.com/juanreyero/llmring.git
cd llmring
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/

# Format code
uv run black src/ tests/
uv run isort src/ tests/
```

### Contributing

1. Fork the repo and create a feature branch
2. Make changes and add tests
3. Ensure tests pass and code is formatted
4. Submit a pull request

Note: The repo may contain symlinks (pgdbm, mcp-client) for local development. These are gitignored and not required.

## License

MIT

Pull requests welcome! Please ensure all tests pass and add new tests for new features.
