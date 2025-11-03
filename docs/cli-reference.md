# CLI Reference

## Overview

The `llmring` command-line interface provides comprehensive tools for managing models, aliases, lockfiles, and interacting with LLMs directly from the terminal.

## Global Options

All commands support these common patterns:

```bash
# Profile selection (environment-specific configs)
export LLMRING_PROFILE=dev
llmring <command>

# Or specify inline
llmring <command> --profile dev

# Custom lockfile path
export LLMRING_LOCKFILE_PATH=/path/to/llmring.lock
llmring <command>
```

---

## Lockfile Management

### `llmring lock init`

Initialize a new lockfile with registry-based defaults.

```bash
# Create in package directory (auto-detected)
llmring lock init

# Create at specific location
llmring lock init --file /path/to/llmring.lock

# Overwrite existing
llmring lock init --force
```

**Options:**
- `--file PATH` - Lockfile path (default: auto-detect package directory)
- `--force` - Overwrite existing lockfile

**Example output:**
```
Creating lockfile with registry-based defaults...
✅ Created lockfile with registry data
✅ Created lockfile: /path/to/llmring.lock

Default aliases:
  fast → openai:gpt-4o-mini
  balanced → anthropic:claude-sonnet-4-5-20250929
  deep → anthropic:claude-sonnet-4-5-20250929

💡 Use 'llmring lock chat' for conversational lockfile management
```

---

### `llmring lock chat`

Conversational lockfile management with AI advisor.

```bash
# Start interactive configuration
llmring lock chat

# Use specific model
llmring lock chat --model advisor

# Connect to custom MCP server
llmring lock chat --server-url "http://localhost:8080"
```

**Options:**
- `--server-url URL` - MCP server URL (default: embedded server)
- `--model ALIAS` - LLM model for conversation (default: advisor)

**Features:**
- Natural language configuration
- Registry-based recommendations
- Cost analysis and tradeoffs
- Multi-model fallback setup
- Profile configuration

**Example session:**
```
You: I need a fast, cheap model for simple tasks
Assistant: I recommend binding 'fast' to openai:gpt-4o-mini. It's 10x cheaper
         than GPT-4o and perfect for simple tasks.

         Would you like me to add this alias?

You: yes, and add a fallback
Assistant: I'll add anthropic:claude-3-haiku as a fallback.
         [Calling tool: add_alias]
         Added alias 'fast' with models: openai:gpt-4o-mini, anthropic:claude-3-haiku
```

See [MCP Integration](mcp.md) for more details on conversational configuration.

---

### `llmring lock validate`

Validate lockfile bindings against registry.

```bash
llmring lock validate
```

**Output:**
```
Validating lockfile bindings...

Profile 'default':
  ✅ fast → openai:gpt-4o-mini
  ✅ balanced → anthropic:claude-sonnet-4-5-20250929
  ❌ old-model → openai:gpt-3.5-turbo-0301

✅ All bindings are valid
```

---

### `llmring lock bump-registry`

Update pinned registry versions to latest.

```bash
llmring lock bump-registry
```

**Output:**
```
Updating registry versions...
  openai: v12 → v13
  anthropic: v8 (unchanged)
  google: v15 → v16

✅ Updated llmring.lock
```

---

## Alias Management

### `llmring bind`

Bind an alias to one or more models.

```bash
# Single model
llmring bind assistant "anthropic:claude-sonnet-4-5-20250929"

# Multiple models with fallbacks
llmring bind assistant "anthropic:claude-sonnet-4-5-20250929,openai:gpt-4o,google:gemini-2.5-pro"

# Specific profile
llmring bind assistant "openai:gpt-4o-mini" --profile dev
```

**Options:**
- `--profile PROFILE` - Profile to use (default: default)

**Example output:**
```
✅ Bound 'assistant' → anthropic:claude-sonnet-4-5-20250929 in profile 'default'
   Fallbacks: openai:gpt-4o, google:gemini-2.5-pro
```

---

### `llmring aliases`

List all aliases from lockfile.

```bash
# List default profile
llmring aliases

# List specific profile
llmring aliases --profile dev

# With environment variable
export LLMRING_PROFILE=prod
llmring aliases
```

**Options:**
- `--profile PROFILE` - Profile to list (default: default or LLMRING_PROFILE)

**Example output:**
```
Aliases in profile 'default':
  fast → openai:gpt-4o-mini
  balanced → anthropic:claude-sonnet-4-5-20250929, openai:gpt-4o
  deep → anthropic:claude-sonnet-4-5-20250929
  vision → google:gemini-2.5-flash
```

---

## Model Information

### `llmring list`

List available models from all providers.

```bash
# List all models
llmring list

# Filter by provider
llmring list --provider openai
llmring list --provider anthropic
```

**Options:**
- `--provider PROVIDER` - Filter by provider (openai, anthropic, google, ollama)

**Example output:**
```
Available Models:

OPENAI:
  - gpt-4o
  - gpt-4o-mini
  - gpt-4-turbo
  - o1-preview
  - o1-mini

ANTHROPIC:
  - claude-sonnet-4-5-20250929
  - claude-3-5-haiku
  - claude-3-opus
```

---

### `llmring info`

Show detailed information about a specific model.

```bash
# Using alias
llmring info fast

# Using provider:model reference
llmring info openai:gpt-4o

# JSON output
llmring info balanced --json
```

**Options:**
- `--json` - Output as JSON

**Example output:**
```
Model: gpt-4o-mini
Provider: openai
Supported: True
Display Name: GPT-4o Mini
Description: Fast and cost-effective model
Max Input: 128,000 tokens
Max Output: 16,384 tokens
Input Cost: $0.15/M tokens
Output Cost: $0.60/M tokens
Supports: Vision
Supports: Function Calling
Supports: JSON Mode
```

---

### `llmring providers`

List configured providers and their status.

```bash
# List providers
llmring providers

# JSON output
llmring providers --json
```

**Options:**
- `--json` - Output as JSON

**Example output:**
```
Configured Providers:
----------------------------------------
✓ openai       OPENAI_API_KEY
✓ anthropic    ANTHROPIC_API_KEY
✗ google       GOOGLE_API_KEY or GEMINI_API_KEY
✗ ollama       (not required)
```

---

## Chat Commands

### `llmring chat`

Send a chat message directly from terminal.

```bash
# Basic chat
llmring chat "What is 2+2?"

# With system prompt
llmring chat "Write a haiku" --system "You are a poet"

# Specific model
llmring chat "Explain quantum physics" --model deep

# With parameters
llmring chat "Count to 10" --temperature 0.5 --max-tokens 100

# JSON output
llmring chat "Hello" --json

# Verbose (show usage stats)
llmring chat "Hello" --verbose

# Streaming
llmring chat "Tell me a story" --stream

# With profile
llmring chat "Hello" --profile dev
```

**Options:**
- `--model ALIAS` - Model alias or provider:model (default: fast)
- `--system TEXT` - System prompt
- `--temperature FLOAT` - Temperature 0.0-2.0 (default: 0.7)
- `--max-tokens INT` - Maximum tokens to generate
- `--json` - Output as JSON
- `--verbose` - Show additional information
- `--stream` - Stream response in real-time
- `--profile PROFILE` - Profile for alias resolution

**Example output:**
```
4

[Model: gpt-4o-mini]
[Tokens: 8 in, 1 out]
[Cost: $0.000002]
```

---

## Usage and Cost Tracking

### `llmring stats`

Show usage statistics (local).

```bash
# Basic stats
llmring stats

# Verbose (show recent requests)
llmring stats --verbose

# JSON output
llmring stats --json
```

**Options:**
- `--verbose` - Show detailed statistics
- `--json` - Output as JSON

**Example output:**
```
Local usage statistics (15 requests):
----------------------------------------
Total requests: 15
Total tokens: 23,450
Total cost: $0.045210

Recent requests:
  2024-01-15 10:30: fast → openai:gpt-4o-mini ($0.000123)
  2024-01-15 10:28: balanced → anthropic:claude-sonnet-4-5-20250929 ($0.012450)
```

**Note:** Full statistics require server connection.

---

## Cache Management

### `llmring cache info`

Show registry cache information.

```bash
llmring cache info
```

**Example output:**
```
📁 Cache directory: /Users/user/.cache/llmring/registry
⏱️  Cache TTL: 24 hours

📄 Cached files (3 total):
  • openai_models.json: 2.5h old (✅ valid)
  • anthropic_models.json: 5.2h old (✅ valid)
  • google_models.json: 26.8h old (❌ stale)

💾 Total cache size: 45.3 KB
```

---

### `llmring cache clear`

Clear the registry cache.

```bash
llmring cache clear
```

**Example output:**
```
✅ Registry cache cleared successfully
Next model lookups will fetch fresh data from the registry
```

**When to use:**
- After registry updates
- To force fresh model data
- When troubleshooting model information issues

---

## Server Management

### `llmring server init`

Bootstrap local environment variables for a running llmring-server instance.

```bash
# With docker compose running in ../llmring-server
llmring server init --env-file .env.llmring
source .env.llmring
```

**Options:**
- `--server URL` – Override server URL (default: detect from env or http://localhost:8000)
- `--api-key KEY` – Provide an explicit API key (default: auto-generate)
- `--env-file PATH` – Output path for env file (default: ./.env.llmring)
- `--force` – Overwrite existing env file
- `--skip-health-check` – Skip connectivity validation (useful if the server is not running yet)

### `llmring server status`

Display current server configuration and run health checks.

```bash
llmring server status --env-file .env.llmring
```

Output includes the detected server URL, API key presence, health endpoint response, and API key validation against `/api/v1/stats`.

### `llmring server key create`

Generate a new API key. Use `--update-env` to write it into an env file.

```bash
llmring server key create --update-env --env-file .env.llmring
```

### `llmring server key rotate`

Rotate the key stored in the env file and export it to the current shell.

```bash
llmring server key rotate --env-file .env.llmring
source .env.llmring
```

### `llmring server key list`

Show active key information from the current environment and env file.

```bash
llmring server key list --env-file .env.llmring
```

---

### `llmring server stats`

Summarise usage for the current API key. By default output is human-readable; add `--json` for structured output.

```bash
llmring server stats --group-by week --json
```

**Options:**
- `--start-date`, `--end-date` – ISO timestamps or dates (defaults: last 30 days)
- `--group-by` – `day | week | month`
- `--json` – Emit raw JSON

### `llmring server logs`

Fetch raw usage log entries. Supports table (default), JSON, or CSV output.

```bash
# Recent entries in table form
llmring server logs --limit 20

# Export as CSV
llmring server logs --output csv --output-file usage.csv
```

**Options:**
- `--limit`, `--offset` – Pagination controls (limit capped at 500)
- `--alias`, `--model`, `--origin` – Filters
- `--start-date`, `--end-date` – ISO timestamps or dates
- `--output [table|json|csv]` – Output format (`--json` is shorthand)
- `--output-file` – Write JSON/CSV output to file instead of stdout

### `llmring server conversations`

List stored conversations or inspect a single conversation with message history.

```bash
llmring server conversations
llmring server conversations --conversation-id 1234-uuid --messages --json
```

**Options:**
- `--limit`, `--offset` – Pagination for listing mode
- `--conversation-id` – Retrieve a specific conversation
- `--messages` – Include message history when fetching a single conversation
- `--json` – Emit raw JSON (list or conversation detail)

---

## Server Integration

### `llmring register`

Register with LLMRing server (for SaaS features).

```bash
llmring register --email user@example.com --org "My Company"
```

**Options:**
- `--email EMAIL` - Email address for registration
- `--org ORG` - Organization name

**Current status:**
```
⚠️  The 'register' command requires a server connection.
This feature is not yet available in the local-only version.

LLMRing SaaS features coming soon:
  • Central binding management
  • Usage analytics and cost tracking
  • Team collaboration
```

---

## Environment Variables

LLMRing CLI respects these environment variables:

### Provider API Keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...
export GEMINI_API_KEY=AIza...  # Alternative for Google
export OLLAMA_BASE_URL=http://localhost:11434  # Default for Ollama
```

### LLMRing Configuration

```bash
# Lockfile path (must exist)
export LLMRING_LOCKFILE_PATH=/path/to/llmring.lock

# Active profile
export LLMRING_PROFILE=dev

# Remote URL access (security)
export LLMRING_ALLOW_REMOTE_URLS=true
```

---

## Complete Workflow Examples

### Setup New Project

```bash
# 1. Navigate to your project
cd my-project

# 2. Initialize lockfile
llmring lock init

# 3. Configure with AI advisor
llmring lock chat

# 4. Verify configuration
llmring aliases

# 5. Test it out
llmring chat "Hello, world!"
```

---

### Multi-Environment Setup

```bash
# 1. Set up development profile
llmring bind assistant "openai:gpt-4o-mini" --profile dev

# 2. Set up production profile
llmring bind assistant "anthropic:claude-sonnet-4-5-20250929,openai:gpt-4o" --profile prod

# 3. Test dev profile
export LLMRING_PROFILE=dev
llmring chat "Test message"

# 4. Switch to production
export LLMRING_PROFILE=prod
llmring chat "Production message"
```

---

### Cost Analysis Workflow

```bash
# 1. Check current usage
llmring stats --verbose

# 2. Export for analysis
llmring export --output usage_report.json

# 3. Get model pricing info
llmring info fast
llmring info balanced

# 4. Optimize with advisor
llmring lock chat
# Ask: "Can you help me reduce costs?"
```

---

## Tips and Best Practices

### General Usage

1. **Use Aliases**: Prefer semantic aliases over direct model references
2. **Profile Strategy**: Set up profiles for dev/staging/prod environments
3. **Cost Monitoring**: Regularly check `llmring stats` to track usage
4. **Cache Management**: Clear cache after major registry updates

### Performance

1. **Stream Long Responses**: Use `--stream` for better UX with long outputs
2. **Cache Hits**: The registry cache improves performance; let it work
3. **Profile ENV Var**: Set `LLMRING_PROFILE` once instead of passing `--profile` repeatedly

### Security

1. **Environment Files**: Use `.env` files for API keys (don't commit!)
2. **Remote URLs**: Keep `LLMRING_ALLOW_REMOTE_URLS` disabled unless needed
3. **Lockfile**: Commit `llmring.lock` to version control for reproducibility

### Troubleshooting

```bash
# Check provider configuration
llmring providers

# Validate lockfile
llmring lock validate

# Clear stale cache
llmring cache clear

# Check specific model info
llmring info <model>

# Verbose output for debugging
llmring chat "test" --verbose --json
```

---

## Related Documentation

- [Lockfile Documentation](lockfile.md) - Comprehensive lockfile guide
- [MCP Integration](mcp.md) - Conversational configuration
- [API Reference](api-reference.md) - Programmatic usage
