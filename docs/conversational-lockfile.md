# Conversational Lockfile Management

LLMRing now supports conversational lockfile management through a natural language chat interface powered by MCP (Model Context Protocol).

## Quick Start

Start the conversational lockfile manager:

```bash
llmring lock chat
```

This will:
1. Start an embedded MCP server with lockfile management tools
2. Launch an interactive chat interface
3. Allow you to manage your lockfile using natural language

## Example Conversations

### Adding Aliases

```
You: I need a fast model for quick responses
Assistant: I'll help you find a fast model. Let me check which models have the best speed and cost balance.
[Calls filter_models_by_requirements with max_price and min_speed criteria]
Based on the available models, 'openai:gpt-4o-mini' offers excellent speed at $0.15 per million tokens.
Would you like me to add this as your 'fast' alias?

You: Yes, please add it
[Adds 'fast' → 'openai:gpt-4o-mini']
```

### Finding Models for Specific Tasks

```
You: What models are available for coding tasks?
Assistant: I'll search for models that support function calling and have good code capabilities.
[Calls filter_models_by_requirements with requires_functions=true]

Here are the top models for coding:
1. anthropic:claude-3-5-sonnet - 200K context, excellent code understanding
2. openai:gpt-4o - 128K context, strong function calling
3. openai:gpt-4o-mini - 128K context, cost-effective with function support
```

### Managing Configuration

```
You: Show me my current aliases
Assistant: Here are your current aliases:
- fast → openai:gpt-4o-mini
- deep → anthropic:claude-3-5-sonnet
- balanced → openai:gpt-4o
```

### Cost Analysis

```
You: How much will my current setup cost per month?
Assistant: Let me analyze the costs based on your current configuration...
[Provides detailed cost breakdown]
```

## Available Natural Language Commands

The conversational interface understands requests like:
- "Add an alias called 'writer' for content creation"
- "Remove the 'test' alias"
- "What models are best for vision tasks?"
- "Show me the cheapest models available"
- "Assess the capabilities of gpt-4o"
- "Save my configuration"

## MCP Tools Available

Behind the scenes, the chat interface uses these MCP tools:
- `add_alias` - Add or update aliases
- `remove_alias` - Remove aliases
- `list_aliases` - Show current configuration
- `assess_model` - Evaluate model capabilities
- `filter_models_by_requirements` - Find models matching specific criteria
- `list_models` - View all available models
- `get_model_details` - Get detailed information about specific models
- `analyze_costs` - Estimate monthly costs
- `save_lockfile` - Save configuration to disk
- `get_configuration` - View full lockfile
- `get_available_providers` - Check which providers have API keys configured

## Persistent History

The chat interface now maintains persistent history across sessions:

### Session Management
- All conversations are automatically saved
- Each session has a unique ID and timestamp
- Resume previous conversations with `/load <session_id>`
- View all sessions with `/sessions`

### History Storage
```
~/.llmring/mcp_chat/
├── command_history.txt        # Command line history
├── conversation_<id>.json     # Individual conversations
└── sessions.json              # Session metadata
```

## Advanced Usage

### Using an External MCP Server

The chat client is completely generic and can connect to any MCP server:

```bash
# Connect via stdio (most common)
llmring mcp chat --server "stdio://python -m your.mcp.server"

# Connect via HTTP
llmring mcp chat --server "http://localhost:8080"

# Connect via WebSocket
llmring mcp chat --server "ws://localhost:8080"
```

### Choosing a Different Model

By default, the conversational interface uses the `advisor` alias (Claude Opus 4.1) for optimal intelligent recommendations. You can specify a different model:

```bash
llmring lock chat --model deep   # Use deep reasoning model
llmring lock chat --model balanced  # Use balanced model
```

The `advisor` alias is specifically configured for:
- Intelligent lockfile creation
- Optimal model recommendations
- Advanced reasoning about your use cases
- Understanding complex requirements

## Integration with Existing Workflow

The conversational interface intelligently manages your `llmring.lock` file:
1. **Automatic Project Root Discovery**: Finds your project root (where pyproject.toml, setup.py, or .git is located)
2. **Creates/loads lockfile at project root**: Ensures lockfile is in the right place for packaging
3. **Packaging guidance**: When saving, provides instructions for including lockfile in your package distribution
4. **In-memory changes**: Experiment with configurations before committing

This ensures your lockfile is always in the right location for both development and distribution.

## Benefits

1. **Natural Language**: No need to remember exact model names or syntax
2. **Intelligent Recommendations**: Get suggestions based on your needs
3. **Cost Awareness**: Understand the financial implications of your choices
4. **Interactive Exploration**: Try different configurations before saving
5. **Registry-Driven**: Always uses the latest model information

## Data-Focused Design Philosophy

The lockfile manager follows a "data-focused" approach:
- Tools provide data and perform actions, not make decisions
- The LLM is in the driver's seat for all intelligent decisions
- Models are selected based on objective criteria you specify
- No hidden "smart" recommendations - full transparency

## Technical Details

The conversational interface uses:
- **MCP Protocol**: For tool communication
- **LLMRing Service**: For LLM interactions
- **Registry Client**: For up-to-date model information
- **Lockfile Management**: For configuration persistence
- **Persistent History**: For session continuity
- **Generic MCP Client**: Works with any MCP-compliant server

This creates a seamless experience for managing your LLM configuration through natural conversation.

## See Also

- [MCP Integration](mcp.md) - Complete guide to MCP and chat client
- [Lockfile Documentation](lockfile.md) - Complete guide to lockfiles and aliases
- [CLI Reference](cli-reference.md) - All CLI commands
