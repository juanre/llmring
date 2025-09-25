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
Assistant: I'll help you add a fast model. Based on the current registry, I recommend using 'openai:gpt-4o-mini' which is highly cost-effective and quick. Let me add this alias for you.
[Adds 'fast' → 'openai:gpt-4o-mini']
```

### Getting Recommendations

```
You: What model should I use for coding tasks?
Assistant: For coding tasks, I recommend using models with strong code understanding and generation capabilities. Let me analyze the available models...
[Provides recommendations based on registry]
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
- `recommend_alias` - Get recommendations for use cases
- `analyze_costs` - Estimate monthly costs
- `save_lockfile` - Save configuration to disk
- `get_configuration` - View full lockfile

## Advanced Usage

### Using an External MCP Server

If you have an MCP server running elsewhere:

```bash
llmring lock chat --server-url http://localhost:8080
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

The conversational interface works with your existing `llmring.lock` file:
1. It reads your current configuration on startup
2. Makes changes in memory during the conversation
3. Saves changes when you request it

This allows you to experiment with different configurations before committing changes.

## Benefits

1. **Natural Language**: No need to remember exact model names or syntax
2. **Intelligent Recommendations**: Get suggestions based on your needs
3. **Cost Awareness**: Understand the financial implications of your choices
4. **Interactive Exploration**: Try different configurations before saving
5. **Registry-Driven**: Always uses the latest model information

## Technical Details

The conversational interface uses:
- **MCP Protocol**: For tool communication
- **LLMRing Service**: For LLM interactions
- **Registry Client**: For up-to-date model information
- **Lockfile Management**: For configuration persistence

This creates a seamless experience for managing your LLM configuration through natural conversation.