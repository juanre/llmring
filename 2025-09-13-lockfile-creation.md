# Intelligent Lockfile Creation System Design

**Date**: 2025-09-13
**Status**: Design Document
**Prerequisites**: Hand-made lockfile with intelligent model + sensible alias, use our own API

## Overview

An LLM-powered system that guides users through intelligent lockfile creation by analyzing the registry, understanding their needs, and recommending optimal model configurations.

## Prerequisites

### 1. Bootstrap Lockfile
Before implementing the intelligent system, create a hand-made lockfile with a current, capable model to power the system:

```toml
# llmring.lock (bootstrap version)
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "advisor"
provider = "anthropic"
model = "claude-opus-4-20250514"  # Most capable model for analysis

[[profiles.default.bindings]]
alias = "fast"
provider = "openai"
model = "gpt-4o-mini"  # Current fast model

[[profiles.default.bindings]]
alias = "balanced"
provider = "anthropic"
model = "claude-3-5-haiku-20241022"  # Current balanced model
```

### 2. Use Our Own API
The intelligent system must use LLMRing's own API (`service.chat()` with aliases) to bootstrap itself - true "eating our own dog food".

## Architecture

### Core Components

1. **Registry Analysis LLM** - Analyzes current registry data to understand available models
2. **MCP Registry Tools** - Provide real-time access to registry data during conversation
3. **Interactive Conversation Engine** - Guides user through lockfile creation process
4. **Lockfile Generation Service** - Creates optimized lockfile based on conversation

### Data Flow

```
User Intent → Interactive LLM Chat → Registry Analysis → Model Recommendations → Lockfile Generation
     ↑              ↓                      ↑                    ↓                    ↓
User Input → Conversation Engine ← MCP Registry Tools → Optimization Logic → llmring.lock
```

## Implementation Plan

### Phase 1: MCP Registry Tools

Create MCP tools that provide registry access to the LLM:

```python
# src/llmring/mcp/tools/registry_tools.py

@mcp_tool("get_provider_models")
async def get_provider_models(provider: str) -> dict:
    """Get all available models for a provider with capabilities and pricing."""
    registry = RegistryClient()
    models = await registry.fetch_current_models(provider)

    return {
        "provider": provider,
        "models": [
            {
                "name": model.model_name,
                "description": model.description,
                "max_input_tokens": model.max_input_tokens,
                "max_output_tokens": model.max_output_tokens,
                "cost_per_million_input": model.dollars_per_million_tokens_input,
                "cost_per_million_output": model.dollars_per_million_tokens_output,
                "supports_vision": model.supports_vision,
                "supports_function_calling": model.supports_function_calling,
                "supports_json_mode": model.supports_json_mode,
                "is_active": model.is_active,
                "added_date": model.added_date,
                "deprecated_date": model.deprecated_date
            }
            for model in models if model.is_active
        ]
    }

@mcp_tool("compare_models")
async def compare_models(model_names: list[str], criteria: str = "cost") -> dict:
    """Compare models across providers based on specific criteria."""
    # Implementation compares models by cost, performance, capabilities, etc.

@mcp_tool("recommend_for_use_case")
async def recommend_for_use_case(
    use_case: str,
    budget: str = "medium",
    performance_priority: str = "balanced"
) -> dict:
    """Recommend models for specific use cases like 'data extraction', 'creative writing', etc."""
    # Implementation analyzes registry and returns recommendations

@mcp_tool("get_cost_analysis")
async def get_cost_analysis(models: list[str], estimated_usage: dict) -> dict:
    """Analyze cost implications of model choices for expected usage patterns."""
    # Implementation calculates monthly costs based on usage patterns

@mcp_tool("create_lockfile_binding")
async def create_lockfile_binding(alias: str, provider: str, model: str) -> dict:
    """Create a new lockfile binding."""
    # Implementation adds binding to lockfile
```

### Phase 2: Conversation Engine

Create an intelligent conversation engine that uses the MCP tools:

```python
# src/llmring/lockfile/intelligent_creator.py

class IntelligentLockfileCreator:
    """Creates lockfiles through intelligent conversation."""

    def __init__(self):
        # Use our own API with the "advisor" alias
        self.service = LLMRing()
        self.conversation_state = {
            "user_needs": {},
            "recommended_models": {},
            "selected_aliases": {}
        }

    async def create_lockfile_interactively(self) -> Lockfile:
        """Guide user through intelligent lockfile creation."""

        # Phase 1: Understand user needs
        await self._discover_user_needs()

        # Phase 2: Analyze registry and make recommendations
        await self._analyze_and_recommend()

        # Phase 3: Refine choices through conversation
        await self._refine_selections()

        # Phase 4: Generate lockfile
        return await self._generate_lockfile()

    async def _discover_user_needs(self):
        """Discover user's specific needs through conversation."""

        # Use our own service with "advisor" alias
        request = LLMRequest(
            model="advisor",  # Uses the capable model from bootstrap lockfile
            messages=[Message(
                role="system",
                content="""You are an expert LLM configuration advisor with access to real-time registry data.

Your job is to understand the user's needs for LLM usage and recommend optimal model configurations.

Ask focused questions to understand:
1. Primary use cases (analysis, creative writing, code generation, etc.)
2. Budget constraints and usage volume
3. Required capabilities (vision, function calling, large context)
4. Provider preferences or restrictions

Use the available MCP tools to analyze the current registry and provide data-driven recommendations."""
            ), Message(
                role="user",
                content="Help me create an optimal LLM configuration. What do you need to know about my use case?"
            )],
            tools=self._get_mcp_tools()  # Registry analysis tools
        )

        response = await self.service.chat(request)
        # Process conversation...
```

### Phase 3: Registry Analysis System

```python
# Enhanced LLM with registry analysis capabilities

SYSTEM_PROMPT = """
You are an expert LLM configuration advisor with access to real-time registry data.

Your job is to:
1. Analyze the current registry to understand available models
2. Compare models across providers for different use cases
3. Consider cost, performance, and capability trade-offs
4. Recommend optimal model configurations
5. Create semantic aliases that make sense for the user's workflow

You have access to these tools:
- get_provider_models: Get all models for a provider with full details
- compare_models: Compare specific models by criteria
- recommend_for_use_case: Get recommendations for specific use cases
- get_cost_analysis: Analyze cost implications
- create_lockfile_binding: Add bindings to lockfile

Always provide data-driven recommendations based on current registry information.
Consider both cost efficiency and capability when making recommendations.
Suggest aliases that are semantic and workflow-oriented.
"""
```

### Phase 4: Interactive CLI Command

```python
# src/llmring/cli.py - Enhanced lock init command

async def cmd_lock_init_intelligent(args):
    """Initialize lockfile with intelligent recommendations."""

    if not args.interactive:
        # Fall back to current basic init
        return await cmd_lock_init_basic(args)

    print("🤖 Starting intelligent lockfile creation...")
    print("I'll analyze the current registry and help you choose optimal models.\n")

    creator = IntelligentLockfileCreator()

    try:
        lockfile = await creator.create_lockfile_interactively()

        # Save lockfile
        path = Path(args.file) if args.file else Path("llmring.lock")
        lockfile.save(path)

        print(f"\n✅ Created intelligent lockfile: {path}")

        # Show recommendations summary
        print("\n📊 Final Configuration:")
        for binding in lockfile.get_profile("default").bindings:
            cost_info = await get_model_cost_info(binding.model_ref)
            print(f"  {binding.alias:<15} → {binding.model_ref}")
            if cost_info:
                print(f"    Cost: ${cost_info['cost_per_million_input']:.2f}/${cost_info['cost_per_million_output']:.2f} per M tokens")

    except KeyboardInterrupt:
        print("\nLockfile creation cancelled.")
        return 1
    except Exception as e:
        print(f"Error creating lockfile: {e}")
        return 1
```

## Desired User Experience

### Command Interface

```bash
# Basic initialization (current)
llmring lock init

# Intelligent initialization (new)
llmring lock init --interactive

# Guided update of existing lockfile
llmring lock optimize --interactive

# Analysis of current configuration
llmring lock analyze
```

### Interactive Session Example

```
$ llmring lock init --interactive

🤖 Starting intelligent lockfile creation...
I'll analyze the current registry and help you choose optimal models.

Let me start by understanding your needs:

What are your primary use cases for LLMs?
1. Data extraction and analysis
2. Creative writing and content generation
3. Code generation and review
4. Customer support and Q&A
5. Research and summarization
6. Mixed/general purpose

> 6

Great! For general purpose usage, I'll recommend a balanced set of models.

Let me check the current registry...

📊 Registry Analysis Complete:
- OpenAI: 3 active models (gpt-4o-mini most cost-effective)
- Anthropic: 12 active models (claude-opus-4 most capable, claude-haiku-3.5 fastest)
- Google: 9 active models (gemini-1.5-flash good balance)
- Ollama: Local models available

What's your budget preference?
1. Cost-conscious (optimize for price)
2. Balanced (good price/performance)
3. Performance-first (best capabilities)

> 2

Perfect! For balanced usage, here are my recommendations:

🎯 Suggested Aliases:
- "deep": anthropic:claude-opus-4-20250514 ($15/$75 per M tokens)
  └ For complex reasoning, analysis, research

- "balanced": anthropic:claude-haiku-3.5 ($0.25/$1.25 per M tokens)
  └ For general conversations, quick tasks

- "fast": openai:gpt-4o-mini ($0.15/$0.60 per M tokens)
  └ For rapid responses, simple queries

- "creative": anthropic:claude-sonnet-4-20250514 ($3/$15 per M tokens)
  └ For creative writing, content generation

- "local": ollama:llama3.3:latest (free)
  └ For private/offline usage

💡 Special Capabilities:
- "vision": openai:gpt-4o (for image analysis)
- "code": anthropic:claude-sonnet-4 (for programming tasks)

Do you want to customize any of these? (y/N) n

✅ Created intelligent lockfile: llmring.lock

📊 Final Configuration:
  deep           → anthropic:claude-opus-4-20250514
    Cost: $15.00/$75.00 per M tokens, Max context: 200K
  balanced       → anthropic:claude-haiku-3.5
    Cost: $0.25/$1.25 per M tokens, Max context: 200K
  fast           → openai:gpt-4o-mini
    Cost: $0.15/$0.60 per M tokens, Max context: 128K
  creative       → anthropic:claude-sonnet-4-20250514
    Cost: $3.00/$15.00 per M tokens, Max context: 200K
  local          → ollama:llama3.3:latest
    Cost: Free, Local execution

💡 Tip: Use 'llmring lock optimize' to update these recommendations as new models become available.
```

## Technical Implementation

### Bootstrap Requirements

1. **Hand-made Lockfile**: Create initial lockfile with capable model:
   ```toml
   [[profiles.default.bindings]]
   alias = "advisor"
   provider = "anthropic"
   model = "claude-opus-4-20250514"  # Most capable for analysis
   ```

2. **Self-Hosted Approach**: The intelligent system uses LLMRing's own API:
   ```python
   # Use our own service
   service = LLMRing()

   # Power the advisor with our own API
   request = LLMRequest(
       model="advisor",  # Alias from bootstrap lockfile
       messages=[...],
       tools=registry_mcp_tools
   )

   response = await service.chat(request)
   ```

### MCP Server Integration

```python
# src/llmring/mcp/servers/registry_server.py

class RegistryMCPServer:
    """MCP server providing registry analysis tools."""

    def __init__(self):
        self.registry = RegistryClient()
        self.server = MCPServer("registry-advisor")
        self._register_tools()

    def _register_tools(self):
        # Register all registry analysis tools
        self.server.register_tool(get_provider_models)
        self.server.register_tool(compare_models)
        self.server.register_tool(recommend_for_use_case)
        self.server.register_tool(get_cost_analysis)
```

### Conversation Templates

```python
# src/llmring/lockfile/conversation_templates.py

DISCOVERY_QUESTIONS = [
    {
        "category": "use_cases",
        "question": "What are your primary LLM use cases?",
        "options": ["analysis", "creative", "code", "support", "research", "mixed"],
        "follow_up": "Tell me more about your specific workflows"
    },
    {
        "category": "budget",
        "question": "What's your budget preference?",
        "options": ["cost_conscious", "balanced", "performance_first"],
        "follow_up": "What's your expected monthly usage volume?"
    },
    {
        "category": "capabilities",
        "question": "Do you need specific capabilities?",
        "options": ["vision", "function_calling", "large_context", "multilingual"],
        "follow_up": "Any provider preferences or restrictions?"
    }
]

RECOMMENDATION_PROMPTS = {
    "analysis_focused": "Recommend models optimized for data analysis and reasoning",
    "cost_conscious": "Prioritize the most cost-effective models while maintaining quality",
    "performance_first": "Recommend the most capable models regardless of cost"
}
```

## Implementation Phases

### Phase 1: Bootstrap Setup
1. **Create hand-made lockfile** with intelligent model for advisor alias
2. **Verify self-hosting** works: LLMRing can use its own API with aliases
3. **Test registry access** via our own service

### Phase 2: MCP Registry Tools (1-2 days)
- Implement registry analysis tools
- Create MCP server for registry access
- Test tool functionality with our own API

### Phase 3: Conversation Engine (2-3 days)
- Build interactive conversation system using LLMRing service
- Implement recommendation logic powered by "advisor" alias
- Create conversation templates

### Phase 4: CLI Integration (1 day)
- Add `--interactive` flag to lock commands
- Integrate with existing lockfile system
- Add cost analysis features

### Phase 5: Advanced Features (optional)
- Lockfile optimization for existing configs
- Usage-based recommendations
- Multi-environment configuration

## Key Design Principles

### 1. Self-Hosted ("Eating Our Own Dog Food")
```python
# The advisor uses LLMRing's own API
service = LLMRing()
response = await service.chat(LLMRequest(
    model="advisor",  # From bootstrap lockfile
    messages=[...],
    tools=registry_tools
))
```

### 2. Registry-Driven Recommendations
- No hardcoded model lists
- All recommendations based on current registry data
- Dynamic cost and capability analysis

### 3. Semantic Alias Creation
- Aliases reflect workflow intent ("deep", "fast", "creative")
- User-customizable based on their specific needs
- Avoid technical model names in favor of semantic meaning

## Expected Benefits

### For Users
- **Intelligent Recommendations**: Data-driven model selection based on current registry
- **Cost Awareness**: Transparent cost implications for each choice
- **Use Case Optimization**: Models matched to specific workflows
- **Future-Proof**: Automatically suggests current best models, not outdated ones

### For System Architecture
- **Registry Integration**: Ensures lockfiles only reference valid models
- **Consistency**: All lockfiles follow best practices and current recommendations
- **Self-Hosting**: System uses its own capabilities (true dog-fooding)
- **Maintainability**: No more hardcoded defaults that become stale

## File Structure

```
src/llmring/lockfile/
├── intelligent_creator.py      # Main conversation engine using LLMRing API
├── conversation_templates.py   # Question templates and flows
├── recommendation_engine.py    # Model analysis and recommendation logic
└── cost_analyzer.py           # Cost calculation and comparison

src/llmring/mcp/tools/
├── registry_tools.py          # Registry access tools
└── lockfile_tools.py          # Lockfile manipulation tools

src/llmring/mcp/servers/
└── lockfile_advisor_server.py # Complete MCP server for lockfile creation
```

## CLI Integration

```bash
# Current (basic with outdated defaults)
llmring lock init

# Enhanced (intelligent, registry-driven)
llmring lock init --interactive      # Guided intelligent creation
llmring lock optimize                # Optimize existing lockfile using current registry
llmring lock validate --suggest      # Validate and suggest improvements
llmring lock analyze                 # Show cost analysis for current config
```

## Success Criteria

1. **Bootstrap Works**: Hand-made lockfile with "advisor" alias powers the system
2. **Self-Hosting**: System uses `service.chat(model="advisor")` successfully
3. **Registry Integration**: All recommendations come from live registry data
4. **No Hardcoded Models**: System generates lockfiles with current, optimal models
5. **User Experience**: Interactive CLI that's better than manual lockfile editing

This system would transform lockfile creation from a manual, error-prone process with outdated hardcoded defaults into an intelligent, data-driven experience that truly leverages LLMRing's own capabilities.