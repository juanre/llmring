# Provider Guide

## Model Aliases vs Direct Models

This guide shows how to use provider-specific features through `extra_params` and direct SDK access.

**Recommended**: Use semantic aliases like `fast`, `balanced`, `deep` for maintainability and cost optimization. These examples use aliases throughout.

**Advanced**: You can still use direct `provider:model` format when you need specific model versions or provider features.

## Extra Params Usage

Pass provider-specific parameters via `extra_params` in your `LLMRequest`:

### OpenAI Extra Params

```python
from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message

async with LLMRing() as service:
    request = LLMRequest(
        model="fast",
        messages=[Message(role="user", content="Hello")],
        extra_params={
            # For analyzing model confidence and alternative predictions
            "logprobs": True,              # Get log probabilities for tokens
            "top_logprobs": 5,             # Show top 5 alternative tokens at each position

            # For controlling output diversity
            "presence_penalty": 0.1,       # Penalize tokens that have appeared (encourages new topics)
            "frequency_penalty": 0.1,      # Penalize tokens based on frequency (reduces repetition)

            # For reproducibility
            "seed": 12345,                 # Make outputs deterministic for same inputs

            # For nucleus sampling
            "top_p": 0.9                   # Only sample from top 90% probability mass
        }
    )

    response = await service.chat(request)
```

**Use Cases:**
- `logprobs`: Analyze model confidence, debug outputs, measure uncertainty
- `presence_penalty`: Generate diverse content, avoid repetition of topics
- `frequency_penalty`: Reduce word repetition, vary vocabulary
- `seed`: Reproducible outputs for testing, A/B testing, debugging
- `top_p`: Balance creativity vs coherence

### Anthropic Extra Params

```python
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "top_p": 0.9,      # Nucleus sampling - sample from top 90% probability
        "top_k": 40        # Consider only top 40 tokens at each step
    }
)
```

**Use Cases:**
- `top_p`: Control randomness while maintaining quality (0.9-0.95 recommended)
- `top_k`: Limit token candidates to prevent unlikely choices

### Google Gemini Extra Params

```python
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        # Sampling parameters
        "top_p": 0.8,                      # Nucleus sampling threshold
        "top_k": 30,                       # Top-k sampling limit
        "candidate_count": 1,              # Number of response candidates to generate

        # Safety controls (important for production)
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"  # Block medium+ harmful content
            }
        ],

        # Generation controls
        "generation_config": {
            "stop_sequences": ["END"],     # Stop generation at these sequences
            "max_output_tokens": 1000      # Limit output length
        }
    }
)
```

**Use Cases:**
- `safety_settings`: Production applications requiring content moderation
- `stop_sequences`: Control output format, implement custom terminators
- `candidate_count`: Generate multiple options for selection (increases cost)

### Ollama Extra Params

```python
request = LLMRequest(
    model="local",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "options": {
            # Mirostat sampling (adaptive sampling for better quality)
            "mirostat": 1,                 # Enable Mirostat sampling (0=off, 1=v1, 2=v2)
            "mirostat_tau": 0.8,           # Target entropy (lower = more focused)
            "mirostat_eta": 0.1,           # Learning rate for adaptation

            # Context and generation
            "num_ctx": 4096,               # Context window size
            "num_predict": 100,            # Maximum tokens to generate

            # Quality controls
            "repeat_penalty": 1.1,         # Penalize repetition (1.0 = no penalty)
            "temperature": 0.8,            # Randomness (lower = more deterministic)

            # Sampling parameters
            "top_k": 40,                   # Top-k sampling
            "top_p": 0.9,                  # Nucleus sampling
            "tfs_z": 1.0,                  # Tail free sampling

            # Reproducibility
            "seed": 42                     # Random seed for deterministic outputs
        }
    }
)

# Alternative flat format (simpler for common params):
request = LLMRequest(
    model="local",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "seed": 42,              # For reproducibility
        "num_predict": 100       # Limit output length
    }
)
```

**Use Cases:**
- `mirostat`: Improve output quality for local models (recommended for Llama)
- `num_ctx`: Adjust context size for memory/quality tradeoff
- `repeat_penalty`: Reduce repetition in longer outputs
- `seed`: Reproducible testing with local models

## Raw Client Access (Escape Hatch)

For maximum SDK power, access the underlying clients directly:

### OpenAI Raw Client

```python
from llmring.service import LLMRing

async with LLMRing() as service:
    openai_client = service.get_provider("openai").client

    # Use full OpenAI SDK capabilities
    response = await openai_client.chat.completions.create(
        model="fast",  # Use alias or provider:model format when needed
        messages=[{"role": "user", "content": "Hello"}],
        logprobs=True,
        top_logprobs=10,
        stream=True,
        # Any OpenAI parameter supported
    )
```

### Anthropic Raw Client

```python
anthropic_client = service.get_provider("anthropic").client

# Use full Anthropic SDK capabilities
response = await anthropic_client.messages.create(
    model="balanced",  # Use alias or provider:model format when needed
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    top_p=0.9,
    # Any Anthropic parameter supported
)
```

### Google Gemini Raw Client

```python
google_client = service.get_provider("google").client

# Use full Google genai SDK capabilities
response = google_client.models.generate_content(
    model="balanced",  # Use alias or provider:model format when needed
    contents="Hello",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1000,
        "stop_sequences": ["END"]
    },
    safety_settings=[{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }]
)
```

### Ollama Raw Client

```python
ollama_client = service.get_provider("ollama").client

# Use full Ollama SDK capabilities
response = await ollama_client.chat(
    model="local",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    options={
        "mirostat": 1,
        "mirostat_tau": 0.8,
        "mirostat_eta": 0.1,
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "seed": 42
    }
)
```

## OpenAI JSON Schema (Structured Output)

```python
request = LLMRequest(
    model="fast",
    messages=[Message(role="user", content="Generate a person")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        },
        "strict": True
    }
)

response = await service.chat(request)
# Response will be valid JSON matching the schema
```

## Anthropic Prompt Caching

```python
request = LLMRequest(
    model="balanced",
    messages=[
        Message(
            role="system",
            content="Very long system prompt...",  # 1024+ tokens
            metadata={"cache_control": {"type": "ephemeral"}}
        ),
        Message(role="user", content="Hello")
    ]
)

response = await service.chat(request)
# 90% cost savings on cached tokens!
```

This provides both convenient unified access through `extra_params` and full SDK power through raw client access.
