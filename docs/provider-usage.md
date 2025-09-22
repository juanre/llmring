# Provider Usage Guide

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

service = LLMRing()

request = LLMRequest(
    model="fast",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "logprobs": True,
        "top_logprobs": 5,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 12345,
        "top_p": 0.9
    }
)

response = await service.chat(request)
```

### Anthropic Extra Params

```python
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "top_p": 0.9,
        "top_k": 40
    }
)
```

### Google Gemini Extra Params

```python
request = LLMRequest(
    model="balanced",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "top_p": 0.8,
        "top_k": 30,
        "candidate_count": 1,
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ],
        "generation_config": {
            "stop_sequences": ["END"],
            "max_output_tokens": 1000
        }
    }
)
```

### Ollama Extra Params

```python
request = LLMRequest(
    model="local",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "options": {
            "mirostat": 1,
            "mirostat_tau": 0.8,
            "mirostat_eta": 0.1,
            "num_ctx": 4096,
            "repeat_penalty": 1.1,
            "seed": 42,
            "temperature": 0.8,
            "tfs_z": 1.0,
            "num_predict": 100,
            "top_k": 40,
            "top_p": 0.9
        }
    }
)

# Alternative flat format (same effect):
request = LLMRequest(
    model="local",
    messages=[Message(role="user", content="Hello")],
    extra_params={
        "seed": 42,
        "num_predict": 100
    }
)
```

## Raw Client Access (Escape Hatch)

For maximum SDK power, access the underlying clients directly:

### OpenAI Raw Client

```python
from llmring.service import LLMRing

service = LLMRing()
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
