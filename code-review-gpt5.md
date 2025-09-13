### Must-fix issues (before release)
- Base interface mismatch
  - Providers implement `chat(messages, model, ...)`, but `BaseLLMProvider.chat` requires a single `LLMRequest`. This breaks LSP and tooling. Unify the signature across base and providers to one of:
    - A) Base takes explicit args (messages, model, temperature, …) to match current providers, or
    - B) Providers accept a single `LLMRequest` and extract fields internally. 
  - Today, `LLMRing.service` calls provider methods with explicit args, so option A is the least invasive change.

- Broken code paths in `service_extended` and MCP client
  - `service_extended.py` reads `response.choices` and `choice.message.content`, but `LLMResponse` only has `content`/`tool_calls`. This will crash when used.
  - `mcp/client/stateless_engine.py` passes `id_at_origin=...` into `LLMRing.chat`, which doesn’t accept it; will raise a `TypeError`.
  - Fix both to use `LLMResponse.content` and remove unsupported kwargs. Consider adding a documented correlation/telemetry field in `LLMRequest.metadata` if you need `id_at_origin`.

- Error taxonomy not in use
  - Providers raise `ValueError`/generic `Exception`. You already have rich typed exceptions in `llmring.exceptions`. Replace ad-hoc exceptions with:
    - Auth: `ProviderAuthenticationError`
    - Rate limit: `ProviderRateLimitError`
    - Timeouts: `ProviderTimeoutError`
    - Not found/invalid model: `ModelNotFoundError`
    - Generic provider errors: `ProviderResponseError`
  - This improves DX, retries, and observability.

- Incomplete/unused code
  - `providers/openai_api.py`: `_is_chat_model` is unfinished and unused → remove or complete and use.
  - `api/types.py` + `api/__init__.py` are not used by the core API; MCP defines separate `ChatRequest/ChatResponse`. Either:
    - Remove `llmring/api` to avoid duplication, or
    - Replace MCP-local DTOs with the shared ones and wire them up.
  - Audit for other dead exports with lints (ruff F401/F841) and remove.

### SDK usage coverage review
- OpenAI (good overall, a few gaps)
  - Uses `openai>=1.x` `AsyncOpenAI`, streaming via `chat.completions` implemented; tool calling supported; JSON mode handled.
  - o1 models correctly routed via `responses.create` and non-streaming.
  - PDFs: handled via file upload + Responses `input_file`. However, comment refers to “vector stores/file_search” but code doesn’t attach vector stores or enable `file_search`. Either:
    - Update comment to reflect current behavior, or
    - Implement vector stores (`client.vector_stores.*`) and `file_search` attachments for real RAG.
  - JSON schema: add support for `response_format={"type":"json_schema","json_schema":{...},"strict":True}` in addition to `json_object`.
  - Audio: you set `supports_audio=True`, but no audio input/output path is present. Either implement the audio routes or mark unsupported.

- Anthropic (strong)
  - Uses `AsyncAnthropic.messages.create` with streaming, tool use, cache-control headers. JSON “mode” via system prompt is fine for now.
  - Good mapping of tool use/tool result blocks and usage (incl. cache token counters).

- Google Gemini (needs work to reach “full power”)
  - Uses `google-genai` with `generate_content`/`chats.create` and runs sync calls in a thread pool. Works, but:
    - Streaming: currently simulated non-streaming. Implement real streaming via `client.models.generate_content_stream(...)` (supported in `google-genai`) and yield `StreamChunk`s.
    - Tools: don’t prompt-engineer tool calls. Use the official function calling (declare `types.Tool`/`types.FunctionDeclaration` in config and parse `tool_invocations`).
    - Models: mapping 2.x to 1.5 for “stability” surprises users and hides newer features. Prefer honoring the user-specified model and only fall back if explicitly configured. At minimum, document the mapping and provide an opt-out.
    - Advanced params: expose `safety_settings`, `response_mime_type`, `generation_config` fields (top_p, top_k, candidate_count, stop sequences, etc.) via pass-through parameters.
    - JSON mode: you set `response_mime_type="application/json"`, which is good. Add schema-guided JSON if available.

- Ollama (OK but minimal)
  - Streaming: currently faked into one chunk. The SDK supports streaming; add `stream=True` support and yield deltas progressively.
  - Tools: prompt-based only (expected; no native API).
  - Options: surface useful Ollama options (e.g., `mirostat`, `penalty_*`, `num_ctx`, `seed`) via pass-through.

### API design: “full power but pluggable”
- Pass-through provider params
  - Add a typed `extra_params: dict[str, Any]` to `LLMRequest` and propagate to provider calls, so advanced SDK features remain accessible without growing the common surface indefinitely.
  - Document supported pass-through keys per provider (OpenAI: `logprobs`, `top_logprobs`, `parallel_tool_calls`, `reasoning`, `presence_penalty`…; Google: `generation_config`, `safety_settings`…; Ollama: `options` bag).
- Escape hatch to raw SDK clients
  - Document `llmring.get_provider("openai").client` (and for others). This preserves maximum SDK power.
- Tooling schema normalization
  - You already map tools across providers. Add a small, typed `ToolSpec` with fields for name/description/parameters, and helpers to convert to provider shapes (OpenAI, Anthropic, Gemini).

### Quality and correctness
- Context validation
  - Good use of registry limits and token counting. Consider a provider-aware guard for output tokens: if `max_tokens` exceeds model limit, clamp or raise `ModelCapabilityError`.
- Model resolution
  - `_parse_model_string` defaults to the first provider when the string has no prefix and can’t be inferred. That can lead to surprising routing. Prefer:
    - Require `provider:model` or an alias, or
    - If inferring, error out with clear guidance unless `LLMRING_ALLOW_AMBIGUOUS_MODEL=1`.
- Token counting
  - Looks solid. For Google/Ollama you estimate, which is fine. Optionally, add an adapter for Gemini’s token count endpoint if/when exposed in `google-genai`.
- Receipts and costs
  - Receipt generation is clean, with lockfile digest and cost enrichment. Good separation in `receipts.py`.
- File utils
  - Good universal content helpers, strict remote-URL gating by env, and PDF vs. image handling.

### Unused/legacy code to prune or align
- Remove or align:
  - `llmring/api/types.py` and `llmring/api/__init__.py` (unused in core; MCP redefines types).
  - `providers/openai_api.py::_is_chat_model` (unused).
  - `service_extended.py` references to `response.choices` must be fixed or the module gated behind feature flags.
  - `mcp/client/stateless_engine.py` unsupported kwargs and simulated streaming should be fixed or clearly marked experimental.

### DX and lint/test gate (quick wins)
- Add a CI step:
  - Ruff: `ruff check --select F,E,I --unsafe-fixes` (remove unused imports/vars, sort, basic errors).
  - Mypy/Pyright for the public API and providers.
  - Run tests with live SKIP for provider calls unless keys present; keep unit tests deterministic.
- Type hygiene
  - Reduce `Any` in public types; annotate provider `chat` signatures and `extra_params` precisely; avoid silent dict bag drift.
- Logging
  - Standardize provider error logs; include request id/correlation id from `LLMRequest.metadata` if present.

### What to implement to meet your goals
- Unify `BaseLLMProvider.chat` signature with providers and update implementations.
- Fix `service_extended.py` and `stateless_engine.py` to use `LLMResponse.content` and remove unsupported kwargs.
- Implement:
  - Google streaming via `generate_content_stream`.
  - Google function calling via `types.Tool` and parse tool invocations.
  - Ollama streaming via SDK (`stream=True`), yielding `StreamChunk`s.
  - OpenAI JSON schema response_format and (optionally) vector stores/file_search path for PDFs.
- Add `extra_params` passthrough to `LLMRequest` and providers; document supported keys.
- Replace generic exceptions with your typed ones across providers.
- Remove or integrate `llmring/api/*` and dead helpers; finish or delete `_is_chat_model`.

If you’d like, I can implement these changes in a focused PR series (providers API unification first, then provider-specific enhancements, then cleanup + lints).

- Proposed commands to find dead code and unused imports locally (non-interactive):
```bash
uv run ruff check /Users/juanre/prj/llmring-all/llmring/src/llmring --select F401,F841 | cat
uv run ruff check /Users/juanre/prj/llmring-all/llmring/src/llmring --select F401,F841 --fix | cat
uv run pyright /Users/juanre/prj/llmring-all/llmring/src/llmring | cat
```

- Example: extend `LLMRequest` with pass-through
```python
# schemas.py
class LLMRequest(BaseModel):
  ...
  extra_params: dict[str, Any] = Field(default_factory=dict)

# service.py (propagate)
response = await provider.chat(..., **(request.extra_params or {}))
```

- Example: document escape hatch
```python
openai_client = ring.get_provider("openai").client
```

Summary
- Fix provider interface mismatch and broken `service_extended`/MCP callsites.
- Upgrade Google and Ollama to real streaming and proper tools (Gemini).
- Add pass-through params and document raw-client escape hatch for full SDK power.
- Adopt your typed exceptions; remove/finish unused code paths; update comments vs. actual OpenAI PDFs flow.
- Run ruff/pyright to ensure “no unused code” and stronger API quality before release.
