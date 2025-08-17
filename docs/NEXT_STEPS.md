### Next Steps: Privacy-preserving prompt fingerprinting

Goal: Store irreversible fingerprints (hashes) of both system and user prompts to enable analytics/deduplication without persisting sensitive content.

Why:
- Correlate repeated prompts and measure usage without storing raw text
- Identify hot prompts for caching/optimization
- Support privacy/compliance postures

What to hash:
- System prompt (if present)
- User-visible prompt input (aggregate of user message parts only; exclude images/binary and tool responses)

Hashing approach:
- Algorithm: SHA-256 (fast, widely available)
- Normalization before hashing:
  - Strip leading/trailing whitespace, normalize newlines to "\n"
  - Collapse multiple spaces to single space
  - Lowercase? (optional; only if collisions vs semantic needs are acceptable)
  - Remove/replace non-text blocks (e.g., images/documents) with stable tokens like "[image]", "[document]"
- Optional HMAC: Use HMAC-SHA-256 with deployment-specific secret (pepper) to mitigate cross-environment correlation and rainbow-table risks. Recommended for multi-tenant or stricter privacy.

Schema changes:
- Add columns to `{{tables.llm_api_calls}}`:
  - `system_prompt_sha256 CHAR(64)` (nullable)
  - `user_prompt_sha256 CHAR(64)` (nullable)
- Indexes (optional):
  - `CREATE INDEX idx_llm_api_calls_system_prompt_sha256 ON {{tables.llm_api_calls}}(system_prompt_sha256)`
  - `CREATE INDEX idx_llm_api_calls_user_prompt_sha256 ON {{tables.llm_api_calls}}(user_prompt_sha256)`

Service changes:
- In `LLMRing.chat`:
  - Build a normalized user prompt string from `request.messages` with `role == "user"`
  - Build a normalized system prompt string from the first `role == "system"` (if present)
  - Compute hash/HMAC on normalized strings
  - Do NOT store raw prompts; only store hashes
  - Keep behind a feature flag (e.g., `LLMRING_LOG_PROMPT_HASHES=true`)

Security/Privacy:
- Prefer HMAC-SHA-256 using `LLMRING_PROMPT_HASH_SECRET` for production
- Never log or expose the secret
- Make hashing deterministic per deployment; changing secret invalidates joins across time (acceptable)

Testing:
- Unit tests for normalization (text-only, multimodal parts, tool messages excluded)
- Unit tests for hashing (stable outputs, HMAC and non-HMAC modes)
- DB roundtrip tests to ensure values persist and indexes usable

Migration plan:
- Because we have not deployed yet, we can update the initial migration directly
- If deployment happens before this lands: create a new forward migration adding the two columns and indexes

Operational notes:
- Hashes enable cache keys or dedupe heuristics without storing content
- Monitor cardinality of hashes to detect normalization bugs (unexpectedly low/high)

Open questions:
- Lowercasing vs case-preserving: decide based on use-cases (case often not semantically relevant)
- Include assistant messages? Generally no; focus on user/system only
- Multi-turn: hash only the latest user message, or a window? Start with latest user message


