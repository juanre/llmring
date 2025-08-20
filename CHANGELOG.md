# Changelog

## v0.3.1 (2025-08-20) - Source-of-Truth v3.5 Compliance

### Critical Fixes
- **Fixed registry field naming**: Corrected pricing field names from `cost_per_million_*_tokens` to `dollars_per_million_tokens_*` per source-of-truth specification
- **Receipt generation**: Integrated automatic receipt generation for all API calls with usage tracking

### New Features
- **CLI Commands**: Added placeholder commands for server integration:
  - `llmring push` - Push lockfile bindings to server (ready for future server)
  - `llmring pull` - Pull lockfile bindings from server (ready for future server)
  - `llmring stats` - Show usage statistics (works with local receipts)
  - `llmring export` - Export receipts to JSON/CSV
  - `llmring register` - Register with LLMRing server (for future SaaS)

- **Enhanced Default Aliases**:
  - Added `pdf_reader` alias for PDF processing models
  - Added `fast` alias for low-latency models
  - Added `multimodal` alias for vision/document models
  - Added `local` alias for Ollama models

- **Lockfile Enhancements**:
  - Added `calculate_digest()` method for receipt tracking
  - Lockfile digest included in all receipts for audit trail

- **Registry Validation**:
  - Added validation for dictionary format compliance
  - Validates "provider:model" key format for O(1) lookup
  - Better error messages for malformed registry data

### Implementation Details
- Receipts are generated locally when usage information is available
- Receipts stored in memory for current session (server integration will persist)
- All CLI commands follow graceful degradation pattern (work locally, enhance with server)
- Registry client validates model dictionary structure per source-of-truth

### Testing
- All unit tests passing (127 passed)
- New integration tests for receipt generation
- Updated tests for new default aliases
- CLI command tests for new functionality

### Compliance
- ✅ Complies with source-of-truth v3.5 specification
- ✅ Registry schema matches specification
- ✅ Receipts include all required fields
- ✅ Lockfile remains authoritative configuration source

### Migration Notes
- Existing lockfiles remain compatible
- New default aliases will be suggested on `llmring lock init`
- Registry field names updated automatically (backward compatible with caching)

---

## v0.3.0 (Previous)
- Initial release with core functionality
- Basic provider support (OpenAI, Anthropic, Google, Ollama)
- Lockfile-based configuration
- Registry integration