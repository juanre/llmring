# Changelog

All notable changes to llmring will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-01-03

### Added
- **Alias Pool Failover**: Automatic fallback to backup models when primary fails with transient errors (rate limits, timeouts). No fallback on auth errors or after streaming starts.
- **Formalized File API**: All providers now implement `upload_file()`, `delete_file()`, `list_files()`, `get_file()` in the abstract interface.
- **Google File API**: Full binary file support up to 2GB via Google's File API, with automatic re-upload for expired files.
- **Configurable Timeouts**: Instance-level, request-level, and environment variable timeout configuration.
- **OpenAI Responses API**: Support for file registration via OpenAI's Responses API.
- **Unified File Interface**: Provider-agnostic file upload with `register_file()` for cross-provider compatibility.

### Changed
- `BaseLLMProvider.get_default_model()` is now async (was sync in some implementations).
- Improved type safety with `_SupportsRootCause` Protocol for exception chaining.
- Better IP validation using Python's `ipaddress` module.
- Dynamic versioning via `importlib.metadata`.
- Pydantic V2 compliance: `Field(None)` → `Field(default=None)`.

### Fixed
- Authentication error handling in integration tests.
- Missing `effective_timeout` in `chat_stream`.
- Google provider concurrency locks and file handling.
- File deletion race conditions.

### Breaking Changes
- **Third-party providers must implement new abstract methods**:
  - `async def get_default_model() -> str`
  - `async def upload_file(...) -> FileUploadResponse`
  - `async def delete_file(file_id: str) -> bool`
  - `async def list_files(...) -> List[FileMetadata]`
  - `async def get_file(file_id: str) -> FileMetadata`
- Providers that don't support files should raise `ProviderResponseError`.

## [1.3.0] - 2025-11-02

### Added
- Lockfile validation helpers for library pattern.
- MCP manifest apply functionality.
- Claude Code Skills integration.

### Changed
- Updated README examples with current models.
- Renamed `execute_tool` to `record_tool_execution` in HTTP client.

### Fixed
- Syntax error in CLI.

## [1.2.0] - 2025-10-26

### Added
- CLI registration command for LLMRing SaaS.
- Claude Code skills for llmring.
- CLI streaming test.
- `supports_caching` field to provider capabilities.

### Changed
- Lowered Python requirement to 3.10.

## [1.1.1] - 2025-10-14

### Fixed
- Minor bug fixes and stability improvements.

## [1.0.0] - 2025-09-13

### Added
- Initial stable release.
- Support for OpenAI, Anthropic, Google, and Ollama providers.
- Alias-based model routing via lockfiles.
- Registry-based model discovery.
- MCP (Model Context Protocol) client and server.
- CLI tools for model interaction.

[Unreleased]: https://github.com/juanre/llmring/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/juanre/llmring/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/juanre/llmring/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/juanre/llmring/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/juanre/llmring/compare/v1.0.0...v1.1.1
[1.0.0]: https://github.com/juanre/llmring/releases/tag/v1.0.0
