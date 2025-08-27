# MCP Integration Refactoring Plan

## Overview

This document outlines the refactoring needed to make the MCP integration truly database-agnostic, with all persistence handled via HTTP endpoints to llmring-server.

## Current Issues

1. **Database Dependencies in MCP Code**
   - `chat/app.py` imports `AsyncDatabaseManager` (commented but still referenced)
   - Several files still have database-related imports
   - Mix of HTTP-based and database-based approaches

2. **Duplicate HTTP Client Implementations**
   - `MCPHttpClient` in `mcp/http_client.py`
   - `ServerClient` in `server_client.py`
   - Various transport-specific HTTP implementations
   - Each implements similar functionality differently

3. **Inconsistent API Usage**
   - `enhanced_llm.py` tries to use non-existent LLMRing methods
   - Mix of sync and async patterns
   - Unclear separation between MCP client variants

## Solution Architecture

### 1. Unified HTTP Client Base

**File: `llmring/net/http_base.py`** ✅ Created
- Single base class for all HTTP operations
- Consistent error handling
- Shared authentication logic
- Context manager support

### 2. Refactored MCP HTTP Client

**File: `llmring/mcp/http_client_refactored.py`** ✅ Created
- Inherits from `BaseHTTPClient`
- Clean API methods for all MCP operations
- No database dependencies
- Fully async

### 3. Fixed Enhanced LLM Interface

**File: `llmring/mcp/client/enhanced_llm_fixed.py`** ✅ Created
- Properly uses LLMRing's actual API
- Uses HTTP client for all persistence
- Clean tool registration and execution
- Proper conversation management

## Migration Steps

### Phase 1: Remove Database Dependencies ✅ Completed

1. **Remove pgdbm imports**
   ```python
   # OLD - Remove these
   from pgdbm import AsyncDatabaseManager
   
   # NEW - Use HTTP client
   from llmring.mcp.http_client import MCPHttpClient
   ```

2. **Update chat/app.py**
   - Remove all `AsyncDatabaseManager` references
   - Use `MCPHttpClient` for persistence
   - Update pool configuration

3. **Update test files**
   - Remove database fixtures
   - Mock HTTP calls instead

### Phase 2: Consolidate HTTP Clients

1. **Update existing clients to use BaseHTTPClient**
   ```python
   # Update server_client.py
   from llmring.net.http_base import BaseHTTPClient
   
   class ServerClient(BaseHTTPClient):
       # Remove duplicate methods
   ```

2. **Update transport implementations**
   - Use `BaseHTTPClient` for HTTP transports
   - Remove duplicate request handling

3. **Deprecate old implementations**
   - Mark old files as deprecated
   - Add migration warnings

### Phase 3: Fix MCP Client Architecture

1. **Simplify MCP client hierarchy**
   - Keep `AsyncMCPClient` as main class
   - Remove unnecessary variants
   - Use composition over inheritance

2. **Update enhanced_llm.py**
   - Replace with `enhanced_llm_fixed.py`
   - Test all tool execution paths
   - Verify conversation persistence

### Phase 4: Update Tests

1. **Mock HTTP calls**
   ```python
   # Example test with HTTP mocking
   import pytest
   from unittest.mock import AsyncMock, patch
   
   @pytest.mark.asyncio
   async def test_mcp_tool_execution():
       with patch('httpx.AsyncClient') as mock_client:
           mock_client.post.return_value = AsyncMock(
               json=lambda: {"result": "success"}
           )
           # Test implementation
   ```

2. **Remove database test dependencies**
   - No direct database connections in tests
   - Use HTTP mocks for all persistence tests

## File Changes Summary

### Files to Update
- [ ] `mcp/client/chat/app.py` - Remove database references
- [ ] `mcp/client/scripts/list_models.py` - Update to use HTTP
- [ ] `mcp/client/info_service.py` - Remove database dependencies
- [ ] `server_client.py` - Inherit from BaseHTTPClient
- [ ] Transport files - Use BaseHTTPClient

### Files to Replace
- [ ] `mcp/client/enhanced_llm.py` → `enhanced_llm_fixed.py`
- [ ] `mcp/http_client.py` → `http_client_refactored.py`

### Files to Add
- [x] `net/http_base.py` - Base HTTP client
- [x] `mcp/http_client_refactored.py` - Refactored MCP client
- [x] `mcp/client/enhanced_llm_fixed.py` - Fixed enhanced LLM

### Files to Deprecate
- [ ] Duplicate HTTP implementations in transports
- [ ] Old database-dependent MCP modules

## Testing Strategy

1. **Unit Tests**
   - Test each HTTP client method
   - Mock all external HTTP calls
   - Verify error handling

2. **Integration Tests**
   - Test against real llmring-server
   - Verify end-to-end MCP workflows
   - Test conversation persistence

3. **Migration Tests**
   - Ensure backward compatibility where needed
   - Test deprecated code paths still work
   - Verify migration warnings appear

## Benefits

1. **Clean Architecture**
   - Single source of truth for HTTP operations
   - Clear separation of concerns
   - No hidden database dependencies

2. **Maintainability**
   - Less code duplication
   - Consistent error handling
   - Easier to test

3. **Scalability**
   - Stateless MCP clients
   - Can scale horizontally
   - Database connections only in server

## Timeline

- **Week 1**: Remove database dependencies
- **Week 2**: Consolidate HTTP clients
- **Week 3**: Fix MCP client architecture
- **Week 4**: Update tests and documentation

## Success Criteria

- [ ] No database imports in MCP client code
- [ ] All persistence via HTTP endpoints
- [ ] Single base HTTP client used everywhere
- [ ] All tests pass with HTTP mocks
- [ ] Documentation updated

## Notes

- Keep backward compatibility where possible
- Add deprecation warnings for old code
- Update examples to use new patterns
- Consider performance implications of HTTP-only approach