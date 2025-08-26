# Using Real MCP Server in Tests

This guide explains how to use the real mcp-server implementation in mcp-client tests instead of mocks.

## Setup

1. **Install mcp-server as a test dependency**:
   ```bash
   cd /path/to/mcp-client
   uv add --dev mcp-server
   ```

2. **Import the fixtures**:
   ```python
   from tests.fixtures.real_server import (
       http_mcp_client,
       basic_mcp_server,
       stdio_mcp_server,
       make_jsonrpc_request,
       assert_jsonrpc_success,
       assert_jsonrpc_error
   )
   ```

## Available Fixtures

### `basic_mcp_server`
Creates a basic MCP server with test functions, resources, and prompts pre-registered.

### `http_mcp_client`
Creates an HTTP test client with proper MCP initialization already completed.

### `stdio_mcp_server`
Creates an MCP server running as a subprocess with stdio transport.

## Example Usage

### Simple Test
```python
def test_list_tools_with_real_server(http_mcp_client):
    client, server = http_mcp_client
    
    # Make a request
    request = make_jsonrpc_request("tools/list")
    response = client.post("/mcp/", json=request)
    
    # Assert success
    assert response.status_code == 200
    result = assert_jsonrpc_success(response.json())
    
    # Check tools
    tools = result["tools"]
    assert len(tools) > 0
```

### Testing Errors
```python
def test_error_handling(http_mcp_client):
    client, server = http_mcp_client
    
    # Call the error function
    request = make_jsonrpc_request(
        "tools/call",
        {"name": "error", "arguments": {"message": "test error"}}
    )
    response = client.post("/mcp/", json=request)
    
    # Should return MCP error format
    assert response.status_code == 200
    result = assert_jsonrpc_success(response.json())
    assert result["isError"] is True
    assert "test error" in result["content"][0]["text"]
```

### Dynamic Registration
```python
def test_dynamic_function_registration(http_mcp_client):
    client, server = http_mcp_client
    
    # Register a new function
    def custom_function(args):
        return f"Custom result: {args.get('input', '')}"
    
    server.register_function("custom", custom_function)
    
    # Call the new function
    request = make_jsonrpc_request(
        "tools/call",
        {"name": "custom", "arguments": {"input": "test"}}
    )
    response = client.post("/mcp/", json=request)
    
    result = assert_jsonrpc_success(response.json())
    assert "Custom result: test" in result["content"][0]["text"]
```

## Migration Guide

### Before (with mocks):
```python
def test_list_resources(self):
    client = MCPClient("http://test.example.com")
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = [{"uri": "file:///test.txt"}]
        result = client.list_resources()
        
    assert len(result) == 1
```

### After (with real server):
```python
def test_list_resources(self, http_mcp_client):
    test_client, server = http_mcp_client
    
    # The server already has test resources registered
    request = make_jsonrpc_request("resources/list")
    response = test_client.post("/mcp/", json=request)
    
    result = assert_jsonrpc_success(response.json())
    assert len(result["resources"]) == 2  # Pre-registered resources
```

## Benefits

1. **Real Protocol Testing**: Tests verify actual MCP protocol compliance
2. **No Mock Maintenance**: No need to update mocks when server changes
3. **Error Testing**: Real server error responses
4. **Integration Coverage**: Tests the full stack

## Running Tests

Tests using real server fixtures will be automatically skipped if mcp-server is not installed:

```bash
# Run all tests (skips real server tests if mcp-server not available)
pytest

# Run only real server integration tests
pytest tests/integration/test_*_real_server.py

# Run with verbose output to see skipped tests
pytest -v
```

## Debugging

The real server logs to stderr, so you can see server-side logs during test failures.

To enable more verbose logging:
```python
import logging
logging.getLogger("mcp_server").setLevel(logging.DEBUG)
```

## Performance Considerations

Real server tests are slower than mocked tests. Consider:
- Keep real server tests in `tests/integration/`
- Use mocks for unit tests that don't need protocol validation
- Run integration tests separately in CI if needed

## Next Steps

1. Gradually migrate existing mock-based tests
2. Add more complex test scenarios (auth, transport variants)
3. Create transport-specific fixtures (SSE, WebSocket)
4. Document patterns for contributors