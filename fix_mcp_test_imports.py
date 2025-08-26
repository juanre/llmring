#!/usr/bin/env python3
"""Fix MCP test imports to use llmring.mcp instead of mcp_client/mcp_server."""

import os
import re
from pathlib import Path

def fix_imports(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix imports
    replacements = [
        # mcp_server imports
        (r'from mcp_server import', 'from llmring.mcp.server import'),
        (r'import mcp_server', 'import llmring.mcp.server'),
        (r'from mcp_server\.', 'from llmring.mcp.server.'),
        
        # mcp_client imports  
        (r'from mcp_client import', 'from llmring.mcp.client import'),
        (r'import mcp_client', 'import llmring.mcp.client'),
        (r'from mcp_client\.', 'from llmring.mcp.client.'),
        
        # Also fix any Message imports
        (r'from mcp_client.schemas import Message', 'from llmring.schemas import Message'),
        (r'from llmring.mcp.client.schemas import Message', 'from llmring.schemas import Message'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    return False

def main():
    """Fix all test files."""
    test_files = [
        'tests/mcp/server/test_context_support.py',
        'tests/mcp/server/test_protocol_compliance.py', 
        'tests/mcp/server/test_missing_compliance.py',
        'tests/mcp/client/test_transports_integration.py',
        'tests/mcp/client/fixtures/real_server.py',
        'tests/mcp/client/fixtures/real_server_simple.py',
        'tests/mcp/client/fixtures/mcp_test_server.py',
        'tests/mcp/client/conftest.py',
    ]
    
    fixed = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            if fix_imports(file_path):
                fixed += 1
    
    print(f"\nFixed {fixed} files")

if __name__ == '__main__':
    main()