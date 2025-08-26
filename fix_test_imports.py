#!/usr/bin/env python
"""Fix test imports to use llmring.mcp instead of mcp_client."""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace mcp_client imports with llmring.mcp.client
    content = re.sub(r'from mcp_client\.', 'from llmring.mcp.client.', content)
    content = re.sub(r'import mcp_client\.', 'import llmring.mcp.client.', content)
    content = re.sub(r'import mcp_client\b', 'import llmring.mcp.client', content)
    
    # Replace mcp_server imports with llmring.mcp.server
    content = re.sub(r'from mcp_server\.', 'from llmring.mcp.server.', content)
    content = re.sub(r'import mcp_server\.', 'import llmring.mcp.server.', content)
    content = re.sub(r'import mcp_server\b', 'import llmring.mcp.server', content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    return False

def main():
    test_dir = Path('/Users/juanre/prj/llmring-all/llmring/tests/mcp')
    
    fixed_count = 0
    for py_file in test_dir.rglob('*.py'):
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()