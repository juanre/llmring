#!/usr/bin/env python3
"""Fix MCP client test imports to match actual module structure."""

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
        # Fix client imports
        (r'from llmring\.mcp\.client\.client import', 'from llmring.mcp.client.mcp_client import'),
        
        # Fix models.db imports (no longer exists)
        (r'from llmring\.mcp\.client\.models\.db import MCPClientDB', '# MCPClientDB removed - using HTTP'),
        
        # Fix shared_pool imports (no longer exists)
        (r'from llmring\.mcp\.client\.shared_pool import', '# shared_pool removed - using HTTP'),
        
        # Fix Message imports 
        (r'from llmring\.mcp\.client\.schemas import Message', 'from llmring.schemas import Message'),
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
    # Find all Python files in tests/mcp/client
    test_dir = Path('tests/mcp/client')
    test_files = list(test_dir.rglob('*.py'))
    
    fixed = 0
    for file_path in test_files:
        if fix_imports(str(file_path)):
            fixed += 1
    
    print(f"\nFixed {fixed} files")

if __name__ == '__main__':
    main()