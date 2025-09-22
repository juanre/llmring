"""
Example math tools for MCP server testing.

Contains basic mathematical operations that can be used to test the MCP server
functionality. All functions follow the mcp_ naming convention.
"""


def mcp_add(args):
    """Add two numbers."""
    a = args.get("a", 0)
    b = args.get("b", 0)
    return f"Result: {a + b}"


def mcp_multiply(args):
    """Multiply two numbers."""
    a = args.get("a", 1)
    b = args.get("b", 1)
    return f"Result: {a * b}"


def mcp_subtract(args):
    """Subtract two numbers."""
    a = args.get("a", 0)
    b = args.get("b", 0)
    return f"Result: {a - b}"


def mcp_divide(args):
    """Divide two numbers."""
    a = args.get("a", 0)
    b = args.get("b", 1)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return f"Result: {a / b}"


def mcp_echo(args):
    """Echo the input message."""
    message = args.get("message", "")
    return f"Echo: {message}"


def mcp_greet(args):
    """Greet someone by name."""
    name = args.get("name", "World")
    return f"Hello, {name}!"
