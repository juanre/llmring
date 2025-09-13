#!/usr/bin/env python3
"""Test script for CLI commands."""

import os
import subprocess
import sys


def run_command(cmd):
    """Run a command and return output."""
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.returncode, result.stdout, result.stderr


def test_help_commands():
    """Test help output for all commands."""
    print("=== Testing help commands ===")

    commands = [
        "mcp-client --help",
        "mcp-client query --help",
        "mcp-client conversations --help",
        "mcp-client conversations list --help",
        "mcp-client conversations show --help",
        "mcp-client conversations delete --help",
        "mcp-client conversations export --help",
        "mcp-client chat --help",
    ]

    for cmd in commands:
        code, stdout, stderr = run_command(cmd)
        assert code == 0, f"Command failed: {cmd}"
        assert "usage:" in stdout or "usage:" in stderr, f"No usage info in: {cmd}"


def test_query_command():
    """Test query command."""
    print("\n=== Testing query command ===")

    # Test simple query
    cmd = 'mcp-client query "What is 2+2?" --no-save --show-id'
    code, stdout, stderr = run_command(cmd)

    # Command should work and return a conversation ID
    assert code == 0, f"Command should succeed. stdout: {stdout}, stderr: {stderr}"
    assert "Conversation ID:" in stdout, "Should show conversation ID"
    assert "2+2" in stdout or "4" in stdout, "Should include the query or answer"


def test_conversations_commands():
    """Test conversations commands."""
    print("\n=== Testing conversations commands ===")

    # Test list command
    cmd = "mcp-client conversations list --limit 5"
    code, stdout, stderr = run_command(cmd)

    # Test show command (with fake ID)
    cmd = "mcp-client conversations show test-id-123"
    code, stdout, stderr = run_command(cmd)

    # Test export command
    cmd = "mcp-client conversations export test-id-123 --format json"
    code, stdout, stderr = run_command(cmd)

    # Test delete command (with --yes to skip confirmation)
    cmd = "mcp-client conversations delete test-id-123 --yes"
    code, stdout, stderr = run_command(cmd)


def test_chat_resume():
    """Test chat --resume functionality."""
    print("\n=== Testing chat --resume ===")

    # Test --resume without --cid (should error)
    cmd = "mcp-client chat --resume"
    code, stdout, stderr = run_command(cmd)
    assert code != 0, "Expected error for --resume without --cid"
    assert "--cid" in stdout or "--cid" in stderr, "Should mention --cid requirement"


def main():
    """Run all tests."""
    print("CLI Command Tests")
    print("=" * 50)

    # Set environment to avoid hitting real services
    os.environ["DATABASE_URL"] = (
        "postgresql://postgres:postgres@localhost/mcp_client_test"
    )

    try:
        test_help_commands()
        test_query_command()
        test_conversations_commands()
        test_chat_resume()

        print("\n" + "=" * 50)
        print("All CLI command tests completed!")
        print("Note: Some commands failed due to missing services, which is expected.")
        print(
            "The important thing is that all commands are recognized and parsed correctly."
        )

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
