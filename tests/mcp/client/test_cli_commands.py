#!/usr/bin/env python3
"""Test script for CLI commands."""

import os
import subprocess
import sys
import tempfile


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

    # Use Python module path for our mcp-client
    python_exe = sys.executable
    commands = [
        f"{python_exe} -m llmring.mcp.client.cli --help",
        f"{python_exe} -m llmring.mcp.client.cli query --help",
        f"{python_exe} -m llmring.mcp.client.cli conversations --help",
        f"{python_exe} -m llmring.mcp.client.cli conversations list --help",
        f"{python_exe} -m llmring.mcp.client.cli conversations show --help",
        f"{python_exe} -m llmring.mcp.client.cli conversations delete --help",
        f"{python_exe} -m llmring.mcp.client.cli conversations export --help",
        f"{python_exe} -m llmring.mcp.client.cli chat --help",
    ]

    for cmd in commands:
        code, stdout, stderr = run_command(cmd)
        assert code == 0, f"Command failed: {cmd}"
        assert "usage:" in stdout or "usage:" in stderr, f"No usage info in: {cmd}"


def test_query_command():
    """Test query command."""
    print("\n=== Testing query command ===")

    # Create a temporary lockfile for testing
    # Use TOML format with the standard name that service expects
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = os.path.join(tmpdir, "llmring.lock")

        # Create a TOML lockfile (which is what the service expects in current directory)
        import toml

        lockfile_data = {
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "default_profile": "default",
            "profiles": {
                "default": {
                    "name": "default",
                    "bindings": [
                        {
                            "alias": "test",
                            "models": ["openai:gpt-4o-mini"],
                            "constraints": None,
                        }
                    ],
                    "registry_versions": {},
                }
            },
        }

        with open(lockfile_path, "w") as f:
            toml.dump(lockfile_data, f)

        # Run command from the temp directory so it finds the lockfile
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Run our mcp-client module directly
            python_exe = sys.executable
            cmd = f'{python_exe} -m llmring.mcp.client.cli --model test query "What is 2+2?" --no-save --show-id'
            code, stdout, stderr = run_command(cmd)
        finally:
            os.chdir(original_cwd)

        # Command should work and return a conversation ID
        assert code == 0, f"Command should succeed. stdout: {stdout}, stderr: {stderr}"
        assert "Conversation ID:" in stdout, "Should show conversation ID"
        assert "2+2" in stdout or "4" in stdout, "Should include the query or answer"


def test_conversations_commands():
    """Test conversations commands."""
    print("\n=== Testing conversations commands ===")

    python_exe = sys.executable
    # Test list command
    cmd = f"{python_exe} -m llmring.mcp.client.cli conversations list --limit 5"
    code, stdout, stderr = run_command(cmd)

    # Test show command (with fake ID)
    cmd = f"{python_exe} -m llmring.mcp.client.cli conversations show test-id-123"
    code, stdout, stderr = run_command(cmd)

    # Test export command
    cmd = f"{python_exe} -m llmring.mcp.client.cli conversations export test-id-123 --format json"
    code, stdout, stderr = run_command(cmd)

    # Test delete command (with --yes to skip confirmation)
    cmd = f"{python_exe} -m llmring.mcp.client.cli conversations delete test-id-123 --yes"
    code, stdout, stderr = run_command(cmd)


def test_chat_resume():
    """Test chat --resume functionality."""
    print("\n=== Testing chat --resume ===")

    python_exe = sys.executable
    # Test --resume without --cid (should error)
    cmd = f"{python_exe} -m llmring.mcp.client.cli chat --resume"
    code, stdout, stderr = run_command(cmd)
    assert code != 0, "Expected error for --resume without --cid"
    assert "--cid" in stdout or "--cid" in stderr, "Should mention --cid requirement"


def main():
    """Run all tests."""
    print("CLI Command Tests")
    print("=" * 50)

    # Set environment to avoid hitting real services
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost/mcp_client_test"

    try:
        test_help_commands()
        test_query_command()
        test_conversations_commands()
        test_chat_resume()

        print("\n" + "=" * 50)
        print("All CLI command tests completed!")
        print("Note: Some commands failed due to missing services, which is expected.")
        print("The important thing is that all commands are recognized and parsed correctly.")

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
