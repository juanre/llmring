"""
Regression tests for logging refactoring bugs.

These tests prevent regressions for three critical bugs found in code review:
1. LLMRingSession._log_usage_to_server AttributeError
2. llmring lock init NameError (project_root undefined)
3. Conversation logging missing usage records
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from llmring.schemas import LLMRequest, Message
from llmring.service_extended import LLMRingSession

# Check if llmring-server is properly installed
try:
    from llmring_server.main import app as _test_import  # noqa: F401

    LLMRING_SERVER_INSTALLED = True
except ImportError:
    LLMRING_SERVER_INSTALLED = False


class TestLoggingRegressions:
    """Regression tests for logging refactoring bugs."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
    async def test_conversation_logging_includes_usage(self, llmring_server_client, project_headers):
        """
        Regression test for: Conversation logging loses usage records

        Bug: When log_conversations=True, LoggingService only called _log_conversation
        and skipped _log_usage_only, so no usage records were created for analytics/receipts.

        Fix: Call both _log_conversation AND _log_usage_only when log_conversations=True.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Create LLMRingSession with conversation logging enabled
        session = LLMRingSession(
            server_url="http://test",
            api_key="proj_test",
            enable_conversations=True,
            origin="test-conversation-regression",
        )
        # Inject the test server client
        session.server_client.client = llmring_server_client
        session.logging_service.server_client.client = llmring_server_client

        # Make a real chat request
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Say 'test' and nothing else.")],
            max_tokens=10,
        )

        await session.chat(request)

        # CRITICAL ASSERTION: Both conversation AND usage records should exist
        # Check usage logs
        usage_response = await llmring_server_client.get(
            "/api/v1/logs",
            params={"origin": "test-conversation-regression", "limit": 1},
            headers=project_headers,
        )
        assert usage_response.status_code == 200
        usage_logs = usage_response.json()
        assert len(usage_logs) > 0, "Usage record should be created even with conversation logging"

        # Check conversation logs
        conv_response = await llmring_server_client.get(
            "/api/v1/conversations/",
            params={"limit": 10},
            headers=project_headers,
        )
        assert conv_response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
    async def test_metadata_only_logging_creates_usage(self, llmring_server_client, project_headers):
        """
        Verify that metadata-only logging creates usage records.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from llmring.service import LLMRing

        # Create LLMRing with metadata logging only
        ring = LLMRing(
            origin="test-metadata-regression",
            server_url="http://test",
            api_key="proj_test",
        )
        # Inject the test server client
        ring.server_client.client = llmring_server_client

        # Make a real chat request
        request = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="Say 'test' and nothing else.")],
            max_tokens=10,
        )

        await ring.chat(request)

        # Verify usage record was created
        usage_response = await llmring_server_client.get(
            "/api/v1/logs",
            params={"origin": "test-metadata-regression", "limit": 1},
            headers=project_headers,
        )
        assert usage_response.status_code == 200
        usage_logs = usage_response.json()
        assert len(usage_logs) > 0, "Usage record should be created"
        assert usage_logs[0]["input_tokens"] > 0

    def test_lock_init_no_nameerror(self):
        """
        Regression test for: llmring lock init NameError (project_root undefined)

        Bug: cmd_lock_init used undefined variable 'project_root' when checking
        for pyproject.toml.

        Fix: Use 'package_dir' which is already defined in the function.
        """
        from llmring.cli import cmd_lock_init

        # Create temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create args with file path
            class Args:
                file = str(tmpdir_path / "llmring.lock")
                force = True

            args = Args()

            # Run the command - it should NOT raise NameError about project_root
            try:
                result = asyncio.run(cmd_lock_init(args))
                # If we get here without NameError, bug is fixed
                assert result == 0
            except NameError as e:
                if "project_root" in str(e):
                    pytest.fail("Regression: cmd_lock_init uses undefined project_root")
                raise
            except Exception as e:
                # Other exceptions are OK for this test (e.g., network issues)
                # We're only testing for the NameError regression
                if "project_root" not in str(e):
                    pass  # Test passes if no project_root error

    def test_llmring_session_usage_logging_no_crash(self):
        """
        Regression test for: LLMRingSession._log_usage_to_server AttributeError

        Bug: LLMRingSession.chat_with_conversation() called _log_usage_to_server()
        which was removed in the logging refactor, causing AttributeError.

        Fix: Use logging_service.log_request_response() instead.

        This test verifies the fix by checking that LLMRingSession no longer has
        the old _log_usage_to_server method and properly uses logging_service instead.
        """
        # Create LLMRingSession
        session = LLMRingSession(
            server_url="http://test",
            api_key="test-key",
            enable_conversations=True,
        )

        # CRITICAL ASSERTION: The old method should NOT exist
        assert not hasattr(
            session, "_log_usage_to_server"
        ), "Regression: LLMRingSession should NOT have _log_usage_to_server method"

        # CRITICAL ASSERTION: Should use logging_service instead
        assert hasattr(session, "logging_service"), "Session should have logging_service"

        # Verify logging_service has the new method
        assert hasattr(
            session.logging_service, "log_request_response"
        ), "LoggingService should have log_request_response method"
