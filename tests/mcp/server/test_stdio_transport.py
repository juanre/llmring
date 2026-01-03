import asyncio
import json
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest

from llmring.mcp.server.transport.stdio import StdioTransport


class TestStdioTransport:
    """Test suite for StdioTransport implementation."""

    @pytest.fixture
    def transport(self):
        """Create a StdioTransport instance."""
        return StdioTransport()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock message handler."""
        return MagicMock()

    async def test_send_message_valid(self, transport):
        """Test sending a valid JSON-RPC message."""
        message = {
            "jsonrpc": "2.0",
            "method": "test",
            "params": {"key": "value"},
            "id": 1,
        }

        # Start transport first
        with patch("sys.stdin", StringIO("")):
            await transport.start()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = await transport.send_message(message)
            assert result is True

            output = mock_stdout.getvalue()
            assert output.endswith("\n")
            assert output.count("\n") == 1

            # Parse the JSON to ensure it's valid
            parsed = json.loads(output.strip())
            assert parsed == message

        await transport.stop()

    async def test_send_message_with_embedded_newline(self, transport):
        """Test that messages with embedded newlines are rejected."""
        message = {
            "jsonrpc": "2.0",
            "method": "test",
            "params": {"text": "line1\nline2"},
            "id": 1,
        }

        # Start transport first
        with patch("sys.stdin", StringIO("")):
            await transport.start()

        # Create a custom JSON encoder that doesn't escape newlines
        class BadEncoder(json.JSONEncoder):
            def encode(self, o):
                if isinstance(o, dict) and "params" in o:
                    # Force an unescaped newline
                    return (
                        '{"jsonrpc":"2.0","method":"test","params":{"text":"line1\nline2"},"id":1}'
                    )
                return super().encode(o)

        with patch("json.dumps", BadEncoder().encode):
            with patch("sys.stdout", new_callable=StringIO):
                result = await transport.send_message(message)
                assert result is False

        await transport.stop()

    async def test_send_message_invalid_json(self, transport):
        """Test sending invalid JSON data."""
        # Not a valid JSON-RPC message (missing jsonrpc field)
        message = {"method": "test"}

        # Start transport first
        with patch("sys.stdin", StringIO("")):
            await transport.start()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Should still send it (transport doesn't validate structure)
            result = await transport.send_message(message)
            assert result is True

            output = mock_stdout.getvalue()
            parsed = json.loads(output.strip())
            assert parsed == message

        await transport.stop()

    async def test_send_notification(self, transport):
        """Test sending a notification (no id field)."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notification",
            "params": {"data": "test"},
        }

        # Start transport first
        with patch("sys.stdin", StringIO("")):
            await transport.start()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = await transport.send_message(notification)
            assert result is True

            output = mock_stdout.getvalue()
            parsed = json.loads(output.strip())
            assert parsed == notification
            assert "id" not in parsed

        await transport.stop()

    async def test_concurrent_sends(self, transport):
        """Test that concurrent sends are serialized."""
        messages = [
            {"jsonrpc": "2.0", "method": "test1", "id": 1},
            {"jsonrpc": "2.0", "method": "test2", "id": 2},
            {"jsonrpc": "2.0", "method": "test3", "id": 3},
        ]

        # Start transport with proper mocking
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Mock stdin to keep transport running
            mock_stdin = Mock()
            mock_stdin.fileno = Mock(return_value=0)

            # Create a future that never completes to keep transport running
            never_complete = asyncio.Future()

            async def mock_readline():
                await never_complete

            mock_stdin.readline = Mock(side_effect=mock_readline)

            with patch("sys.stdin", mock_stdin):
                await transport.start()

                # Give transport time to start
                await asyncio.sleep(0.1)

                # Send messages concurrently
                agents = [transport.send_message(msg) for msg in messages]
                results = await asyncio.gather(*agents)

                assert all(results)

                # Check that all messages were written
                output = mock_stdout.getvalue()
                lines = output.strip().split("\n")
                assert len(lines) == 3

                # Parse each line to ensure valid JSON
                parsed_messages = [json.loads(line) for line in lines]
                assert len(parsed_messages) == 3

                # Check that all messages are present (order may vary)
                received_ids = [msg["id"] for msg in parsed_messages]
                assert sorted(received_ids) == [1, 2, 3]

                await transport.stop()

    async def test_receive_valid_message(self, transport, mock_handler):
        """Test receiving a valid JSON-RPC message."""
        transport.set_message_callback(mock_handler)

        message = '{"jsonrpc":"2.0","method":"test","params":{},"id":1}\n'

        with patch("sys.stdin", StringIO(message)):
            await transport.start()

            # Give the read agent time to process
            await asyncio.sleep(0.1)

            # Check that handler was called
            mock_handler.assert_called_once()
            received_msg = mock_handler.call_args[0][0]
            assert received_msg["method"] == "test"
            assert received_msg["id"] == 1

            await transport.stop()

    async def test_receive_multiple_messages(self, transport, mock_handler):
        """Test receiving multiple messages in sequence."""
        transport.set_message_callback(mock_handler)

        messages = (
            '{"jsonrpc":"2.0","method":"test1","id":1}\n'
            '{"jsonrpc":"2.0","method":"test2","id":2}\n'
            '{"jsonrpc":"2.0","method":"test3","id":3}\n'
        )

        with patch("sys.stdin", StringIO(messages)):
            await transport.start()

            # Give the read agent time to process all messages
            await asyncio.sleep(0.1)

            # Check that handler was called for each message
            assert mock_handler.call_count == 3

            # Verify each message
            calls = mock_handler.call_args_list
            assert calls[0][0][0]["method"] == "test1"
            assert calls[1][0][0]["method"] == "test2"
            assert calls[2][0][0]["method"] == "test3"

            await transport.stop()

    async def test_receive_invalid_json(self, transport, mock_handler):
        """Test receiving invalid JSON data."""
        transport.set_message_callback(mock_handler)

        # Mix of valid and invalid messages
        messages = (
            '{"jsonrpc":"2.0","method":"valid","id":1}\n'
            "invalid json\n"
            '{"jsonrpc":"2.0","method":"valid2","id":2}\n'
        )

        with patch("sys.stdin", StringIO(messages)):
            await transport.start()

            # Give the read agent time to process
            await asyncio.sleep(0.1)

            # Handler should be called only for valid messages
            assert mock_handler.call_count == 2

            # Verify that valid messages were processed correctly
            calls = mock_handler.call_args_list
            assert calls[0][0][0]["method"] == "valid"
            assert calls[1][0][0]["method"] == "valid2"

            await transport.stop()

    async def test_empty_lines_ignored(self, transport, mock_handler):
        """Test that empty lines are ignored."""
        transport.set_message_callback(mock_handler)

        messages = (
            "\n"
            '{"jsonrpc":"2.0","method":"test","id":1}\n'
            "\n\n"
            '{"jsonrpc":"2.0","method":"test2","id":2}\n'
            "\n"
        )

        with patch("sys.stdin", StringIO(messages)):
            await transport.start()

            # Give the read agent time to process
            await asyncio.sleep(0.1)

            # Handler should be called only for non-empty messages
            assert mock_handler.call_count == 2

            await transport.stop()

    async def test_logging_to_stderr(self, transport):
        """Test that all logging goes to stderr, not stdout."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                # This should log to stderr
                transport._log("Test log message")

                # Check stdout is empty
                assert mock_stdout.getvalue() == ""

                # Check stderr contains the log
                assert "Test log message" in mock_stderr.getvalue()

    async def test_stop_without_start(self, transport):
        """Test stopping transport that was never started."""
        # Should not raise any errors
        await transport.stop()

    async def test_double_start(self, transport):
        """Test starting transport twice."""
        with patch("sys.stdin", StringIO("")):
            await transport.start()

            # Second start should be a no-op
            await transport.start()

            # Only one read agent should be created
            assert transport._read_task is not None

            await transport.stop()

    async def test_stdin_eof(self, transport, mock_handler):
        """Test handling EOF on stdin."""
        transport.set_message_callback(mock_handler)

        # Simulate EOF with empty StringIO
        with patch("sys.stdin", StringIO("")):
            with patch("sys.stderr", new_callable=StringIO):
                await transport.start()

                # Give the read agent time to process EOF
                await asyncio.sleep(0.1)

                # For test mode, the agent should complete normally
                # (EOF is handled in _test_feed_stdin)
                assert transport._test_mode is True

                await transport.stop()

    async def test_message_handler_exception(self, transport):
        """Test that exceptions in message handler don't crash transport."""
        handler_calls = []

        def failing_handler(message):
            handler_calls.append(message)
            raise ValueError("Handler error")

        transport.set_message_callback(failing_handler)

        # Multiple messages to ensure transport continues after error
        messages = (
            '{"jsonrpc":"2.0","method":"test1","id":1}\n{"jsonrpc":"2.0","method":"test2","id":2}\n'
        )

        with patch("sys.stdin", StringIO(messages)):
            with patch("sys.stderr", new_callable=StringIO):
                await transport.start()

                # Give the read agent time to process
                await asyncio.sleep(0.1)

                # Both messages should have been received despite errors
                assert len(handler_calls) == 2
                assert handler_calls[0]["method"] == "test1"
                assert handler_calls[1]["method"] == "test2"

                # Verify handler was called despite errors
                # This proves the transport continues working

                await transport.stop()

    async def test_large_message(self, transport):
        """Test sending and receiving large messages."""
        # Create a large message (1MB of data)
        large_data = "x" * (1024 * 1024)
        message = {
            "jsonrpc": "2.0",
            "method": "test",
            "params": {"data": large_data},
            "id": 1,
        }

        # Start transport first
        with patch("sys.stdin", StringIO("")):
            await transport.start()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = await transport.send_message(message)
            assert result is True

            output = mock_stdout.getvalue()
            assert output.endswith("\n")
            assert output.count("\n") == 1

            # Verify the message can be parsed back
            parsed = json.loads(output.strip())
            assert parsed["params"]["data"] == large_data

        await transport.stop()
