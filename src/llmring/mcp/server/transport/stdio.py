"""
STDIO transport implementation for MCP Server Engine.

Provides communication over standard input/output streams, ideal for local
subprocess-based integrations. Follows the MCP STDIO transport specification.
"""

import asyncio
import io
import json
import logging
import sys
from typing import Any, Dict, Optional

from llmring.mcp.server.transport.base import Transport

logger = logging.getLogger(__name__)


class StdioTransport(Transport):
    """
    STDIO transport implementation for MCP.

    Enables communication through standard input/output streams,
    which is particularly useful for local integrations where the
    server runs as a subprocess of the client.

    Key features:
    - Messages sent via stdout (newline-delimited JSON)
    - Messages received via stdin
    - Logging sent to stderr
    - No authentication required (process-based security)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the STDIO transport.

        Args:
            logger: Optional logger for transport-specific logging
        """
        super().__init__(logger=logger or logging.getLogger(__name__))

        # Ensure logging goes to stderr, not stdout
        self._configure_stderr_logging()

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer_lock = asyncio.Lock()
        self._running = False
        self._read_task: Optional[asyncio.Agent] = None
        self._test_mode = False
        self._pipe_transport = None  # Store transport for cleanup

    async def _test_feed_stdin(self) -> None:
        """Feed data from stdin in test mode (StringIO)."""
        try:
            while self._running:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                # Feed to reader
                self._reader.feed_data(line.encode("utf-8"))
                await asyncio.sleep(0.01)  # Small delay to simulate I/O
            self._reader.feed_eof()
        except Exception as e:
            self.logger.error(f"Test feed error: {e}")

    def _log(self, message: str) -> None:
        """Log message to stderr for testing."""
        print(message, file=sys.stderr)

    def _configure_stderr_logging(self):
        """Configure logging to use stderr exclusively."""
        # Remove any existing handlers that might write to stdout
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if hasattr(handler, "stream") and handler.stream in (
                sys.stdout,
                sys.__stdout__,
            ):
                root_logger.removeHandler(handler)

        # Ensure we have a stderr handler
        if not any(
            hasattr(h, "stream") and h.stream in (sys.stderr, sys.__stderr__)
            for h in root_logger.handlers
        ):
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root_logger.addHandler(stderr_handler)

    async def start(self) -> bool:
        """
        Start the STDIO transport.

        Sets up asyncio streams for stdin/stdout and begins
        listening for incoming messages.

        Returns:
            True if successfully started, False otherwise
        """
        if self._running:
            self.logger.warning("STDIO transport already running")
            return True

        try:
            # Set up stdin reader
            loop = asyncio.get_event_loop()
            self._reader = asyncio.StreamReader()

            # Check if stdin is a real file or a test mock
            try:
                # Try to get fileno - this will fail for StringIO
                sys.stdin.fileno()
                protocol = asyncio.StreamReaderProtocol(self._reader)
                self._pipe_transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)
            except (AttributeError, io.UnsupportedOperation):
                # For testing with StringIO
                self._test_mode = True
                asyncio.create_task(self._test_feed_stdin())

            # Start read agent
            self._running = True
            self._read_task = asyncio.create_task(self._read_loop())

            self.logger.info("STDIO transport started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start STDIO transport: {e}")
            self._handle_error(e)
            return False

    async def stop(self) -> None:
        """
        Stop the STDIO transport.

        Cancels the read agent and cleans up resources.
        """
        if not self._running:
            return

        self._running = False

        # Cancel read agent
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        # Clean up pipe transport if it exists
        if self._pipe_transport:
            self._pipe_transport.close()
            self._pipe_transport = None

        self.logger.info("STDIO transport stopped")
        self._handle_close()

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a JSON-RPC message to stdout.

        Messages are JSON-encoded and written as a single line
        (newline-terminated) to stdout.

        Args:
            message: The JSON-RPC message to send

        Returns:
            True if successfully sent, False otherwise
        """
        if not self._running:
            self.logger.warning("Cannot send message: transport not running")
            return False

        async with self._writer_lock:
            try:
                # Ensure no embedded newlines
                json_str = json.dumps(message, separators=(",", ":"))
                if "\n" in json_str:
                    raise ValueError("Message contains embedded newline")

                # Write to stdout with exactly one newline
                sys.stdout.write(json_str + "\n")
                sys.stdout.flush()

                self.logger.debug(f"Sent message: {message.get('method') or message.get('id')}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to send message: {e}")
                self._handle_error(e)
                return False

    async def _read_loop(self) -> None:
        """
        Read messages from stdin continuously.

        Handles partial reads and message boundaries correctly.
        Messages are expected to be newline-delimited JSON.
        """
        if not self._reader:
            self.logger.error("Cannot read messages: stdin not connected")
            return

        buffer = ""

        try:
            while self._running:
                try:
                    # Read chunk from stdin
                    data = await self._reader.read(4096)
                    if not data:  # EOF
                        self.logger.info("stdin closed, shutting down")
                        break

                    # Decode and add to buffer
                    buffer += data.decode("utf-8")

                    # Process complete messages
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)

                        if line.strip():  # Skip empty lines
                            await self._process_line(line)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Error reading from stdin: {e}")
                    self._handle_error(e)
                    await asyncio.sleep(0.1)  # Prevent tight loop on errors

        except asyncio.CancelledError:
            self.logger.debug("Read loop cancelled")
        finally:
            # Process any remaining data
            if buffer.strip():
                await self._process_line(buffer)

            # If we exited the loop naturally, stop the transport
            if self._running:
                await self.stop()

    async def _process_line(self, line: str) -> None:
        """
        Process a single line of input.

        Args:
            line: A line containing a JSON-RPC message
        """
        try:
            message = json.loads(line)

            # Validate basic JSON-RPC structure
            if not isinstance(message, dict):
                raise ValueError("Message must be a JSON object")

            if "jsonrpc" not in message or message["jsonrpc"] != "2.0":
                raise ValueError("Invalid or missing JSON-RPC version")

            self.logger.debug(f"Received message: {message.get('method') or message.get('id')}")

            # Deliver to handler (synchronously)
            self._handle_message(message)

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON received: {e}")
            # Send parse error response as per JSON-RPC 2.0 spec
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": {"line": line[:100]},
                },
            }
            # Create agent to send response
            asyncio.create_task(self.send_message(error_response))
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")


class StdioServerTransport(StdioTransport):
    """
    STDIO transport implementation optimized for MCP servers.

    This is the same as StdioTransport but exists for clarity
    and potential future server-specific enhancements.
    """

    pass
