"""
STDIO transport implementation for MCP client.

Enables communication with MCP servers running as local processes via
stdin/stdout, as required by the MCP specification for local integrations.
"""

import asyncio
import contextlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

from llmring.mcp.client.transports.base import ConnectionState, Transport

logger = logging.getLogger(__name__)

# Default allowed commands for security
DEFAULT_ALLOWED_COMMANDS = {
    "python",
    "python3",
    "python.exe",
    "node",
    "node.exe",
    "npm",
    "npx",
    "deno",
    "bun",
    "ruby",
    "ruby.exe",
    "java",
    "java.exe",
    "dotnet",
    "dotnet.exe",
    "go",
    "go.exe",
    "php",
    "php.exe",
    "perl",
    "perl.exe",
    "julia",
    "julia.exe",
    "bash",
    "sh",
    "zsh",  # Unix shells
    "cmd.exe",
    "powershell.exe",  # Windows shells
    "mcp-server",  # MCP server entry point
}

# Environment variables that are safe to pass to subprocesses
SAFE_ENV_PREFIXES = (
    "PATH",
    "HOME",
    "USER",
    "USERNAME",
    "LANG",
    "LC_",
    "PYTHON",
    "NODE",
    "NPM",
    "JAVA_HOME",
    "GOPATH",
    "RUBY",
    "GEM",
    "CARGO",
    "RUSTUP",
    "TEMP",
    "TMP",
    "TMPDIR",
    "SYSTEMROOT",
    "WINDIR",  # Windows system paths
)

# Dangerous environment variables to always exclude
DANGEROUS_ENV_VARS = {
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "__CF_USER_TEXT_ENCODING",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "DATABASE_URL",
    "DB_PASSWORD",
    "API_KEY",
    "API_SECRET",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "SSH_AUTH_SOCK",
}


class STDIOTransport(Transport):
    """
    STDIO transport implementation for MCP client.

    Communicates with MCP servers running as local processes via stdin/stdout
    using newline-delimited JSON-RPC messages.
    """

    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        restart_on_failure: bool = True,
        max_restart_attempts: int = 3,
        allowed_commands: set[str] | None = None,
        allow_unsafe_commands: bool = False,
        filter_env: bool = True,
        resource_limits: dict[str, Any] | None = None,
    ):
        """
        Initialize STDIO transport with security features.

        Args:
            command: Command and arguments to start the MCP server process
                    e.g., ["python", "server.py"] or ["node", "server.js"]
            cwd: Working directory for the process
            env: Environment variables for the process
            timeout: Timeout for process operations in seconds
            restart_on_failure: Whether to automatically restart failed processes
            max_restart_attempts: Maximum number of restart attempts
            allowed_commands: Set of allowed command names (defaults to DEFAULT_ALLOWED_COMMANDS)
            allow_unsafe_commands: If True, disables command validation (security risk!)
            filter_env: If True, filters environment variables for security
            resource_limits: Resource limits for subprocess (memory, cpu, etc.)
        """
        super().__init__()

        # Security settings
        self.allowed_commands = allowed_commands or DEFAULT_ALLOWED_COMMANDS
        self.allow_unsafe_commands = allow_unsafe_commands
        self.filter_env = filter_env
        self.resource_limits = resource_limits or {}

        # Validate command before storing
        self._validate_command(command)
        self.command = command

        # Validate and normalize cwd
        self.cwd = self._validate_cwd(cwd)
        self.env = env
        self.timeout = timeout
        self.restart_on_failure = restart_on_failure
        self.max_restart_attempts = max_restart_attempts

        # Process management
        self.process: asyncio.subprocess.Process | None = None
        self.restart_count = 0

        # Message handling
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._reader_task: asyncio.Agent | None = None
        self._protocol_version: str | None = None

    async def start(self) -> None:
        """Start the STDIO transport and subprocess."""
        logger.debug(f"STDIOTransport.start() called, current state: {self.state}")
        self._set_state(ConnectionState.CONNECTING)
        logger.debug("Set state to CONNECTING")

        try:
            await self._start_process()
            self._set_state(ConnectionState.CONNECTED)
            logger.info(f"STDIO transport connected to process: {self.command}")
            logger.debug(f"Set state to CONNECTED, current state: {self.state}")
        except Exception as e:
            self._set_state(ConnectionState.ERROR)
            logger.error(f"Failed to start process: {e}")
            self._handle_error(e)
            raise

    def _validate_command(self, command: list[str]) -> None:
        """Validate command for security risks."""
        if not command:
            raise ValueError("Command cannot be empty")

        if self.allow_unsafe_commands:
            logger.warning("Command validation disabled - security risk!")
            return

        # Extract the base command (first element)
        base_command = command[0]

        # Handle absolute paths by extracting just the command name
        command_name = os.path.basename(base_command).lower()

        # Remove common extensions for comparison
        for ext in [".exe", ".bat", ".cmd", ".sh"]:
            if command_name.endswith(ext):
                command_name = command_name[: -len(ext)]

        # Check if command is in allowed list
        if command_name not in self.allowed_commands:
            raise ValueError(
                f"Command '{command_name}' is not in allowed commands list. "
                f"Use allow_unsafe_commands=True to bypass (not recommended) or "
                f"add to allowed_commands set."
            )

        # Additional security checks for shell injection
        for arg in command[1:]:
            if any(char in arg for char in ["&", "|", ";", "$", "`", "\n", "\r"]):
                logger.warning(f"Potentially dangerous character in command argument: {arg}")

    def _validate_cwd(self, cwd: str | None) -> str | None:
        """Validate and normalize working directory."""
        if cwd is None:
            return None

        try:
            # Resolve to absolute path and check it exists
            cwd_path = Path(cwd).resolve()
            if not cwd_path.exists():
                raise ValueError(f"Working directory does not exist: {cwd}")
            if not cwd_path.is_dir():
                raise ValueError(f"Working directory is not a directory: {cwd}")
            return str(cwd_path)
        except Exception as e:
            raise ValueError(f"Invalid working directory: {e}")

    def _filter_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Filter environment variables for security."""
        if not self.filter_env:
            return env

        filtered_env = {}

        for key, value in env.items():
            # Skip dangerous variables
            if key.upper() in DANGEROUS_ENV_VARS:
                logger.debug(f"Filtering out dangerous environment variable: {key}")
                continue

            # Skip variables that don't match safe prefixes
            if not any(key.upper().startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
                # Allow explicitly passed variables from self.env
                if self.env and key in self.env:
                    filtered_env[key] = value
                else:
                    logger.debug(f"Filtering out environment variable: {key}")
                continue

            filtered_env[key] = value

        # Always include explicitly passed environment variables
        if self.env:
            filtered_env.update(self.env)

        return filtered_env

    async def _start_process(self) -> None:
        """Start the MCP server subprocess with security measures."""
        # Prepare environment with security filtering
        if self.filter_env:
            process_env = self._filter_environment(os.environ.copy())
        else:
            process_env = os.environ.copy()

        if self.env:
            process_env.update(self.env)

        # Platform-specific process creation
        kwargs = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": self.cwd,
            "env": process_env,
        }

        # Add resource limits on Unix-like systems
        if platform.system() != "Windows" and self.resource_limits:
            import resource

            def set_limits():
                """Set resource limits for the subprocess."""
                if "memory" in self.resource_limits:
                    # Memory limit in bytes
                    resource.setrlimit(
                        resource.RLIMIT_AS,
                        (
                            self.resource_limits["memory"],
                            self.resource_limits["memory"],
                        ),
                    )

                if "cpu" in self.resource_limits:
                    # CPU time limit in seconds
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (self.resource_limits["cpu"], self.resource_limits["cpu"]),
                    )

                if "files" in self.resource_limits:
                    # Open file descriptor limit
                    resource.setrlimit(
                        resource.RLIMIT_NOFILE,
                        (self.resource_limits["files"], self.resource_limits["files"]),
                    )

            kwargs["preexec_fn"] = set_limits

        try:
            # Create subprocess with security measures
            self.process = await asyncio.create_subprocess_exec(*self.command, **kwargs)

            # Start background agent to read responses
            self._reader_task = asyncio.create_task(self._read_responses())

            # Also start reading stderr for debugging
            asyncio.create_task(self._read_stderr())

            logger.debug(f"Started subprocess with PID: {self.process.pid}")

            # Give the server process time to start
            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Failed to start subprocess: {e}")
            raise

    async def _read_stderr(self) -> None:
        """Read stderr for debugging."""
        if not self.process or not self.process.stderr:
            return

        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.debug(f"STDERR: {line.decode('utf-8').strip()}")
        except Exception as e:
            logger.error(f"Error reading stderr: {e}")

    async def _read_responses(self) -> None:
        """Background agent to read JSON-RPC responses from process stdout."""
        if not self.process:
            logger.error("No process available for reading")
            return
        if not self.process.stdout:
            logger.error("Process has no stdout")
            return

        logger.debug(f"Starting to read responses from process PID: {self.process.pid}")

        try:
            while True:
                # Read line from stdout
                line = await self.process.stdout.readline()

                if line:
                    # We got data, process it
                    try:
                        # Parse JSON-RPC message
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue

                        logger.debug(f"Received line: {line_str}")
                        message = json.loads(line_str)
                        await self._handle_response(message)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from process: {line_str} - {e}")
                    except Exception as e:
                        logger.error(f"Error processing response: {e}")
                else:
                    # Check if process is still running
                    # For asyncio subprocess, returncode is automatically updated
                    if self.process.returncode is not None:
                        # Process has actually ended
                        logger.debug(f"Process ended with return code: {self.process.returncode}")
                        break
                    else:
                        # EOF but process still running - this happens with stdio servers
                        # They don't output anything until they receive input
                        # Just wait a bit and continue
                        await asyncio.sleep(0.1)
                        continue

        except Exception as e:
            logger.error(f"Error reading responses: {e}", exc_info=True)
            if self.restart_on_failure and self.restart_count < self.max_restart_attempts:
                await self._attempt_restart()
        except asyncio.CancelledError:
            logger.debug("Reader agent was cancelled")
            raise
        finally:
            # Only mark as disconnected if the process actually ended
            logger.debug(f"Reader agent ending, current state: {self.state}")
            if self.process and self.process.returncode is not None:
                # Process has ended
                if self.state == ConnectionState.CONNECTED:
                    logger.debug(
                        f"Setting state to DISCONNECTED because process ended with code {self.process.returncode}"
                    )
                    self._set_state(ConnectionState.DISCONNECTED)
                    self._handle_close()
            else:
                logger.debug("Reader agent ended but process still running, keeping connection")

    async def _handle_response(self, message: dict[str, Any]) -> None:
        """Handle JSON-RPC response or notification from process."""
        logger.debug(f"Received message: {message}")
        if "id" in message:
            # This is a response to a request
            request_id = str(message["id"])
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    # JSON-RPC error response
                    error = message["error"]
                    if isinstance(error, dict):
                        error_msg = error.get("message", str(error))
                        code = error.get("code", -32000)
                        future.set_exception(ValueError(f"JSON-RPC error {code}: {error_msg}"))
                    else:
                        future.set_exception(ValueError(f"JSON-RPC error: {error}"))
                else:
                    # Successful response
                    future.set_result(message.get("result", {}))
            else:
                logger.warning(f"Received response for unknown request ID: {request_id}")
        else:
            # This is a notification (no id field)
            self._handle_message(message)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Send JSON-RPC message to process and wait for response.

        Args:
            message: JSON-RPC message dictionary

        Returns:
            JSON-RPC response result

        Raises:
            RuntimeError: If transport is not connected
            ValueError: If there's a JSON-RPC error
            TimeoutError: If response times out
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")

        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not available")

        request_id = message.get("id")
        if request_id is None:
            raise ValueError("Message must have an 'id' field for STDIO transport")
        request_id = str(request_id)

        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Send message as newline-delimited JSON
            message_json = json.dumps(message) + "\n"
            message_bytes = message_json.encode("utf-8")

            logger.debug(f"Sending message: {message_json.strip()}")
            self.process.stdin.write(message_bytes)
            await self.process.stdin.drain()

            logger.debug(
                f"Sent STDIO message: {message.get('method', 'unknown')}, waiting for response with id {request_id}"
            )

            # Wait for response
            result = await asyncio.wait_for(response_future, timeout=self.timeout)
            return result

        except asyncio.TimeoutError:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {request_id} timed out after {self.timeout}s")
        except Exception as e:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            logger.error(f"Error sending STDIO message: {e}")
            raise

    def set_protocol_version(self, version: str) -> None:
        """No-op for STDIO (no headers), but stored for introspection."""
        self._protocol_version = version

    async def send_notification(self, message: dict[str, Any]) -> None:
        """
        Send JSON-RPC notification (no response expected).

        Args:
            message: JSON-RPC notification dictionary

        Raises:
            RuntimeError: If transport is not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")

        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not available")

        # Notifications should not have an id
        if "id" in message:
            raise ValueError("Notifications must not have an 'id' field")

        try:
            # Send message as newline-delimited JSON
            message_json = json.dumps(message) + "\n"
            message_bytes = message_json.encode("utf-8")

            self.process.stdin.write(message_bytes)
            await self.process.stdin.drain()

            logger.debug(f"Sent STDIO notification: {message.get('method', 'unknown')}")

        except Exception as e:
            logger.error(f"Error sending STDIO notification: {e}")
            raise

    async def send_response(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC response back to the server over STDIO."""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Transport not connected (state: {self.state})")
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not available")

        # Must include id and either result or error
        if "id" not in message:
            raise ValueError("Response must include 'id'")

        message_json = json.dumps(message) + "\n"
        self.process.stdin.write(message_json.encode("utf-8"))
        await self.process.stdin.drain()

    async def close(self) -> None:
        """Close the STDIO transport and terminate subprocess."""
        self._set_state(ConnectionState.DISCONNECTED)

        # Cancel reader agent
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        # Close subprocess
        if self.process:
            await self._close_process()

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        logger.debug("STDIO transport closed")

    async def _close_process(self) -> None:
        """Gracefully close the subprocess."""
        if not self.process:
            return

        try:
            # Close stdin to signal process to exit
            if self.process.stdin:
                self.process.stdin.close()
                await self.process.stdin.wait_closed()

            # Wait for process to exit gracefully
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.debug(f"Process {self.process.pid} exited gracefully")
            except asyncio.TimeoutError:
                # Force termination
                logger.warning(f"Process {self.process.pid} did not exit gracefully, terminating")
                await self._terminate_process()

        except Exception as e:
            logger.error(f"Error closing process: {e}")
        finally:
            self.process = None

    async def _terminate_process(self) -> None:
        """Force terminate the subprocess."""
        if not self.process:
            return

        try:
            if platform.system() == "Windows":
                # On Windows, use terminate()
                self.process.terminate()
            else:
                # On Unix, try SIGTERM first, then SIGKILL
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Process {self.process.pid} did not respond to SIGTERM, using SIGKILL"
                    )
                    self.process.kill()

            # Wait for process to actually exit
            await self.process.wait()
            logger.debug(f"Process {self.process.pid} terminated")

        except Exception as e:
            logger.error(f"Error terminating process: {e}")

    async def _attempt_restart(self) -> None:
        """Attempt to restart the failed process."""
        if self.restart_count >= self.max_restart_attempts:
            logger.error(f"Max restart attempts ({self.max_restart_attempts}) reached")
            return

        self.restart_count += 1
        logger.info(
            f"Attempting to restart process (attempt {self.restart_count}/{self.max_restart_attempts})"
        )

        try:
            # Clean up old process
            if self.process:
                await self._close_process()

            # Wait before restart (exponential backoff)
            wait_time = min(2**self.restart_count, 30)
            await asyncio.sleep(wait_time)

            # Restart process
            await self._start_process()
            self._set_state(ConnectionState.CONNECTED)
            logger.info("Process restarted successfully")

        except Exception as e:
            logger.error(f"Failed to restart process: {e}")
            self._set_state(ConnectionState.ERROR)
            self._handle_error(e)

    def is_process_running(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process is not None and self.process.returncode is None
