"""
Main chat application for MCP client.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from llmring.mcp.client import MCPClient
from llmring.mcp.client.chat.styles import PROMPT_STYLE, RICH_THEME
from llmring.mcp.client.llmring_integration import MCPLLMRingIntegration
from llmring.mcp.client.pool_config import CHAT_APP_POOL
from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing

# Load environment variables from .env file
load_dotenv()


class CommandCompleter(Completer):
    """Completer for chat commands."""

    def __init__(self, chat_app: "MCPChatApp"):
        """
        Initialize the command completer.

        Args:
            chat_app: The chat app instance
        """
        self.chat_app = chat_app
        self.commands = {
            "/help": "Show help",
            "/clear": "Clear the conversation history",
            "/history": "Show conversation history",
            "/sessions": "List saved sessions",
            "/load": "Load a previous session",
            "/model": "Change the LLM model",
            "/models": "List available models",
            "/connect": "Connect to MCP server",
            "/tools": "List available MCP tools",
            "/exit": "Exit the chat",
        }

    def get_completions(self, document, complete_event):
        """Get command completions."""
        text = document.text

        # Only complete for commands
        if text.startswith("/"):
            word = text.split()[0] if text else ""

            # Complete command names
            for command in self.commands:
                if command.startswith(word):
                    display = HTML(
                        f"<b>{command}</b> - <style fg='#888888'>{self.commands[command]}</style>"
                    )
                    yield Completion(command, start_position=-len(word), display=display)

            # For /model command, offer model suggestions
            if text.startswith("/model "):
                # Get what's typed after the command
                typed = text[7:].strip()

                # Offer model suggestions
                for model in self.chat_app.get_available_models():
                    if model["model_key"].startswith(typed) or model[
                        "display_name"
                    ].lower().startswith(typed.lower()):
                        display = HTML(
                            f"<b>{model['model_key']}</b> - <style fg='#888888'>{model['display_name']}</style>"
                        )
                        yield Completion(
                            model["model_key"],
                            start_position=-len(typed),
                            display=display,
                        )


class MCPChatApp:
    """Main chat application for interactive MCP and LLM usage."""

    def __init__(
        self,
        mcp_server_url: str | None = None,
        llm_model: str = "balanced",
        session_id: str | None = None,
        enable_telemetry: bool = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the chat application.

        Args:
            mcp_server_url: URL of the MCP server
            llm_model: LLM model to use
            session_id: Optional session ID to load
            enable_telemetry: Enable llmring-server telemetry (auto-detects if None)
            system_prompt: Optional system prompt to include in conversations
        """
        # Rich console for output
        self.console = Console(theme=RICH_THEME)

        # LLMRing will be initialized in async context
        self.llmring = None
        self.model = llm_model
        self.system_prompt = system_prompt

        # Telemetry integration (optional)
        self.integration = None
        if enable_telemetry is None:
            # Auto-detect based on environment
            enable_telemetry = bool(os.getenv("LLMRING_SERVER_URL"))

        if enable_telemetry:
            try:
                self.integration = MCPLLMRingIntegration(
                    origin="mcp-chat",
                    llmring_server_url=os.getenv("LLMRING_SERVER_URL"),
                    api_key=os.getenv("LLMRING_API_KEY"),
                )
                self.console.print(
                    f"[info]Telemetry enabled: {os.getenv('LLMRING_SERVER_URL', 'default')}[/info]"
                )
            except Exception as e:
                self.console.print(f"[warning]Telemetry disabled: {e}[/warning]")
                self.integration = None

        # MCP client
        self.mcp_client = None
        self.mcp_server_url = mcp_server_url

        if mcp_server_url:
            self.connect_to_server(mcp_server_url)

        # Chat state
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation: list[Message] = []
        self.available_tools: dict[str, Any] = {}
        self.resolved_model_shown = False  # Track if we've shown the resolved model

        # Persistent storage paths
        self.data_dir = Path.home() / ".llmring" / "mcp_chat"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Command history file (for prompt_toolkit)
        self.history_file = self.data_dir / "command_history.txt"

        # Conversation history file (for chat messages)
        self.conversation_file = self.data_dir / f"conversation_{self.session_id}.json"
        self.conversations_dir = self.data_dir / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)

        # Load previous conversation if it exists
        self._load_conversation()

        # Command history and session
        self.history = FileHistory(str(self.history_file))
        self.completer = CommandCompleter(self)
        self.session = PromptSession(
            history=self.history,
            completer=self.completer,
            style=PROMPT_STYLE,
        )

        # Command handlers
        self.command_handlers = {
            "/help": self.cmd_help,
            "/clear": self.cmd_clear,
            "/history": self.cmd_history,
            "/sessions": self.cmd_sessions,
            "/load": self.cmd_load,
            "/model": self.cmd_model,
            "/models": self.cmd_models,
            "/connect": self.cmd_connect,
            "/tools": self.cmd_tools,
            "/exit": self.cmd_exit,
        }

    def connect_to_server(
        self,
        server_url: str,
    ) -> bool:
        """
        Connect to an MCP server.

        Args:
            server_url: The server URL

        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse server URL to determine transport type
            if server_url.startswith("http"):
                self.mcp_client = MCPClient.http(server_url)
            elif server_url.startswith("ws://") or server_url.startswith("wss://"):
                self.mcp_client = MCPClient.websocket(server_url)
            elif server_url.startswith("stdio://"):
                # Extract command from URL
                # Format: stdio://command args or stdio://path/to/command
                command_str = server_url.replace("stdio://", "")
                # Handle Python module invocation
                if command_str.startswith("python -m"):
                    command = command_str.split()
                else:
                    command = command_str.split()
                self.mcp_client = MCPClient.stdio(command=command)
            else:
                # Default to HTTP
                self.mcp_client = MCPClient.http(server_url)

            # Initialize and get server info
            self.mcp_client.initialize()

            # Get available tools
            self._refresh_tools()

            return True
        except Exception as e:
            self.console.print(f"[error]Failed to connect to MCP server:[/error] {e!s}")
            self.mcp_client = None
            self.available_tools = {}
            return False

    def _refresh_tools(self) -> None:
        """Refresh the list of available tools from the MCP server."""
        if not self.mcp_client:
            self.available_tools = {}
            return

        try:
            tools = self.mcp_client.list_tools()
            self.available_tools = {tool["name"]: tool for tool in tools}
        except Exception as e:
            self.console.print(f"[error]Failed to fetch tools:[/error] {e!s}")
            self.available_tools = {}

    def _convert_tools_for_llm(self) -> list[dict[str, Any]]:
        """Convert MCP tools to format expected by LLMs."""
        if not self.available_tools:
            return []

        llm_tools = []
        for tool in self.available_tools.values():
            # Convert MCP tool format to OpenAI-style function format
            llm_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            llm_tools.append(llm_tool)

        return llm_tools

    def get_available_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model information dictionaries
        """
        # Get models from llmring - returns dict of provider -> list of models
        models_by_provider = self.llmring.get_available_models()

        # Flatten into list of model info dicts
        models = []
        for provider, model_list in models_by_provider.items():
            for model_name in model_list:
                models.append(
                    {
                        "provider": provider,
                        "model": model_name,
                        "model_key": f"{provider}:{model_name}",
                        "display_name": f"{provider}:{model_name}",
                        "full_name": f"{provider}:{model_name}",
                        "context_length": "N/A",  # We don't have this info readily available
                    }
                )

        return models

    def initialize(self) -> None:
        """Initialize the chat application."""
        # Load available tools silently if MCP client is configured
        if self.mcp_client:
            try:
                # Refresh tools without verbose output
                self._refresh_tools()
            except Exception as e:
                # Only show error if critical
                if "connection" in str(e).lower():
                    self.console.print(f"[error]MCP server connection failed[/error]")

        # Simple prompt to start
        self.console.print(
            f"[dim]Model: {self.model}[/dim] • Type [bold]/help[/bold] for commands\n"
        )

    async def initialize_async(self) -> None:
        """Initialize async resources."""
        # Create LLMRing instance
        self.llmring = LLMRing(origin="mcp-client-chat")

    async def cleanup(self) -> None:
        """Clean up resources."""
        # No database resources to clean up
        pass

    async def run(self) -> None:
        """Run the chat application."""
        # Initialize async resources
        await self.initialize_async()

        # Initialize UI
        self.initialize()

        try:
            while True:
                try:
                    # Get user input
                    user_input = await self.session.prompt_async("You: ")

                    # Skip empty input
                    if not user_input.strip():
                        continue

                    # Handle commands (starting with /)
                    if user_input.startswith("/"):
                        await self.handle_command(user_input)
                        continue

                    # Add user message to conversation
                    self.conversation.append(Message(role="user", content=user_input))
                    # Save after each user message
                    self._save_conversation()

                    # Format system message
                    system_message = self.create_system_message()

                    # Convert tools for LLM
                    tools = self._convert_tools_for_llm()

                    # Create LLM request with native tool support
                    request = LLMRequest(
                        messages=[
                            Message(role="system", content=system_message),
                            *self.conversation,
                        ],
                        model=self.model,
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None,
                    )

                    # Get response from LLM
                    with self.console.status("[info]Thinking...[/info]"):
                        response = await self.llmring.chat(request)

                    # Show the resolved model on first response
                    if not self.resolved_model_shown and response.model:
                        self.console.print(f"[dim]Using model: {response.model}[/dim]\n")
                        self.resolved_model_shown = True

                    # Process response for potential tool calls
                    await self.process_response(response)

                except KeyboardInterrupt:
                    # Ctrl-C pressed - interrupt current operation but don't exit
                    self.console.print("\n[dim]Interrupted. Press Ctrl-D to exit.[/dim]")
                    continue
                except EOFError:
                    # Ctrl-D pressed - exit the chat
                    self.console.print("\n[warning]Exiting...[/warning]")
                    # Save conversation on exit
                    self._save_conversation()
                    break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.console.print(f"[error]Error:[/error] {e!s}")
        finally:
            # Clean up resources
            await self.cleanup()

    def create_system_message(self) -> str:
        """
        Create system message with tool instructions.

        Returns:
            System message content
        """
        # Use custom system prompt if provided
        if self.system_prompt:
            return self.system_prompt

        if not self.available_tools:
            return "You are a helpful assistant. Respond directly to the user's questions."

        # Create simple system message - tools will be passed separately
        return """You are a helpful assistant that can use tools to help answer questions about lockfile management.

When the user asks about aliases, models, or configurations, use the appropriate tools to provide accurate information."""

    def _parse_json_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try to parse tool calls from JSON text response.

        Args:
            content: Response content that may contain JSON

        Returns:
            List of tool calls if found, None otherwise
        """
        if not content:
            return None

        try:
            # Try to extract JSON from the content
            json_content = content.strip()

            # Handle markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    json_content = content[start:end].strip()
            elif "```" in content and "{" in content:
                start = content.find("```") + 3
                newline = content.find("\n", start)
                if newline != -1:
                    start = newline + 1
                end = content.find("```", start)
                if end > start:
                    json_content = content[start:end].strip()

            # Parse JSON
            data = json.loads(json_content)

            # Check if it has tool_calls
            if isinstance(data, dict) and "tool_calls" in data:
                tool_calls = data["tool_calls"]

                # Convert to native format
                native_calls = []
                for call in tool_calls:
                    native_call = {
                        "id": call.get("id", f"call_{len(native_calls)}"),
                        "type": "function",
                        "function": {
                            "name": call.get("tool", call.get("name", "")),
                            "arguments": json.dumps(call.get("arguments", {})),
                        },
                    }
                    native_calls.append(native_call)

                return native_calls

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return None

    async def process_response(self, response: LLMResponse, depth: int = 0) -> None:
        """
        Process and display LLM response, handling potential tool calls.

        Args:
            response: LLM response to process
            depth: Recursion depth for tool calling loop
        """
        # Check for native tool calls in the response
        if response.tool_calls:
            await self.process_tool_calls(response, depth)
            return

        # Try to parse JSON tool calls from content (fallback)
        if response.content:
            tool_calls = self._parse_json_tool_calls(response.content)
            if tool_calls:
                # Create a new response with parsed tool calls
                response.tool_calls = tool_calls
                # Try to extract content message if present
                try:
                    json_data = json.loads(response.content.strip())
                    if isinstance(json_data, dict) and "content" in json_data:
                        response.content = json_data["content"]
                    else:
                        response.content = ""
                except (json.JSONDecodeError, KeyError):
                    response.content = ""
                await self.process_tool_calls(response, depth)
                return

        # Display regular response
        content = response.content
        if content:
            self.console.print("[assistant]Assistant:[/assistant]")
            self.console.print(Markdown(content))

            # Add to conversation
            self.conversation.append(Message(role="assistant", content=content))
            # Save after assistant response
            self._save_conversation()

    async def reconnect(self) -> bool:
        """Attempt to reconnect to MCP server."""
        if not self.mcp_server_url:
            return False

        self.console.print("[info]Attempting to reconnect to MCP server...[/info]")

        for attempt in range(3):
            try:
                success = self.connect_to_server(self.mcp_server_url)
                if success:
                    self.console.print("[success]Reconnected successfully[/success]")
                    return True
            except Exception as e:
                if attempt < 2:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    self.console.print(
                        f"[warning]Reconnection attempt {attempt + 1} failed, waiting {wait_time}s...[/warning]"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.console.print(f"[error]Failed to reconnect after 3 attempts: {e}[/error]")

        return False

    async def process_tool_calls(self, response: LLMResponse, depth: int = 0) -> None:
        """
        Process tool calls from LLM response with recursive loop support.

        Args:
            response: LLM response containing tool calls
            depth: Current recursion depth (to prevent infinite loops)
        """
        # Prevent infinite recursion
        if depth > 5:
            self.console.print("[warning]Maximum tool calling depth reached[/warning]")
            return
        # Display content if any
        if response.content:
            self.console.print("[assistant]Assistant:[/assistant]")
            self.console.print(Markdown(response.content))

        # No MCP client means no tool calls
        if not self.mcp_client:
            self.console.print("[error]Cannot execute tools: No MCP server connected[/error]")
            # Add to conversation
            self.conversation.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            # Save after response
            self._save_conversation()
            return

        # Add assistant message with tool calls to conversation
        self.conversation.append(
            Message(
                role="assistant", content=response.content or "", tool_calls=response.tool_calls
            )
        )
        # Save after adding message
        self._save_conversation()

        # Process each tool call
        tool_results = []
        for call in response.tool_calls:
            # Extract tool name and arguments from the tool call
            # Tool calls from LLMs typically have structure like:
            # {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if "function" in call:
                tool_name = call["function"]["name"]
                # Arguments might be a JSON string that needs parsing
                args_str = call["function"].get("arguments", "{}")
                if isinstance(args_str, str):
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = args_str
            else:
                # Fallback for simpler format
                tool_name = call.get("name", call.get("tool", ""))
                arguments = call.get("arguments", {})

            # Display concise tool call information
            self.console.print(f"[dim]Calling tool: {tool_name}[/dim]", end="")

            try:
                # Execute the tool
                result = self.mcp_client.call_tool(tool_name, arguments)

                # Display success indicator
                self.console.print(" [success]✓[/success]")

                # Only show verbose output if it's an error or warning
                if isinstance(result, dict):
                    # Check if result indicates an error
                    if result.get("success") is False or result.get("error"):
                        # Show error details
                        self.console.print(
                            f"[error]Error: {result.get('error') or result.get('message', 'Unknown error')}[/error]"
                        )
                    elif result.get("warning"):
                        # Show warning
                        self.console.print(f"[warning]Warning: {result.get('warning')}[/warning]")

                # Add to results with tool_call_id if present
                tool_result = {
                    "tool_call_id": call.get("id"),
                    "tool": tool_name,
                    "result": result,
                    "success": True,
                }
                tool_results.append(tool_result)

            except Exception as e:
                # Check if it's a connection error
                error_str = str(e).lower()
                if any(
                    conn_err in error_str
                    for conn_err in ["connection", "transport", "disconnected", "timeout"]
                ):
                    # Try to reconnect once
                    self.console.print(" [warning]reconnecting...[/warning]", end="")
                    if await self.reconnect():
                        # Retry the tool call
                        try:
                            result = self.mcp_client.call_tool(tool_name, arguments)
                            self.console.print(" [success]✓[/success]")

                            tool_result = {
                                "tool_call_id": call.get("id"),
                                "tool": tool_name,
                                "result": result,
                                "success": True,
                            }
                            tool_results.append(tool_result)
                            continue  # Skip to next tool
                        except Exception as retry_e:
                            e = retry_e  # Use the retry error

                # Handle error (connection or otherwise)
                self.console.print(f" [error]✗ {e!s}[/error]")
                tool_result = {
                    "tool_call_id": call.get("id"),
                    "tool": tool_name,
                    "error": str(e),
                    "success": False,
                }
                tool_results.append(tool_result)

        # Add tool results to conversation as tool messages
        for result in tool_results:
            # Create tool result message
            content = json.dumps(result.get("result", result.get("error", "")))
            self.conversation.append(
                Message(role="tool", content=content, tool_call_id=result.get("tool_call_id"))
            )
            # Save after tool result
            self._save_conversation()

        # Get follow-up response from LLM with tools still available
        system_message = self.create_system_message()
        tools = self._convert_tools_for_llm()

        request = LLMRequest(
            messages=[
                Message(role="system", content=system_message),
                *self.conversation,
            ],
            model=self.model,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )

        with self.console.status("[info]Getting follow-up response...[/info]"):
            follow_up_response = await self.llmring.chat(request)

        # Process the follow-up response (which might contain more tool calls)
        await self.process_response(follow_up_response, depth + 1)

    async def handle_command(self, command: str) -> None:
        """
        Handle chat commands.

        Args:
            command: The command string
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Execute command handler if it exists
        handler = self.command_handlers.get(cmd)
        if handler:
            await handler(args)
        else:
            self.console.print(f"[error]Unknown command:[/error] {cmd}")
            self.console.print("Type [command]/help[/command] for available commands")

    async def cmd_help(self, args: str) -> None:
        """
        Show help command.

        Args:
            args: Command arguments (unused)
        """
        help_text = """
[heading]Available commands:[/heading]

[command]/help[/command]               Show this help message
[command]/clear[/command]              Clear the conversation history
[command]/history[/command]            Show conversation history
[command]/sessions[/command]           List saved conversation sessions
[command]/load[/command] <session_id>  Load a previous session
[command]/model[/command] <model_name> Change the LLM model
[command]/models[/command]             List available models
[command]/connect[/command] <url>      Connect to MCP server
[command]/tools[/command]              List available MCP tools
[command]/exit[/command]               Exit the chat
"""
        self.console.print(Panel(help_text, title="Help"))

    def _save_conversation(self) -> None:
        """Save the current conversation to disk."""
        try:
            # Create conversation data
            conversation_data = {
                "session_id": self.session_id,
                "model": self.model,
                "created_at": datetime.now().isoformat(),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", datetime.now().isoformat()),
                    }
                    for msg in self.conversation
                ],
            }

            # Save to the session-specific file
            conversation_file = self.conversations_dir / f"session_{self.session_id}.json"
            with open(conversation_file, "w") as f:
                json.dump(conversation_data, f, indent=2)

        except Exception as e:
            # Don't crash on save errors, just log
            self.console.print(f"[warning]Could not save conversation: {e}[/warning]")

    def _load_conversation(self) -> None:
        """Load a previous conversation from disk."""
        try:
            conversation_file = self.conversations_dir / f"session_{self.session_id}.json"
            if conversation_file.exists():
                with open(conversation_file, "r") as f:
                    data = json.load(f)

                # Restore conversation messages
                self.conversation = []
                for msg_data in data.get("messages", []):
                    msg = Message(role=msg_data["role"], content=msg_data["content"])
                    self.conversation.append(msg)

                if self.conversation:
                    self.console.print(
                        f"[info]Loaded {len(self.conversation)} messages from previous session[/info]"
                    )
                    self.console.print("[dim]Use /history to view conversation history[/dim]")

        except Exception as e:
            # Don't crash on load errors
            self.console.print(f"[warning]Could not load previous conversation: {e}[/warning]")

    def _list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved conversation sessions."""
        sessions = []
        try:
            for session_file in self.conversations_dir.glob("session_*.json"):
                try:
                    with open(session_file, "r") as f:
                        data = json.load(f)
                        sessions.append(
                            {
                                "session_id": data.get("session_id", "unknown"),
                                "created_at": data.get("created_at", "unknown"),
                                "message_count": len(data.get("messages", [])),
                                "model": data.get("model", "unknown"),
                            }
                        )
                except:
                    continue

            # Sort by creation date
            sessions.sort(key=lambda x: x["created_at"], reverse=True)

        except Exception:
            pass

        return sessions

    async def cmd_history(self, args: str) -> None:
        """
        Show conversation history.

        Args:
            args: Command arguments (unused)
        """
        if not self.conversation:
            self.console.print("[dim]No conversation history[/dim]")
            return

        self.console.print(Panel("[bold]Conversation History[/bold]"))
        for msg in self.conversation:
            if msg.role == "user":
                self.console.print(f"[user]You:[/user] {msg.content}")
            else:
                self.console.print(f"[assistant]Assistant:[/assistant] {msg.content[:500]}...")

    async def cmd_sessions(self, args: str) -> None:
        """
        List saved conversation sessions.

        Args:
            args: Command arguments (unused)
        """
        sessions = self._list_sessions()
        if not sessions:
            self.console.print("[dim]No saved sessions found[/dim]")
            return

        table = Table(title="Saved Sessions")
        table.add_column("Session ID", style="cyan", width=20)
        table.add_column("Created", style="yellow")
        table.add_column("Messages", style="green", justify="right")
        table.add_column("Model", style="blue")

        for session in sessions[:10]:  # Show latest 10
            session_id = session["session_id"][:8] + "..."
            created = (
                session["created_at"][:19]
                if isinstance(session["created_at"], str)
                else str(session["created_at"])
            )
            table.add_row(session_id, created, str(session["message_count"]), session["model"])

        self.console.print(table)
        self.console.print("\n[dim]Use /load <session_id> to load a session[/dim]")

    async def cmd_load(self, args: str) -> None:
        """
        Load a previous conversation session.

        Args:
            args: Session ID or part of it
        """
        if not args.strip():
            self.console.print("[error]Please specify a session ID[/error]")
            self.console.print("[dim]Use /sessions to list available sessions[/dim]")
            return

        # Find matching session
        sessions = self._list_sessions()
        matching = [s for s in sessions if s["session_id"].startswith(args.strip())]

        if not matching:
            self.console.print(f"[error]No session found matching: {args}[/error]")
            return

        if len(matching) > 1:
            self.console.print(
                f"[warning]Multiple sessions match. Please be more specific:[/warning]"
            )
            for s in matching[:5]:
                self.console.print(f"  - {s['session_id'][:12]}... ({s['message_count']} messages)")
            return

        # Load the session
        session = matching[0]
        self.session_id = session["session_id"]
        self._load_conversation()
        self.resolved_model_shown = False  # Reset flag when loading a session
        self.console.print(
            f"[success]Loaded session with {len(self.conversation)} messages[/success]"
        )

    async def cmd_clear(self, args: str) -> None:
        """
        Clear conversation history.

        Args:
            args: Command arguments (unused)
        """
        self.conversation = []
        self.resolved_model_shown = False  # Reset flag when clearing conversation
        # Save the empty conversation
        self._save_conversation()
        self.console.print("[success]Conversation cleared[/success]")

    async def cmd_model(self, args: str) -> None:
        """
        Change the current model.

        Args:
            args: Model name
        """
        if not args:
            self.console.print("[error]Model name required[/error]")
            self.console.print("Usage: [command]/model[/command] <model_name>")
            return

        new_model = args.strip()
        try:
            # Validate model exists
            self.llmring.get_model_info(new_model)
            self.model = new_model
            self.resolved_model_shown = False  # Reset flag so new model is shown
            self.console.print(f"[success]Model changed to {new_model}[/success]")
        except Exception as e:
            self.console.print(f"[error]Error changing model:[/error] {e!s}")
            self.console.print("Use [command]/models[/command] to see available models")

    async def cmd_models(self, args: str) -> None:
        """
        List available models.

        Args:
            args: Command arguments (unused)
        """
        models = self.get_available_models()

        if not models:
            self.console.print("[warning]No models available[/warning]")
            return

        # Create a table
        table = Table(title="Available Models")
        table.add_column("Provider", style="cyan")
        table.add_column("Model Key", style="green")
        table.add_column("Display Name", style="blue")
        table.add_column("Context Length", style="magenta")

        # Add rows
        for model in models:
            table.add_row(
                model["provider"],
                model["model_key"],
                model["display_name"],
                str(model["context_length"]),
            )

        self.console.print(table)

    async def cmd_connect(self, args: str) -> None:
        """
        Connect to an MCP server.

        Args:
            args: Server URL
        """
        if not args:
            self.console.print("[error]Server URL required[/error]")
            self.console.print("Usage: [command]/connect[/command] <server_url>")
            return

        server_url = args.strip()

        with self.console.status(f"[info]Connecting to {server_url}...[/info]"):
            success = self.connect_to_server(server_url)

        if success:
            self.console.print(f"[success]Connected to {server_url}[/success]")
            self.console.print(f"Found {len(self.available_tools)} tools")

            # Update stored URL
            self.mcp_server_url = server_url

    async def cmd_tools(self, args: str) -> None:
        """
        List available MCP tools.

        Args:
            args: Command arguments (unused)
        """
        if not self.available_tools:
            self.console.print("[warning]No tools available[/warning]")
            if not self.mcp_client:
                self.console.print(
                    "Connect to an MCP server first with [command]/connect[/command] <url>"
                )
            return

        # Create a table
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")

        # Add rows
        for name, tool in self.available_tools.items():
            table.add_row(
                name,
                tool.get("description", ""),
            )

        self.console.print(table)

    async def cmd_exit(self, args: str) -> None:
        """
        Exit the chat application.

        Args:
            args: Command arguments (unused)
        """
        # Save conversation before exiting
        self._save_conversation()
        self.console.print("[info]Conversation saved[/info]")
        raise EOFError()  # Use EOFError to exit, since KeyboardInterrupt just interrupts now


def main():
    """Entry point for the chat application."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Chat Interface")
    parser.add_argument("--server", help="MCP server URL")
    parser.add_argument(
        "--model",
        default="balanced",
        help="LLM model alias (fast, balanced, deep) or provider:model format",
    )

    args = parser.parse_args()

    # Create and run the chat app
    app = MCPChatApp(
        mcp_server_url=args.server,
        llm_model=args.model,
    )

    # Run the chat app
    try:
        asyncio.run(app.run())
    except EOFError:
        print("\nExiting...")
    except KeyboardInterrupt:
        # KeyboardInterrupt is now handled inside the app.run() loop
        pass


if __name__ == "__main__":
    main()
