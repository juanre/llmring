"""
Main chat application for MCP client.
"""

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from llmring.mcp.client import MCPClient
from llmring.mcp.client.chat.styles import PROMPT_STYLE, RICH_THEME
from llmring.mcp.client.pool_config import CHAT_APP_POOL
from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing

# Load environment variables from .env file
load_dotenv()

# Database imports - optional, only if database support is needed
try:
    from pgdbm import AsyncDatabaseManager, DatabaseConfig, MonitoredAsyncDatabaseManager

    HAS_DATABASE = True
except ImportError:
    # Database support is optional
    AsyncDatabaseManager = None
    DatabaseConfig = None
    MonitoredAsyncDatabaseManager = None
    HAS_DATABASE = False


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
        db_connection_string: str | None = None,
        session_id: str | None = None,
        db_manager: Any | None = None,  # AsyncDatabaseManager if database is available
    ):
        """
        Initialize the chat application.

        Args:
            mcp_server_url: URL of the MCP server
            llm_model: LLM model to use
            db_connection_string: Database connection string (for standalone mode)
            session_id: Optional session ID to load
            db_manager: External database manager (for integrated mode)
        """
        # Rich console for output
        self.console = Console(theme=RICH_THEME)

        # Store database configuration
        self.db_connection_string = db_connection_string
        self.db_manager = db_manager
        self._external_db = db_manager is not None
        self.shared_pool = None  # Will be set in initialize_async if standalone

        # LLMRing will be initialized in async context
        self.llmring = None
        self.model = llm_model

        # MCP client
        self.mcp_client = None
        self.mcp_server_url = mcp_server_url

        if mcp_server_url:
            self.connect_to_server(mcp_server_url)

        # Chat state
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation: list[Message] = []
        self.available_tools: dict[str, Any] = {}

        # Command history and session
        self.history = InMemoryHistory()
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
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                }
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
                        "context_length": "N/A"  # We don't have this info readily available
                    }
                )

        return models

    def initialize(self) -> None:
        """Initialize the chat application."""
        self.console.print(Panel("MCP Chat Interface", style="heading"))

        # Load available tools if MCP client is configured
        if self.mcp_client:
            self.console.print("[info]Connecting to MCP server...[/info]")
            try:
                # Refresh tools
                self._refresh_tools()

                self.console.print(
                    f"[success]Connected![/success] Found {len(self.available_tools)} tools"
                )

                # Display first few tools
                if self.available_tools:
                    max_display = 5
                    displayed_tools = list(self.available_tools.values())[:max_display]

                    for tool in displayed_tools:
                        self.console.print(
                            f"  [tool]{tool['name']}[/tool]: {tool.get('description', '')}"
                        )

                    # Show count of remaining tools
                    remaining = len(self.available_tools) - max_display
                    if remaining > 0:
                        self.console.print(f"  [info]... and {remaining} more tools[/info]")
            except Exception as e:
                self.console.print(f"[error]Failed to connect to MCP server:[/error] {e!s}")

        # Display available models
        models = self.get_available_models()

        if models:
            self.console.print(
                f"[info]Available models:[/info] {', '.join(m['display_name'] for m in models[:3])}..."
            )
        else:
            self.console.print("[warning]No LLM models found in database[/warning]")

        self.console.print(f"[info]Current model:[/info] [bold]{self.model}[/bold]")
        self.console.print("\nType [command]/help[/command] for available commands")

    async def initialize_async(self) -> None:
        """Initialize async resources like database pool and LLMRing."""
        if not self._external_db:
            # Create our own shared pool in standalone mode
            connection_string = self.db_connection_string or os.getenv(
                "DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres"
            )

            if not HAS_DATABASE:
                raise ImportError(
                    "Database support not available. Install pgdbm to use database features."
                )

            config = DatabaseConfig(
                connection_string=connection_string,
                min_connections=CHAT_APP_POOL.min_connections,
                max_connections=CHAT_APP_POOL.max_connections,
            )

            # Create a shared pool
            if AsyncDatabaseManager:
                self.shared_pool = await AsyncDatabaseManager.create_shared_pool(config)
            else:
                raise ImportError("Database support not available")

            # Create schema-isolated manager for mcp_client and run migrations
            if AsyncDatabaseManager:
                self.db_manager = AsyncDatabaseManager(pool=self.shared_pool, schema="mcp_client")
            else:
                raise ImportError("Database support not available")
            # MCPClientDB functionality removed - database operations now optional
            # Database migrations would go here if MCPClientDB was available
        else:
            # In integrated mode, use provided db_manager; do not create a shared_pool here
            self.shared_pool = None  # Will be managed externally

        # Create LLMRing instance without database features for now
        self.llmring = LLMRing(origin="mcp-client-chat")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if not self._external_db and self.shared_pool:
            await self.shared_pool.close()

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

                    # Process response for potential tool calls
                    await self.process_response(response)

                except KeyboardInterrupt:
                    self.console.print("\n[warning]Exiting...[/warning]")
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
                            "arguments": json.dumps(call.get("arguments", {}))
                        }
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
            self.conversation.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))
            return

        # Add assistant message with tool calls to conversation
        self.conversation.append(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        ))

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

            # Display tool call information
            self.console.print(f"[tool]Calling tool:[/tool] {tool_name}")
            if arguments:
                self.console.print(JSON(json.dumps(arguments), indent=2))

            try:
                # Execute the tool
                with self.console.status(f"[info]Executing {tool_name}...[/info]"):
                    result = self.mcp_client.call_tool(tool_name, arguments)

                # Display result
                self.console.print("[success]Tool result:[/success]")

                # Format result based on type
                if isinstance(result, dict | list):
                    self.console.print(JSON(json.dumps(result), indent=2))
                elif isinstance(result, str) and (result.startswith("{") or result.startswith("[")):
                    try:
                        # Already JSON string, display as-is
                        self.console.print(JSON(result, indent=2))
                    except Exception:
                        # Not valid JSON, print as text
                        self.console.print(result)
                else:
                    # Regular text
                    self.console.print(result)

                # Add to results with tool_call_id if present
                tool_result = {
                    "tool_call_id": call.get("id"),
                    "tool": tool_name,
                    "result": result,
                    "success": True
                }
                tool_results.append(tool_result)

            except Exception as e:
                # Handle error
                self.console.print(f"[error]Tool error:[/error] {e!s}")
                tool_result = {
                    "tool_call_id": call.get("id"),
                    "tool": tool_name,
                    "error": str(e),
                    "success": False
                }
                tool_results.append(tool_result)

        # Add tool results to conversation as tool messages
        for result in tool_results:
            # Create tool result message
            content = json.dumps(result.get("result", result.get("error", "")))
            self.conversation.append(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=result.get("tool_call_id")
                )
            )

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
[command]/model[/command] <model_name> Change the LLM model
[command]/models[/command]             List available models
[command]/connect[/command] <url>      Connect to MCP server
[command]/tools[/command]              List available MCP tools
[command]/exit[/command]               Exit the chat
"""
        self.console.print(Panel(help_text, title="Help"))

    async def cmd_clear(self, args: str) -> None:
        """
        Clear conversation history.

        Args:
            args: Command arguments (unused)
        """
        self.conversation = []
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
        raise KeyboardInterrupt()


async def run_with_shared_pool(args):
    """Run the chat app with shared database pool."""
    from mcp_client.models.db import MCPClientDB
    from mcp_client.shared_pool import shared_pool_context

    async with shared_pool_context(
        connection_string=args.db,
        min_connections=CHAT_APP_POOL.min_connections,
        max_connections=CHAT_APP_POOL.max_connections,
        enable_monitoring=True,
    ) as pool:
        # Create schema-specific managers
        if AsyncDatabaseManager:
            mcp_db_manager = AsyncDatabaseManager(pool=pool, schema="mcp_client")
            llm_db_manager = AsyncDatabaseManager(pool=pool, schema="llmring")
        else:
            mcp_db_manager = None
            llm_db_manager = None

        # Create MCP client database with schema-specific manager
        mcp_db = MCPClientDB.from_manager(mcp_db_manager)
        await mcp_db.initialize()

        # Create LLM service with schema-specific manager
        llmring = LLMRing(db_manager=llm_db_manager, origin="mcp-client", enable_db_logging=True)

        # Create and configure chat app
        app = MCPChatApp(
            mcp_server_url=args.server,
            llm_model=args.model,
            db_connection_string=args.db,
        )

        # Replace the LLM service with our shared pool version
        app.llmring = llmring

        # Run the app
        await app.run()


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
    parser.add_argument("--db", help="Database connection string")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the database and recreate with default models",
    )
    parser.add_argument(
        "--use-shared-pool",
        action="store_true",
        help="Use shared database connection pool",
    )

    args = parser.parse_args()

    # Handle database reset if requested
    if args.reset_db:
        from mcp_client.models.db import MCPClientDB

        print("Resetting database...")
        db = MCPClientDB(connection_string=args.db)
        asyncio.run(db.initialize())
        asyncio.run(db.reset_database())
        asyncio.run(db.close())
        print("Database reset complete with default models.")
        return

    # Use shared pool if requested or by default
    if args.use_shared_pool or os.getenv("MCP_USE_SHARED_POOL", "true").lower() == "true":
        # Run with shared pool
        try:
            asyncio.run(run_with_shared_pool(args))
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        # Legacy mode - each service creates its own pool
        app = MCPChatApp(
            mcp_server_url=args.server,
            llm_model=args.model,
            db_connection_string=args.db,
        )

        # Run the chat app
        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
