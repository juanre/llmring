"""
CLI entry point for MCP client.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from llmbridge.service import LLMBridge
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from llmring.mcp.server.client.chat.app import MCPChatApp
from llmring.mcp.server.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.server.client.db import create_mcp_db_manager
from llmring.mcp.server.client.pool_config import CLI_LIST_POOL, CLI_QUERY_POOL
from llmring.mcp.server.client.stateless_engine import ChatRequest, StatelessChatEngine

# Load environment variables from .env file
load_dotenv()


async def setup_database(connection_string: str, reset: bool = False):
    """Setup database schema."""
    from llmbridge.db import LLMDatabase

    from mcp_client.models.db import MCPClientDB

    # Setup MCP Client database
    print("Setting up MCP Client database...")
    mcp_db = MCPClientDB(connection_string=connection_string)
    await mcp_db.initialize()

    if reset:
        print("  Resetting mcp_client schema...")
        await mcp_db.reset_database()
    else:
        result = await mcp_db.init_schema()
        print(f"  MCP Client setup: {result}")

    await mcp_db.close()

    # Setup LLMBridge database
    print("\nSetting up LLMBridge database...")
    llm_db = LLMDatabase(connection_string=connection_string)
    await llm_db.initialize()

    if reset:
        print("  Resetting llmbridge schema...")
        # Drop and recreate the llmbridge schema
        await llm_db.db.execute("DROP SCHEMA IF EXISTS llmbridge CASCADE")
        await llm_db.db.execute("CREATE SCHEMA IF NOT EXISTS llmbridge")
        # Re-run migrations
        await llm_db.apply_migrations()
        print("  ✓ LLMBridge schema reset and migrations applied")
    else:
        await llm_db.apply_migrations()
        print("  ✓ LLMBridge migrations applied")

    await llm_db.close()

    print("\nDatabase setup complete!")


def run_chat():
    """Run the MCP chat interface."""
    parser = argparse.ArgumentParser(description="MCP Chat Interface")
    parser.add_argument("--server", help="MCP server URL")
    parser.add_argument(
        "--model",
        default="anthropic:claude-3-7-sonnet",
        help="LLM model to use (provider:model format)",
    )
    parser.add_argument("--db", help="Database connection string")
    parser.add_argument(
        "--setup-db", action="store_true", help="Run database setup before starting"
    )

    args = parser.parse_args()

    # If setup-db flag is set, run setup_db
    if args.setup_db:
        asyncio.run(setup_database(args.db))

    # Create chat app
    app = MCPChatApp(
        mcp_server_url=args.server, llm_model=args.model, db_connection_string=args.db
    )

    # Run the chat app
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nExiting...")


async def run_query(args):
    """Run a single-turn query with optional conversation ID."""

    console = Console()

    # Get database URL
    db_url = args.db or os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/mcp_client"
    )

    # Fast path: if --no-save, avoid database entirely and call the LLM directly
    if args.no_save:
        try:
            from llmbridge.schemas import LLMRequest, Message

            # Initialize LLM without DB logging/providers must be configured
            llmbridge = LLMBridge(origin="mcp-client-cli", enable_db_logging=False)

            # Build request
            messages = []
            if args.system_prompt:
                messages.append(Message(role="system", content=args.system_prompt))
            messages.append(Message(role="user", content=args.message))

            request = LLMRequest(
                messages=messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            with console.status("Processing query..."):
                response = await llmbridge.chat(request)

            console.print(f"\n[bold]Assistant:[/bold] {response.content}")
            # Provide a synthetic conversation id for UX consistency
            import uuid as _uuid

            conv_id = args.cid or str(_uuid.uuid4())
            console.print(f"\n[dim]Conversation ID: {conv_id}[/dim]")
            if response.usage and "total_tokens" in response.usage:
                console.print(f"[dim]Tokens: {response.usage['total_tokens']}[/dim]")
            return
        except Exception as e:
            console.print(
                f"[yellow]Provider configuration may be missing or other error:[/yellow] {e!s}"
            )
            sys.exit(1)

    # Use shared pool for query with standardized CLI configuration
    from mcp_client.shared_pool import shared_pool_context

    async with shared_pool_context(
        connection_string=db_url,
        min_connections=CLI_QUERY_POOL.min_connections,
        max_connections=CLI_QUERY_POOL.max_connections,
        enable_monitoring=True,
    ) as pool:
        # Create schema-specific managers
        mcp_db_manager = AsyncDatabaseManager(pool=pool, schema="mcp_client")
        llm_db_manager = AsyncDatabaseManager(pool=pool, schema="llmbridge")

        # Ensure mcp_client schema is migrated
        from mcp_client.models.db import MCPClientDB

        mcp_db = MCPClientDB.from_manager(mcp_db_manager)
        await mcp_db.initialize()
        await mcp_db.run_migrations()

        # Initialize services with shared pool
        llmbridge = LLMBridge(
            db_manager=llm_db_manager, origin="mcp-client-cli", enable_db_logging=True
        )
        engine = StatelessChatEngine(mcp_db_manager, llmbridge)

        # Create auth context (simplified for CLI)
        auth_context = {
            "user_id": os.getenv("USER", "cli-user"),
            "user_info": {"type": "cli"},
        }

        # Build chat request
        request = ChatRequest(
            conversation_id=args.cid,
            message=args.message,
            model=args.model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            save_to_db=not args.no_save,
            auth_context=auth_context,
        )

        try:
            # Process request
            if args.stream:
                # Streaming response
                console.print("[dim]Processing query...[/dim]")
                full_response = ""
                async for chunk in engine.process_request_stream(request):
                    if chunk.delta:
                        console.print(chunk.delta, end="")
                        full_response += chunk.delta
                    if chunk.finished and chunk.usage:
                        console.print(f"\n\n[dim]Tokens: {chunk.usage['total_tokens']}[/dim]")
            else:
                # Regular response
                with console.status("Processing query..."):
                    response = await engine.process_request(request)

                console.print(f"\n[bold]Assistant:[/bold] {response.message.content}")
                console.print(f"\n[dim]Conversation ID: {response.conversation_id}[/dim]")
                console.print(f"[dim]Tokens: {response.usage['total_tokens']}[/dim]")

                if args.show_id:
                    console.print("\n[green]To continue this conversation, use:[/green]")
                    console.print(
                        f'mcp-client query --cid {response.conversation_id} "Your next message"'
                    )

        except Exception as e:
            console.print(f"[red]Error:[/red] {e!s}")
            sys.exit(1)


async def run_conversations_list(args):
    """List conversations."""
    from mcp_client.shared_pool import shared_pool_context

    console = Console()

    # Get database URL
    db_url = args.db or os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/mcp_client"
    )

    # Use shared pool for listing with standardized configuration
    async with shared_pool_context(
        connection_string=db_url,
        min_connections=CLI_LIST_POOL.min_connections,
        max_connections=CLI_LIST_POOL.max_connections,
        enable_monitoring=False,
    ) as pool:
        # Initialize services with shared pool (schema-specific)
        mcp_db_manager = AsyncDatabaseManager(pool=pool, schema="mcp_client")
        from mcp_client.models.db import MCPClientDB

        mcp_db = MCPClientDB.from_manager(mcp_db_manager)
        await mcp_db.initialize()
        await mcp_db.run_migrations()
        manager = AsyncConversationManager(mcp_db_manager)

        # Create auth context
        auth_context = {
            "user_id": os.getenv("USER", "cli-user"),
            "user_info": {"type": "cli"},
        }

        try:
            conversations = await manager.list_conversations(
                user_id=auth_context["user_id"], limit=args.limit, offset=args.offset
            )

            if not conversations:
                console.print("[yellow]No conversations found.[/yellow]")
                return

            # Create table
            table = Table(title="Conversations")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="green")
            table.add_column("Model", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Updated", style="yellow")

            for conv in conversations:
                # Format timestamp
                updated = datetime.fromisoformat(conv["updated_at"])
                updated_str = updated.strftime("%Y-%m-%d %H:%M")

                # Truncate title if needed
                title = conv.get("title", "Untitled")
                if len(title) > 50:
                    title = title[:47] + "..."

                table.add_row(
                    conv["conversation_id"][:8] + "...",
                    title,
                    conv.get("model", "unknown"),
                    str(conv.get("message_count", 0)),
                    updated_str,
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e!s}")
            sys.exit(1)


async def run_conversations_show(args):
    """Show a specific conversation."""
    console = Console()

    # Get database URL
    db_url = args.db or os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/mcp_client"
    )

    # Initialize services
    db = create_mcp_db_manager(connection_string=db_url)
    await db.connect()  # Connect to the database
    # Ensure schema is migrated
    from mcp_client.models.db import MCPClientDB

    mcp_db = MCPClientDB.from_manager(db)
    await mcp_db.initialize()
    await mcp_db.run_migrations()
    manager = AsyncConversationManager(db)

    # Create auth context
    auth_context = {
        "user_id": os.getenv("USER", "cli-user"),
        "user_info": {"type": "cli"},
    }

    try:
        conversation = await manager.get_conversation(args.cid, auth_context)

        if not conversation:
            console.print(f"[red]Conversation {args.cid} not found.[/red]")
            sys.exit(1)

        # Print conversation info
        console.print(f"\n[bold]Conversation:[/bold] {conversation['conversation_id']}")
        console.print(f"[bold]Title:[/bold] {conversation.get('title', 'Untitled')}")
        console.print(f"[bold]Model:[/bold] {conversation.get('model', 'unknown')}")
        console.print(f"[bold]Created:[/bold] {conversation['created_at']}")
        console.print(f"[bold]Updated:[/bold] {conversation['updated_at']}")
        console.print(f"[bold]Messages:[/bold] {conversation.get('message_count', 0)}")

        if conversation.get("system_prompt"):
            console.print("\n[bold]System Prompt:[/bold]")
            console.print(conversation["system_prompt"])

        # Print messages
        if args.full and conversation.get("messages"):
            console.print("\n[bold]Messages:[/bold]")
            for msg in conversation["messages"]:
                role = msg["role"].upper()
                if role == "USER":
                    console.print(f"\n[blue]{role}:[/blue]")
                elif role == "ASSISTANT":
                    console.print(f"\n[green]{role}:[/green]")
                else:
                    console.print(f"\n[yellow]{role}:[/yellow]")

                console.print(msg["content"])

                if msg.get("tool_calls"):
                    console.print("\n[dim]Tool Calls:[/dim]")
                    for tool_call in msg["tool_calls"]:
                        console.print(
                            f"  - {tool_call['tool_name']}({json.dumps(tool_call['arguments'])})"
                        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!s}")
        sys.exit(1)
    finally:
        # Cleanup database connection
        await db.disconnect()


async def run_conversations_delete(args):
    """Delete a conversation."""
    console = Console()

    # Get database URL
    db_url = args.db or os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/mcp_client"
    )

    # Initialize services
    db = create_mcp_db_manager(connection_string=db_url)
    await db.connect()  # Connect to the database
    # Ensure schema is migrated
    from mcp_client.models.db import MCPClientDB

    mcp_db = MCPClientDB.from_manager(db)
    await mcp_db.initialize()
    await mcp_db.run_migrations()
    manager = AsyncConversationManager(db)

    # Create auth context
    auth_context = {
        "user_id": os.getenv("USER", "cli-user"),
        "user_info": {"type": "cli"},
    }

    try:
        # Confirm deletion
        if not args.yes:
            confirm = console.input(f"[yellow]Delete conversation {args.cid}? (y/N):[/yellow] ")
            if confirm.lower() != "y":
                console.print("Cancelled.")
                return

        await manager.delete_conversation(args.cid, auth_context)
        console.print(f"[green]Conversation {args.cid} deleted.[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!s}")
        sys.exit(1)
    finally:
        # Cleanup database connection
        await db.disconnect()


async def run_conversations_export(args):
    """Export a conversation."""
    console = Console()

    # Get database URL
    db_url = args.db or os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/mcp_client"
    )

    # Initialize services
    db = create_mcp_db_manager(connection_string=db_url)
    await db.connect()  # Connect to the database
    manager = AsyncConversationManager(db)

    # Create auth context
    auth_context = {
        "user_id": os.getenv("USER", "cli-user"),
        "user_info": {"type": "cli"},
    }

    try:
        export_data = await manager.export_conversation(args.cid, args.format, auth_context)

        if args.output:
            # Write to file
            with open(args.output, "w") as f:
                f.write(export_data)
            console.print(f"[green]Exported to {args.output}[/green]")
        else:
            # Print to console
            if args.format == "json":
                syntax = Syntax(export_data, "json", theme="monokai", line_numbers=True)
                console.print(syntax)
            else:
                console.print(export_data)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!s}")
        sys.exit(1)
    finally:
        # Cleanup database connection
        await db.disconnect()


def main():
    """Main entry point for the MCP client CLI."""
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="MCP Client - CLI tools for Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  mcp-client chat --server http://localhost:8000/mcp
  mcp-client chat --resume --cid abc123def456

  # Single-turn query
  mcp-client query "What is the weather today?"
  mcp-client query --cid abc123def456 "Tell me more about that"

  # Manage conversations
  mcp-client conversations list
  mcp-client conversations show abc123def456
  mcp-client conversations export abc123def456 --format markdown

  # Setup
  mcp-client setup-db --db postgresql://user:pass@localhost/dbname
""",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Chat command (updated with --resume support)
    chat_parser = subparsers.add_parser("chat", help="Start the chat interface")
    chat_parser.add_argument("--server", help="MCP server URL")
    chat_parser.add_argument(
        "--model",
        default="anthropic:claude-3-7-sonnet",
        help="LLM model to use (provider:model format)",
    )
    chat_parser.add_argument("--db", help="Database connection string")
    chat_parser.add_argument(
        "--resume", action="store_true", help="Resume an existing conversation"
    )
    chat_parser.add_argument("--cid", help="Conversation ID to resume")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single-turn query")
    query_parser.add_argument("message", help="Message to send")
    query_parser.add_argument("--cid", help="Conversation ID to continue")
    query_parser.add_argument(
        "--model", default="anthropic:claude-3-7-sonnet", help="LLM model to use"
    )
    query_parser.add_argument("--system-prompt", help="System prompt for new conversations")
    query_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0-1)")
    query_parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    query_parser.add_argument("--stream", action="store_true", help="Stream response")
    query_parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    query_parser.add_argument(
        "--show-id", action="store_true", help="Show conversation ID for continuation"
    )
    query_parser.add_argument("--db", help="Database connection string")

    # Conversations command group
    conv_parser = subparsers.add_parser("conversations", help="Manage conversations")
    conv_subparsers = conv_parser.add_subparsers(dest="conv_command", help="Conversation command")

    # Conversations list
    conv_list_parser = conv_subparsers.add_parser("list", help="List conversations")
    conv_list_parser.add_argument("--limit", type=int, default=10, help="Number to show")
    conv_list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    conv_list_parser.add_argument("--db", help="Database connection string")

    # Conversations show
    conv_show_parser = conv_subparsers.add_parser("show", help="Show conversation details")
    conv_show_parser.add_argument("cid", help="Conversation ID")
    conv_show_parser.add_argument("--full", action="store_true", help="Show full messages")
    conv_show_parser.add_argument("--db", help="Database connection string")

    # Conversations delete
    conv_delete_parser = conv_subparsers.add_parser("delete", help="Delete a conversation")
    conv_delete_parser.add_argument("cid", help="Conversation ID")
    conv_delete_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    conv_delete_parser.add_argument("--db", help="Database connection string")

    # Conversations export
    conv_export_parser = conv_subparsers.add_parser("export", help="Export a conversation")
    conv_export_parser.add_argument("cid", help="Conversation ID")
    conv_export_parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Export format"
    )
    conv_export_parser.add_argument("-o", "--output", help="Output file")
    conv_export_parser.add_argument("--db", help="Database connection string")

    # Setup DB command
    setup_db_parser = subparsers.add_parser("setup-db", help="Set up the database")
    setup_db_parser.add_argument("--db", help="Database connection string")
    setup_db_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database by dropping all schemas first (WARNING: destroys all data)",
    )

    # If no arguments were provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()

    # Execute appropriate command
    if args.command == "chat":

        # Handle resume functionality
        session_id = None
        if args.resume and args.cid:
            session_id = args.cid
        elif args.resume:
            # If --resume without --cid, show error
            print("Error: --resume requires --cid to specify conversation ID")
            sys.exit(1)

        # Create and run chat app
        app = MCPChatApp(
            mcp_server_url=args.server,
            llm_model=args.model,
            db_connection_string=args.db,
            session_id=session_id,
        )

        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            print("\nExiting...")

    elif args.command == "query":
        # Run single-turn query
        try:
            asyncio.run(run_query(args))
        except KeyboardInterrupt:
            print("\nCancelled.")

    elif args.command == "conversations":
        # Handle conversations subcommands
        if args.conv_command == "list":
            asyncio.run(run_conversations_list(args))
        elif args.conv_command == "show":
            asyncio.run(run_conversations_show(args))
        elif args.conv_command == "delete":
            asyncio.run(run_conversations_delete(args))
        elif args.conv_command == "export":
            asyncio.run(run_conversations_export(args))
        else:
            conv_parser.print_help()

    elif args.command == "setup-db":
        asyncio.run(setup_database(args.db, reset=args.reset))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
