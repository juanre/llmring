#!/usr/bin/env python3
"""
MCP Client CLI - Database-agnostic command-line interface.

This CLI provides easy access to the Enhanced LLM functionality without
requiring any database setup. All persistence is handled via HTTP endpoints.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, UTC

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.table import Table

from llmring.mcp.client.enhanced_llm import create_enhanced_llm

# Load environment variables
load_dotenv()

# Initialize Rich console for beautiful terminal output
console = Console()


async def cmd_query(args: argparse.Namespace) -> None:
    """
    Execute a single query and return the response.

    Args:
        args: Command-line arguments containing the query and options
    """
    try:
        # Create Enhanced LLM instance
        llm = create_enhanced_llm(
            llm_model=args.model,
            llmring_server_url=args.server_url,
            api_key=args.api_key,
            origin="mcp-cli-query",
        )

        # Initialize if needed
        await llm.initialize()

        # Build messages
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.query})

        # Handle conversation context
        if args.conversation_id and not args.no_save:
            try:
                await llm.load_conversation(args.conversation_id)
                console.print(
                    f"[dim]Continuing conversation {args.conversation_id}[/dim]"
                )
            except Exception:
                # Create new conversation if ID doesn't exist
                args.conversation_id = await llm.create_conversation(
                    title=args.query[:50],
                    system_prompt=args.system,
                )
                console.print(f"[dim]Created conversation {args.conversation_id}[/dim]")
        elif not args.no_save:
            # Create new conversation
            args.conversation_id = await llm.create_conversation(
                title=args.query[:50],
                system_prompt=args.system,
            )

        # Execute query
        with console.status("[bold green]Thinking..."):
            response = await llm.chat(
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        # Display response
        if args.format == "markdown":
            console.print(Markdown(response.content))
        else:
            console.print(response.content)

        # Show metadata if requested
        if args.show_id:
            if args.conversation_id:
                console.print(f"\nConversation ID: {args.conversation_id}")
            elif args.no_save:
                console.print("\nConversation ID: Not saved (--no-save flag)")

        if args.show_usage and response.usage:
            console.print(
                f"[dim]Tokens used: {response.usage.get('total_tokens', 'N/A')}[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        await llm.close()


async def cmd_chat(args: argparse.Namespace) -> None:
    """
    Start an interactive chat session.

    Args:
        args: Command-line arguments for chat configuration
    """
    try:
        # Create Enhanced LLM instance
        llm = create_enhanced_llm(
            llm_model=args.model,
            llmring_server_url=args.server_url,
            mcp_server_url=args.mcp_server,
            api_key=args.api_key,
            origin="mcp-cli-chat",
        )

        # Initialize
        await llm.initialize()

        # Handle conversation resumption
        if args.resume:
            if not args.conversation_id:
                console.print(
                    "[red]Error: --resume requires --conversation-id (--cid)[/red]"
                )
                sys.exit(1)
            try:
                await llm.load_conversation(args.conversation_id)
                console.print(
                    f"[green]Resumed conversation {args.conversation_id}[/green]"
                )

                # Show last few messages for context
                if llm.conversation_history:
                    console.print("\n[dim]Recent messages:[/dim]")
                    for msg in llm.conversation_history[-3:]:
                        role_color = "cyan" if msg.role == "user" else "green"
                        console.print(
                            f"[{role_color}]{msg.role}:[/{role_color}] {msg.content[:100]}..."
                        )
                    console.print()
            except Exception as e:
                console.print(f"[yellow]Could not resume conversation: {e}[/yellow]")
                args.conversation_id = None

        # Create new conversation if needed
        if not args.conversation_id:
            args.conversation_id = await llm.create_conversation(
                title="Interactive Chat Session",
                system_prompt=args.system,
            )
            console.print(f"[dim]Started conversation {args.conversation_id}[/dim]")

        # Set system prompt if provided
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        console.print(
            "[bold]MCP Interactive Chat[/bold] (Type 'exit' or Ctrl+C to quit)\n"
        )

        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[cyan]You[/cyan]")

                if user_input.lower() in ["exit", "quit", "bye"]:
                    break

                # Add user message
                messages.append({"role": "user", "content": user_input})

                # Get response
                with console.status("[dim]Thinking...[/dim]"):
                    response = await llm.chat(
                        messages=[{"role": "user", "content": user_input}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )

                # Display response
                console.print(f"[green]Assistant:[/green] {response.content}\n")

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response.content})

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

        console.print("\n[dim]Chat session ended.[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        await llm.close()


async def cmd_conversations_list(args: argparse.Namespace) -> None:
    """
    List recent conversations.

    Args:
        args: Command-line arguments with limit and filters
    """
    try:
        # Create Enhanced LLM instance
        llm = create_enhanced_llm(
            llmring_server_url=args.server_url,
            api_key=args.api_key,
            origin="mcp-cli-list",
        )

        # Get conversations
        conversations = await llm.list_conversations(limit=args.limit)

        if not conversations:
            console.print("[dim]No conversations found.[/dim]")
            return

        # Create table
        table = Table(title="Recent Conversations")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Created", style="dim")
        table.add_column("Messages", justify="right", style="dim")

        for conv in conversations:
            table.add_row(
                conv.get("id", "")[:8] + "...",
                conv.get("title", "Untitled")[:50],
                conv.get("created_at", ""),
                str(conv.get("message_count", 0)),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        await llm.close()


async def cmd_conversations_show(args: argparse.Namespace) -> None:
    """
    Show a specific conversation.

    Args:
        args: Command-line arguments with conversation ID
    """
    try:
        # Create Enhanced LLM instance
        llm = create_enhanced_llm(
            llmring_server_url=args.server_url,
            api_key=args.api_key,
            origin="mcp-cli-show",
        )

        # Load conversation
        await llm.load_conversation(args.conversation_id)

        console.print(f"[bold]Conversation: {args.conversation_id}[/bold]\n")

        # Display messages
        if llm.conversation_history:
            for msg in llm.conversation_history:
                role_color = {
                    "system": "yellow",
                    "user": "cyan",
                    "assistant": "green",
                    "tool": "magenta",
                }.get(msg.role, "white")

                console.print(f"[{role_color}]{msg.role.upper()}[/{role_color}]")

                if args.format == "markdown" and msg.role == "assistant":
                    console.print(Markdown(msg.content))
                else:
                    console.print(msg.content)
                console.print()
        else:
            console.print("[dim]No messages in conversation.[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        await llm.close()


async def cmd_conversations_delete(args: argparse.Namespace) -> None:
    """
    Delete a conversation.

    Args:
        args: Command-line arguments with conversation ID
    """
    try:
        # Confirm deletion if not forced
        if not args.yes:
            confirm = Confirm.ask(
                f"[yellow]Delete conversation {args.conversation_id}?[/yellow]"
            )
            if not confirm:
                console.print("Cancelled.")
                return

        # Create conversation manager
        from llmring.mcp.client.conversation_manager_async import (
            AsyncConversationManager,
        )

        manager = AsyncConversationManager(
            llmring_server_url=args.server_url,
            api_key=args.api_key,
        )

        # Delete conversation
        await manager.delete_conversation(
            conversation_id=args.conversation_id,
            user_id=os.getenv("USER", "cli-user"),
        )

        console.print(f"[green]Conversation {args.conversation_id} deleted.[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


async def cmd_conversations_export(args: argparse.Namespace) -> None:
    """
    Export a conversation.

    Args:
        args: Command-line arguments with conversation ID and format
    """
    try:
        # Create Enhanced LLM instance
        llm = create_enhanced_llm(
            llmring_server_url=args.server_url,
            api_key=args.api_key,
            origin="mcp-cli-export",
        )

        # Load conversation
        await llm.load_conversation(args.conversation_id)

        # Prepare export data
        export_data = {
            "conversation_id": args.conversation_id,
            "exported_at": datetime.now(UTC).isoformat(),
            "messages": [],
        }

        for msg in llm.conversation_history:
            export_data["messages"].append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
            )

        # Format output
        if args.format == "json":
            output = json.dumps(export_data, indent=2)
        elif args.format == "markdown":
            output = f"# Conversation {args.conversation_id}\n\n"
            output += f"*Exported: {export_data['exported_at']}*\n\n"
            for msg in export_data["messages"]:
                output += f"## {msg['role'].upper()}\n\n{msg['content']}\n\n"
        else:
            output = str(export_data)

        # Write to file or stdout
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            console.print(f"[green]Exported to {args.output}[/green]")
        else:
            console.print(output)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        await llm.close()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Client - Enhanced LLM with tool capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  mcp-client query "What is the weather today?"
  
  # Query with system prompt
  mcp-client query "Explain quantum computing" --system "You are a physics teacher"
  
  # Interactive chat
  mcp-client chat
  
  # Resume a chat session
  mcp-client chat --resume --conversation-id abc123
  
  # List conversations
  mcp-client conversations list
  
  # Show a conversation
  mcp-client conversations show abc123
  
  # Export conversation
  mcp-client conversations export abc123 --format markdown -o chat.md
  
  # Delete conversation
  mcp-client conversations delete abc123 --yes
""",
    )

    # Global options
    parser.add_argument(
        "--server-url",
        default=os.getenv("LLMRING_SERVER_URL", "http://localhost:8000"),
        help="LLMRing server URL for persistence",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLMRING_API_KEY"),
        help="API key for LLMRing server",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "anthropic:claude-3-haiku-20240307"),
        help="LLM model to use (provider:model format)",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a single query")
    query_parser.add_argument("query", help="The query to send")
    query_parser.add_argument("--system", help="System prompt")
    query_parser.add_argument(
        "--conversation-id", "--cid", help="Continue in existing conversation"
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation",
    )
    query_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens in response",
    )
    query_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to conversation history",
    )
    query_parser.add_argument(
        "--show-id",
        action="store_true",
        help="Show conversation ID",
    )
    query_parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Show token usage",
    )
    query_parser.add_argument(
        "--format",
        choices=["plain", "markdown"],
        default="plain",
        help="Output format",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing conversation (requires --cid)",
    )
    chat_parser.add_argument(
        "--conversation-id",
        "--cid",
        help="Conversation ID to resume",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation",
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens per response",
    )
    chat_parser.add_argument(
        "--mcp-server",
        help="Optional MCP server URL for tools",
    )

    # Conversations command with subcommands
    conv_parser = subparsers.add_parser("conversations", help="Manage conversations")
    conv_subparsers = conv_parser.add_subparsers(
        dest="subcommand", help="Conversation commands"
    )

    # List conversations
    list_parser = conv_subparsers.add_parser("list", help="List conversations")
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of conversations to show",
    )

    # Show conversation
    show_parser = conv_subparsers.add_parser("show", help="Show a conversation")
    show_parser.add_argument("conversation_id", help="Conversation ID")
    show_parser.add_argument(
        "--format",
        choices=["plain", "markdown"],
        default="plain",
        help="Output format",
    )

    # Delete conversation
    delete_parser = conv_subparsers.add_parser("delete", help="Delete a conversation")
    delete_parser.add_argument("conversation_id", help="Conversation ID")
    delete_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation",
    )

    # Export conversation
    export_parser = conv_subparsers.add_parser("export", help="Export a conversation")
    export_parser.add_argument("conversation_id", help="Conversation ID")
    export_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "query":
        asyncio.run(cmd_query(args))
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "conversations":
        if args.subcommand == "list":
            asyncio.run(cmd_conversations_list(args))
        elif args.subcommand == "show":
            asyncio.run(cmd_conversations_show(args))
        elif args.subcommand == "delete":
            asyncio.run(cmd_conversations_delete(args))
        elif args.subcommand == "export":
            asyncio.run(cmd_conversations_export(args))
        else:
            conv_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
