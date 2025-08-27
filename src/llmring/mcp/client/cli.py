"""
CLI entry point for MCP client.
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from llmring.service import LLMRing
from rich.console import Console
from rich.table import Table

from llmring.mcp.client.chat.app import MCPChatApp
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.client.stateless_engine import ChatRequest, StatelessChatEngine

# Load environment variables from .env file
load_dotenv()


def run_chat():
    """Run the MCP chat interface."""
    parser = argparse.ArgumentParser(description="MCP Chat Interface")
    parser.add_argument("--server", help="MCP server URL")
    parser.add_argument(
        "--model",
        default="anthropic:claude-3-7-sonnet",
        help="LLM model to use (provider:model format)",
    )
    parser.add_argument(
        "--llmring-server",
        default=os.getenv("LLMRING_SERVER_URL", "http://localhost:8000"),
        help="LLMRing server URL for persistence",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLMRING_API_KEY"),
        help="API key for LLMRing server",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    app = MCPChatApp(
        mcp_server_url=args.server,
        llmring_server_url=args.llmring_server,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )
    asyncio.run(app.run())


def run_query():
    """Run a single query through the stateless engine."""
    parser = argparse.ArgumentParser(description="Run a single MCP query")
    parser.add_argument("query", help="The query to run")
    parser.add_argument("--server", help="MCP server URL")
    parser.add_argument(
        "--model",
        default="anthropic:claude-3-7-sonnet",
        help="LLM model to use",
    )
    parser.add_argument(
        "--llmring-server",
        default=os.getenv("LLMRING_SERVER_URL", "http://localhost:8000"),
        help="LLMRing server URL for persistence",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLMRING_API_KEY"),
        help="API key for LLMRing server",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--system", help="System prompt")
    parser.add_argument("--json", action="store_true", help="Output JSON response")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    async def _run():
        # Create LLMRing service
        llmring = LLMRing(origin="mcp-query")

        # Create stateless engine
        engine = StatelessChatEngine(
            llmring=llmring,
            default_model=args.model,
            mcp_server_url=args.server,
            llmring_server_url=args.llmring_server,
            api_key=args.api_key,
            debug=args.debug,
        )

        # Create request
        request = ChatRequest(
            messages=[{"role": "user", "content": args.query}],
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
        )

        # Process request
        response = await engine.process_request(request)

        # Output response
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            console = Console()
            if response.content:
                console.print(response.content)
            if response.error:
                console.print(f"[red]Error: {response.error}[/red]")

    asyncio.run(_run())


def list_conversations():
    """List recent conversations."""
    parser = argparse.ArgumentParser(description="List recent conversations")
    parser.add_argument(
        "--llmring-server",
        default=os.getenv("LLMRING_SERVER_URL", "http://localhost:8000"),
        help="LLMRing server URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLMRING_API_KEY"),
        help="API key for LLMRing server",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of conversations to show"
    )
    parser.add_argument("--user", help="Filter by user ID")

    args = parser.parse_args()

    async def _run():
        manager = AsyncConversationManager(
            llmring_server_url=args.llmring_server,
            api_key=args.api_key,
        )

        user_id = args.user or "default"
        conversations = await manager.list_conversations(user_id, limit=args.limit)

        if not conversations:
            print("No conversations found.")
            return

        console = Console()
        table = Table(title="Recent Conversations")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Messages", justify="right")
        table.add_column("Created", style="green")
        table.add_column("Last Message", style="yellow")

        for conv in conversations:
            created = conv.created_at.strftime("%Y-%m-%d %H:%M")
            last_preview = conv.last_message_preview or ""
            if len(last_preview) > 50:
                last_preview = last_preview[:47] + "..."

            table.add_row(
                conv.id[:8],
                conv.title or "Untitled",
                str(conv.message_count),
                created,
                last_preview,
            )

        console.print(table)
        
        await manager.close()

    asyncio.run(_run())


def show_conversation():
    """Show a conversation with its messages."""
    parser = argparse.ArgumentParser(description="Show a conversation")
    parser.add_argument("conversation_id", help="Conversation ID")
    parser.add_argument(
        "--llmring-server",
        default=os.getenv("LLMRING_SERVER_URL", "http://localhost:8000"),
        help="LLMRing server URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLMRING_API_KEY"),
        help="API key for LLMRing server",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    async def _run():
        manager = AsyncConversationManager(
            llmring_server_url=args.llmring_server,
            api_key=args.api_key,
        )

        conversation = await manager.get_conversation(args.conversation_id)

        if not conversation:
            print(f"Conversation {args.conversation_id} not found.")
            return

        if args.json:
            # Convert to JSON-serializable format
            data = {
                "id": conversation.id,
                "title": conversation.title,
                "model": conversation.model,
                "created_at": conversation.created_at.isoformat(),
                "message_count": conversation.message_count,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                    for msg in conversation.messages
                ],
            }
            print(json.dumps(data, indent=2))
        else:
            console = Console()
            console.print(f"\n[bold cyan]Conversation: {conversation.title or 'Untitled'}[/bold cyan]")
            console.print(f"ID: {conversation.id}")
            console.print(f"Model: {conversation.model}")
            console.print(f"Messages: {conversation.message_count}")
            console.print(f"Created: {conversation.created_at}")
            console.print("\n" + "=" * 80 + "\n")

            for msg in conversation.messages:
                role_color = {
                    "system": "yellow",
                    "user": "cyan",
                    "assistant": "green",
                    "tool": "magenta",
                }.get(msg.role, "white")

                console.print(f"[{role_color}]{msg.role.upper()}[/{role_color}]")
                console.print(msg.content)
                console.print("")
        
        await manager.close()

    asyncio.run(_run())


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Client CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  chat              Start interactive chat interface
  query             Run a single query
  list              List recent conversations
  show              Show a conversation
        """,
    )

    parser.add_argument(
        "command",
        choices=["chat", "query", "list", "show"],
        help="Command to run",
    )

    # Parse only the command
    args, remaining = parser.parse_known_args()

    # Reset sys.argv to include only the remaining arguments
    sys.argv = [sys.argv[0]] + remaining

    # Route to appropriate command
    if args.command == "chat":
        run_chat()
    elif args.command == "query":
        run_query()
    elif args.command == "list":
        list_conversations()
    elif args.command == "show":
        show_conversation()


if __name__ == "__main__":
    main()