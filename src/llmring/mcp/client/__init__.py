"""
MCP Client Library

A Python library for interacting with Model Context Protocol (MCP) servers,
providing both synchronous and asynchronous clients with conversation management,
and an enhanced LLM interface for modules.
"""

__version__ = "0.1.0"

from mcp_client.client import AsyncMCPClient, ConnectionState, MCPClient

# from mcp_client.conversation_manager import ConversationManager  # TODO: Convert to async
from llmring.mcp.client.conversation_manager_async import AsyncConversationManager
from llmring.mcp.client.enhanced_llm import EnhancedLLM, create_enhanced_llm
from llmring.mcp.client.info_service import MCPClientInfoService, create_info_service
from llmring.mcp.client.stateless_engine import StatelessChatEngine

__all__ = [
    # "ConversationManager",  # TODO: Convert to async
    "AsyncConversationManager",
    "AsyncMCPClient",
    "ConnectionState",
    "EnhancedLLM",
    "MCPClient",
    "MCPClientInfoService",
    "StatelessChatEngine",
    "create_enhanced_llm",
    "create_info_service",
]
