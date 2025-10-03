"""
Decorators for logging LLM calls to llmring-server.

Enables LLMRing logging for any LLM SDK without requiring migration to LLMRing abstraction.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

from llmring.logging.normalizers import detect_provider, normalize_response
from llmring.server_client import ServerClient

logger = logging.getLogger(__name__)


def log_llm_call(
    server_url: str,
    api_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    alias: Optional[str] = None,
    log_metadata: bool = True,
    log_conversations: bool = False,
    origin: str = "llmring-decorator",
) -> Callable:
    """
    Decorator to log LLM calls to llmring-server.

    This decorator can wrap any async function that calls an LLM provider SDK
    and automatically logs the request/response to llmring-server.

    Args:
        server_url: URL of llmring-server instance
        api_key: Optional API key for llmring-server/SaaS
        provider: Provider name ("openai", "anthropic", "google") or "auto" for auto-detection
        model: Model name (if not auto-detected from response)
        alias: Optional alias name for tracking
        log_metadata: Whether to log usage metadata (default: True)
        log_conversations: Whether to log full conversations (default: False)
        origin: Origin identifier for tracking (default: "llmring-decorator")

    Returns:
        Decorated async function

    Example:
        ```python
        from openai import AsyncOpenAI
        from llmring.logging import log_llm_call

        client = AsyncOpenAI()

        @log_llm_call(
            server_url="http://localhost:8000",
            provider="openai",
            log_conversations=True,
        )
        async def chat_with_gpt(messages, model="gpt-4o"):
            return await client.chat.completions.create(
                model=model,
                messages=messages,
            )

        # Use normally - logging happens automatically
        response = await chat_with_gpt([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    def decorator(func: Callable) -> Callable:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"@log_llm_call can only decorate async functions, got {func}")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute the wrapped function
            response = await func(*args, **kwargs)

            # Log asynchronously in background (non-blocking)
            asyncio.create_task(
                _log_response(
                    response=response,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    server_url=server_url,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    alias=alias,
                    log_metadata=log_metadata,
                    log_conversations=log_conversations,
                    origin=origin,
                )
            )

            # Return original response unchanged
            return response

        return wrapper

    return decorator


def log_llm_stream(
    server_url: str,
    api_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    alias: Optional[str] = None,
    log_metadata: bool = True,
    log_conversations: bool = False,
    origin: str = "llmring-decorator",
) -> Callable:
    """
    Decorator to log streaming LLM calls to llmring-server.

    This decorator wraps async generator functions that stream LLM responses,
    accumulates the full response, and logs it to llmring-server after completion.

    Args:
        server_url: URL of llmring-server instance
        api_key: Optional API key for llmring-server/SaaS
        provider: Provider name ("openai", "anthropic", "google") or "auto" for auto-detection
        model: Model name (if not auto-detected from response)
        alias: Optional alias name for tracking
        log_metadata: Whether to log usage metadata (default: True)
        log_conversations: Whether to log full conversations (default: False)
        origin: Origin identifier for tracking (default: "llmring-decorator")

    Returns:
        Decorated async generator function

    Example:
        ```python
        from openai import AsyncOpenAI
        from llmring.logging import log_llm_stream

        client = AsyncOpenAI()

        @log_llm_stream(
            server_url="http://localhost:8000",
            provider="openai",
            log_metadata=True,
        )
        async def stream_chat(messages, model="gpt-4o"):
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                yield chunk

        # Use normally - chunks are yielded, logging happens after completion
        async for chunk in stream_chat([{"role": "user", "content": "Hello!"}]):
            print(chunk.choices[0].delta.content)
        ```
    """

    def decorator(func: Callable) -> Callable:
        if not inspect.isasyncgenfunction(func):
            raise TypeError(f"@log_llm_stream can only decorate async generator functions, got {func}")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncIterator[Any]:
            # Storage for accumulating stream data
            accumulated_content = []
            last_chunk = None
            final_usage = None

            # Execute and yield chunks
            async for chunk in func(*args, **kwargs):
                # Store chunk data for logging
                last_chunk = chunk

                # Try to extract content and usage from chunk
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        if choice.delta.content:
                            accumulated_content.append(choice.delta.content)

                # Check for usage in final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    final_usage = chunk.usage

                # Yield chunk unchanged
                yield chunk

            # After stream completes, log the full response
            if last_chunk:
                asyncio.create_task(
                    _log_stream_response(
                        last_chunk=last_chunk,
                        accumulated_content="".join(accumulated_content),
                        final_usage=final_usage,
                        func=func,
                        args=args,
                        kwargs=kwargs,
                        server_url=server_url,
                        api_key=api_key,
                        provider=provider,
                        model=model,
                        alias=alias,
                        log_metadata=log_metadata,
                        log_conversations=log_conversations,
                        origin=origin,
                    )
                )

        return wrapper

    return decorator


async def _log_response(
    response: Any,
    func: Callable,
    args: tuple,
    kwargs: dict,
    server_url: str,
    api_key: Optional[str],
    provider: str,
    model: Optional[str],
    alias: Optional[str],
    log_metadata: bool,
    log_conversations: bool,
    origin: str,
) -> None:
    """
    Internal function to log non-streaming response to server.

    Args:
        response: Provider SDK response object
        func: The wrapped function
        args: Positional arguments passed to function
        kwargs: Keyword arguments passed to function
        server_url: Server URL
        api_key: API key
        provider: Provider name or "auto"
        model: Model name
        alias: Alias name
        log_metadata: Whether to log metadata
        log_conversations: Whether to log conversations
        origin: Origin identifier
    """
    try:
        # Auto-detect provider if needed
        detected_provider = provider
        if provider == "auto":
            detected_provider = detect_provider(response, func)
            if not detected_provider:
                logger.warning("Could not auto-detect provider. Skipping logging.")
                return

        # Normalize response to common format
        content, response_model, usage, finish_reason = normalize_response(
            response, detected_provider
        )

        # Use provided model or detected model
        final_model = model or response_model

        # Extract messages from function arguments
        messages = _extract_messages_from_args(args, kwargs)

        # Create server client
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Calculate cost if possible (basic estimation)
        cost_info = None
        if usage and usage.get("prompt_tokens"):
            # For now, we don't have registry access in decorator context
            # Cost will be calculated server-side if needed
            cost_info = {"total_cost": 0.0}

        # Prepare log data
        if log_conversations and messages:
            # Log full conversation
            conversation_data = {
                "messages": messages,
                "response": {
                    "content": content,
                    "model": final_model,
                    "finish_reason": finish_reason,
                    "usage": usage or {},
                },
                "metadata": {
                    "provider": detected_provider,
                    "model": final_model,
                    "alias": alias,
                    "origin": origin,
                },
            }

            # Add usage tokens
            if usage:
                conversation_data["metadata"]["input_tokens"] = usage.get("prompt_tokens", 0)
                conversation_data["metadata"]["output_tokens"] = usage.get("completion_tokens", 0)
                conversation_data["metadata"]["cached_tokens"] = usage.get("cached_tokens", 0)

            await client.post("/api/v1/conversations/log", json=conversation_data)
            logger.debug(
                f"Logged conversation to server: {detected_provider}:{final_model}"
            )

        elif log_metadata and usage:
            # Log metadata only
            log_data = {
                "model": final_model,
                "provider": detected_provider,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cached_input_tokens": usage.get("cached_tokens", 0),
                "origin": origin,
                "metadata": {
                    "model_alias": f"{detected_provider}:{final_model}",
                    "finish_reason": finish_reason,
                },
            }

            if alias:
                log_data["alias"] = alias

            await client.post("/api/v1/log", json=log_data)
            logger.debug(
                f"Logged usage to server: {detected_provider}:{final_model} "
                f"({log_data['input_tokens']} in, {log_data['output_tokens']} out)"
            )

    except Exception as e:
        # Logging failures should not break the main flow
        logger.warning(f"Failed to log LLM call to server: {e}")


async def _log_stream_response(
    last_chunk: Any,
    accumulated_content: str,
    final_usage: Any,
    func: Callable,
    args: tuple,
    kwargs: dict,
    server_url: str,
    api_key: Optional[str],
    provider: str,
    model: Optional[str],
    alias: Optional[str],
    log_metadata: bool,
    log_conversations: bool,
    origin: str,
) -> None:
    """
    Internal function to log streaming response to server.

    Args:
        last_chunk: Last chunk from stream
        accumulated_content: Accumulated text content
        final_usage: Usage information from final chunk
        func: The wrapped function
        args: Positional arguments
        kwargs: Keyword arguments
        server_url: Server URL
        api_key: API key
        provider: Provider name or "auto"
        model: Model name
        alias: Alias name
        log_metadata: Whether to log metadata
        log_conversations: Whether to log conversations
        origin: Origin identifier
    """
    try:
        # Auto-detect provider if needed
        detected_provider = provider
        if provider == "auto":
            detected_provider = detect_provider(last_chunk, func)
            if not detected_provider:
                logger.warning("Could not auto-detect provider for stream. Skipping logging.")
                return

        # Extract model and finish reason from last chunk
        response_model = getattr(last_chunk, "model", "unknown")
        finish_reason = None
        if hasattr(last_chunk, "choices") and last_chunk.choices:
            choice = last_chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

        # Use provided model or detected model
        final_model = model or response_model

        # Extract usage
        usage = {}
        if final_usage:
            usage = {
                "prompt_tokens": getattr(final_usage, "prompt_tokens", 0),
                "completion_tokens": getattr(final_usage, "completion_tokens", 0),
                "total_tokens": getattr(final_usage, "total_tokens", 0),
            }
            if hasattr(final_usage, "prompt_tokens_details"):
                details = final_usage.prompt_tokens_details
                if hasattr(details, "cached_tokens"):
                    usage["cached_tokens"] = details.cached_tokens

        # Extract messages from function arguments
        messages = _extract_messages_from_args(args, kwargs)

        # Create server client
        client = ServerClient(server_url=server_url, api_key=api_key)

        # Prepare log data
        if log_conversations and messages:
            # Log full conversation
            conversation_data = {
                "messages": messages,
                "response": {
                    "content": accumulated_content,
                    "model": final_model,
                    "finish_reason": finish_reason,
                    "usage": usage,
                },
                "metadata": {
                    "provider": detected_provider,
                    "model": final_model,
                    "alias": alias,
                    "origin": origin,
                },
            }

            # Add usage tokens
            if usage:
                conversation_data["metadata"]["input_tokens"] = usage.get("prompt_tokens", 0)
                conversation_data["metadata"]["output_tokens"] = usage.get("completion_tokens", 0)
                conversation_data["metadata"]["cached_tokens"] = usage.get("cached_tokens", 0)

            await client.post("/api/v1/conversations/log", json=conversation_data)
            logger.debug(
                f"Logged streaming conversation to server: {detected_provider}:{final_model}"
            )

        elif log_metadata and usage:
            # Log metadata only
            log_data = {
                "model": final_model,
                "provider": detected_provider,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cached_input_tokens": usage.get("cached_tokens", 0),
                "origin": origin,
                "metadata": {
                    "model_alias": f"{detected_provider}:{final_model}",
                    "finish_reason": finish_reason,
                },
            }

            if alias:
                log_data["alias"] = alias

            await client.post("/api/v1/log", json=log_data)
            logger.debug(
                f"Logged streaming usage to server: {detected_provider}:{final_model} "
                f"({log_data['input_tokens']} in, {log_data['output_tokens']} out)"
            )

    except Exception as e:
        # Logging failures should not break the main flow
        logger.warning(f"Failed to log streaming LLM call to server: {e}")


def _extract_messages_from_args(args: tuple, kwargs: dict) -> Optional[list]:
    """
    Extract messages from function arguments.

    Tries common parameter names: messages, message, prompt, input

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        List of messages or None if not found
    """
    # Try to find messages in kwargs
    for param_name in ["messages", "message", "prompt", "input"]:
        if param_name in kwargs:
            messages = kwargs[param_name]
            # Normalize to list
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]
            elif isinstance(messages, list):
                # Validate list format
                if messages and not isinstance(messages[0], dict):
                    logger.debug(
                        f"Found '{param_name}' parameter but format is unexpected "
                        f"(expected list of dicts, got {type(messages[0]).__name__}). "
                        "Skipping message extraction."
                    )
                    return None
                return messages
            else:
                logger.debug(
                    f"Found '{param_name}' parameter but type is unexpected "
                    f"(got {type(messages).__name__}). Skipping message extraction."
                )

    # Try first positional argument if it looks like messages
    if args and isinstance(args[0], (list, str)):
        messages = args[0]
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Validate list format
            if messages and not isinstance(messages[0], dict):
                logger.debug(
                    f"First argument is a list but format is unexpected "
                    f"(expected list of dicts, got {type(messages[0]).__name__}). "
                    "Skipping message extraction."
                )
                return None
            return messages

    return None
