"""
Exception hierarchy for LLMRing.

Clean, well-structured exception classes for proper error handling.
Enhanced with specific exceptions for all components.
"""

from typing import Any, Optional


class LLMRingError(Exception):
    """Base exception for all LLMRing errors with original exception preservation."""

    def __init__(
        self,
        message: str,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.original = original
        self.details = details or {}

    def __str__(self):
        parts = [self.message]
        if self.original:
            parts.append(f"(caused by {type(self.original).__name__}: {self.original})")
        return " ".join(parts)

    @property
    def root_cause(self) -> Optional[Exception]:
        """Get the root cause exception."""
        if self.original and hasattr(self.original, "root_cause"):
            return self.original.root_cause
        return self.original or self


# Configuration Errors
class ConfigurationError(LLMRingError):
    """Error in configuration (missing keys, invalid values, etc.)."""

    def __init__(
        self,
        message: str,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, original, details)


class InitializationError(LLMRingError):
    """Raised when initialization of a component fails."""

    def __init__(
        self,
        message: str,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, original, details)


# Provider Errors
class ProviderError(LLMRingError):
    """Base error for provider-related issues."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, original, details)
        self.provider = provider
        self.status_code = status_code

    def __str__(self):
        parts = []
        if self.provider:
            parts.append(f"[{self.provider}]")
        parts.append(self.message)
        if self.original:
            parts.append(f"(caused by {type(self.original).__name__}: {self.original})")
        return " ".join(parts)


class ProviderNotFoundError(ProviderError):
    """Requested provider is not available."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Provider authentication failed (invalid API key, etc.)."""

    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, provider, original=original, details=details)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    pass


class ProviderResponseError(ProviderError):
    """Invalid or unexpected response from provider."""

    pass


# Model Errors
class ModelError(LLMRingError):
    """Base error for model-related issues."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        original: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, original, details)
        self.model_name = model_name
        self.provider = provider


class ModelNotFoundError(ModelError):
    """Requested model is not available."""

    pass


class ModelCapabilityError(ModelError):
    """Model doesn't support requested capability (e.g., vision, tools)."""

    def __init__(
        self,
        message: str,
        required_capability: Optional[str] = None,
        model_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, model_name, details=details)
        self.required_capability = required_capability


# Registry Errors
class RegistryError(LLMRingError):
    """Base error for registry-related issues."""

    pass


class RegistryConnectionError(RegistryError):
    """Cannot connect to registry."""

    pass


class RegistryValidationError(RegistryError):
    """Registry data validation failed."""

    pass


# Lockfile Errors
class LockfileError(LLMRingError):
    """Base error for lockfile-related issues."""

    pass


class LockfileNotFoundError(LockfileError):
    """Lockfile not found."""

    pass


class LockfileParseError(LockfileError):
    """Cannot parse lockfile."""

    pass


class LockfileVersionError(LockfileError):
    """Lockfile version incompatible."""

    pass


# Server Errors
class ServerError(LLMRingError):
    """Base error for server communication issues."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.endpoint = endpoint


class ServerConnectionError(ServerError):
    """Cannot connect to llmring-server."""

    pass


class ServerAuthenticationError(ServerError):
    """Server authentication failed."""

    pass


class ServerResponseError(ServerError):
    """Invalid response from server."""

    pass


# Conversation Errors
class ConversationError(LLMRingError):
    """Base error for conversation-related issues."""

    def __init__(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.conversation_id = conversation_id


class ConversationNotFoundError(ConversationError):
    """Requested conversation not found."""

    pass


class ConversationAccessError(ConversationError):
    """No access to requested conversation."""

    pass


# Message Errors
class MessageError(LLMRingError):
    """Base error for message-related issues."""

    pass


class MessageValidationError(MessageError):
    """Message validation failed."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.field = field
        self.value = value


class MessageStorageError(MessageError):
    """Failed to store message."""

    pass


# Receipt Errors
class ReceiptError(LLMRingError):
    """Base error for receipt-related issues."""

    pass


class ReceiptSignatureError(ReceiptError):
    """Receipt signature validation failed."""

    pass


class ReceiptStorageError(ReceiptError):
    """Failed to store receipt."""

    pass


# MCP-specific Errors
class MCPError(LLMRingError):
    """Base class for MCP-related errors."""

    pass


class MCPProtocolError(MCPError):
    """MCP protocol violation or invalid message."""

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        method: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.error_code = error_code
        self.method = method


class MCPTransportError(MCPError):
    """MCP transport layer error."""

    pass


class MCPToolError(MCPError):
    """MCP tool execution error."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.tool_name = tool_name
        self.arguments = arguments


class MCPResourceError(MCPError):
    """MCP resource access error."""

    def __init__(
        self,
        message: str,
        resource_uri: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.resource_uri = resource_uri


class MCPServerError(MCPError):
    """MCP server-side error."""

    pass


class MCPInitializationError(MCPError):
    """MCP initialization failed."""

    pass


# File Processing Errors
class FileProcessingError(LLMRingError):
    """Base class for file processing errors."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.filename = filename


class InvalidFileFormatError(FileProcessingError):
    """File format is invalid or unsupported."""

    pass


class FileSizeError(FileProcessingError):
    """File size exceeds limits."""

    def __init__(
        self,
        message: str,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
        filename: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, filename, details)
        self.file_size = file_size
        self.max_size = max_size


class FileAccessError(FileProcessingError):
    """Cannot access or read file."""

    pass


# Tool and Function Errors
class ToolError(LLMRingError):
    """Base class for tool-related errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Requested tool is not found."""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, tool_name, details)
        self.error_type = error_type


class ToolValidationError(ToolError):
    """Tool arguments validation failed."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        invalid_args: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, tool_name, details)
        self.invalid_args = invalid_args


# Circuit Breaker Errors
class CircuitBreakerError(LLMRingError):
    """Circuit breaker is open."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
        failure_count: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.provider = provider
        self.retry_after = retry_after
        self.failure_count = failure_count


# Network Errors
class NetworkError(LLMRingError):
    """Network-related error."""

    pass


class TimeoutError(NetworkError):
    """Operation timed out."""

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.timeout = timeout
        self.operation = operation


class ConnectionError(NetworkError):
    """Connection failed."""

    pass


# Validation Errors
class ValidationError(LLMRingError):
    """Data validation failed."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.expected_type = expected_type


class SchemaValidationError(ValidationError):
    """Schema validation failed."""

    pass


# Session Errors
class SessionError(LLMRingError):
    """Session-related error."""

    pass


class SessionExpiredError(SessionError):
    """Session has expired."""

    pass


class SessionNotFoundError(SessionError):
    """Session not found."""

    pass


# Utility Functions
def wrap_exception(
    exc: Exception,
    wrapper_class: type[LLMRingError],
    message: Optional[str] = None,
    **kwargs,
) -> LLMRingError:
    """
    Wrap a generic exception in a LLMRing-specific exception.

    Args:
        exc: The original exception
        wrapper_class: The LLMRing exception class to use
        message: Optional custom message
        **kwargs: Additional arguments for the wrapper exception

    Returns:
        A LLMRing-specific exception wrapping the original
    """
    if isinstance(exc, LLMRingError):
        return exc

    msg = message or str(exc)
    details = kwargs.pop("details", {})
    details["original_error"] = str(exc)
    details["original_type"] = type(exc).__name__

    return wrapper_class(msg, details=details, **kwargs)


def is_retryable_error(exc: Exception) -> bool:
    """
    Check if an exception is retryable.

    Args:
        exc: The exception to check

    Returns:
        True if the error is temporary and can be retried
    """
    retryable_types = (
        ProviderRateLimitError,
        ProviderTimeoutError,
        CircuitBreakerError,
        TimeoutError,
        ConnectionError,
        ServerConnectionError,
        MCPTransportError,
    )
    return isinstance(exc, retryable_types)
