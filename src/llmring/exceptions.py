"""Exception hierarchy for LLMRing.

Clean, well-structured exception classes for proper error handling.
"""


class LLMRingError(Exception):
    """Base exception for all LLMRing errors."""
    pass


class ConfigurationError(LLMRingError):
    """Error in configuration (missing keys, invalid values, etc.)."""
    pass


class ProviderError(LLMRingError):
    """Base error for provider-related issues."""
    pass


class ProviderNotFoundError(ProviderError):
    """Requested provider is not available."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Provider authentication failed (invalid API key, etc.)."""
    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""
    pass


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""
    pass


class ModelError(LLMRingError):
    """Base error for model-related issues."""
    pass


class ModelNotFoundError(ModelError):
    """Requested model is not available."""
    pass


class ModelCapabilityError(ModelError):
    """Model doesn't support requested capability (e.g., vision, tools)."""
    pass


class RegistryError(LLMRingError):
    """Base error for registry-related issues."""
    pass


class RegistryConnectionError(RegistryError):
    """Cannot connect to registry."""
    pass


class RegistryValidationError(RegistryError):
    """Registry data validation failed."""
    pass


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


class ServerError(LLMRingError):
    """Base error for server communication issues."""
    pass


class ServerConnectionError(ServerError):
    """Cannot connect to llmring-server."""
    pass


class ServerAuthenticationError(ServerError):
    """Server authentication failed."""
    pass


class ServerResponseError(ServerError):
    """Invalid response from server."""
    pass


class ConversationError(LLMRingError):
    """Base error for conversation-related issues."""
    pass


class ConversationNotFoundError(ConversationError):
    """Requested conversation not found."""
    pass


class ConversationAccessError(ConversationError):
    """No access to requested conversation."""
    pass


class MessageError(LLMRingError):
    """Base error for message-related issues."""
    pass


class MessageValidationError(MessageError):
    """Message validation failed."""
    pass


class MessageStorageError(MessageError):
    """Failed to store message."""
    pass


class ReceiptError(LLMRingError):
    """Base error for receipt-related issues."""
    pass


class ReceiptSignatureError(ReceiptError):
    """Receipt signature validation failed."""
    pass


class ReceiptStorageError(ReceiptError):
    """Failed to store receipt."""
    pass