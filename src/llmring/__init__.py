"""The llmring package."""

__version__ = "0.4.0"

from .base import BaseLLMProvider

# Import file utilities
from .file_utils import (  # Core functions; Content creation; Convenience functions
    analyze_image,
    compare_images,
    create_base64_image_content,
    create_data_url,
    create_image_content,
    create_multi_image_content,
    encode_file_to_base64,
    extract_text_from_image,
    get_file_mime_type,
    validate_image_file,
)
from .schemas import LLMRequest, LLMResponse, Message

# Import main components
from .service import LLMRing
from .service_extended import LLMRingExtended, ConversationManager

# Import exceptions
from .exceptions import (
    LLMRingError,
    ConfigurationError,
    ProviderError,
    ProviderNotFoundError,
    ModelNotFoundError,
    ConversationNotFoundError,
    ServerConnectionError,
)

__all__ = [
    # Core classes
    "LLMRing",
    "LLMRingExtended",
    "ConversationManager",
    "BaseLLMProvider",
    # Exceptions
    "LLMRingError",
    "ConfigurationError",
    "ProviderError",
    "ProviderNotFoundError",
    "ModelNotFoundError",
    "ConversationNotFoundError",
    "ServerConnectionError",
    # Schemas
    "LLMRequest",
    "LLMResponse",
    "Message",
    # File utilities
    "encode_file_to_base64",
    "create_data_url",
    "get_file_mime_type",
    "validate_image_file",
    "create_image_content",
    "create_multi_image_content",
    "create_base64_image_content",
    "analyze_image",
    "extract_text_from_image",
    "compare_images",
]
