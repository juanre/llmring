"""Input validation for LLM requests with size limits and format checks. Validates messages, base64 data, file sizes, and registry URLs."""

import base64
import ipaddress
import logging
from collections.abc import Sequence
from typing import Any, Dict
from urllib.parse import urlparse

from llmring.schemas import Message

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates user inputs to prevent resource exhaustion and security issues."""

    # Size limits (configurable via environment variables)
    MAX_BASE64_SIZE_MB = 50  # Maximum size for base64 data before decoding
    MAX_DECODED_SIZE_MB = 100  # Maximum size for decoded document data
    MAX_MESSAGE_CONTENT_LENGTH = 1_000_000  # 1MB of text content per message
    MAX_MESSAGES = 1000  # Maximum number of messages in a conversation

    @staticmethod
    def validate_base64_size(base64_data: str, context: str = "document") -> None:
        """
        Validate base64 data size before decoding to prevent OOM.

        Args:
            base64_data: Base64 encoded string
            context: Context for error messages (e.g., "document", "image")

        Raises:
            ValueError: If base64 data exceeds size limits
        """
        # Calculate encoded size in MB
        encoded_size_mb = len(base64_data) / (1024 * 1024)

        if encoded_size_mb > InputValidator.MAX_BASE64_SIZE_MB:
            raise ValueError(
                f"Base64 {context} data too large: {encoded_size_mb:.2f}MB "
                f"(max: {InputValidator.MAX_BASE64_SIZE_MB}MB). "
                f"Consider reducing file size or using file upload instead."
            )

        logger.debug(f"Base64 {context} data size validated: {encoded_size_mb:.2f}MB")

    @staticmethod
    def validate_decoded_size(decoded_data: bytes, context: str = "document") -> None:
        """
        Validate decoded data size to prevent excessive memory usage.

        Args:
            decoded_data: Decoded binary data
            context: Context for error messages

        Raises:
            ValueError: If decoded data exceeds size limits
        """
        decoded_size_mb = len(decoded_data) / (1024 * 1024)

        if decoded_size_mb > InputValidator.MAX_DECODED_SIZE_MB:
            raise ValueError(
                f"Decoded {context} data too large: {decoded_size_mb:.2f}MB "
                f"(max: {InputValidator.MAX_DECODED_SIZE_MB}MB)."
            )

        logger.debug(f"Decoded {context} data size validated: {decoded_size_mb:.2f}MB")

    @staticmethod
    def safe_decode_base64(base64_data: str, context: str = "document") -> bytes:
        """
        Safely decode base64 data with size validation.

        Args:
            base64_data: Base64 encoded string
            context: Context for error messages

        Returns:
            Decoded binary data

        Raises:
            ValueError: If base64 data is invalid or too large
        """
        # Validate encoded size first
        InputValidator.validate_base64_size(base64_data, context)

        # Decode
        try:
            decoded = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Invalid base64 {context} data: {e}")

        # Validate decoded size
        InputValidator.validate_decoded_size(decoded, context)

        return decoded

    @staticmethod
    def validate_message_content(messages: Sequence[Dict[str, Any] | Message]) -> None:
        """
        Validate message content to prevent abuse and resource exhaustion.

        Args:
            messages: List of message dictionaries

        Raises:
            ValueError: If message content is invalid or too large
        """
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise ValueError("Messages must be a sequence of messages")

        if len(messages) == 0:
            raise ValueError("Messages list cannot be empty")

        if len(messages) > InputValidator.MAX_MESSAGES:
            raise ValueError(
                f"Too many messages: {len(messages)} (max: {InputValidator.MAX_MESSAGES})"
            )

        for i, message in enumerate(messages):
            if isinstance(message, Message):
                message_dict: Dict[str, Any] = message.model_dump()
            elif isinstance(message, dict):
                message_dict = message
            else:
                raise ValueError(f"Message {i} must be a dictionary or Message object")

            # Validate role
            role = message_dict.get("role")
            if not role:
                raise ValueError(f"Message {i} missing required 'role' field")

            # Allow 'tool' role for tool-calling conversations
            if role not in ["user", "assistant", "system", "tool"]:
                raise ValueError(
                    f"Message {i} has invalid role: '{role}'. "
                    f"Must be 'user', 'assistant', 'system', or 'tool'"
                )

            # Validate content
            content = message_dict.get("content")
            if content is None:
                raise ValueError(f"Message {i} missing 'content' field")

            # Check content size
            if isinstance(content, str):
                if len(content) > InputValidator.MAX_MESSAGE_CONTENT_LENGTH:
                    raise ValueError(
                        f"Message {i} content too large: {len(content)} chars "
                        f"(max: {InputValidator.MAX_MESSAGE_CONTENT_LENGTH})"
                    )
            elif isinstance(content, list):
                # Multimodal content - validate each part
                for j, part in enumerate(content):
                    if not isinstance(part, dict):
                        raise ValueError(f"Message {i} content part {j} must be a dictionary")

                    part_type = part.get("type")
                    if not part_type:
                        raise ValueError(f"Message {i} content part {j} missing 'type' field")

                    # Validate text parts
                    if part_type == "text":
                        text = part.get("text", "")
                        if len(text) > InputValidator.MAX_MESSAGE_CONTENT_LENGTH:
                            raise ValueError(
                                f"Message {i} content part {j} text too large: {len(text)} chars"
                            )

                    # Validate document parts with base64 data
                    elif part_type == "document":
                        source = part.get("source", {})
                        if source.get("type") == "base64":
                            base64_data = source.get("data", "")
                            InputValidator.validate_base64_size(base64_data, "document")

                    # Validate image parts with base64 data
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:"):
                            # Extract base64 part from data URL
                            if ";base64," in url:
                                base64_data = url.split(";base64,", 1)[1]
                                InputValidator.validate_base64_size(base64_data, "image")

        logger.debug(f"Validated {len(messages)} messages")

    @staticmethod
    def validate_registry_url(url: str) -> None:
        """
        Validate registry URL to prevent SSRF and other security issues.

        Args:
            url: Registry URL to validate

        Raises:
            ValueError: If URL is invalid or potentially unsafe
        """
        if not url:
            raise ValueError("Registry URL cannot be empty")

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid registry URL: {e}")

        # Allow file:// for local development/testing
        if parsed.scheme == "file":
            logger.debug("Registry URL uses file:// scheme (local development)")
            return

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Registry URL must use HTTPS (got: {parsed.scheme}://). "
                f"Use file:// for local registries."
            )

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Registry URL must have a hostname")

        normalized_hostname = hostname.lower()
        is_localhost_name = normalized_hostname == "localhost"

        parsed_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
        try:
            parsed_ip = ipaddress.ip_address(hostname)
        except ValueError:
            parsed_ip = None

        is_loopback_ip = bool(parsed_ip and parsed_ip.is_loopback)
        is_unspecified_ip = bool(parsed_ip and parsed_ip.is_unspecified)

        # For remote URLs, require HTTPS unless it's a localhost/loopback URL or a test/example domain.
        if parsed.scheme == "http":
            test_domains = ["example.com", "example.org", "test.com", "localhost"]
            is_test_domain = (
                any(normalized_hostname.endswith(td) for td in test_domains)
                or normalized_hostname in test_domains
            )
            if not (is_localhost_name or is_loopback_ip or is_test_domain):
                raise ValueError(
                    f"Registry URL must use HTTPS (got: {parsed.scheme}://). "
                    f"Use file:// for local registries."
                )
            logger.debug(f"Allowing HTTP registry URL for host: {hostname}")

        if is_unspecified_ip:
            raise ValueError(f"Registry URL cannot point to an unspecified address: {hostname}")

        # Warn about internal networks (allowed; can be intentional for enterprise registries).
        if parsed_ip:
            if parsed_ip.is_private or parsed_ip.is_link_local:
                logger.warning(
                    f"Registry URL points to a non-public IP address: {hostname}. "
                    f"This may be intentional for internal registries."
                )
            elif parsed_ip.is_loopback:
                logger.debug(f"Registry URL points to loopback address: {hostname}")

        logger.debug(f"Registry URL validated: {url}")
