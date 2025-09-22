"""
JSON-RPC 2.0 protocol implementation for MCP.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request structure."""

    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONRPCRequest":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params", {}),
            id=data.get("id"),
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response structure."""

    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response


class JSONRPCError:
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    APPLICATION_ERROR = -32000
    UNAUTHORIZED = -32001
    NOT_INITIALIZED = -32002  # MCP-specific: Server not initialized
    NOT_FOUND = -32003

    @staticmethod
    def parse_error(message: str = "Parse error") -> Dict[str, Any]:
        return {"code": JSONRPCError.PARSE_ERROR, "message": message}

    @staticmethod
    def invalid_request(message: str = "Invalid request") -> Dict[str, Any]:
        return {"code": JSONRPCError.INVALID_REQUEST, "message": message}

    @staticmethod
    def method_not_found(method: str) -> Dict[str, Any]:
        return {
            "code": JSONRPCError.METHOD_NOT_FOUND,
            "message": f"Method not found: {method}",
        }

    @staticmethod
    def invalid_params(message: str = "Invalid parameters") -> Dict[str, Any]:
        return {"code": JSONRPCError.INVALID_PARAMS, "message": message}

    @staticmethod
    def internal_error(message: str = "Internal error") -> Dict[str, Any]:
        return {"code": JSONRPCError.INTERNAL_ERROR, "message": message}

    @staticmethod
    def application_error(message: str, data: Any = None) -> Dict[str, Any]:
        error = {"code": JSONRPCError.APPLICATION_ERROR, "message": message}
        if data is not None:
            error["data"] = data
        return error

    @staticmethod
    def unauthorized(message: str = "Unauthorized") -> Dict[str, Any]:
        return {"code": JSONRPCError.UNAUTHORIZED, "message": message}

    @staticmethod
    def forbidden(message: str = "Forbidden") -> Dict[str, Any]:
        return {"code": JSONRPCError.FORBIDDEN, "message": message}

    @staticmethod
    def not_found(resource: str) -> Dict[str, Any]:
        return {
            "code": JSONRPCError.NOT_FOUND,
            "message": f"Resource not found: {resource}",
        }

    @staticmethod
    def not_initialized(method: str = None) -> Dict[str, Any]:
        message = "Server not initialized"
        if method:
            message += f". Please call 'initialize' first. (method: {method})"
        return {"code": JSONRPCError.NOT_INITIALIZED, "message": message}
