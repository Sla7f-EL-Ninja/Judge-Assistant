"""Shared error types for MCP server/client boundary."""

import json
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    QUERY_VALIDATION = "QUERY_VALIDATION"
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"
    LLM_BUDGET = "LLM_BUDGET"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    OFF_TOPIC = "OFF_TOPIC"
    INVALID_ARG = "INVALID_ARG"
    INTERNAL = "INTERNAL"


class ToolError(Exception):
    """Structured error returned by an MCP tool (not a transport error)."""

    def __init__(self, code: ErrorCode, message: str, **details: Any):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"{code}: {message}")


class MCPUnavailable(Exception):
    """Raised when the MCP child process cannot be reached after one respawn."""

    def __init__(self, server: str, cause: Exception):
        self.server = server
        self.cause = cause
        super().__init__(f"MCP server '{server}' unavailable: {cause}")


def raise_tool_error(code: ErrorCode, message: str, **details: Any) -> None:
    """Raise a FastMCP ToolError with JSON-encoded payload so the client can
    deserialize the typed error code and details."""
    from mcp.server.fastmcp.exceptions import ToolError as _FastMCPToolError

    payload = json.dumps({"code": code.value, "message": message, "details": details})
    raise _FastMCPToolError(payload)
