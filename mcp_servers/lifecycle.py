"""MCP server process lifecycle — spawn, registry, shutdown."""

from __future__ import annotations

import atexit
import logging
from typing import Dict

from mcp_servers.client import MCPClient

logger = logging.getLogger(__name__)

_MCP_CLIENTS: Dict[str, MCPClient] = {}


def start_mcp_servers() -> None:
    """Spawn both MCP child processes eagerly. Idempotent."""
    import time
    if _MCP_CLIENTS:
        return

    t_total = time.monotonic()

    legal = MCPClient("mcp_servers.legal_rag_server")
    logger.info("[TRACE] lifecycle — starting legal_rag server")
    t0 = time.monotonic()
    legal.start()
    logger.info("[TRACE] lifecycle — legal_rag ready (%.2fs)", time.monotonic() - t0)

    logger.info("[TRACE] lifecycle — starting case_doc_rag server")
    t0 = time.monotonic()
    case = MCPClient("mcp_servers.case_doc_server")
    case.start()
    logger.info("[TRACE] lifecycle — case_doc_rag ready (%.2fs)", time.monotonic() - t0)

    _MCP_CLIENTS["legal_rag"] = legal
    _MCP_CLIENTS["case_doc_rag"] = case

    atexit.register(legal.shutdown)
    atexit.register(case.shutdown)

    logger.info("MCP servers started: legal_rag, case_doc_rag — total=%.2fs",
                time.monotonic() - t_total)


def get_client(name: str) -> MCPClient:
    """Return the named MCPClient. Raises RuntimeError if not started."""
    if not _MCP_CLIENTS:
        raise RuntimeError(
            "MCP servers not started. Call start_mcp_servers() first "
            "(normally done in Supervisor/__init__.py)."
        )
    client = _MCP_CLIENTS.get(name)
    if client is None:
        raise KeyError(f"Unknown MCP server '{name}'. Known: {list(_MCP_CLIENTS)}")
    return client