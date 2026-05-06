"""MCPClient — synchronous JSON-RPC client over subprocess stdio.

The MCP SDK (mcp 1.x) uses newline-delimited JSON: each JSON object is one
line on stdout/stdin. This client speaks that protocol directly.

Threading model: one threading.Lock per instance serialises all stdio framing.
Tier-0 parallel dispatch uses one MCPClient per server (different processes),
so civil_law_rag and case_doc_rag calls never share a lock.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from typing import Any, Dict
import queue

from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError

logger = logging.getLogger(__name__)

_RESPAWN_LIMIT = 1
_CALL_TIMEOUT = 120      # 5 min — covers first cold RAG + LLM call
_HANDSHAKE_TIMEOUT = 600  # first-run model download (bge-reranker-v2-m3) can take ~300s


class MCPClient:
    def __init__(self, server_module: str, call_timeout: int = _CALL_TIMEOUT):
        self._server_module = server_module
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._id = 0
        self._respawns = 0
        self._call_timeout = call_timeout

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        import time
        t0 = time.monotonic()
        logger.info("[TRACE] %s — spawning child process", self._server_module)
        self._proc = self._spawn()
        logger.info("[TRACE] %s — child spawned (pid=%d) in %.2fs",
                    self._server_module, self._proc.pid, time.monotonic() - t0)
        logger.info("[TRACE] %s — starting handshake (timeout=%ds)",
                    self._server_module, _HANDSHAKE_TIMEOUT)
        t1 = time.monotonic()
        self._handshake()
        logger.info("[TRACE] %s — handshake complete in %.2fs",
                    self._server_module, time.monotonic() - t1)
        logger.info("MCP server started: %s (pid=%d) — total start time %.2fs",
                    self._server_module, self._proc.pid, time.monotonic() - t0)

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            logger.info("MCP server stopped: %s", self._server_module)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", self._server_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit parent stderr so server logs are visible
            bufsize=-1,
        )

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, obj: Dict[str, Any]) -> None:
        line = (json.dumps(obj) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _recv(self, timeout: int | None = None) -> Dict[str, Any]:
        effective_timeout = timeout if timeout is not None else self._call_timeout
        result_q: queue.Queue = queue.Queue()

        def _read():
            try:
                line = self._proc.stdout.readline()
                result_q.put(("ok", line))
            except Exception as e:
                result_q.put(("err", e))

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        try:
            kind, value = result_q.get(timeout=effective_timeout)
        except queue.Empty:
            raise BrokenPipeError(
                f"MCP server '{self._server_module}' timed out after {effective_timeout}s"
            )

        if kind == "err":
            raise BrokenPipeError(f"MCP server '{self._server_module}' read error: {value}")

        line = value
        if not line:
            raise BrokenPipeError(f"MCP server '{self._server_module}' closed stdout")

        return json.loads(line.decode("utf-8"))

    def _handshake(self) -> None:
        import time
        req_id = self._next_id()
        logger.debug("[TRACE] %s — sending initialize (req_id=%d)", self._server_module, req_id)
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hakim-supervisor", "version": "1.0"},
            },
        })
        logger.debug("[TRACE] %s — waiting for initialize response (timeout=%ds)",
                     self._server_module, _HANDSHAKE_TIMEOUT)
        t0 = time.monotonic()
        resp = self._recv(timeout=_HANDSHAKE_TIMEOUT)
        logger.debug("[TRACE] %s — initialize response received in %.2fs",
                     self._server_module, time.monotonic() - t0)
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed for {self._server_module}: {resp['error']}")
        # Send the required initialized notification (no id = notification)
        logger.debug("[TRACE] %s — sending initialized notification", self._server_module)
        self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })
        logger.debug("[TRACE] %s — handshake done", self._server_module)

    def _do_call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        import time
        req_id = self._next_id()
        logger.debug("[TRACE] %s — sending tools/call (tool=%s req_id=%d)",
                     self._server_module, tool, req_id)
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        })
        logger.info("[TRACE] %s — waiting for tool response (tool=%s timeout=%ds)",
                    self._server_module, tool, self._call_timeout)
        t0 = time.monotonic()
        resp = self._recv()
        logger.info("[TRACE] %s — tool response received (tool=%s elapsed=%.2fs)",
                    self._server_module, tool, time.monotonic() - t0)
        return resp

    def _parse_response(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error from {self._server_module}: {resp['error']}")

        result = resp.get("result", {})
        content = result.get("content", [])
        text = content[0].get("text", "{}") if content else "{}"

        if result.get("isError"):
            try:
                err = json.loads(text)
            except json.JSONDecodeError:
                # FastMCP wraps ToolError: "ErrorCode.INTERNAL — Error executing tool <name>: <json>"
                # Extract the embedded JSON payload if present.
                brace = text.find("{")
                if brace != -1:
                    try:
                        err = json.loads(text[brace:])
                    except json.JSONDecodeError:
                        raise ToolError(ErrorCode.INTERNAL, text)
                else:
                    raise ToolError(ErrorCode.INTERNAL, text)

            code_str = err.get("code", "INTERNAL")
            try:
                code = ErrorCode(code_str)
            except ValueError:
                code = ErrorCode.INTERNAL
            raise ToolError(code, err.get("message", text), **err.get("details", {}))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"answer": text}

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def call(self, tool: str, **args: Any) -> Dict[str, Any]:
        """Call *tool* with keyword *args*. Returns parsed dict on success.

        Transport errors trigger one respawn + one retry.
        Persistent failure → MCPUnavailable.
        ToolErrors are NOT retried — they are re-raised as-is.
        """
        with self._lock:
            try:
                resp = self._do_call(tool, args)
                result = self._parse_response(resp)
                self._respawns = 0
                return result
            except ToolError:
                raise  # tool-level errors never trigger respawn
            except (BrokenPipeError, OSError, ConnectionError, RuntimeError) as transport_err:
                logger.warning(
                    "MCP transport error (server=%s tool=%s): %s — respawning",
                    self._server_module, tool, transport_err,
                )
                if self._respawns >= _RESPAWN_LIMIT:
                    raise MCPUnavailable(self._server_module, transport_err)
                self._respawns += 1
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = self._spawn()
                self._handshake()
                try:
                    resp = self._do_call(tool, args)
                    result = self._parse_response(resp)
                    self._respawns = 0
                    return result
                except ToolError:
                    raise
                except Exception as retry_err:
                    raise MCPUnavailable(self._server_module, retry_err) from retry_err