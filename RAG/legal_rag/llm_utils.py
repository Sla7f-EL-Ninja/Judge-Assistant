"""
llm_utils.py
------------
LLM invocation helper: budget guard + timeout enforcement in one call.

Timeout strategy: uses concurrent.futures to run the blocking LLM call on a
worker thread and waits with a wall-clock deadline.  The underlying call cannot
be cancelled (LangChain synchronous providers don't support cooperative
cancellation), but the calling node returns promptly and the budget guard
prevents further calls.  This is the most portable approach for Windows where
signal.alarm is unavailable.
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Optional

from config.legal_rag import MAX_LLM_CALLS, LLM_TIMEOUT
from RAG.legal_rag.errors import LLMBudgetExceededError, LLMTimeoutError
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm_invoke")


def invoke_with_budget_and_timeout(
    state: dict,
    llm: Any,
    prompt: Any,
    *,
    node: str,
    timeout: Optional[float] = None,
) -> Any:
    """Invoke `llm` with budget + timeout guards.

    Budget check: raises LLMBudgetExceededError if llm_call_count >= MAX_LLM_CALLS.
    Timeout: submits call to thread pool, waits up to `timeout` seconds, raises
             LLMTimeoutError if exceeded.  The underlying call continues to
             completion on the worker thread but its result is discarded.

    Increments state["llm_call_count"] on success.

    Args:
        state:   Mutable LangGraph state dict.
        llm:     LangChain LLM / ChatModel instance.
        prompt:  Prompt string or message list.
        node:    Calling node name, embedded in any raised exception.
        timeout: Override for LLM_TIMEOUT (seconds).

    Returns:
        LangChain AIMessage response.

    Raises:
        LLMBudgetExceededError: budget exhausted.
        LLMTimeoutError:        provider call timed out.
    """
    effective_timeout = timeout if timeout is not None else LLM_TIMEOUT

    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        log_event(logger, "llm_budget_exceeded", node=node, level=logging.WARNING)
        raise LLMBudgetExceededError(f"LLM budget exhausted at node '{node}' (limit={MAX_LLM_CALLS})")

    future = _executor.submit(llm.invoke, prompt)
    try:
        response = future.result(timeout=effective_timeout)
    except concurrent.futures.TimeoutError as exc:
        log_event(logger, "llm_timeout", node=node, timeout=effective_timeout, level=logging.WARNING)
        raise LLMTimeoutError(
            f"LLM call timed out after {effective_timeout}s at node '{node}'"
        ) from exc

    state["llm_call_count"] = state.get("llm_call_count", 0) + 1
    return response
