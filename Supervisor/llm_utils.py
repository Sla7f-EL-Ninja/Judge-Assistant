"""
llm_utils.py

Utilities for Supervisor LLM calls: wall-clock timeout and transient-error
retry with exponential backoff.

Addresses:
  P1.5.1 — No LLM retry/backoff: 429/503 went straight to catch block.
  P1.5.3 — No LLM timeouts: a hanging LLM call hung the whole turn.
  P1.4.9 — No typed exception handling: is_transient_error() lets callers
            distinguish retryable from permanent failures.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from supervisor config; fall back to safe defaults
# ---------------------------------------------------------------------------

try:
    from config import cfg as _cfg
    _LLM_TIMEOUT_S: int = int(_cfg.supervisor.get("llm_timeout_seconds", 60))
    _LLM_MAX_RETRIES: int = int(_cfg.supervisor.get("llm_max_retries", 2))
    _LLM_BACKOFF_S: float = float(_cfg.supervisor.get("llm_retry_backoff_seconds", 2.0))
except Exception:
    _LLM_TIMEOUT_S = 60
    _LLM_MAX_RETRIES = 2
    _LLM_BACKOFF_S = 2.0

# Substrings (lowercased) that flag a transient / retryable error
_TRANSIENT_MARKERS = (
    "429",
    "503",
    "rate limit",
    "quota exceeded",
    "unavailable",
    "deadline exceeded",
    "overloaded",
    "too many requests",
    "resource exhausted",
    "connection",
)


def is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a retryable LLM / network error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def llm_invoke(
    fn: Callable,
    *args: Any,
    timeout_s: int = _LLM_TIMEOUT_S,
    max_retries: int = _LLM_MAX_RETRIES,
    backoff_s: float = _LLM_BACKOFF_S,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with a wall-clock timeout and retry.

    Transient errors (rate-limit, 503, timeout) are retried with exponential
    backoff up to *max_retries* times.  Permanent errors propagate immediately.

    Parameters
    ----------
    fn:
        Callable to invoke (e.g. ``llm.invoke``, ``structured_llm.invoke``).
    timeout_s:
        Per-attempt wall-clock limit in seconds.
    max_retries:
        Extra attempts after the first failure.
    backoff_s:
        Base backoff between retries; doubles each attempt.
    """
    for attempt in range(max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_s)
                except FuturesTimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"LLM call timed out after {timeout_s}s (attempt {attempt + 1})"
                    )

        except Exception as exc:
            transient = isinstance(exc, TimeoutError) or is_transient_error(exc)

            if transient and attempt < max_retries:
                wait = backoff_s * (2 ** attempt)
                logger.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, wait, exc,
                )
                time.sleep(wait)
            else:
                raise
