"""
Shared utilities for the Hakim Summarization pipeline.

Provides:
  - escape_braces: Escape curly braces in user content for ChatPromptTemplate safety
  - normalize_arabic_for_matching: Normalize Arabic text for robust keyword matching
  - get_logger: Return a named logger configured for the pipeline
  - llm_invoke_with_retry: Invoke an LLM parser with simple exponential-backoff retry
"""

import logging
import re
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a pipeline module."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Curly-brace safety for LangChain prompt templates
# ---------------------------------------------------------------------------

def escape_braces(text: str) -> str:
    """Escape { and } in user-provided content before embedding in a
    ChatPromptTemplate template string."""
    return text.replace("{", "{{").replace("}", "}}")


# ---------------------------------------------------------------------------
# Arabic text normalization for keyword matching
# ---------------------------------------------------------------------------

_HAMZA_RE = re.compile(r"[أإآ]")
_ALEF_MAKSURA_RE = re.compile(r"ى")


def normalize_arabic_for_matching(text: str) -> str:
    """Normalize Arabic text variations for robust keyword matching."""
    text = _HAMZA_RE.sub("ا", text)
    text = _ALEF_MAKSURA_RE.sub("ي", text)
    return text


# ---------------------------------------------------------------------------
# LLM retry wrapper
# ---------------------------------------------------------------------------

def llm_invoke_with_retry(
    parser: Any,
    messages: Any,
    max_retries: int = 2,
    base_delay: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Invoke ``parser.invoke(messages)`` with exponential-backoff retry.

    Retries on transient errors (rate limits, timeouts, connection resets).
    Non-transient errors (schema validation, auth failures) are raised immediately.
    """
    _TRANSIENT_SIGNALS = (
        "rate limit", "ratelimit", "rate_limit",
        "too many requests",
        "timeout", "timed out",
        "deadline",           # gRPC DeadlineExceeded — Gemini 504
        "deadlineexceeded",
        "connection", "connect",
        "503", "429", "502", "504",
    )

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return parser.invoke(messages)
        except Exception as exc:
            exc_lower = str(exc).lower()
            is_transient = any(sig in exc_lower for sig in _TRANSIENT_SIGNALS)

            if not is_transient or attempt >= max_retries:
                raise

            delay = base_delay * (2 ** attempt)
            if logger:
                logger.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc,
                )
            time.sleep(delay)
            last_exc = exc

    # Should never be reached, but keeps type checkers happy
    raise RuntimeError("llm_invoke_with_retry exhausted all retries") from last_exc
