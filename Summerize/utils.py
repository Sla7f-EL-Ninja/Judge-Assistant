"""
utils.py

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
    """Return a named logger for a pipeline module.

    Example:
        logger = get_logger("hakim.node_0")
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Curly-brace safety for LangChain prompt templates
# ---------------------------------------------------------------------------

def escape_braces(text: str) -> str:
    """Escape { and } in user-provided content before embedding in a
    ChatPromptTemplate template string.

    When user content is passed directly as part of a template string (not as a
    named format variable), Python's str.format will try to interpret any {…}
    in that content as variable placeholders.  Replacing { with {{ and } with }}
    prevents the KeyError while the final formatted string still contains single
    braces, because Python's str.format unescapes {{ → { in the template.

    NOTE: Do NOT apply this to content that is passed as a named keyword
    argument value to format_messages() — in that case the braces are already
    safe.  Apply it only when you are building the template string itself by
    f-string concatenation of user-supplied text.
    """
    return text.replace("{", "{{").replace("}", "}}")


# ---------------------------------------------------------------------------
# Arabic text normalization for keyword matching
# ---------------------------------------------------------------------------

_HAMZA_RE = re.compile(r"[أإآ]")
_ALEF_MAKSURA_RE = re.compile(r"ى")


def normalize_arabic_for_matching(text: str) -> str:
    """Normalize Arabic text variations for robust keyword matching.

    Normalizes:
      • Hamza forms  (أ / إ / آ → ا)
      • Alef Maksura (ى → ي)

    Apply to both the search text AND the keyword list.  Do NOT apply to the
    text that will be displayed or sent to the LLM — this is for matching only.
    """
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

    Args:
        parser:      An object with an ``invoke(messages)`` method (e.g. a
                     LangChain chain created by ``llm.with_structured_output()``).
        messages:    The input to pass to ``parser.invoke()``.
        max_retries: Number of additional attempts after the first failure.
                     Default 2 → up to 3 total attempts.
        base_delay:  Seconds to wait before the first retry; doubles each time.
        logger:      Optional logger for retry warnings.

    Returns:
        The return value of ``parser.invoke(messages)`` on success.

    Raises:
        The last exception when all retries are exhausted, or immediately for
        non-transient errors.
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