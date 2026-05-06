"""
metrics.py

Prometheus metrics for the Supervisor workflow (P1.7.2).

All metrics are no-op stubs when prometheus_client is not installed or
PROMETHEUS_ENABLED=false, so the rest of the code can call them
unconditionally.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from config.supervisor import PROMETHEUS_ENABLED as _ENABLED
except Exception:
    _ENABLED = True

_active = False

try:
    if _ENABLED:
        from prometheus_client import Counter, Histogram

        # Total turns processed, labelled by terminal status
        TURN_COUNTER = Counter(
            "hakim_supervisor_turns_total",
            "Supervisor turns processed",
            ["status"],           # pass | partial_pass | fallback | off_topic
        )

        # Validation retries
        RETRY_COUNTER = Counter(
            "hakim_supervisor_retries_total",
            "Validation retry attempts",
            ["reason"],           # fail_hallucination | fail_relevance | fail_completeness | validator_error
        )

        # Fallback responses triggered
        FALLBACK_COUNTER = Counter(
            "hakim_supervisor_fallbacks_total",
            "Fallback responses returned",
        )

        # Per-agent errors
        AGENT_ERROR_COUNTER = Counter(
            "hakim_supervisor_agent_errors_total",
            "Agent invocation errors",
            ["agent"],
        )

        # LLM call latency per node (seconds)
        LLM_LATENCY = Histogram(
            "hakim_supervisor_llm_seconds",
            "LLM call wall-clock time",
            ["node"],
        )

        _active = True
        logger.debug("Prometheus metrics registered")

except ImportError:
    logger.warning("prometheus_client not installed; metrics disabled")
    _ENABLED = False

if not _active:
    # ---------------------------------------------------------------------------
    # No-op stubs — identical API, zero cost
    # ---------------------------------------------------------------------------
    import contextlib

    class _NoopLabelled:
        def inc(self, amount: float = 1) -> None:
            pass

    class _NoopCounter:
        def labels(self, *args, **kwargs) -> "_NoopLabelled":
            return _NoopLabelled()

        def inc(self, amount: float = 1) -> None:
            pass

    class _NoopHistogram:
        def labels(self, *args, **kwargs) -> "_NoopLabelled":
            return _NoopLabelled()

        def observe(self, amount: float) -> None:
            pass

        @contextlib.contextmanager
        def time(self):
            yield

    TURN_COUNTER = _NoopCounter()
    RETRY_COUNTER = _NoopCounter()
    FALLBACK_COUNTER = _NoopCounter()
    AGENT_ERROR_COUNTER = _NoopCounter()
    LLM_LATENCY = _NoopHistogram()
