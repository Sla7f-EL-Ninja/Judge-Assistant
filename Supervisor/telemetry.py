"""
telemetry.py

One-time startup wiring for Sentry (P1.7.6), LangSmith (P1.7.3), and
structured JSON logging (P1.7.4).

Call ``setup_telemetry()`` once at process start.  All setup is opt-in:
missing env vars or missing optional libraries produce a warning, not a crash.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_telemetry() -> None:
    """Configure all observability backends."""
    _setup_logging()
    _setup_sentry()
    _setup_langsmith()


def _setup_logging() -> None:
    """Configure JSON logging when LOG_FORMAT=json (P1.7.4)."""
    try:
        from config.supervisor import LOG_FORMAT
    except Exception:
        LOG_FORMAT = "text"

    if LOG_FORMAT != "json":
        return

    try:
        from pythonjsonlogger import jsonlogger
        root = logging.getLogger()
        if not any(isinstance(h.formatter, jsonlogger.JsonFormatter) for h in root.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s"
                )
            )
            root.addHandler(handler)
        logger.debug("JSON logging configured")
    except ImportError:
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.stdlib.add_log_level,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.BoundLogger,
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
            )
            logger.debug("structlog JSON logging configured")
        except ImportError:
            logger.warning(
                "LOG_FORMAT=json requested but neither python-json-logger "
                "nor structlog is installed; falling back to text logging"
            )


def _setup_sentry() -> None:
    """Initialise Sentry error tracking if SENTRY_DSN is set (P1.7.6)."""
    try:
        from config.supervisor import SENTRY_DSN
    except Exception:
        SENTRY_DSN = os.getenv("SENTRY_DSN", "")

    if not SENTRY_DSN:
        return

    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,
            environment=os.getenv("APP_ENV", "production"),
        )
        logger.info("Sentry initialised (environment=%s)", os.getenv("APP_ENV", "production"))
    except ImportError:
        logger.warning("sentry-sdk not installed; Sentry error tracking disabled")
    except Exception as exc:
        logger.warning("Sentry init failed (non-fatal): %s", exc)


def _setup_langsmith() -> None:
    """Configure LangSmith tracing if LANGCHAIN_API_KEY is present (P1.7.3)."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        return

    try:
        from config.supervisor import LANGSMITH_PROJECT
    except Exception:
        LANGSMITH_PROJECT = "hakim-supervisor"

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    logger.info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])
