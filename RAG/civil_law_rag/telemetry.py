"""
telemetry.py
------------
Structured logging helpers for the Civil Law RAG pipeline.

All node-level log calls go through log_event() which emits a JSON-
compatible dict via the standard logging framework.  Production log
aggregators (Datadog, Loki, CloudWatch) can parse structured fields
without regex fragility.

Usage::

    from RAG.civil_law_rag.telemetry import get_logger, log_event

    logger = get_logger(__name__)
    log_event(logger, "retrieval", query=query, docs=len(docs), confidence=0.87)
"""

from __future__ import annotations

import logging
import json
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits root-level configuration.

    Never hardcode setLevel(DEBUG) inside library code — let the
    application's logging config drive verbosity.
    """
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log line.

    All keyword arguments are serialized as JSON fields alongside the
    event name, making log lines machine-parseable.

    Example output::
        {"event": "retrieval", "query": "...", "docs": 15, "confidence": 0.87}
    """
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))
