"""
Supervisor Agent - LangGraph-based orchestrator for the Judge Assistant.

Sits on top of 4 worker agents (Summarization, Civil Law RAG,
Case Doc RAG, Case Reasoner), classifies judge intent, dispatches to
one or more workers, validates outputs, and maintains conversation history.
"""

from Supervisor.telemetry import setup_telemetry as _setup_telemetry

_setup_telemetry()

from Supervisor.graph import get_app, get_app_persistent  # noqa: E402,F401
