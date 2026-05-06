"""
chat_reasoner

Multi-step reasoning sub-agent for the Hakim Supervisor.

Exports the compiled LangGraph app and the local AgentAdapter / AgentResult
interface so the adapter file and tests can import from one place.
"""

from chat_reasoner.graph import app
from chat_reasoner.interface import AgentAdapter, AgentResult
from chat_reasoner.state import ChatReasonerState

__all__ = ["app", "AgentAdapter", "AgentResult", "ChatReasonerState"]
