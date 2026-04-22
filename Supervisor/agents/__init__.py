"""Agent adapters providing a uniform interface to the worker agents."""

from Supervisor.agents.base import AgentAdapter, AgentResult
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from Supervisor.agents.case_reasoner_adapter import CaseReasonerAdapter

__all__ = [
    "AgentAdapter",
    "AgentResult",
    "CivilLawRAGAdapter",
    "CaseDocRAGAdapter",
    "CaseReasonerAdapter",
]
