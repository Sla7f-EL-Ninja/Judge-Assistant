"""Agent adapters providing a uniform interface to the 5 worker agents."""

from Supervisor.agents.base import AgentAdapter, AgentResult
from Supervisor.agents.ocr_adapter import OCRAdapter
from Supervisor.agents.summarize_adapter import SummarizeAdapter
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from Supervisor.agents.case_reasoner_adapter import CaseReasonerAdapter

__all__ = [
    "AgentAdapter",
    "AgentResult",
    "OCRAdapter",
    "SummarizeAdapter",
    "CivilLawRAGAdapter",
    "CaseDocRAGAdapter",
    "CaseReasonerAdapter",
]
