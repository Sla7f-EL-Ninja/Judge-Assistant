"""case_doc_rag.state -- TypedDict definitions for graph state.

Defines AgentState (main graph) and SubQuestionState (per-branch fan-out).
No functions, no logic. No imports from inside the package.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


def _last_value(existing, new):
    """Reducer that keeps the latest value. Required for fields shared between
    AgentState and SubQuestionState so that concurrent fan-out branches do not
    cause INVALID_CONCURRENT_GRAPH_UPDATE at merge time.
    """
    return new


class AgentState(TypedDict):
    """Main graph state contract between all nodes and the Supervisor.

    Input fields are set once by the caller via run() and never modified.
    Processing fields are written by specific nodes as documented.
    """

    # -- Input fields (set once, never modified by nodes) --
    query: str
    case_id: Annotated[str, _last_value]
    conversation_history: Annotated[List[Dict[str, str]], _last_value]
    request_id: Annotated[str, _last_value]

    # -- Query processing fields --
    sub_questions: List[str]
    on_topic: bool

    # -- Document selection fields --
    doc_selection_mode: Annotated[
        Literal["retrieve_specific_doc", "restrict_to_doc", "no_doc_specified"],
        _last_value,
    ]
    selected_doc_id: Annotated[Optional[str], _last_value]
    doc_titles: List[str]

    # -- Fan-out result field (CRITICAL: annotated reducer required) --
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]

    # -- Output and error fields --
    final_answer: str
    error: Optional[str]


class SubQuestionState(TypedDict):
    """Per-branch state for parallel fan-out.

    Each parallel branch spawned by dispatchQuestions gets its own
    independent SubQuestionState. No field from AgentState is
    automatically inherited -- all required values are explicitly
    copied by dispatchQuestions at dispatch time.
    """

    # -- Fields copied from AgentState at dispatch time (read-only) --
    sub_question: str
    case_id: Annotated[str, _last_value]
    conversation_history: Annotated[List[Dict[str, str]], _last_value]
    selected_doc_id: Annotated[Optional[str], _last_value]
    doc_selection_mode: Annotated[str, _last_value]
    request_id: Annotated[str, _last_value]

    # -- Retrieval field --
    retrieved_docs: List[Document]

    # -- Retry and grading fields (branch-local) --
    proceedToGenerate: bool
    rephraseCount: int

    # -- Output fields --
    sub_answer: str
    sources: List[str]
    found: bool
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]
