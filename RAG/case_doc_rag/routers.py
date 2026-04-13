"""case_doc_rag.routers -- Pure routing functions for graph conditional edges.

Zero side effects. No LLM calls. No imports from nodes/, infrastructure, or prompts.
Only imports: state types, langgraph Send, and standard library.

Rule 5: Router functions return a string key (or a list of Send objects for
fan-out) and nothing else. They read from state, compute the routing decision,
and return it. They never write to state.
"""

import logging
from typing import List, Union

from langgraph.types import Send

from RAG.case_doc_rag.state import AgentState, SubQuestionState

logger = logging.getLogger("case_doc_rag.routers")

# Single source of truth for the retry ceiling.
# proceedRouter is the sole enforcer. refineQuestion must not import this.
_MAX_REPHRASE: int = 2


def onTopicRouter(state: AgentState) -> str:
    """Route after questionClassifier: documentSelector / offTopicResponse / errorResponse."""
    if state.get("error"):
        return "errorResponse"
    if state.get("on_topic", False) is True:
        return "documentSelector"
    return "offTopicResponse"


def docSelectorDispatchRouter(
    state: AgentState,
) -> Union[str, List[Send]]:
    """Conditional-edge function after documentSelector (title-fetch step).

    Phase 8 refactor: the retrieve_specific_doc → DocumentFinalizer short-circuit
    has been removed from the main graph. That decision is now branch-local, made
    by branchDocSelector + branchDocSelectorRouter inside each parallel branch.

    This router now has two responsibilities only:
      1. Guard against pipeline errors  → return "errorResponse"
      2. Guard against empty sub_questions → return "errorResponse"
      3. Otherwise: unconditionally fan-out every sub-question to retrieve_branch
         via Send(), injecting doc_titles so branches never hit MongoDB.

    doc_titles is read from AgentState (populated by documentSelector) and copied
    verbatim into every branch's initial SubQuestionState.
    """
    if state.get("error"):
        logger.error(
            "[%s] docSelectorDispatchRouter: routing to errorResponse due to error: %s",
            state.get("request_id", ""), state.get("error"),
        )
        return "errorResponse"

    request_id = state.get("request_id", "")
    sub_questions = state.get("sub_questions", [])
    doc_titles = state.get("doc_titles", [])

    if not sub_questions:
        logger.error(
            "[%s] docSelectorDispatchRouter: no sub_questions, routing to errorResponse",
            request_id,
        )
        return "errorResponse"

    sends: List[Send] = []
    for sub_q in sub_questions:
        sends.append(
            Send(
                "retrieve_branch",
                {
                    # Sub-question for this branch
                    "sub_question": sub_q,
                    # Copied from AgentState (read-only inside branch)
                    "case_id": state.get("case_id", ""),
                    "conversation_history": state.get("conversation_history", []),
                    "request_id": request_id,
                    # Pre-fetched titles -- branchDocSelector reads this instead
                    # of hitting MongoDB, giving zero extra DB calls at fan-out.
                    "doc_titles": doc_titles,
                    # Sentinel defaults; overwritten by branchDocSelector
                    "selected_doc_id": None,
                    "doc_selection_mode": "no_doc_specified",
                    # Retrieval / grading fields (branch-local)
                    "retrieved_docs": [],
                    "proceedToGenerate": False,
                    "rephraseCount": 0,
                    # Output fields
                    "sub_answer": "",
                    "sources": [],
                    "found": False,
                    "sub_answers": [],
                },
            )
        )

    logger.info(
        "[%s] docSelectorDispatchRouter: dispatching %d branches",
        request_id, len(sends),
    )
    return sends


def branchDocSelectorRouter(state: SubQuestionState) -> str:
    """Route after branchDocSelector inside the branch sub-graph.

    Phase 5: pure function, zero side effects.

    Returns
    -------
    "BranchDocumentFinalizer"
        When doc_selection_mode == "retrieve_specific_doc".
        The branch short-circuits to BranchDocumentFinalizer → END without
        touching retrieval or grading.

    "retrieve"
        For "restrict_to_doc" or "no_doc_specified". The branch continues
        through the normal retrieve → retrievalGrader → ... pipeline.
        In the "restrict_to_doc" case, selected_doc_id has been set by
        branchDocSelector and retrieve() will scope its Qdrant filter to that doc.
    """
    mode = state.get("doc_selection_mode", "no_doc_specified")
    if mode == "retrieve_specific_doc":
        logger.debug(
            "[%s] branchDocSelectorRouter -> BranchDocumentFinalizer (mode=%s)",
            state.get("request_id", ""), mode,
        )
        return "BranchDocumentFinalizer"

    logger.debug(
        "[%s] branchDocSelectorRouter -> retrieve (mode=%s)",
        state.get("request_id", ""), mode,
    )
    return "retrieve"


def proceedRouter(state: SubQuestionState) -> str:
    """Route after retrievalGrader: generateAnswer / refineQuestion / cannotAnswer.

    This is the ONLY location that checks the rephraseCount ceiling.
    """
    if state.get("proceedToGenerate", False) is True:
        return "generateAnswer"
    if state.get("rephraseCount", 0) >= _MAX_REPHRASE:
        return "cannotAnswer"
    return "refineQuestion"


# ---------------------------------------------------------------------------
# DEPRECATED: docSelectorRouter
# ---------------------------------------------------------------------------


def docSelectorRouter(state: AgentState) -> str:
    """DEPRECATED -- superseded by docSelectorDispatchRouter.

    Kept for backwards-compatibility / reference only. Not wired into any graph.
    """
    if state.get("error"):
        return "errorResponse"
    if state.get("doc_selection_mode") == "retrieve_specific_doc":
        return "DocumentFinalizer"
    return "dispatchQuestions"
