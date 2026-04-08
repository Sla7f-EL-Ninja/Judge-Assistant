"""case_doc_rag.routers -- Pure routing functions for graph conditional edges.

Zero side effects. No LLM calls. No imports from nodes/, infrastructure, or prompts.
Only imports: state types and standard library.

Rule 5: Router functions return a string key and nothing else. They read from
state, compute one string key, and return it. They never write to state.
"""

from RAG.case_doc_rag.state import AgentState, SubQuestionState

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


def docSelectorRouter(state: AgentState) -> str:
    """Route after documentSelector: DocumentFinalizer / dispatchQuestions / errorResponse."""
    if state.get("error"):
        return "errorResponse"
    if state.get("doc_selection_mode") == "retrieve_specific_doc":
        return "DocumentFinalizer"
    return "dispatchQuestions"


def proceedRouter(state: SubQuestionState) -> str:
    """Route after retrievalGrader: generateAnswer / refineQuestion / cannotAnswer.

    This is the ONLY location that checks the rephraseCount ceiling.
    """
    if state.get("proceedToGenerate", False) is True:
        return "generateAnswer"
    if state.get("rephraseCount", 0) >= _MAX_REPHRASE:
        return "cannotAnswer"
    return "refineQuestion"
