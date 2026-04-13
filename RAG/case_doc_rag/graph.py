"""case_doc_rag.graph -- Graph wiring and compilation.

No node logic. No prompts. No infrastructure calls. Imports all node
functions and routers, wires them, and returns the compiled app from
build_graph(). No graph compilation at module import time.

Post-refactor wiring
--------------------

Branch sub-graph (SubQuestionState):
    START
      → branchDocSelector
      → [branchDocSelectorRouter]
          "BranchDocumentFinalizer" → BranchDocumentFinalizer → END   (short-circuit)
          "retrieve"                → retrieve
                                      → retrievalGrader
                                      → [proceedRouter]
                                          "generateAnswer" → generateAnswer → END
                                          "refineQuestion" → refineQuestion → retrieve
                                          "cannotAnswer"   → cannotAnswer   → END

Main graph (AgentState):
    START
      → questionRewriter
      → questionClassifier
      → [onTopicRouter]
          "documentSelector"  → documentSelector (title-fetch only)
          "offTopicResponse"  → offTopicResponse → END
          "errorResponse"     → errorResponse    → END
    documentSelector
      → [docSelectorDispatchRouter]
          "errorResponse"  → errorResponse → END
          <Send list>      → retrieve_branch × N → mergeAnswers → END

Changes from the previous wiring:
  - DocumentFinalizer node and its edges removed from main graph.
  - documentSelector → DocumentFinalizer edge removed.
  - docSelectorDispatchRouter map reduced to {"errorResponse": "errorResponse"}.
  - branch: START now goes to branchDocSelector (not retrieve directly).
  - branch: conditional edge added after branchDocSelector.
  - branch: BranchDocumentFinalizer node added with → END edge.
"""

from langgraph.graph import END, START, StateGraph

from RAG.case_doc_rag.nodes.generation_nodes import (
    cannotAnswer,
    errorResponse,
    generateAnswer,
    mergeAnswers,
    refineQuestion,
)
from RAG.case_doc_rag.nodes.query_nodes import (
    offTopicResponse,
    questionClassifier,
    questionRewriter,
)
from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve, retrievalGrader
from RAG.case_doc_rag.nodes.selection_nodes import (
    BranchDocumentFinalizer,
    branchDocSelector,
    documentSelector,
)
from RAG.case_doc_rag.routers import (
    branchDocSelectorRouter,
    docSelectorDispatchRouter,
    onTopicRouter,
    proceedRouter,
)
from RAG.case_doc_rag.state import AgentState, SubQuestionState


def build_graph():
    """Build and compile the Case Doc RAG LangGraph workflow.

    Returns the compiled StateGraph app with .invoke() method.
    """
    # -----------------------------------------------------------------------
    # Branch sub-graph (operates on SubQuestionState)
    # -----------------------------------------------------------------------
    branch = StateGraph(SubQuestionState)

    # Nodes
    branch.add_node("branchDocSelector", branchDocSelector)
    branch.add_node("BranchDocumentFinalizer", BranchDocumentFinalizer)
    branch.add_node("retrieve", retrieve)
    branch.add_node("retrievalGrader", retrievalGrader)
    branch.add_node("refineQuestion", refineQuestion)
    branch.add_node("generateAnswer", generateAnswer)
    branch.add_node("cannotAnswer", cannotAnswer)

    # branchDocSelector is the new entry point of every branch
    branch.add_edge(START, "branchDocSelector")

    # After classification: short-circuit to BranchDocumentFinalizer OR enter retrieval
    branch.add_conditional_edges(
        "branchDocSelector",
        branchDocSelectorRouter,
        {
            "BranchDocumentFinalizer": "BranchDocumentFinalizer",
            "retrieve": "retrieve",
        },
    )

    # Short-circuit terminal edge
    branch.add_edge("BranchDocumentFinalizer", END)

    # Normal retrieval flow
    branch.add_edge("retrieve", "retrievalGrader")
    branch.add_conditional_edges(
        "retrievalGrader",
        proceedRouter,
        {
            "generateAnswer": "generateAnswer",
            "refineQuestion": "refineQuestion",
            "cannotAnswer": "cannotAnswer",
        },
    )
    branch.add_edge("refineQuestion", "retrieve")  # retry loop
    branch.add_edge("generateAnswer", END)
    branch.add_edge("cannotAnswer", END)

    branch_graph = branch.compile()

    # -----------------------------------------------------------------------
    # Main graph (operates on AgentState)
    # -----------------------------------------------------------------------
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("questionRewriter", questionRewriter)
    workflow.add_node("questionClassifier", questionClassifier)
    workflow.add_node("offTopicResponse", offTopicResponse)
    # documentSelector is now a lightweight title-fetch step only (no LLM call)
    workflow.add_node("documentSelector", documentSelector)
    workflow.add_node("retrieve_branch", branch_graph)
    workflow.add_node("mergeAnswers", mergeAnswers)
    workflow.add_node("errorResponse", errorResponse)
    # NOTE: DocumentFinalizer (main-graph) is intentionally NOT registered here.
    # The branch-local BranchDocumentFinalizer handles retrieve_specific_doc
    # short-circuits. The deprecated DocumentFinalizer symbol still exists in
    # selection_nodes.py for import compatibility until Phase 10 test cleanup.

    # Edges
    workflow.add_edge(START, "questionRewriter")
    workflow.add_edge("questionRewriter", "questionClassifier")

    workflow.add_conditional_edges(
        "questionClassifier",
        onTopicRouter,
        {
            "documentSelector": "documentSelector",
            "offTopicResponse": "offTopicResponse",
            "errorResponse": "errorResponse",
        },
    )
    workflow.add_edge("offTopicResponse", END)

    # documentSelector (title-fetch) → fan-out or errorResponse
    # The "retrieve_specific_doc" key is intentionally absent: that routing
    # decision is now made per-branch inside branchDocSelectorRouter.
    workflow.add_conditional_edges(
        "documentSelector",
        docSelectorDispatchRouter,
        {"errorResponse": "errorResponse"},
    )

    workflow.add_edge("retrieve_branch", "mergeAnswers")
    workflow.add_edge("mergeAnswers", END)
    workflow.add_edge("errorResponse", END)

    return workflow.compile()
