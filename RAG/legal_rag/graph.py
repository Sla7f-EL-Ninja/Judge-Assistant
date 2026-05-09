"""
graph.py
--------
Builds and returns the compiled legal_rag LangGraph.

This module is PURE — it has no side-effects at import time.
It does NOT call ensure_indexed() or open any network connections.
Those are the responsibility of the startup lifespan in api/app.py.

Architecture (unified graph):
    The graph is corpus-agnostic at startup.  preprocessor_node runs first
    so off-topic queries (non-Arabic, too short, clearly off-domain) are
    rejected cheaply before any corpus LLM call.  corpus_classifier_node
    runs second and injects state["corpus_config"] for queries that passed
    the preprocessor's off-topic check.

    There is exactly ONE compiled graph shared by all corpora.
    build_graph() is cached after the first call.

Flow::
    START
      → preprocessor_node
          → [off_topic_node | corpus_classifier_node]
                → [off_topic_node | textual_node | scope_classifier_node]
                      → retrieve_node → rule_grader_node → ...
                                                         → generate_answer_node
                                                         → cannot_answer_node

Usage::

    from RAG.legal_rag.graph import build_graph

    app = build_graph()
    state = make_initial_state()
    state["last_query"] = "..."
    result = app.invoke(state)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from RAG.legal_rag.state import State
from RAG.legal_rag.nodes import (
    cannot_answer_node,
    generate_answer_node,
    llm_grader_node,
    off_topic_node,
    preprocessor_node,
    refine_node,
    retrieve_node,
    rule_grader_node,
    scope_classifier_node,
    textual_node,
)
from RAG.legal_rag.nodes.corpus_router import corpus_router_node
from RAG.legal_rag.routers import (
    post_preprocessor_router,
    corpus_classifier_router,
    llm_grader_router,
    rule_grader_router,
)

# Singleton compiled graph — built once, shared across all corpora.
_compiled_app = None


def build_graph():
    """Build and compile the unified legal RAG graph.

    The compiled graph is cached after the first call; subsequent calls
    return the same object instantly.

    corpus_config is not an argument — it is resolved at runtime inside
    corpus_classifier_node and stored in state["corpus_config"].
    """
    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    graph = StateGraph(State)

    # ── Nodes ─────────────────────────────────────────────────────────────
    graph.add_node("preprocessor_node",      preprocessor_node)
    graph.add_node("corpus_classifier_node", corpus_router_node)   # function renamed internally
    graph.add_node("off_topic_node",         off_topic_node)
    graph.add_node("textual_node",           textual_node)
    graph.add_node("scope_classifier_node",  scope_classifier_node)
    graph.add_node("retrieve_node",          retrieve_node)
    graph.add_node("rule_grader_node",       rule_grader_node)
    graph.add_node("refine_node",            refine_node)
    graph.add_node("llm_grader_node",        llm_grader_node)
    graph.add_node("generate_answer_node",   generate_answer_node)
    graph.add_node("cannot_answer_node",     cannot_answer_node)

    # ── Entry point: preprocessor first ───────────────────────────────────
    graph.add_edge(START, "preprocessor_node")

    # ── After preprocessor: off_topic short-circuit or corpus classification
    graph.add_conditional_edges(
        "preprocessor_node",
        post_preprocessor_router,
        {
            "off_topic_node":        "off_topic_node",
            "corpus_classifier_node": "corpus_classifier_node",
            "cannot_answer_node":    "cannot_answer_node",
        },
    )

    # ── After corpus classifier: dispatch by classification + corpus_config ─
    graph.add_conditional_edges(
        "corpus_classifier_node",
        corpus_classifier_router,
        {
            "off_topic_node":        "off_topic_node",
            "textual_node":          "textual_node",
            "scope_classifier_node": "scope_classifier_node",
            "cannot_answer_node":    "cannot_answer_node",
        },
    )

    # ── Terminal / pass-through edges ─────────────────────────────────────
    graph.add_edge("off_topic_node",         END)
    graph.add_edge("textual_node",           END)
    graph.add_edge("scope_classifier_node",  "retrieve_node")
    graph.add_edge("retrieve_node",          "rule_grader_node")

    graph.add_conditional_edges(
        "rule_grader_node",
        rule_grader_router,
        {
            "generate_answer_node": "generate_answer_node",
            "refine_node":          "refine_node",
            "llm_grader_node":      "llm_grader_node",
            "cannot_answer_node":   "cannot_answer_node",
        },
    )

    graph.add_edge("refine_node", "scope_classifier_node")

    graph.add_conditional_edges(
        "llm_grader_node",
        llm_grader_router,
        {
            "generate_answer_node": "generate_answer_node",
            "refine_node":          "refine_node",
            "cannot_answer_node":   "cannot_answer_node",
        },
    )

    graph.add_edge("generate_answer_node", END)
    graph.add_edge("cannot_answer_node",   END)

    _compiled_app = graph.compile()
    return _compiled_app
