"""
graph.py
--------
Builds and returns the compiled Civil Law RAG LangGraph.

This module is PURE — it has no side-effects at import time.
Specifically, it does NOT call ensure_civil_law_indexed() or open any
network connections.  Those are the responsibility of the startup
lifespan in api/app.py.

Usage::

    from RAG.civil_law_rag.graph import build_graph
    app = build_graph()
    result = app.invoke(make_initial_state())
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from RAG.civil_law_rag.state import State
from RAG.civil_law_rag.nodes import (
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
from RAG.civil_law_rag.routers import (
    llm_grader_router,
    rule_grader_router,
    top_level_router,
)

_compiled_app = None


def build_graph():
    """Build and compile the Civil Law RAG graph.  Cached after first call."""
    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    graph = StateGraph(State)

    # Nodes
    graph.add_node("preprocessor_node",      preprocessor_node)
    graph.add_node("off_topic_node",         off_topic_node)
    graph.add_node("textual_node",           textual_node)
    graph.add_node("scope_classifier_node",  scope_classifier_node)
    graph.add_node("retrieve_node",          retrieve_node)
    graph.add_node("rule_grader_node",       rule_grader_node)
    graph.add_node("refine_node",            refine_node)
    graph.add_node("llm_grader_node",        llm_grader_node)
    graph.add_node("generate_answer_node",   generate_answer_node)
    graph.add_node("cannot_answer_node",     cannot_answer_node)

    # Edges
    graph.add_edge(START, "preprocessor_node")

    graph.add_conditional_edges(
        "preprocessor_node",
        top_level_router,
        {
            "off_topic_node":        "off_topic_node",
            "textual_node":          "textual_node",
            "scope_classifier_node": "scope_classifier_node",
            "cannot_answer_node":    "cannot_answer_node",
        },
    )

    graph.add_edge("off_topic_node",          END)
    graph.add_edge("textual_node",            END)
    graph.add_edge("scope_classifier_node",   "retrieve_node")
    graph.add_edge("retrieve_node",           "rule_grader_node")

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
