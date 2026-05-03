# """
# graph.py
# --------
# Builds and returns the compiled legal_rag LangGraph.

# This module is PURE — it has no side-effects at import time.
# It does NOT call ensure_indexed() or open any network connections.
# Those are the responsibility of the startup lifespan in api/app.py.

# build_graph() is cached per corpus name so each corpus compiles
# its graph exactly once, regardless of how many times it is called.

# Usage::

#     from RAG.legal_rag.graph import build_graph
#     from RAG.civil_law_rag.corpus import CIVIL_LAW_CORPUS

#     app = build_graph(CIVIL_LAW_CORPUS)
#     result = app.invoke(make_initial_state(CIVIL_LAW_CORPUS))
# """

# from __future__ import annotations

# from typing import Dict

# from langgraph.graph import END, START, StateGraph

# from RAG.legal_rag.corpus_config import CorpusConfig
# from RAG.legal_rag.state import State
# from RAG.legal_rag.nodes import (
#     cannot_answer_node,
#     generate_answer_node,
#     llm_grader_node,
#     off_topic_node,
#     preprocessor_node,
#     refine_node,
#     retrieve_node,
#     rule_grader_node,
#     scope_classifier_node,
#     textual_node,
# )
# from RAG.legal_rag.routers import (
#     llm_grader_router,
#     rule_grader_router,
#     top_level_router,
# )

# # Per-corpus compiled graph cache  {corpus_name: CompiledGraph}
# _compiled_apps: Dict[str, object] = {}


# def build_graph(corpus_config: CorpusConfig):
#     """Build and compile the legal RAG graph for *corpus_config*.

#     The compiled graph is cached by corpus name after the first call.
#     The graph topology is identical for all corpora; only the state
#     (specifically state["corpus_config"]) distinguishes them at runtime.
#     """
#     if corpus_config.name in _compiled_apps:
#         return _compiled_apps[corpus_config.name]

#     graph = StateGraph(State)

#     graph.add_node("preprocessor_node",     preprocessor_node)
#     graph.add_node("off_topic_node",        off_topic_node)
#     graph.add_node("textual_node",          textual_node)
#     graph.add_node("scope_classifier_node", scope_classifier_node)
#     graph.add_node("retrieve_node",         retrieve_node)
#     graph.add_node("rule_grader_node",      rule_grader_node)
#     graph.add_node("refine_node",           refine_node)
#     graph.add_node("llm_grader_node",       llm_grader_node)
#     graph.add_node("generate_answer_node",  generate_answer_node)
#     graph.add_node("cannot_answer_node",    cannot_answer_node)

#     graph.add_edge(START, "preprocessor_node")

#     graph.add_conditional_edges(
#         "preprocessor_node",
#         top_level_router,
#         {
#             "off_topic_node":        "off_topic_node",
#             "textual_node":          "textual_node",
#             "scope_classifier_node": "scope_classifier_node",
#             "cannot_answer_node":    "cannot_answer_node",
#         },
#     )

#     graph.add_edge("off_topic_node",         END)
#     graph.add_edge("textual_node",           END)
#     graph.add_edge("scope_classifier_node",  "retrieve_node")
#     graph.add_edge("retrieve_node",          "rule_grader_node")

#     graph.add_conditional_edges(
#         "rule_grader_node",
#         rule_grader_router,
#         {
#             "generate_answer_node": "generate_answer_node",
#             "refine_node":          "refine_node",
#             "llm_grader_node":      "llm_grader_node",
#             "cannot_answer_node":   "cannot_answer_node",
#         },
#     )

#     graph.add_edge("refine_node", "scope_classifier_node")

#     graph.add_conditional_edges(
#         "llm_grader_node",
#         llm_grader_router,
#         {
#             "generate_answer_node": "generate_answer_node",
#             "refine_node":          "refine_node",
#             "cannot_answer_node":   "cannot_answer_node",
#         },
#     )

#     graph.add_edge("generate_answer_node", END)
#     graph.add_edge("cannot_answer_node",   END)

#     compiled = graph.compile()
#     _compiled_apps[corpus_config.name] = compiled
#     return compiled


"""
graph.py
--------
Builds and returns the compiled legal_rag LangGraph.

This module is PURE — it has no side-effects at import time.
It does NOT call ensure_indexed() or open any network connections.
Those are the responsibility of the startup lifespan in api/app.py.

Architecture (unified graph):
    The graph is now corpus-agnostic at startup.  corpus_router_node runs
    first and injects state["corpus_config"] before the preprocessor ever
    runs, so all downstream nodes (which already read corpus_config from
    state) are unchanged.

    There is now exactly ONE compiled graph shared by all corpora.
    build_graph() is cached after the first call.

Usage::

    from RAG.legal_rag.graph import build_graph

    app = build_graph()
    state = make_initial_state()      # no corpus_config needed upfront
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
    corpus_router_router,
    llm_grader_router,
    rule_grader_router,
    top_level_router,
)

# Singleton compiled graph — built once, shared across all corpora.
_compiled_app = None


def build_graph():
    """Build and compile the unified legal RAG graph.

    The compiled graph is cached after the first call; subsequent calls
    return the same object instantly.

    corpus_config is no longer an argument — it is resolved at runtime
    inside corpus_router_node and stored in state["corpus_config"].
    """
    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    graph = StateGraph(State)

    # ── Nodes ─────────────────────────────────────────────────────────────
    graph.add_node("corpus_router_node",    corpus_router_node)
    graph.add_node("preprocessor_node",     preprocessor_node)
    graph.add_node("off_topic_node",        off_topic_node)
    graph.add_node("textual_node",          textual_node)
    graph.add_node("scope_classifier_node", scope_classifier_node)
    graph.add_node("retrieve_node",         retrieve_node)
    graph.add_node("rule_grader_node",      rule_grader_node)
    graph.add_node("refine_node",           refine_node)
    graph.add_node("llm_grader_node",       llm_grader_node)
    graph.add_node("generate_answer_node",  generate_answer_node)
    graph.add_node("cannot_answer_node",    cannot_answer_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.add_edge(START, "corpus_router_node")

    # ── Corpus router → preprocessor or off_topic ─────────────────────────
    graph.add_conditional_edges(
        "corpus_router_node",
        corpus_router_router,
        {
            "preprocessor_node": "preprocessor_node",
            "off_topic_node":    "off_topic_node",
        },
    )

    # ── Preprocessor → type classification branch ─────────────────────────
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