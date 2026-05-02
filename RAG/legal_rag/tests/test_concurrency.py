# """
# test_concurrency.py
# -------------------
# Concurrency safety test for the Civil Law RAG state factory.

# The critical invariant: each invocation must receive a completely
# independent state — no shared mutable lists between concurrent calls.

# This test simulates 50 concurrent invocations and asserts that
# query_history, last_results, and answer_history are never shared.
# """

# from __future__ import annotations

# import threading
# from typing import List

# import pytest
# from langchain_core.documents import Document

# from RAG.civil_law_rag.state import make_initial_state


# def _mutate_state(state: dict, query: str, results: list) -> None:
#     """Simulate what nodes do to state."""
#     state["query_history"].append(query)
#     state["last_results"].extend(results)
#     state["answer_history"].append(f"answer for {query}")


# def test_make_initial_state_deep_copy():
#     """States returned by make_initial_state must not share list references."""
#     s1 = make_initial_state()
#     s2 = make_initial_state()

#     s1["query_history"].append("query1")
#     s1["last_results"].append(Document(page_content="doc", metadata={}))

#     assert s2["query_history"] == [], "query_history shared between states!"
#     assert s2["last_results"] == [], "last_results shared between states!"


# def test_concurrent_state_isolation():
#     """50 threads each mutate their own state — no cross-contamination."""
#     states: List[dict] = []
#     lock = threading.Lock()

#     def worker(idx: int):
#         state = make_initial_state()
#         _mutate_state(
#             state,
#             query=f"query_{idx}",
#             results=[Document(page_content=f"doc_{idx}", metadata={"index": idx})],
#         )
#         with lock:
#             states.append(state)

#     threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()

#     # Each state must have exactly 1 query in its history
#     for state in states:
#         assert len(state["query_history"]) == 1, (
#             f"Expected 1 query in history, got {len(state['query_history'])}: "
#             f"{state['query_history']}"
#         )
#         assert len(state["last_results"]) == 1, (
#             f"Expected 1 result, got {len(state['last_results'])}"
#         )
#         assert len(state["answer_history"]) == 1


# def test_llm_call_count_starts_at_zero():
#     """LLM call budget counter must start fresh for every invocation."""
#     states = [make_initial_state() for _ in range(10)]
#     for s in states:
#         assert s["llm_call_count"] == 0


# def test_retry_count_starts_at_zero():
#     state = make_initial_state()
#     assert state["retry_count"] == 0


# def test_max_retries_default_is_three():
#     state = make_initial_state()
#     assert state["max_retries"] == 3


"""
test_concurrency.py
-------------------
Concurrency safety test for the Legal RAG state factory.
"""

from __future__ import annotations

import threading
from typing import List

import pytest
from langchain_core.documents import Document

# UPDATE: Pointing to legal_rag
from RAG.legal_rag.state import make_initial_state


def _mutate_state(state: dict, query: str, results: list) -> None:
    state["query_history"].append(query)
    state["last_results"].extend(results)
    state["answer_history"].append(f"answer for {query}")


def test_make_initial_state_deep_copy():
    s1 = make_initial_state()
    s2 = make_initial_state()

    s1["query_history"].append("query1")
    s1["last_results"].append(Document(page_content="doc", metadata={}))

    assert s2["query_history"] == [], "query_history shared between states!"
    assert s2["last_results"] == [], "last_results shared between states!"


def test_concurrent_state_isolation():
    states: List[dict] = []
    lock = threading.Lock()

    def worker(idx: int):
        state = make_initial_state()
        _mutate_state(
            state,
            query=f"query_{idx}",
            results=[Document(page_content=f"doc_{idx}", metadata={"index": idx})],
        )
        with lock:
            states.append(state)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for state in states:
        assert len(state["query_history"]) == 1
        assert len(state["last_results"]) == 1
        assert len(state["answer_history"]) == 1


def test_llm_call_count_starts_at_zero():
    states = [make_initial_state() for _ in range(10)]
    for s in states:
        assert s["llm_call_count"] == 0


def test_retry_count_starts_at_zero():
    state = make_initial_state()
    assert state["retry_count"] == 0


def test_max_retries_default_is_three():
    state = make_initial_state()
    assert state["max_retries"] == 3