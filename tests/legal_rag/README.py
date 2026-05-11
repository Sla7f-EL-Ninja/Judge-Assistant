"""
README for legal_rag test suite
================================

File layout
-----------
RAG/legal_rag/tests/
    conftest.py              shared fixtures, mock factories, SAMPLE_TOC
    test_routers.py          unit — all 4 router functions (pure logic, no mocks)
    test_fallback.py         unit — off_topic_node, cannot_answer_node
    test_service.py          unit — validate_query, _extract_sources, ask_question
    test_corpus_router.py    unit — corpus_router_node (mock LLM + registry)
    test_preprocessor.py     unit — preprocessor_node (mock LLM)
    test_graders.py          unit — rule_grader_node, llm_grader_node
    test_refine.py           unit — refine_node (mock LLM)
    test_generate.py         unit — generate_answer_node + citation helpers
    test_scope_classifier.py unit — scope_classifier_node (mock LLM + TOC)
    test_retrieve.py         unit — retrieve_node (mock VS + reranker)
    test_textual.py          unit — textual_node (mock Qdrant client)
    test_graph_flow.py       integration — full compiled graph, all major paths
    test_e2e.py              e2e — real LLM + Qdrant (opt-in via -m e2e)

Running
-------
# All unit + integration tests (default, no live services):
    pytest RAG/legal_rag/tests/ -v

# Only unit tests (fast):
    pytest RAG/legal_rag/tests/ -v -m "not e2e"

# Only e2e tests (requires live LLM + Qdrant):
    pytest RAG/legal_rag/tests/test_e2e.py -m e2e -v

# Run a single file:
    pytest RAG/legal_rag/tests/test_graders.py -v

# Run a specific test:
    pytest RAG/legal_rag/tests/test_routers.py::TestRuleGraderRouter::test_max_retries_reached_overrides_grade -v

# Show coverage:
    pytest RAG/legal_rag/tests/ --cov=RAG.legal_rag --cov-report=term-missing -v

Add to pytest.ini
-----------------
[pytest]
markers =
    e2e: end-to-end tests requiring live LLM and Qdrant services
    slow: tests that take longer than 5 seconds

# To skip e2e by default, add:
addopts = -m "not e2e"
"""
