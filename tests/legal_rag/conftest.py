"""
conftest.py
-----------
Shared pytest fixtures and helpers for the legal_rag test suite.

Add these lines to your pytest.ini under [pytest]:
    markers =
        e2e: end-to-end tests that require live LLM + Qdrant services
        slow: tests that take longer than 5 seconds

Run only unit tests (default):
    pytest RAG/legal_rag/tests/

Run e2e tests:
    pytest RAG/legal_rag/tests/ -m e2e

Skip e2e (explicit):
    pytest RAG/legal_rag/tests/ -m "not e2e"
"""

from __future__ import annotations

import json
import os
from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Disable LangSmith tracing globally for all tests — must happen before any
# node imports so @traceable decorators see the env var at decoration time.
# ---------------------------------------------------------------------------
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("LANGCHAIN_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Register custom markers
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests requiring live LLM and Qdrant services",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take longer than 5 seconds",
    )


# ---------------------------------------------------------------------------
# CorpusConfig fixtures
# ---------------------------------------------------------------------------
from RAG.legal_rag.corpus_config import CorpusConfig


@pytest.fixture
def civil_corpus() -> CorpusConfig:
    return CorpusConfig(
        name="civil",
        collection_name="civil_law_docs",
        source_filter_value="civil_law",
        docs_path="/fake/civil_law.txt",
        law_display_name="القانون المدني المصري",
        corpus_version="1.0.0",
        prompts_version="1.0.0",
    )


@pytest.fixture
def evidence_corpus() -> CorpusConfig:
    return CorpusConfig(
        name="evidence",
        collection_name="evidence_docs",
        source_filter_value="evidence_law",
        docs_path="/fake/evidence_law.txt",
        law_display_name="قانون الإثبات المصري",
        corpus_version="1.0.0",
        prompts_version="1.0.0",
    )


@pytest.fixture
def procedures_corpus() -> CorpusConfig:
    return CorpusConfig(
        name="procedures",
        collection_name="procedures_docs",
        source_filter_value="procedures_law",
        docs_path="/fake/procedures_law.txt",
        law_display_name="قانون المرافعات المصري",
        corpus_version="1.0.0",
        prompts_version="1.0.0",
    )


@pytest.fixture
def mock_registry(civil_corpus, evidence_corpus, procedures_corpus) -> dict:
    """Fake corpus registry matching the shape of _get_registry()."""
    return {
        "civil": civil_corpus,
        "evidence": evidence_corpus,
        "procedures": procedures_corpus,
    }


# ---------------------------------------------------------------------------
# State factory fixtures
# ---------------------------------------------------------------------------
from RAG.legal_rag.state import make_initial_state


@pytest.fixture
def fresh_state() -> dict:
    """Bare initial state — no corpus_config (as produced by make_initial_state)."""
    return make_initial_state()


@pytest.fixture
def base_state(civil_corpus) -> dict:
    """Initial state with civil corpus already resolved (simulates post-corpus-router state)."""
    state = make_initial_state()
    state["corpus_config"] = civil_corpus
    state["last_query"] = "ما هي شروط صحة العقد؟"
    return state


# ---------------------------------------------------------------------------
# Document factory
# ---------------------------------------------------------------------------
@pytest.fixture
def make_doc():
    """Factory for langchain Document objects with legal metadata."""
    def _factory(
        content: str = "نص المادة القانونية",
        index: int = 89,
        source: str = "civil_law",
        chapter: str = "الفصل الأول",
        title: str | None = None,
        section: str | None = None,
        doc_type: str = "article",
        **extra_meta,
    ) -> Document:
        meta: dict = {
            "index": index,
            "source": source,
            "type": doc_type,
            "chapter": chapter,
            "title": title or f"المادة {index}",
        }
        if section:
            meta["section"] = section
        meta.update(extra_meta)
        return Document(page_content=content, metadata=meta)

    return _factory


# ---------------------------------------------------------------------------
# LLM mock helpers — used by individual test modules
# ---------------------------------------------------------------------------
def make_llm_response(content: str) -> MagicMock:
    """Return a mock that mimics a LangChain LLM response."""
    resp = MagicMock()
    resp.content = content
    return resp


def make_mock_llm(*response_contents: str) -> MagicMock:
    """Return a mock LLM.

    - Single string  → always returns that response.
    - Multiple strings → returns them in order via side_effect.
    """
    llm = MagicMock()
    if len(response_contents) == 1:
        llm.invoke.return_value = make_llm_response(response_contents[0])
    else:
        llm.invoke.side_effect = [make_llm_response(c) for c in response_contents]
    return llm


# ---------------------------------------------------------------------------
# Sample Table-of-Contents structure (used by scope_classifier tests)
# ---------------------------------------------------------------------------
SAMPLE_TOC = [
    {
        "id": "1",
        "title": "الفصل الأول: العقد",
        "book": "الكتاب الأول",
        "part": "الجزء الأول",
        "sections": [
            {"id": "1", "title": "القسم الأول: الانعقاد"},
            {"id": "2", "title": "القسم الثاني: الصحة"},
        ],
    },
    {
        "id": "2",
        "title": "الفصل الثاني: المسؤولية",
        "book": "الكتاب الأول",
        "part": "الجزء الثاني",
        "sections": [
            {"id": "1", "title": "القسم الأول: المسؤولية العقدية"},
        ],
    },
]
