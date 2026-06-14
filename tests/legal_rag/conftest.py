# """
# conftest.py
# -----------
# Shared pytest fixtures and helpers for the legal_rag test suite.

# Add these lines to your pytest.ini under [pytest]:
#     markers =
#         e2e: end-to-end tests that require live LLM + Qdrant services
#         slow: tests that take longer than 5 seconds

# Run only unit tests (default):
#     pytest RAG/legal_rag/tests/

# Run e2e tests:
#     pytest RAG/legal_rag/tests/ -m e2e

# Skip e2e (explicit):
#     pytest RAG/legal_rag/tests/ -m "not e2e"
# """

# from __future__ import annotations

# import json
# import os
# from typing import List
# from unittest.mock import MagicMock

# import pytest
# from langchain_core.documents import Document

# # ---------------------------------------------------------------------------
# # Disable LangSmith tracing globally for all tests — must happen before any
# # node imports so @traceable decorators see the env var at decoration time.
# # ---------------------------------------------------------------------------
# # os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ.setdefault("LANGCHAIN_API_KEY", "test-key")
# os.environ.setdefault("LANGSMITH_API_KEY", "test-key")


# # ---------------------------------------------------------------------------
# # Register custom markers
# # ---------------------------------------------------------------------------
# def pytest_configure(config: pytest.Config) -> None:
#     config.addinivalue_line(
#         "markers",
#         "e2e: end-to-end tests requiring live LLM and Qdrant services",
#     )
#     config.addinivalue_line(
#         "markers",
#         "slow: tests that take longer than 5 seconds",
#     )


# # ---------------------------------------------------------------------------
# # CorpusConfig fixtures
# # ---------------------------------------------------------------------------
# from RAG.legal_rag.corpus_config import CorpusConfig


# @pytest.fixture
# def civil_corpus() -> CorpusConfig:
#     return CorpusConfig(
#         name="civil",
#         collection_name="civil_law_docs",
#         source_filter_value="civil_law",
#         docs_path="/fake/civil_law.txt",
#         law_display_name="القانون المدني المصري",
#         corpus_version="1.0.0",
#         prompts_version="1.0.0",
#     )


# @pytest.fixture
# def evidence_corpus() -> CorpusConfig:
#     return CorpusConfig(
#         name="evidence",
#         collection_name="evidence_docs",
#         source_filter_value="evidence_law",
#         docs_path="/fake/evidence_law.txt",
#         law_display_name="قانون الإثبات المصري",
#         corpus_version="1.0.0",
#         prompts_version="1.0.0",
#     )


# @pytest.fixture
# def procedures_corpus() -> CorpusConfig:
#     return CorpusConfig(
#         name="procedures",
#         collection_name="procedures_docs",
#         source_filter_value="procedures_law",
#         docs_path="/fake/procedures_law.txt",
#         law_display_name="قانون المرافعات المصري",
#         corpus_version="1.0.0",
#         prompts_version="1.0.0",
#     )


# @pytest.fixture
# def mock_registry(civil_corpus, evidence_corpus, procedures_corpus) -> dict:
#     """Fake corpus registry matching the shape of _get_registry()."""
#     return {
#         "civil": civil_corpus,
#         "evidence": evidence_corpus,
#         "procedures": procedures_corpus,
#     }


# # ---------------------------------------------------------------------------
# # State factory fixtures
# # ---------------------------------------------------------------------------
# from RAG.legal_rag.state import make_initial_state


# @pytest.fixture
# def fresh_state() -> dict:
#     """Bare initial state — no corpus_config (as produced by make_initial_state)."""
#     return make_initial_state()


# @pytest.fixture
# def base_state(civil_corpus) -> dict:
#     """Initial state with civil corpus already resolved (simulates post-corpus-router state)."""
#     state = make_initial_state()
#     state["corpus_config"] = civil_corpus
#     state["last_query"] = "ما هي شروط صحة العقد؟"
#     return state


# # ---------------------------------------------------------------------------
# # Document factory
# # ---------------------------------------------------------------------------
# @pytest.fixture
# def make_doc():
#     """Factory for langchain Document objects with legal metadata."""
#     def _factory(
#         content: str = "نص المادة القانونية",
#         index: int = 89,
#         source: str = "civil_law",
#         chapter: str = "الفصل الأول",
#         title: str | None = None,
#         section: str | None = None,
#         doc_type: str = "article",
#         **extra_meta,
#     ) -> Document:
#         meta: dict = {
#             "index": index,
#             "source": source,
#             "type": doc_type,
#             "chapter": chapter,
#             "title": title or f"المادة {index}",
#         }
#         if section:
#             meta["section"] = section
#         meta.update(extra_meta)
#         return Document(page_content=content, metadata=meta)

#     return _factory


# # ---------------------------------------------------------------------------
# # LLM mock helpers — used by individual test modules
# # ---------------------------------------------------------------------------
# def make_llm_response(content: str) -> MagicMock:
#     """Return a mock that mimics a LangChain LLM response."""
#     resp = MagicMock()
#     resp.content = content
#     return resp


# def make_mock_llm(*response_contents: str) -> MagicMock:
#     """Return a mock LLM.

#     - Single string  → always returns that response.
#     - Multiple strings → returns them in order via side_effect.
#     """
#     llm = MagicMock()
#     if len(response_contents) == 1:
#         llm.invoke.return_value = make_llm_response(response_contents[0])
#     else:
#         llm.invoke.side_effect = [make_llm_response(c) for c in response_contents]
#     return llm


# # ---------------------------------------------------------------------------
# # Sample Table-of-Contents structure (used by scope_classifier tests)
# # ---------------------------------------------------------------------------
# SAMPLE_TOC = [
#     {
#         "id": "1",
#         "title": "الفصل الأول: العقد",
#         "book": "الكتاب الأول",
#         "part": "الجزء الأول",
#         "sections": [
#             {"id": "1", "title": "القسم الأول: الانعقاد"},
#             {"id": "2", "title": "القسم الثاني: الصحة"},
#         ],
#     },
#     {
#         "id": "2",
#         "title": "الفصل الثاني: المسؤولية",
#         "book": "الكتاب الأول",
#         "part": "الجزء الثاني",
#         "sections": [
#             {"id": "1", "title": "القسم الأول: المسؤولية العقدية"},
#         ],
#     },
# ]


"""
conftest.py
-----------
Shared pytest fixtures and helpers for the legal_rag test suite.

Plugin wiring
-------------
LegalRAGReportPlugin is registered here so it is active on every run without
needing an `-p` flag or an `addopts` entry.  The plugin writes a JSON report
to <this_directory>/legal_rag_test_report.json by default.
Override the path with:
    pytest --legal-rag-report=/path/to/report.json

Markers
-------
Add these lines to your pytest.ini under [pytest]:
    markers =
        e2e:    end-to-end tests that require live LLM + Qdrant services
        slow:   tests that take longer than 5 seconds
        golden: curated golden-set evaluation (subset of e2e)

Run only unit tests (default):
    pytest RAG/legal_rag/tests/

Run e2e tests:
    pytest RAG/legal_rag/tests/ -m e2e

Run golden-set evaluation (live services):
    pytest RAG/legal_rag/tests/test_golden_set.py -m "golden and e2e" -v

Skip e2e (explicit):
    pytest RAG/legal_rag/tests/ -m "not e2e"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Disable LangSmith tracing globally — must happen before any node imports
# so @traceable decorators see the env var at decoration time.
# ---------------------------------------------------------------------------
os.environ.setdefault("LANGCHAIN_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# CLI option — registered here so pytest resolves it before configure runs
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption(
            "--legal-rag-report",
            action="store",
            default=None,
            metavar="PATH",
            help=(
                "Output path for the legal_rag JSON report. "
                "Defaults to <conftest_dir>/legal_rag_test_report.json"
            ),
        )
    except ValueError:
        # Raised when the option is already registered (plugin loaded twice).
        pass


# ---------------------------------------------------------------------------
# Register custom markers + the report plugin
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    # ── markers ────────────────────────────────────────────────────────────
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests requiring live LLM and Qdrant services",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take longer than 5 seconds",
    )
    config.addinivalue_line(
        "markers",
        "golden: golden-set evaluation against curated Q&A pairs (subset of e2e)",
    )

    # ── report plugin ──────────────────────────────────────────────────────
    _this_dir = Path(__file__).parent

    # Ensure this directory is importable so the plugin can be found.
    if str(_this_dir) not in sys.path:
        sys.path.insert(0, str(_this_dir))

    # Resolve the output path: CLI flag → default next to conftest.
    try:
        cli_path = config.getoption("--legal-rag-report", default=None)
    except (ValueError, AttributeError):
        cli_path = None

    report_path = Path(cli_path) if cli_path else (_this_dir / "legal_rag_test_report.json")

    try:
        from legal_rag_report_plugin import LegalRAGReportPlugin

        # Guard against double-registration (e.g. worker processes in xdist).
        if not config.pluginmanager.get_plugin("legal_rag_report_plugin_instance"):
            plugin = LegalRAGReportPlugin(report_path)
            config.pluginmanager.register(plugin, "legal_rag_report_plugin_instance")
            print(
                f"\n✅ [LegalRAG Report] Plugin active → report will be written to:\n"
                f"   {report_path.absolute()}\n"
            )
    except ImportError as exc:
        print(
            f"\n⚠️  [LegalRAG Report] Plugin not found — "
            f"place legal_rag_report_plugin.py next to conftest.py.\n"
            f"   ({exc})\n"
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
        "civil":      civil_corpus,
        "evidence":   evidence_corpus,
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
    """Initial state with civil corpus resolved (simulates post-corpus-router state)."""
    state = make_initial_state()
    state["corpus_config"] = civil_corpus
    state["last_query"]    = "ما هي شروط صحة العقد؟"
    return state


# ---------------------------------------------------------------------------
# Document factory
# ---------------------------------------------------------------------------
@pytest.fixture
def make_doc():
    """Factory for langchain Document objects with legal metadata."""
    def _factory(
        content:  str = "نص المادة القانونية",
        index:    int = 89,
        source:   str = "civil_law",
        chapter:  str = "الفصل الأول",
        title:    str | None = None,
        section:  str | None = None,
        doc_type: str = "article",
        **extra_meta,
    ) -> Document:
        meta: dict = {
            "index":   index,
            "source":  source,
            "type":    doc_type,
            "chapter": chapter,
            "title":   title or f"المادة {index}",
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
    """
    Return a mock LLM.
    · Single string  → always returns that response.
    · Multiple strings → returns them in order via side_effect.
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
        "id":    "1",
        "title": "الفصل الأول: العقد",
        "book":  "الكتاب الأول",
        "part":  "الجزء الأول",
        "sections": [
            {"id": "1", "title": "القسم الأول: الانعقاد"},
            {"id": "2", "title": "القسم الثاني: الصحة"},
        ],
    },
    {
        "id":    "2",
        "title": "الفصل الثاني: المسؤولية",
        "book":  "الكتاب الأول",
        "part":  "الجزء الثاني",
        "sections": [
            {"id": "1", "title": "القسم الأول: المسؤولية العقدية"},
        ],
    },
]