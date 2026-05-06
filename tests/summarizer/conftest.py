"""
conftest.py — Shared fixtures for the Hakim Summarizer test suite.

Path setup ensures the project root is on sys.path so that the summarize
package is importable without installing it.
"""

import pathlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — project root must be on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixture directory
# ---------------------------------------------------------------------------
FIXTURE_DIR = _REPO_ROOT / "tests" / "CASE_RAG" / "fixtures"

FIXTURE_FILENAMES = [
    "صحيفة_دعوى.txt",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
    "تقرير_الخبير.txt",
    "تقرير_الطب_الشرعي.txt",
    "محضر_جلسة_25_03_2024.txt",
    "حكم_المحكمة.txt",
]


# ---------------------------------------------------------------------------
# Session-scoped: raw fixture texts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_fixture_texts() -> dict:
    """Read all 7 Arabic legal fixture files → {filename: text}."""
    texts = {}
    for fname in FIXTURE_FILENAMES:
        fpath = FIXTURE_DIR / fname
        if fpath.exists():
            texts[fname] = fpath.read_text(encoding="utf-8")
        else:
            texts[fname] = ""
    return texts


@pytest.fixture(scope="session")
def fixture_documents(raw_fixture_texts) -> list:
    """Convert fixture texts to pipeline input format."""
    return [
        {"doc_id": fname.replace(".txt", ""), "raw_text": text}
        for fname, text in raw_fixture_texts.items()
        if text
    ]


# ---------------------------------------------------------------------------
# Function-scoped: mock LLM
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm():
    """MagicMock LLM where .with_structured_output() returns a configurable parser mock."""
    llm = MagicMock()
    parser = MagicMock()
    llm.with_structured_output.return_value = parser
    return llm


def make_mock_llm_with_result(result: Any):
    """Create a mock LLM whose parser returns *result* on invoke()."""
    llm = MagicMock()
    parser = MagicMock()
    parser.invoke.return_value = result
    llm.with_structured_output.return_value = parser
    return llm


def make_mock_llm_raising(exc: Exception):
    """Create a mock LLM whose parser raises *exc* on invoke()."""
    llm = MagicMock()
    parser = MagicMock()
    parser.invoke.side_effect = exc
    llm.with_structured_output.return_value = parser
    return llm


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_chunk():
    """Factory for NormalizedChunk dicts (as returned by model_dump())."""

    def _make(
        chunk_id="chunk-001",
        doc_id="doc-test",
        clean_text="نص قانوني تجريبي",
        doc_type="صحيفة دعوى",
        party="المدعي",
        page_number=1,
        paragraph_number=1,
    ) -> dict:
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page_number": page_number,
            "paragraph_number": paragraph_number,
            "clean_text": clean_text,
            "doc_type": doc_type,
            "party": party,
        }

    return _make


@pytest.fixture()
def make_classified_chunk(make_chunk):
    """Factory for ClassifiedChunk dicts."""

    def _make(role="الوقائع", confidence=1.0, **kwargs) -> dict:
        return {**make_chunk(**kwargs), "role": role, "confidence": confidence}

    return _make


@pytest.fixture()
def make_bullet():
    """Factory for LegalBullet dicts (as returned by model_dump())."""

    def _make(
        bullet_id="bullet-001",
        role="الوقائع",
        bullet="ادعى المدعي بأن العقد قد أبرم في تاريخ كذا",
        source=None,
        party="المدعي",
        chunk_id="chunk-001",
    ) -> dict:
        return {
            "bullet_id": bullet_id,
            "role": role,
            "bullet": bullet,
            "source": source or ["doc-test ص1 ف1"],
            "party": party,
            "chunk_id": chunk_id,
        }

    return _make


@pytest.fixture()
def make_role_aggregation():
    """Factory for RoleAggregation dicts."""

    def _make(
        role="الوقائع",
        agreed=None,
        disputed=None,
        party_specific=None,
    ) -> dict:
        return {
            "role": role,
            "agreed": agreed or [],
            "disputed": disputed or [],
            "party_specific": party_specific or [],
        }

    return _make


@pytest.fixture()
def make_theme_cluster():
    """Factory for ThemeCluster dicts."""

    def _make(
        theme_name="موضوع تجريبي",
        agreed=None,
        disputed=None,
        party_specific=None,
        bullet_count=1,
    ) -> dict:
        return {
            "theme_name": theme_name,
            "agreed": agreed or [],
            "disputed": disputed or [],
            "party_specific": party_specific or [],
            "bullet_count": bullet_count,
        }

    return _make


@pytest.fixture()
def make_themed_role(make_theme_cluster):
    """Factory for ThemedRole dicts."""

    def _make(role="الوقائع", themes=None) -> dict:
        if themes is None:
            themes = [make_theme_cluster()]
        return {"role": role, "themes": themes}

    return _make


@pytest.fixture()
def make_theme_summary():
    """Factory for ThemeSummary dicts."""

    def _make(
        theme="موضوع تجريبي",
        summary="ملخص الموضوع القانوني",
        key_disputes=None,
        sources=None,
    ) -> dict:
        return {
            "theme": theme,
            "summary": summary,
            "key_disputes": key_disputes or [],
            "sources": sources or ["doc-test ص1 ف1"],
        }

    return _make


@pytest.fixture()
def make_role_theme_summaries(make_theme_summary):
    """Factory for RoleThemeSummaries dicts."""

    def _make(role="الوقائع", theme_summaries=None) -> dict:
        if theme_summaries is None:
            theme_summaries = [make_theme_summary()]
        return {"role": role, "theme_summaries": theme_summaries}

    return _make


# ---------------------------------------------------------------------------
# Node instance fixtures (with mock LLM)
# ---------------------------------------------------------------------------


@pytest.fixture()
def node0(mock_llm):
    from summarize.nodes.intake import Node0_DocumentIntake
    return Node0_DocumentIntake(mock_llm)


@pytest.fixture()
def node1(mock_llm):
    from summarize.nodes.classifier import Node1_RoleClassifier
    return Node1_RoleClassifier(mock_llm)


@pytest.fixture()
def node2(mock_llm):
    from summarize.nodes.extractor import Node2_BulletExtractor
    return Node2_BulletExtractor(mock_llm)


@pytest.fixture()
def node3(mock_llm):
    from summarize.nodes.aggregator import Node3_Aggregator
    return Node3_Aggregator(mock_llm)


@pytest.fixture()
def node4a(mock_llm):
    from summarize.nodes.clustering import Node4A_ThematicClustering
    return Node4A_ThematicClustering(mock_llm)


@pytest.fixture()
def node4b(mock_llm):
    from summarize.nodes.synthesis import Node4B_ThemeSynthesis
    return Node4B_ThemeSynthesis(mock_llm)


@pytest.fixture()
def node5(mock_llm):
    from summarize.nodes.brief import Node5_BriefGenerator
    return Node5_BriefGenerator(mock_llm)
