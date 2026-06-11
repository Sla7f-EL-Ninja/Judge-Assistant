"""
conftest.py — Shared fixtures for the Hakim Summarizer test suite.

PDF ingestion
-------------
Fixture documents are now read from a directory of PDF files.  Each PDF is
run through ``DocumentProcessor.pipeline.process_document`` (full ingest:
text extraction / OCR → classification → MongoDB → Qdrant) before the test
session starts.  The extracted text is then fed to the summarizer pipeline
exactly as the old .txt fixtures were.

PDF source directory
--------------------
``FIXTURE_DIR`` points at the local PDF folder.  All ``*.pdf`` files found
there are ingested; the stem of each filename becomes its ``doc_id``.

Path setup
----------
The project root is added to ``sys.path`` so that both the ``summarize`` and
``DocumentProcessor`` packages are importable without installing them.
"""

import pathlib
import sys
import logging
from typing import Any
from unittest.mock import MagicMock
import os
import pytest

# Ensure poppler is always found on Windows regardless of environment
import os
os.environ.setdefault("POPPLER_PATH", r"C:\poppler\Library\bin")

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Path setup — project root must be on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# PDF fixture directory
# ---------------------------------------------------------------------------

FIXTURE_DIR = pathlib.Path(r"D:\FUCK!!\Grad\TESTING DATA\First Case\PDFS")

# Case ID used when ingesting test PDFs into MongoDB / Qdrant.
# Change this if you want the test documents stored under a different case.
TEST_CASE_ID = "hakim-test-case"

@pytest.fixture(scope="session", autouse=True)
def ingest_pdf_fixtures(raw_fixture_texts):
    """
    Force PDF ingestion to run at session start for every test run.
    autouse=True means this fires even for unit tests that don't
    directly request fixture_documents.
    """
    ingested = sum(1 for t in raw_fixture_texts.values() if t)
    skipped  = sum(1 for t in raw_fixture_texts.values() if not t)
    logger.info(
        "PDF ingestion complete — %d ingested, %d failed/empty",
        ingested, skipped,
    )
    yield
    logger.info("Test session complete.")

# ---------------------------------------------------------------------------
# Session-scoped: ingest PDFs and return extracted texts
# ---------------------------------------------------------------------------


def _ingest_pdf(pdf_path: pathlib.Path) -> str:
    """
    Run a single PDF through the full DocumentProcessor pipeline.

    Returns the canonical extracted text (empty string on failure).
    Side-effects: document is stored in MongoDB and indexed in Qdrant.
    """
    from DocumentProcessor.pipeline import process_document

    logger.info("Ingesting PDF fixture: %s", pdf_path.name)
    try:
        result = process_document(
            file_path=str(pdf_path),
            case_id=TEST_CASE_ID,
            file_id=pdf_path.stem,
        )
        text = result.get("text", "")
        doc_type = result.get("classification", {}).get("final_type", "غير محدد")
        confidence = result.get("classification", {}).get("confidence", 0)
        logger.info(
            "  ✓ %s → doc_type='%s' confidence=%d chars=%d",
            pdf_path.name, doc_type, confidence, len(text),
        )
        return text
    except Exception as exc:
        logger.error("  ✗ Failed to ingest '%s': %s", pdf_path.name, exc)
        return ""


@pytest.fixture(scope="session")
def raw_fixture_texts() -> dict:
    """
    Ingest all PDFs in FIXTURE_DIR and return {filename: extracted_text}.

    Each PDF is processed exactly once per test session via
    ``DocumentProcessor.pipeline.process_document``.  If the directory does
    not exist or contains no PDFs the fixture returns an empty dict (tests
    that depend on real documents will then naturally fail or skip).
    """
    if not FIXTURE_DIR.exists():
        pytest.skip(
            f"PDF fixture directory not found: {FIXTURE_DIR}\n"
            "Create the directory and populate it with the case PDFs."
        )

    pdf_paths = sorted(FIXTURE_DIR.glob("*.pdf"))
    if not pdf_paths:
        pytest.skip(f"No *.pdf files found in {FIXTURE_DIR}")

    texts: dict = {}
    for pdf_path in pdf_paths:
        text = _ingest_pdf(pdf_path)
        texts[pdf_path.name] = text   # key = "filename.pdf"

    return texts


@pytest.fixture(scope="session")
def fixture_documents(raw_fixture_texts) -> list:
    """
    Convert ingested PDF texts to the pipeline input format.

    Returns a list of ``{"doc_id": <stem>, "raw_text": <text>}`` dicts,
    filtered to non-empty texts only (skips PDFs that failed extraction).
    """
    docs = []
    for filename, text in raw_fixture_texts.items():
        if not text:
            logger.warning("Skipping empty fixture document: %s", filename)
            continue
        stem = pathlib.Path(filename).stem   # "صحيفة_دعوى.pdf" → "صحيفة_دعوى"
        docs.append({"doc_id": stem, "raw_text": text})
    return docs


# ---------------------------------------------------------------------------
# Convenience fixture: list of discovered PDF stems (for parametrize / IDs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixture_doc_ids(raw_fixture_texts) -> list:
    """Return the list of doc_id strings for all successfully ingested PDFs."""
    return [
        pathlib.Path(fname).stem
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


# === Add this to the very bottom of conftest.py ===

# AFTER
def pytest_configure(config):
    import sys
    import logging
    from pathlib import Path

    # 1. Ensure the directory containing this conftest is in the Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # 2. Silence noisy third-party DEBUG loggers
    for noisy_logger in (
        "urllib3",
        "urllib3.connectionpool",
        "langsmith",
        "langsmith.client",
        "pydot",
        "pydot.core",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    # 2. Import the plugin directly
    try:
        from hakim_report_plugin import HakimReportPlugin
        report_path = current_dir / "hakim_test_report.json"

        # 3. Register the instance explicitly and print a success message
        plugin = HakimReportPlugin(report_path)
        config.pluginmanager.register(plugin, "hakim_report_instance_forced")
        
        # This will print to your console the moment you start the test suite
        print(f"\n✅ [Hakim Report] Plugin successfully hooked! Report will drop at:\n   {report_path}\n")
    except ImportError as e:
        print(f"\n❌ [Hakim Report Error] Could not find the plugin file: {e}\n")