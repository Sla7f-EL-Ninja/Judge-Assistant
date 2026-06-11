"""
tests/CASE_RAG/conftest.py

Session-scoped fixtures for the case_doc_rag integration test suite.
All tests share one ingestion run; data is isolated by TEST_CASE_ID and
cleaned up in the autouse teardown fixture.

Plugin wiring
-------------
pytest_configure() at the bottom of this file manually registers
CaseDocRagReportPlugin so the report always drops next to this conftest,
even when the plugin is not listed in addopts.

register_result fixture
-----------------------
Every test that wants structured data captured in the report should
inject ``register_result`` and call it once with a CaseRagTestResult dict:

    def test_something(app, register_result):
        result = invoke_graph(...)
        register_result({
            "layer":              "B",
            "test_id":            "B1",
            "query":              query,
            "expected_keywords":  expected,
            "answer_text":        _get_answer_text(result),
            "sub_answers":        result.get("sub_answers", []),
            "doc_selection_mode": result.get("doc_selection_mode"),
            "expected_doc_mode":  "no_doc_specified",
            "on_topic":           result.get("on_topic"),
        })

All keys are optional except ``layer`` and ``test_id``.
"""

from __future__ import annotations

import logging
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — project root must be on sys.path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_CASE_ID: str = f"test_case_rag_{uuid4().hex[:12]}"

# Keep CASE_RAG on the same PDF fixture source used by the summarizer suite.
FIXTURE_DIR: Path = Path(r"D:\FUCK!!\Grad\TESTING DATA\First Case\PDFS")
FIXTURE_GLOB: str = "*.pdf"


def _discover_fixture_names() -> list[str]:
    if not FIXTURE_DIR.exists():
        return []
    return sorted(path.name for path in FIXTURE_DIR.glob(FIXTURE_GLOB))


FIXTURE_FILES: list[str] = _discover_fixture_names()

# PDF fixtures are classified by the production DocumentProcessor pipeline.
# Add per-file ground truth here only when the PDF inventory has been labeled.
EXPECTED_DOC_TYPES: dict[str, str] = {}


from config.taxonomy import get_unknown_label

UNKNOWN_DOC_TYPE = get_unknown_label()


def _wait_for_tcp(host: str, port: int, timeout_s: int = 120) -> None:
    """Wait until a local test dependency accepts TCP connections."""
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)

    pytest.skip(
        f"Qdrant is not reachable at {host}:{port} after {timeout_s}s. "
        f"Start Qdrant and wait for HTTP port {port} before running CASE_RAG. "
        f"Last error: {last_error}"
    )


def _shape_ingestion_result(file_path: str, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize DocumentProcessor output to the legacy test result shape."""
    classification = result.get("classification", {}) or {}
    metadata = result.get("metadata", {}) or {}
    doc_type = classification.get("final_type") or UNKNOWN_DOC_TYPE
    return {
        "file": file_path,
        "mongo_id": metadata.get("mongo_id"),
        "title": doc_type,
        "doc_type": doc_type,
        "confidence": classification.get("confidence", 0),
        "classification": classification,
        "text": result.get("text", ""),
        "file_type": result.get("file_type"),
        "source_file": metadata.get("source_file", file_path),
        "qdrant_chunks": metadata.get("qdrant_chunks", 0),
        "metadata": metadata,
    }


@dataclass
class _PipelineIngestorAdapter:
    """Compatibility wrapper around DocumentProcessor.pipeline for tests."""

    _pipeline: Any = field(init=False, repr=False)
    mongo_collection: Any = field(init=False)
    vectorstore: Any = field(init=False)
    _qdrant_collection_name: str = field(init=False)
    _minio_endpoint: str = field(init=False, default="")
    _minio_access_key: str = field(init=False, default="")
    _minio_secret_key: str = field(init=False, default="")
    _minio_bucket: str = field(init=False, default="")
    _minio_secure: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        from DocumentProcessor import pipeline

        self._pipeline = pipeline
        # Local test Qdrant often exposes HTTP on 6333 without gRPC on 6334.
        # Force REST here so fixture setup does not fail before tests run.
        pipeline.QDRANT_PREFER_GRPC = False
        if pipeline.QDRANT_HOST in {"localhost", "0.0.0.0"}:
            pipeline.QDRANT_HOST = "127.0.0.1"

        _wait_for_tcp(pipeline.QDRANT_HOST, pipeline.QDRANT_PORT)
        self.mongo_collection = pipeline._get_mongo_collection()
        self.vectorstore = pipeline._get_vectorstore()
        self._qdrant_collection_name = (
            getattr(self.vectorstore, "collection_name", None)
            or getattr(self.vectorstore, "_collection_name", None)
            or "case_docs"
        )

        minio_cfg = getattr(pipeline, "_minio_config", {})
        self._minio_endpoint = minio_cfg.get("endpoint", "localhost:9000")
        self._minio_access_key = minio_cfg.get("access_key", "minioadmin")
        self._minio_secret_key = minio_cfg.get("secret_key", "minioadmin")
        self._minio_bucket = minio_cfg.get("bucket", "hakim-files")
        self._minio_secure = minio_cfg.get("secure", False)

    def ingest_file(self, file_path: str, case_id: str) -> dict[str, Any]:
        pdf_path = Path(file_path)
        logger.info("Ingesting CASE_RAG PDF fixture: %s", pdf_path.name)
        result = self._pipeline.process_document(
            file_path=str(pdf_path),
            case_id=case_id,
            file_id=pdf_path.stem,
        )
        return _shape_ingestion_result(str(pdf_path), result)

    def ingest_files(self, file_paths: list[str], case_id: str) -> list[dict[str, Any]]:
        return [self.ingest_file(path, case_id=case_id) for path in file_paths]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_case_id() -> str:
    return TEST_CASE_ID


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    if not FIXTURE_DIR.exists():
        pytest.skip(
            f"PDF fixture directory not found: {FIXTURE_DIR}\n"
            "Create the directory and populate it with the case PDFs."
        )
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def fixture_files(fixture_dir) -> list[Path]:
    pdf_paths = sorted(fixture_dir.glob(FIXTURE_GLOB))
    if not pdf_paths:
        pytest.skip(f"No {FIXTURE_GLOB} files found in {fixture_dir}")
    return pdf_paths


@pytest.fixture(scope="session")
def file_ingestor(fixture_dir):
    """Expose a DocumentProcessor-backed ingestor with legacy test attributes."""
    return _PipelineIngestorAdapter()


@pytest.fixture(scope="session")
def ingestion_results(file_ingestor, fixture_files, test_case_id):
    """Ingest all PDF fixture files. Hard-fail if any basic invariant is broken."""
    file_paths = [str(pdf_path) for pdf_path in fixture_files]
    results = file_ingestor.ingest_files(file_paths, case_id=test_case_id)
    expected_count = len(file_paths)

    # --- Hard-fail assertions (abort session on failure) ---
    assert len(results) == expected_count, (
        f"Expected {expected_count} ingestion results, got {len(results)}. "
        f"Files ingested: {[r.get('file') for r in results]}"
    )
    for r in results:
        assert r.get("mongo_id") is not None, (
            f"mongo_id is None for file: {r.get('file')}"
        )
        assert r.get("title", "") != "", (
            f"Empty title for file: {r.get('file')}"
        )
        assert r.get("text", "").strip() != "", (
            f"No extracted text for file: {r.get('file')}"
        )
        assert r.get("qdrant_chunks", 0) > 0, (
            f"No Qdrant chunks indexed for file: {r.get('file')}"
        )

    return results


@pytest.fixture(scope="session")
def vectorstore_ready(file_ingestor, ingestion_results):
    """Inject the ingestor's vectorstore into the RAG infrastructure singleton."""
    import RAG.case_doc_rag.infrastructure as infra

    infra.set_vectorstore(file_ingestor.vectorstore)
    infra._mongo_collection = file_ingestor.mongo_collection
    infra._qdrant_client = file_ingestor.vectorstore.client
    return True


@pytest.fixture(scope="session")
def app(vectorstore_ready):
    """Build and return the compiled case_doc_rag LangGraph app."""
    from RAG.case_doc_rag.graph import build_graph
    return build_graph()


@pytest.fixture(scope="session")
def mongo_collection(file_ingestor, ingestion_results):
    """Return the MongoDB collection used during this test session."""
    return file_ingestor.mongo_collection


# ---------------------------------------------------------------------------
# register_result fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def register_result(request):
    """Function-scoped fixture: register a structured result payload with the
    CaseDocRagReportPlugin so it appears in the JSON report.

    Usage inside a test::

        def test_something(app, register_result):
            result = invoke_graph(...)
            register_result({
                "layer":              "B",
                "test_id":            "B1",
                "query":              query,
                "expected_keywords":  expected,
                "answer_text":        _get_answer_text(result),
                "sub_answers":        result.get("sub_answers", []),
                "doc_selection_mode": result.get("doc_selection_mode"),
                "expected_doc_mode":  "no_doc_specified",
                "on_topic":           result.get("on_topic"),
                "test_case_id":       TEST_CASE_ID,
            })

    Falls back gracefully if the plugin is not registered (e.g. during
    isolated unit runs) — so tests never fail because of this fixture.
    """
    plugin = request.config.pluginmanager.get_plugin(
        "case_doc_rag_report_plugin_instance"
    )
    nodeid = request.node.nodeid

    def _register(payload: dict) -> None:
        # Always stamp the nodeid and test_case_id into the payload
        payload.setdefault("test_case_id", TEST_CASE_ID)
        payload["_nodeid"] = nodeid
        if plugin is not None:
            plugin.register(nodeid, payload)
        else:
            logger.debug(
                "[register_result] Plugin not found; payload for %s dropped.", nodeid
            )

    return _register


# ---------------------------------------------------------------------------
# Session-scoped autouse teardown
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def cleanup(file_ingestor, ingestion_results, test_case_id):
    """Run all tests, then clean up every trace of TEST_CASE_ID."""
    yield  # ← all tests execute here

    logger.info("=== TEARDOWN: cleaning up test_case_id=%s ===", test_case_id)

    # 1. MongoDB: delete all docs for this case
    try:
        deleted = file_ingestor.mongo_collection.delete_many(
            {"case_id": test_case_id}
        )
        logger.info("MongoDB: deleted %d doc(s) for case_id=%s",
                    deleted.deleted_count, test_case_id)
    except Exception as exc:
        logger.warning("MongoDB cleanup failed: %s", exc)

    # 2. Qdrant: delete all vectors tagged with this case_id
    try:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        qdrant_client = file_ingestor.vectorstore.client
        qdrant_client.delete(
            collection_name=file_ingestor._qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.case_id",
                            match=MatchValue(value=test_case_id),
                        )
                    ]
                )
            ),
        )
        logger.info("Qdrant: deleted vectors for case_id=%s", test_case_id)
    except Exception as exc:
        logger.warning("Qdrant cleanup failed: %s", exc)

    # 3. MinIO: best-effort delete objects under TEST_CASE_ID/ prefix
    try:
        from minio import Minio

        minio_client = Minio(
            endpoint=file_ingestor._minio_endpoint,
            access_key=file_ingestor._minio_access_key,
            secret_key=file_ingestor._minio_secret_key,
            secure=file_ingestor._minio_secure,
        )
        objects = list(
            minio_client.list_objects(
                file_ingestor._minio_bucket,
                prefix=f"{test_case_id}/",
                recursive=True,
            )
        )
        for obj in objects:
            minio_client.remove_object(file_ingestor._minio_bucket, obj.object_name)
        logger.info("MinIO: removed %d object(s) under prefix=%s/",
                    len(objects), test_case_id)
    except Exception as exc:
        logger.warning("MinIO cleanup failed (non-fatal): %s", exc)

    # 4. Reset infrastructure singletons
    try:
        import RAG.case_doc_rag.infrastructure as infra
        infra._vectorstore = None
        infra._mongo_collection = None
        infra._qdrant_client = None
        infra._embedding_fn = None
        infra._llm_cache.clear()
        logger.info("Infrastructure singletons reset")
    except Exception as exc:
        logger.warning("Infrastructure reset failed: %s", exc)

    # 5. Clear titles cache
    try:
        from RAG.case_doc_rag.nodes.selection_nodes import (
            _titles_cache,
            _titles_cache_ts,
        )
        _titles_cache.clear()
        _titles_cache_ts.clear()
        logger.info("Titles cache cleared")
    except Exception as exc:
        logger.warning("Titles cache clear failed: %s", exc)

    logger.info("=== TEARDOWN COMPLETE for test_case_id=%s ===", test_case_id)


# ---------------------------------------------------------------------------
# Helper: invoke graph
# ---------------------------------------------------------------------------

def invoke_graph(app, query: str, case_id: str, **overrides) -> dict:
    """Build a full AgentState dict and invoke the compiled graph.

    Parameters
    ----------
    app : compiled LangGraph app
    query : str
        The judge's natural-language query.
    case_id : str
        Case identifier used for Qdrant/MongoDB filtering.
    **overrides
        Any AgentState field to override from the defaults.

    Returns
    -------
    dict
        The final AgentState after graph execution.
    """
    state = {
        "query": query,
        "case_id": case_id,
        "conversation_history": [],
        "request_id": str(uuid4()),
        "sub_questions": [],
        "on_topic": False,
        "doc_selection_mode": "no_doc_specified",
        "selected_doc_id": None,
        "doc_titles": [],
        "sub_answers": [],
        "final_answer": "",
        "error": None,
    }
    state.update(overrides)
    return app.invoke(state)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

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
    # 3. Register the report plugin
    try:
        from case_doc_rag_report_plugin import CaseDocRagReportPlugin

        report_path = current_dir / "case_doc_rag_test_report.json"
        plugin = CaseDocRagReportPlugin(report_path)
        config.pluginmanager.register(
            plugin, "case_doc_rag_report_plugin_instance"
        )
        print(
            f"\n✅ [CaseDocRag Report] Plugin hooked! Report will drop at:\n"
            f"   {report_path}\n"
        )
    except ImportError as exc:
        print(
            f"\n❌ [CaseDocRag Report] Could not import plugin: {exc}\n"
            f"   Place case_doc_rag_report_plugin.py next to conftest.py.\n"
        )
