"""
tests/CASE_RAG/conftest.py

Session-scoped fixtures for the case_doc_rag integration test suite.
All tests share one ingestion run; data is isolated by TEST_CASE_ID and
cleaned up in the autouse teardown fixture.
"""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from types import MethodType
from uuid import uuid4

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_CASE_ID: str = f"test_case_rag_{uuid4().hex[:12]}"

FIXTURE_DIR: Path = Path(__file__).parent / "fixtures"

FIXTURE_FILES = [
    "صحيفة_دعوى.txt",
    "تقرير_الخبير.txt",
    "تقرير_الطب_الشرعي.txt",
    "حكم_المحكمة.txt",
    "محضر_جلسة_25_03_2024.txt",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
]

# Ground-truth doc_type per fixture filename
EXPECTED_DOC_TYPES: dict[str, str] = {
    "صحيفة_دعوى.txt": "صحيفة دعوى",
    "تقرير_الخبير.txt": "تقرير خبير",
    "تقرير_الطب_الشرعي.txt": "تقرير خبير",
    "حكم_المحكمة.txt": "حكم",
    "محضر_جلسة_25_03_2024.txt": "محضر جلسة",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt": "مذكرة بدفاع",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt": "مذكرة بدفاع",
}


def _build_fixture_classifier():
    """Return a deterministic classifier stub for CASE_RAG fixture files."""
    current_file: dict[str, str | None] = {"path": None}

    def classify_document(_: str):
        file_path = current_file["path"]
        file_name = Path(file_path).name if file_path else ""
        doc_type = EXPECTED_DOC_TYPES.get(file_name, "مستند غير معروف")
        confidence = 100 if doc_type != "مستند غير معروف" else 0
        return {
            "final_type": doc_type,
            "confidence": confidence,
            "explanation": "CASE_RAG test fixture mapping",
        }

    def wrap_ingest_file(original_ingest_file):
        @wraps(original_ingest_file)
        def wrapped(self, file_path: str, *args, **kwargs):
            current_file["path"] = file_path
            try:
                return original_ingest_file(file_path, *args, **kwargs)
            finally:
                current_file["path"] = None

        return wrapped

    return classify_document, wrap_ingest_file


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_case_id() -> str:
    return TEST_CASE_ID


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    assert FIXTURE_DIR.exists(), f"Fixture directory missing: {FIXTURE_DIR}"
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def file_ingestor(fixture_dir):
    """Construct FileIngestor with production config and a test classifier."""
    from config import cfg
    from Supervisor.services.file_ingestor import FileIngestor

    ingestor = FileIngestor(
        mongo_uri=cfg.mongodb["uri"],
        mongo_db=cfg.mongodb["database"],
        mongo_collection=cfg.mongodb.get("collection", "Document Storage"),
        embedding_model=cfg.embedding.get("model", "BAAI/bge-m3"),
        qdrant_host=cfg.qdrant.get("host", "localhost"),
        qdrant_port=cfg.qdrant.get("port", 6333),
        qdrant_grpc_port=cfg.qdrant.get("grpc_port", 6334),
        qdrant_prefer_grpc=cfg.qdrant.get("prefer_grpc", True),
        qdrant_collection=cfg.qdrant.get("collection", "judicial_docs"),
    )

    classifier_stub, wrap_ingest_file = _build_fixture_classifier()
    ingestor._classifier = classifier_stub
    ingestor.ingest_file = MethodType(
        wrap_ingest_file(ingestor.ingest_file),
        ingestor,
    )

    return ingestor


@pytest.fixture(scope="session")
def ingestion_results(file_ingestor, fixture_dir, test_case_id):
    """Ingest all 7 fixture files. Hard-fail if any basic invariant is broken."""
    file_paths = [str(fixture_dir / fname) for fname in FIXTURE_FILES]
    results = file_ingestor.ingest_files(file_paths, case_id=test_case_id)

    # --- Hard-fail assertions (abort session on failure) ---
    assert len(results) == 7, (
        f"Expected 7 ingestion results, got {len(results)}. "
        f"Files ingested: {[r.get('file') for r in results]}"
    )
    for r in results:
        assert r.get("mongo_id") is not None, (
            f"mongo_id is None for file: {r.get('file')}"
        )
        assert r.get("title", "") != "", (
            f"Empty title for file: {r.get('file')}"
        )
        assert r.get("doc_type") != "مستند غير معروف", (
            f"Classifier returned unknown type for: {r.get('file')}"
        )

    return results


@pytest.fixture(scope="session")
def vectorstore_ready(file_ingestor, ingestion_results):
    """Inject the ingestor's vectorstore into the RAG infrastructure singleton."""
    from RAG.case_doc_rag.infrastructure import set_vectorstore
    set_vectorstore(file_ingestor.vectorstore)
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
