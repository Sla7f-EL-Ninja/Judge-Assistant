"""
tests/chat_reasoner/conftest.py

Session-scoped fixtures: preflight connectivity checks, document seeding
for case_id="1234" (which already has a summary in MongoDB), and cleanup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("langsmith").setLevel(logging.INFO)

CASE_ID = "1234"

FIXTURE_DIR = Path(__file__).parent.parent / "CASE_RAG" / "fixtures"

FIXTURE_FILES = [
    "صحيفة_دعوى.txt",
    "تقرير_الخبير.txt",
    "تقرير_الطب_الشرعي.txt",
    "محضر_جلسة_25_03_2024.txt",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
]

EXPECTED_DOC_TYPES: dict[str, str] = {
    "صحيفة_دعوى.txt": "صحيفة دعوى",
    "تقرير_الخبير.txt": "تقرير خبير",
    "تقرير_الطب_الشرعي.txt": "تقرير خبير",
    "محضر_جلسة_25_03_2024.txt": "محضر جلسة",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt": "مذكرة بدفاع",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt": "مذكرة بدفاع",
}


def _build_fixture_classifier():
    """Deterministic doc-type classifier for the 7 fixture files."""
    from functools import wraps
    from types import MethodType

    current_file: dict[str, str | None] = {"path": None}

    def classify_document(_: str):
        name = Path(current_file["path"]).name if current_file["path"] else ""
        doc_type = EXPECTED_DOC_TYPES.get(name, "مستند غير معروف")
        return {
            "final_type": doc_type,
            "confidence": 100 if doc_type != "مستند غير معروف" else 0,
            "explanation": "chat_reasoner test fixture",
        }

    def wrap_ingest_file(original):
        @wraps(original)
        def wrapped(self, file_path: str, *args, **kwargs):
            current_file["path"] = file_path
            try:
                return original(file_path, *args, **kwargs)
            finally:
                current_file["path"] = None
        return wrapped

    return classify_document, wrap_ingest_file, MethodType


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _civil_law_index():
    """Ensure civil_law_docs collection is populated before any test runs."""
    from RAG.civil_law_rag.indexing.indexer import ensure_civil_law_indexed
    ensure_civil_law_indexed()


@pytest.fixture(scope="session", autouse=True)
def _preflight():
    """Verify all external dependencies before any test runs."""
    try:
        from config.supervisor import MONGO_URI, MONGO_DB
        from config import get_llm
        from qdrant_client import QdrantClient
        from config import cfg
    except ImportError as exc:
        pytest.skip(f"Config import failed: {exc}")

    # MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client[MONGO_DB]
        summary = db["summaries"].find_one({"case_id": CASE_ID})
        client.close()
        if summary is None:
            pytest.skip(
                f"Pre-seeded summary not found in {MONGO_DB}.summaries for case_id={CASE_ID!r}. "
                "fetch_summary_report tests require it."
            )
    except Exception as exc:
        pytest.skip(f"MongoDB unreachable: {exc}")

    # Qdrant
    try:
        qc = QdrantClient(
            host=cfg.qdrant.get("host", "localhost"),
            port=cfg.qdrant.get("port", 6333),
            timeout=3,
        )
        qc.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable: {exc}")

    # LLM
    try:
        get_llm("low").invoke("ping")
    except Exception as exc:
        pytest.skip(f"LLM unavailable: {exc}")

    yield


# ---------------------------------------------------------------------------
# Document seeding
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def seeded_case():
    """Ingest 7 Arabic fixture docs under CASE_ID into Mongo + Qdrant.

    Yields {"case_id": CASE_ID, "mongo_ids": [str, ...], "ingestor": FileIngestor}.
    Teardown deletes only the seeded docs (by mongo_id), leaving the pre-existing
    summary and any other data for CASE_ID untouched.
    """
    assert FIXTURE_DIR.exists(), f"Fixture dir missing: {FIXTURE_DIR}"

    from config import cfg
    from Supervisor.services.file_ingestor import FileIngestor
    from types import MethodType

    classifier_stub, wrap_ingest_file, MT = _build_fixture_classifier()

    try:
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
    except Exception as exc:
        pytest.skip(f"FileIngestor construction failed: {exc}")

    ingestor._classifier = classifier_stub
    ingestor.ingest_file = MT(wrap_ingest_file(ingestor.ingest_file), ingestor)

    file_paths = [str(FIXTURE_DIR / f) for f in FIXTURE_FILES]
    try:
        results = ingestor.ingest_files(file_paths, case_id=CASE_ID)
    except Exception as exc:
        pytest.skip(f"Document ingestion failed: {exc}")

    if len(results) != len(FIXTURE_FILES):
        pytest.skip(f"Expected {len(FIXTURE_FILES)} ingestion results, got {len(results)}")

    mongo_ids = []
    for r in results:
        mid = r.get("mongo_id")
        if mid is None:
            pytest.skip(f"mongo_id is None for file: {r.get('file')}")
        mongo_ids.append(str(mid))

    yield {"case_id": CASE_ID, "mongo_ids": mongo_ids, "ingestor": ingestor}

    # --- Teardown: delete only the seeded documents ---
    logger.info("chat_reasoner seeded_case teardown: removing %d docs", len(mongo_ids))

    # MongoDB
    try:
        from bson import ObjectId
        ids_as_oids = [ObjectId(mid) for mid in mongo_ids]
        deleted = ingestor.mongo_collection.delete_many({"_id": {"$in": ids_as_oids}})
        logger.info("MongoDB: deleted %d seeded doc(s)", deleted.deleted_count)
    except Exception as exc:
        logger.warning("MongoDB seeded doc cleanup failed: %s", exc)

    # Qdrant — filter by case_id AND doc_id to avoid touching other data
    try:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny, MatchValue

        qdrant_client = ingestor.vectorstore.client
        qdrant_client.delete(
            collection_name=ingestor._qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.case_id",
                            match=MatchValue(value=CASE_ID),
                        ),
                        FieldCondition(
                            key="metadata.doc_id",
                            match=MatchAny(any=mongo_ids),
                        ),
                    ]
                )
            ),
        )
        logger.info("Qdrant: deleted vectors for seeded mongo_ids")
    except Exception as exc:
        logger.warning("Qdrant seeded doc cleanup failed: %s", exc)
