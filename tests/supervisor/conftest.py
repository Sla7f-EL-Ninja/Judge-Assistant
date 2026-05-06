"""
conftest.py — session-scoped real-stack fixtures for the Supervisor test suite.

All external services (MongoDB, Qdrant, Redis, MinIO) are real — no mocks.
Requires env vars: MONGO_URI, GOOGLE_API_KEY (others have defaults).
"""

import os
import uuid
from pathlib import Path
from typing import Generator

import pymongo
import pytest
from dotenv import load_dotenv
load_dotenv()


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.expensive unless RUN_EXPENSIVE=1."""
    if os.getenv("RUN_EXPENSIVE", "").strip() not in ("1", "true", "yes"):
        skip_expensive = pytest.mark.skip(reason="Set RUN_EXPENSIVE=1 to run LLM/DB tests")
        for item in items:
            if item.get_closest_marker("expensive"):
                item.add_marker(skip_expensive)

from tests.supervisor.helpers.env_check import (
    assert_env,
    get_mongo_uri,
    get_qdrant_host,
    get_qdrant_port,
    get_test_db_name,
    get_minio_endpoint,
    get_minio_access_key,
    get_minio_secret_key,
)
from tests.supervisor.helpers.db_seed import FIXTURE_FILES, seed_case_summary
from tests.supervisor.helpers.db_cleanup import cleanup_mongo, cleanup_qdrant

# ---------------------------------------------------------------------------
# Env validation — fail fast before any test runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def env_check():
    assert_env()


# ---------------------------------------------------------------------------
# Shared test identifiers
# ---------------------------------------------------------------------------

TEST_CASE_ID = "test-case-2847-2024"


@pytest.fixture(scope="session")
def test_case_id() -> str:
    return TEST_CASE_ID


@pytest.fixture
def fresh_case_id() -> str:
    return f"test-case-{uuid.uuid4()}"


@pytest.fixture
def fresh_correlation_id() -> str:
    return f"test-cid-{uuid.uuid4()}"


# ---------------------------------------------------------------------------
# MongoDB (pymongo — supervisor uses pymongo, not motor)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mongo_client() -> Generator[pymongo.MongoClient, None, None]:
    client = pymongo.MongoClient(get_mongo_uri(), serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        pytest.skip(f"MongoDB unreachable: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def test_db_name() -> str:
    return get_test_db_name()


@pytest.fixture(scope="session")
def mongo_db(mongo_client, test_db_name):
    return mongo_client[test_db_name]


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant_client not installed")
    host = get_qdrant_host()
    port = get_qdrant_port()
    client = QdrantClient(host=host, port=port, timeout=10)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable at {host}:{port}: {exc}")
    yield client


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minio_client():
    try:
        from minio import Minio
    except ImportError:
        pytest.skip("minio not installed")
    endpoint = get_minio_endpoint()
    client = Minio(
        endpoint,
        access_key=get_minio_access_key(),
        secret_key=get_minio_secret_key(),
        secure=False,
    )
    try:
        client.list_buckets()
    except Exception as exc:
        pytest.skip(f"MinIO unreachable at {endpoint}: {exc}")
    yield client


# ---------------------------------------------------------------------------
# Supervisor graph (singleton)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def supervisor_app():
    from Supervisor.graph import get_app
    return get_app()


# ---------------------------------------------------------------------------
# Case docs — seed once per session, cleanup after
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def seeded_case(mongo_client, test_db_name, qdrant_client):
    """Ingest all 6 fixture files into Qdrant + Mongo for test-case-2847-2024."""
    existing_files = [f for f in FIXTURE_FILES if f.exists()]
    if not existing_files:
        pytest.skip("Fixture files not found — cannot seed case docs")

    case_id = TEST_CASE_ID

    # Check if already seeded (avoid re-ingestion in same session)
    try:
        seed_case_summary(mongo_client, test_db_name, case_id)
    except Exception:
        pass  # summary may already exist

    try:
        from tests.supervisor.helpers.db_seed import seed_case_docs
        seed_case_docs(case_id, existing_files)
    except Exception as exc:
        pytest.skip(f"Case seeding failed: {exc}")

    yield case_id

    # Cleanup
    cleanup_mongo(mongo_client, test_db_name, case_id)
    if qdrant_client:
        cleanup_qdrant(qdrant_client, case_id)


# ---------------------------------------------------------------------------
# Per-test audit log cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_audit_logs(mongo_client, test_db_name):
    yield
    if not os.getenv("KEEP_EVIDENCE"):
        try:
            mongo_client[test_db_name]["audit_log"].delete_many(
                {"correlation_id": {"$regex": "^test-cid-"}}
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CIVIL_LAW_DOCS_PRESEEDED skip guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def judicial_docs_available(qdrant_client):
    """Skip civil_law_rag tests if civil_law_docs collection is empty."""
    try:
        info = qdrant_client.get_collection("civil_law_docs")
        if info.points_count == 0:
            pytest.skip("civil_law_docs collection is empty — set CIVIL_LAW_DOCS_PRESEEDED=1 after seeding")
    except Exception as exc:
        pytest.skip(f"civil_law_docs collection check failed: {exc}")
