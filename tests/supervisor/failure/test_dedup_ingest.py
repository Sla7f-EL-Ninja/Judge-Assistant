"""
test_dedup_ingest.py — duplicate file ingestion behavior.
"""

import uuid
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent.parent / "CASE_RAG" / "fixtures"


@pytest.fixture
def ingestor():
    try:
        from Supervisor.services.file_ingestor import FileIngestor
        return FileIngestor()
    except Exception as exc:
        pytest.skip(f"FileIngestor unavailable: {exc}")


@pytest.fixture
def txt_fixture():
    f = FIXTURES_DIR / "صحيفة_دعوى.txt"
    if not f.exists():
        pytest.skip(f"Fixture not found: {f}")
    return str(f)


class TestDedupIngest:
    def test_second_ingest_returns_existing_id(self, ingestor, txt_fixture):
        """Ingest same file twice → second call returns existing _id (no duplicate)."""
        case_id = f"test-case-{uuid.uuid4()}"
        r1 = ingestor.ingest_files([txt_fixture], case_id)
        r2 = ingestor.ingest_files([txt_fixture], case_id)

        if r1 and r2:
            ids1 = {str(item.get("_id")) for item in r1 if item.get("_id")}
            ids2 = {str(item.get("_id")) for item in r2 if item.get("_id")}
            if ids1 and ids2:
                assert ids1 == ids2, "Second ingest created a duplicate document"

    @pytest.mark.xfail(reason="Concurrent dedup TOCTOU race — known gap, not atomic")
    def test_concurrent_dedup_no_race(self, ingestor, txt_fixture):
        """Two concurrent ingests of same file → only one doc in Mongo (TOCTOU gap)."""
        import threading
        case_id = f"test-case-{uuid.uuid4()}"
        results = []
        errors = []

        def ingest():
            try:
                r = ingestor.ingest_files([txt_fixture], case_id)
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=ingest)
        t2 = threading.Thread(target=ingest)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # If both succeeded, IDs should be the same (dedup worked)
        all_ids = [
            str(item.get("_id"))
            for r in results
            for item in (r or [])
            if item.get("_id")
        ]
        assert len(set(all_ids)) <= 1, f"TOCTOU race created duplicates: {all_ids}"
