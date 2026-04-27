"""db_seed.py — seeds Mongo + Qdrant per scenario."""

import os
from pathlib import Path
from typing import List, Optional
from config.supervisor import QDRANT_COLLECTION_CASE

FIXTURES_DIR = Path(__file__).parent.parent.parent / "CASE_RAG" / "fixtures"

FIXTURE_FILES = [
    FIXTURES_DIR / "صحيفة_دعوى.txt",
    FIXTURES_DIR / "محضر_جلسة_25_03_2024.txt",
    FIXTURES_DIR / "تقرير_الخبير.txt",
    FIXTURES_DIR / "تقرير_الطب_الشرعي.txt",
    FIXTURES_DIR / "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
    FIXTURES_DIR / "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
]


def seed_case_docs(case_id: str, files: Optional[List[Path]] = None) -> None:
    """Ingest fixture files into Mongo + Qdrant for a given case_id."""
    from Supervisor.services.file_ingestor import FileIngestor

    ingestor = FileIngestor(qdrant_collection=QDRANT_COLLECTION_CASE)
    paths = [str(f) for f in (files or FIXTURE_FILES) if f.exists()]
    if not paths:
        raise FileNotFoundError(f"No fixture files found under {FIXTURES_DIR}")
    ingestor.ingest_files(paths, case_id)


def seed_case_summary(mongo_client, db_name: str, case_id: str) -> None:
    """Insert a canned case summary into Mongo for enrich_context tests."""
    db = mongo_client[db_name]
    from datetime import datetime, timezone

    db["summaries"].insert_one({
        "case_id": case_id,
        "summary": (
            "قضية رقم 2847/2024 — دعوى فسخ عقد بيع وتعويض بمبلغ 2,000,000 جنيه مصري. "
            "المدعي: أحمد محمد عبد الله. المدعى عليهم: محمود سعيد إبراهيم (أول) وشركة العقارات الحديثة (ثانية)."
        ),
        "generated_at": datetime.now(timezone.utc),
    })
 