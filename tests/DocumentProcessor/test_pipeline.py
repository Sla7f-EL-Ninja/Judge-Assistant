"""
Standalone verification script for DocumentProcessor.process_document.

Run:
    python tests/DocumentProcessor/test_pipeline.py

Fill in IMAGE_FILE and TXT_FILE before running.
"""

import sys
import uuid
from pathlib import Path

# Ensure repo root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DocumentProcessor import process_document
from config.supervisor import (
    MONGO_URI, MONGO_DB, MONGO_COLLECTION,
    QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT, QDRANT_PREFER_GRPC,
    QDRANT_COLLECTION_CASE,
)

# ── Fill these in before running ──────────────────────────────────────────────
IMAGE_FILE = r"D:\FUCK!!\Grad\TESTING DATA\WhatsApp Image 2026-02-15 at 7.14.30 PM.jpeg"   # e.g. r"D:\samples\scan.jpg"
TXT_FILE   = r"D:\FUCK!!\Grad\Code\tests\CASE_RAG\fixtures\صحيفة_دعوى.txt"  # e.g. r"D:\samples\contract.txt"
# ─────────────────────────────────────────────────────────────────────────────


def _verify_mongo(case_id: str) -> bool:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]
    docs = list(col.find({"case_id": case_id}))
    print(f"  Mongo docs found: {len(docs)}")
    for d in docs:
        print(f"    _id={d['_id']}  type={d.get('doc_type')}  file={d.get('source_file')}")
    return len(docs) >= 1


def _verify_qdrant(case_id: str) -> bool:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )
    results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION_CASE,
        scroll_filter=Filter(
            must=[FieldCondition(key="metadata.case_id", match=MatchValue(value=case_id))]
        ),
        limit=200,
    )
    print(f"  Qdrant vectors found: {len(results)}")
    return len(results) >= 1


def _run_test(label: str, file_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"File: {file_path}")

    if not file_path:
        print("  SKIP — file path not set")
        return

    case_id = f"test_doc_processor_{uuid.uuid4().hex[:8]}"
    print(f"Case ID: {case_id}")

    result = process_document(file_path, case_id=case_id)
    print("\nReturned dict:")
    print(f"  file_type:      {result['file_type']}")
    print(f"  text length:    {len(result['text'])} chars")
    print(f"  classification: {result['classification']}")
    print(f"  metadata:       {result['metadata']}")

    print("\nMongo check:")
    mongo_ok = False
    try:
        mongo_ok = _verify_mongo(case_id)
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\nQdrant check:")
    qdrant_ok = False
    try:
        qdrant_ok = _verify_qdrant(case_id)
    except Exception as exc:
        print(f"  ERROR: {exc}")

    mongo_status = "PASS" if mongo_ok else "FAIL"
    qdrant_status = "PASS" if qdrant_ok else "FAIL"
    print(f"\n  Mongo:  {mongo_status}")
    print(f"  Qdrant: {qdrant_status}")


if __name__ == "__main__":
    _run_test("Image file (OCR path)", IMAGE_FILE)
    _run_test("Text file (direct path)", TXT_FILE)
