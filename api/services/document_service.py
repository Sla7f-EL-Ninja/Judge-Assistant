"""
document_service.py

Orchestrates document ingestion via DocumentProcessor.process_document.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import FILES, DOCUMENTS
from api.services.case_service import add_document_to_case

logger = logging.getLogger(__name__)


def _str_id(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


async def ingest_files(
    db: AsyncIOMotorDatabase,
    settings,
    case_id: str,
    file_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Ingest the given files into a case.

    Returns ``{"ingested": [...], "errors": [...]}``.
    """
    from DocumentProcessor import process_document

    ingested: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for file_id in file_ids:
        file_rec = await db[FILES].find_one({"_id": file_id})
        if file_rec is None:
            errors.append({"file_id": file_id, "error": "File not found"})
            continue

        # -- Resolve file path: MinIO first, local disk fallback -------------
        resolved_path: str = ""
        tmp_file = None

        if file_rec.get("storage_backend") == "minio" and file_rec.get("minio_object"):
            try:
                from api.db.minio_client import get_minio, get_bucket

                minio_client = get_minio()
                if minio_client:
                    bucket = get_bucket()
                    ext = os.path.splitext(file_rec.get("filename", ""))[1]
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tmp_file.close()  # ← must close before fget_object on Windows
                    await asyncio.to_thread(
                        minio_client.fget_object, bucket, file_rec["minio_object"], tmp_file.name
                    )
                    resolved_path = tmp_file.name
                    logger.info("Resolved file %s from MinIO -> %s", file_id, resolved_path)
            except Exception as exc:
                logger.warning("MinIO download failed for %s: %s — trying local disk", file_id, exc)

        if not resolved_path:
            resolved_path = file_rec.get("disk_path", "")

        if not resolved_path:
            errors.append({"file_id": file_id, "error": "File not found in MinIO or local disk"})
            continue

        try:
            result = await asyncio.to_thread(
                process_document, resolved_path, case_id, file_id
            )

            classification = result.get("classification", {})
            doc_type = classification.get("final_type", "")

            ingested.append({
                "file_id": file_id,
                "doc_type": doc_type,
                "classification": classification,
                "status": "success",
            })

            await add_document_to_case(
                db,
                case_id,
                {
                    "file_id": file_id,
                    "filename": file_rec.get("filename", ""),
                    "classification": classification,
                    "ingested_at": datetime.now(timezone.utc),
                },
            )

        except Exception as exc:
            logger.exception("Ingestion failed for file %s: %s", file_id, exc)
            errors.append({"file_id": file_id, "error": str(exc), "status": "failed"})

        finally:
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass

    return {"ingested": ingested, "errors": errors}


async def list_documents(db: AsyncIOMotorDatabase, case_id: str) -> list:
    cursor = db[DOCUMENTS].find(
        {"case_id": case_id},
        {
            "_id": 1, "title": 1, "source_file": 1, "created_at": 1,
            "doc_type": 1, "file_type": 1, "file_id": 1,
        }
    ).sort("created_at", -1)
    docs = await cursor.to_list(length=200)
    result = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        result.append(d)
    return result


async def get_document(
    db: AsyncIOMotorDatabase, case_id: str, doc_id: str
) -> Optional[dict]:
    """Fetch a single document by _id, scoped to case_id."""
    try:
        oid = ObjectId(doc_id)
    except Exception:
        oid = doc_id
    doc = await db[DOCUMENTS].find_one({"_id": oid, "case_id": case_id})
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def get_document_ocr(
    db: AsyncIOMotorDatabase, case_id: str, identifier: str
) -> Optional[dict]:
    """Resolve by Mongo _id OR file_id, scoped to case_id."""
    doc = None
    try:
        oid = ObjectId(identifier)
        doc = await db[DOCUMENTS].find_one({"_id": oid, "case_id": case_id})
    except Exception:
        pass
    if doc is None:
        doc = await db[DOCUMENTS].find_one({"file_id": identifier, "case_id": case_id})
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def correct_document_ocr(
    db: AsyncIOMotorDatabase,
    case_id: str,
    identifier: str,
    new_text: str,
    corrected_by: Optional[str] = None,
) -> Optional[dict]:
    """Update OCR text, re-index in Qdrant. Returns updated doc or None if not found."""
    from DocumentProcessor import reindex_document

    doc = await get_document_ocr(db, case_id, identifier)
    if doc is None:
        return None

    mongo_id = doc["id"]
    now = datetime.now(timezone.utc)

    update: Dict[str, Any] = {
        "text": new_text,
        "corrected": True,
        "corrected_at": now,
    }
    if corrected_by:
        update["corrected_by"] = corrected_by
    if not doc.get("original_text"):
        update["original_text"] = doc.get("text", "")

    try:
        oid = ObjectId(mongo_id)
    except Exception:
        oid = mongo_id

    await db[DOCUMENTS].update_one({"_id": oid}, {"$set": update})

    doc_meta = {
        "title": doc.get("title", ""),
        "doc_type": doc.get("doc_type", ""),
        "case_id": case_id,
        "source_file": doc.get("source_file", ""),
        "file_id": doc.get("file_id"),
    }
    try:
        await asyncio.to_thread(reindex_document, mongo_id, new_text, doc_meta)
    except Exception as exc:
        logger.exception("Qdrant reindex failed for doc %s: %s", mongo_id, exc)
        raise RuntimeError(f"QDRANT_REINDEX_FAILED: {exc}") from exc

    doc.update(update)
    return doc


async def delete_document(
    db: AsyncIOMotorDatabase,
    case_id: str,
    doc_id: str,
    qdrant_client=None,
    minio_client=None,
) -> None:
    """Delete document from Mongo, Qdrant chunks, and MinIO object."""
    from DocumentProcessor.pipeline import _delete_qdrant_chunks_by_mongo_id

    try:
        oid = ObjectId(doc_id)
    except Exception:
        oid = doc_id

    doc = await db[DOCUMENTS].find_one({"_id": oid, "case_id": case_id})
    if doc is None:
        return None

    # Delete Qdrant chunks
    try:
        await asyncio.to_thread(_delete_qdrant_chunks_by_mongo_id, str(doc["_id"]))
    except Exception as exc:
        logger.warning("Qdrant delete failed for %s: %s", doc_id, exc)

    # Delete MinIO object
    minio_object = doc.get("minio_object")
    if minio_object and doc.get("storage_backend") == "minio":
        try:
            from api.db.minio_client import get_minio, get_bucket
            minio_cli = get_minio()
            if minio_cli:
                bucket = get_bucket()
                await asyncio.to_thread(minio_cli.remove_object, bucket, minio_object)
        except Exception as exc:
            logger.warning("MinIO delete failed for %s: %s", minio_object, exc)

    await db[DOCUMENTS].delete_one({"_id": oid})