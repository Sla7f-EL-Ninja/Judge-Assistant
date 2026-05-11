"""
document_service.py

Orchestrates document ingestion via DocumentProcessor.process_document.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import FILES, DOCUMENTS
from api.services.case_service import add_document_to_case

logger = logging.getLogger(__name__)


def _str_id(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


async def _resolve_file_path(
    db: AsyncIOMotorDatabase, file_id: str
) -> tuple:
    """Return (file_rec, resolved_path, tmp_file_or_None).

    Caller must delete tmp_file.name when done (if tmp_file is not None).
    Returns (None, "", None) when file not found in DB.
    Returns (file_rec, "", None) when file exists in DB but not on disk/MinIO.
    """
    file_rec = await db[FILES].find_one({"_id": file_id})
    if file_rec is None:
        return None, "", None

    resolved_path = ""
    tmp_file = None

    if file_rec.get("storage_backend") == "minio" and file_rec.get("minio_object"):
        try:
            from api.db.minio_client import get_minio, get_bucket

            minio_client = get_minio()
            if minio_client:
                bucket = get_bucket()
                ext = os.path.splitext(file_rec.get("filename", ""))[1]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp_file.close()  # must close before fget_object on Windows
                await asyncio.to_thread(
                    minio_client.fget_object, bucket, file_rec["minio_object"], tmp_file.name
                )
                resolved_path = tmp_file.name
                logger.info("Resolved file %s from MinIO -> %s", file_id, resolved_path)
        except Exception as exc:
            logger.warning("MinIO download failed for %s: %s — trying local disk", file_id, exc)
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass
                tmp_file = None

    if not resolved_path:
        resolved_path = file_rec.get("disk_path", "")

    return file_rec, resolved_path, tmp_file


async def ingest_files(
    db: AsyncIOMotorDatabase,
    settings,
    case_id: str,
    file_ids: Optional[List[str]] = None,
    groups=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Ingest files into a case, returning ``{"ingested": [...], "errors": [...]}``.

    Accepts either ``file_ids`` (legacy: one doc per file) or ``groups``
    (list of IngestGroup: one doc per group). The caller normalises via
    ``IngestRequest.resolved_groups``.
    """
    from DocumentProcessor import process_document
    from DocumentProcessor.pipeline import process_document_group

    # Normalise to groups
    if groups is not None:
        resolved_groups = groups
    elif file_ids is not None:
        class _G:
            def __init__(self, fids):
                self.file_ids = fids
        resolved_groups = [_G([fid]) for fid in file_ids]
    else:
        return {"ingested": [], "errors": []}

    ingested: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for group in resolved_groups:
        group_file_ids: List[str] = list(group.file_ids)
        tmp_files = []
        resolved_paths: List[str] = []
        file_recs: List[dict] = []
        group_error = None

        try:
            for fid in group_file_ids:
                file_rec, path, tmp_file = await _resolve_file_path(db, fid)
                if tmp_file is not None:
                    tmp_files.append(tmp_file)
                if file_rec is None:
                    group_error = f"File not found: {fid}"
                    break
                if not path:
                    group_error = f"File not found in MinIO or local disk: {fid}"
                    break
                file_recs.append(file_rec)
                resolved_paths.append(path)

            if group_error:
                errors.append({
                    "file_ids": group_file_ids,
                    "file_id": group_file_ids[0] if group_file_ids else None,
                    "error": group_error,
                    "status": "failed",
                })
                continue

            if len(group_file_ids) == 1:
                result = await asyncio.to_thread(
                    process_document, resolved_paths[0], case_id, group_file_ids[0]
                )
            else:
                result = await asyncio.to_thread(
                    process_document_group, resolved_paths, case_id, group_file_ids
                )

            classification = result.get("classification", {})
            doc_type = classification.get("final_type", "")
            metadata = result.get("metadata", {})
            doc_id = metadata.get("mongo_id")

            ingested.append({
                "file_ids": group_file_ids,
                "file_id": group_file_ids[0],
                "doc_id": doc_id,
                "doc_type": doc_type,
                "classification": classification,
                "status": "success",
            })

            await add_document_to_case(
                db,
                case_id,
                {
                    "file_ids": group_file_ids,
                    "file_id": group_file_ids[0],
                    "filenames": [r.get("filename", "") for r in file_recs],
                    "classification": classification,
                    "ingested_at": datetime.now(timezone.utc),
                },
            )

        except Exception as exc:
            logger.exception("Ingestion failed for group %s: %s", group_file_ids, exc)
            errors.append({
                "file_ids": group_file_ids,
                "file_id": group_file_ids[0] if group_file_ids else None,
                "error": str(exc),
                "status": "failed",
            })

        finally:
            for tmp_file in tmp_files:
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass

    return {"ingested": ingested, "errors": errors}


async def list_documents(db: AsyncIOMotorDatabase, case_id: str) -> list:
    cursor = db[DOCUMENTS].find(
        {"case_id": case_id},
        {
            "_id": 1, "title": 1, "source_file": 1, "source_files": 1,
            "created_at": 1, "doc_type": 1, "file_type": 1,
            "file_id": 1, "file_ids": 1,
        }
    ).sort("created_at", -1)
    docs = await cursor.to_list(length=200)
    result = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        # Backfill list fields for docs ingested before this change
        if "file_ids" not in d:
            d["file_ids"] = [d["file_id"]] if d.get("file_id") else []
        if "source_files" not in d:
            d["source_files"] = [d["source_file"]] if d.get("source_file") else []
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


async def bulk_correct_document_ocr(
    db: AsyncIOMotorDatabase,
    case_id: str,
    items: List[Dict[str, Any]],
    default_corrected_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Apply correct_document_ocr to each item sequentially with per-item error isolation.

    Each item dict must have keys: doc_id, text, corrected_by (optional).
    Returns list of result dicts: {doc_id, status, result, error}.

    Note: if Mongo update succeeds but Qdrant reindex fails, the doc text is
    already updated in Mongo — no rollback occurs (same as single PATCH semantics).
    """
    from api.errors import DOCUMENT_NOT_FOUND, QDRANT_REINDEX_FAILED, INTERNAL_ERROR

    results = []
    for item in items:
        doc_id = item["doc_id"]
        text = item["text"]
        corrected_by = item.get("corrected_by") or default_corrected_by

        try:
            doc = await correct_document_ocr(db, case_id, doc_id, text, corrected_by)
        except RuntimeError as exc:
            code = QDRANT_REINDEX_FAILED if "QDRANT_REINDEX_FAILED" in str(exc) else INTERNAL_ERROR
            logger.exception("bulk OCR correction failed for doc %s in case %s", doc_id, case_id)
            results.append({
                "doc_id": doc_id,
                "status": "failed",
                "result": None,
                "error": {"code": code, "message": str(exc)},
            })
            continue
        except Exception as exc:
            logger.exception("bulk OCR correction failed for doc %s in case %s", doc_id, case_id)
            results.append({
                "doc_id": doc_id,
                "status": "failed",
                "result": None,
                "error": {"code": INTERNAL_ERROR, "message": str(exc)},
            })
            continue

        if doc is None:
            results.append({
                "doc_id": doc_id,
                "status": "failed",
                "result": None,
                "error": {"code": DOCUMENT_NOT_FOUND, "message": f"Document not found: {doc_id}"},
            })
            continue

        results.append({"doc_id": doc_id, "status": "success", "result": doc, "error": None})

    return results


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