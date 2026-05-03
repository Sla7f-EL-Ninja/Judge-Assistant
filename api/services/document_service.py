"""
document_service.py

Orchestrates document ingestion via DocumentProcessor.process_document.
"""

import asyncio
import logging
from typing import Any, Dict, List

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import FILES
from api.services.case_service import add_document_to_case

logger = logging.getLogger(__name__)


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

        disk_path: str = file_rec.get("disk_path", "")
        if not disk_path:
            errors.append({"file_id": file_id, "error": "File has no disk path"})
            continue

        try:
            result = await asyncio.to_thread(process_document, disk_path, case_id)

            classification = result.get("classification", {})
            doc_type = classification.get("final_type", "")

            ingested.append({
                "file_id": file_id,
                "doc_type": doc_type,
                "classification": classification,
                "status": "success",
            })

            from datetime import datetime, timezone

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
            errors.append({"file_id": file_id, "error": str(exc)})

    return {"ingested": ingested, "errors": errors}

async def list_documents(db, case_id: str) -> list:
    cursor = db.documents.find(
        {"case_id": case_id},
        {"_id": 1, "title": 1, "source_file": 1, "created_at": 1}
    ).sort("created_at", -1)
    docs = await cursor.to_list(length=100)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs