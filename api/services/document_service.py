"""
document_service.py

Orchestrates document ingestion by wrapping the existing
``Supervisor.services.file_ingestor.FileIngestor``.

Because ``FileIngestor`` uses synchronous pymongo and blocking I/O,
calls are wrapped with ``asyncio.to_thread`` to avoid blocking the
event loop.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import Settings
from api.db.collections import FILES
from api.services.case_service import add_document_to_case

logger = logging.getLogger(__name__)


def _get_ingestor(settings: Settings):
    """Lazily import and construct the synchronous FileIngestor."""
    from Supervisor.services.file_ingestor import FileIngestor

    return FileIngestor(
        mongo_uri=settings.mongo_uri,
        mongo_db=settings.mongo_db,
        mongo_collection=settings.mongo_collection,
        embedding_model=settings.embedding_model,
        chroma_collection=settings.chroma_collection,
        chroma_persist_dir=settings.chroma_persist_dir,
    )


async def ingest_files(
    db: AsyncIOMotorDatabase,
    settings: Settings,
    case_id: str,
    file_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Ingest the given files into a case.

    Returns ``{"ingested": [...], "errors": [...]}``.
    """
    ingested: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Resolve file records
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
            ingestor = _get_ingestor(settings)
            # Run blocking ingestor in a thread
            result = await asyncio.to_thread(
                ingestor.ingest_file, disk_path, case_id
            )

            classification = ""
            doc_type = ""
            if isinstance(result, dict):
                classification = result.get("classification", "")
                doc_type = result.get("doc_type", "")

            ingested.append(
                {
                    "file_id": file_id,
                    "doc_type": doc_type,
                    "classification": classification,
                    "status": "success",
                }
            )

            # Add document reference to the case
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
