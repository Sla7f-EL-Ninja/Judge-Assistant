"""
report_service.py

Async report pipeline: run summarizer then Case Reasoner, persist both,
track job status in report_jobs collection.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import DOCUMENTS, REPORT_JOBS
from api.services.summary_service import save_summary
from CR.pipeline import run_case_reasoning

logger = logging.getLogger("hakim.api.reports")


async def create_report_job(db: AsyncIOMotorDatabase, case_id: str) -> str:
    """Insert a queued report_jobs row; return the new job_id string."""
    doc = {
        "case_id": case_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "completed_at": None,
        "error": None,
    }
    result = await db[REPORT_JOBS].insert_one(doc)
    return str(result.inserted_id)


async def get_report_job(
    db: AsyncIOMotorDatabase, case_id: str, report_id: str
) -> Optional[dict]:
    try:
        oid = ObjectId(report_id)
    except Exception:
        return None
    doc = await db[REPORT_JOBS].find_one({"_id": oid, "case_id": case_id})
    if doc:
        doc["report_id"] = str(doc.pop("_id"))
    return doc


async def list_report_jobs(
    db: AsyncIOMotorDatabase, case_id: str, skip: int, limit: int
) -> tuple[list, int]:
    cursor = db[REPORT_JOBS].find({"case_id": case_id}).sort("created_at", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=None)
    total = await db[REPORT_JOBS].count_documents({"case_id": case_id})
    for d in docs:
        d["report_id"] = str(d.pop("_id"))
    return docs, total


async def run_report_pipeline(
    db: AsyncIOMotorDatabase,
    case_id: str,
    report_id: str,
) -> None:
    """Background task: summarize → CR → persist → update job row."""
    try:
        oid = ObjectId(report_id)
    except Exception:
        logger.error("run_report_pipeline: invalid report_id=%s", report_id)
        return

    async def _set_status(status: str, error: Optional[str] = None, completed: bool = False):
        update: dict = {"status": status}
        if error is not None:
            update["error"] = error
        if completed:
            update["completed_at"] = datetime.now(timezone.utc)
        await db[REPORT_JOBS].update_one({"_id": oid}, {"$set": update})

    await _set_status("running")

    # 1. Load documents
    raw_docs = await db[DOCUMENTS].find(
        {"case_id": case_id}, {"_id": 1, "text": 1}
    ).to_list(length=None)

    if not raw_docs:
        await _set_status("failed", error="No documents found for case", completed=True)
        return

    documents = [{"doc_id": str(d["_id"]), "raw_text": d.get("text", "")} for d in raw_docs]

    # 2. Run summarization (sync, blocking — offload to thread)
    try:
        from summarize.pipeline import run_summarization
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_summarization(documents=documents, case_id=case_id, save_to_db=False),
        )
    except Exception as exc:
        logger.exception("Summarization failed  case_id=%s", case_id)
        await _set_status("failed", error=f"Summarization failed: {exc}", completed=True)
        return

    # 3. Persist summary
    try:
        await save_summary(
            db=db,
            case_id=case_id,
            rendered_brief=result.rendered_brief,
            all_sources=result.all_sources,
            case_brief=result.case_brief,
        )
    except Exception as exc:
        logger.exception("Failed to save summary  case_id=%s", case_id)
        await _set_status("failed", error=f"Failed to save summary: {exc}", completed=True)
        return

    # 4 & 5. Run Case Reasoner and Persist
    try:
        
        cr_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_case_reasoning(
                case_id=case_id,
                case_brief=result.case_brief or {},
                rendered_brief=result.rendered_brief or "",
                save_to_db=True # Handles its own persistence via PyMongo
            ),
        )
    except Exception as exc:
        logger.exception("Case Reasoner failed  case_id=%s", case_id)
        await _set_status("failed", error=f"Case Reasoner failed: {exc}", completed=True)
        return

    await _set_status("completed", completed=True)
    logger.info("Report pipeline complete  case_id=%s  report_id=%s", case_id, report_id)

    await _set_status("completed", completed=True)
    logger.info("Report pipeline complete  case_id=%s  report_id=%s", case_id, report_id)
