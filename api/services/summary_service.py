"""
summary_service.py

Read and write case summaries in MongoDB, and handle the execution
of the summarization pipeline.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import SUMMARIES, DOCUMENTS
from summarize.pipeline import run_summarization

logger = logging.getLogger("hakim.api.summaries")


async def get_summary(
    db: AsyncIOMotorDatabase, case_id: str
) -> Optional[dict]:
    """Retrieve the stored summary for a case, or None if not found[cite: 14]."""
    return await db[SUMMARIES].find_one({"case_id": case_id})


async def get_case_brief(
    db: AsyncIOMotorDatabase, case_id: str
) -> Optional[dict]:
    """Return {case_id, case_brief, generated_at} or None if no summary exists[cite: 14]."""
    doc = await db[SUMMARIES].find_one(
        {"case_id": case_id},
        {"_id": 0, "case_id": 1, "case_brief": 1, "generated_at": 1},
    )
    return doc


async def save_summary(
    db: AsyncIOMotorDatabase,
    case_id: str,
    rendered_brief: str,
    all_sources: list,
    case_brief: Optional[dict] = None,
) -> None:
    """
    Upsert the generated summary into MongoDB, linked to case_id[cite: 14].
    Overwrites existing summaries for the same case[cite: 14].
    """
    doc: dict = {
        "case_id": case_id,
        "summary": rendered_brief,
        "sources": all_sources,
        "generated_at": datetime.now(timezone.utc),
    }
    if case_brief:
        doc["case_brief"] = case_brief

    await db[SUMMARIES].update_one(
        {"case_id": case_id},
        {"$set": doc},
        upsert=True,
    )


async def generate_summary(
    db: AsyncIOMotorDatabase, 
    case_id: str
) -> dict:
    """
    Standalone orchestrator to fetch documents and run the summarization pipeline.
    """
    logger.info("generate_summary: fetching documents for case_id='%s'", case_id)

    # 1. Load raw documents from the database
    raw_docs = await db[DOCUMENTS].find(
        {"case_id": case_id}, {"_id": 1, "text": 1}
    ).to_list(length=None)

    if not raw_docs:
        raise ValueError(f"No documents found for case: {case_id}")

    # 2. Map 'text' to 'raw_text' to satisfy the pipeline's validation
    # This prevents the 'all documents have empty raw_text' 422 error.
    documents = [
        {"doc_id": str(d["_id"]), "raw_text": d.get("text", "").strip()} 
        for d in raw_docs
    ]

    # Double check that we still have content after stripping[cite: 11]
    if not any(d["raw_text"] for d in documents):
        raise ValueError(f"All documents for case {case_id} contain no readable text.")

    # 3. Run the sync pipeline in a thread to keep the API responsive[cite: 12]
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_summarization(
                documents=documents, 
                case_id=case_id, 
                save_to_db=False  # We use the service's save_summary for consistency
            ),
        )
    except Exception as exc:
        logger.exception("Summarization pipeline failed for case_id='%s'", case_id)
        raise exc

    # 4. Persist the result to the 'summaries' collection[cite: 13, 14]
    await save_summary(
        db=db,
        case_id=case_id,
        rendered_brief=result.rendered_brief,
        all_sources=result.all_sources,
        case_brief=result.case_brief,
    )

    return {
        "case_id": case_id,
        "sources_count": len(result.all_sources),
        "message": "Summary generated and saved successfully."
    }