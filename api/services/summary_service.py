"""
summary_service.py

Read and write case summaries in MongoDB.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import SUMMARIES


async def get_summary(
    db: AsyncIOMotorDatabase, case_id: str
) -> Optional[dict]:
    """Retrieve the stored summary for a case, or None if not found."""
    return await db[SUMMARIES].find_one({"case_id": case_id})


async def save_summary(
    db: AsyncIOMotorDatabase,
    case_id: str,
    rendered_brief: str,
    all_sources: list,
    case_brief: Optional[dict] = None,
) -> None:
    """Upsert the generated summary into MongoDB, linked to case_id.

    If a summary already exists for this case it is overwritten, so
    re-running the pipeline always reflects the latest result.

    Args:
        db:             Motor async database handle.
        case_id:        The case this summary belongs to.
        rendered_brief: Full Arabic-markdown brief produced by Node 5.
        all_sources:    Unique citation strings collected across the brief.
        case_brief:     Optional structured CaseBrief dict (7 sections).
                        Stored as-is for programmatic access if provided.
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
