"""
summary_service.py

Read stored summaries from MongoDB.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import SUMMARIES


async def get_summary(
    db: AsyncIOMotorDatabase, case_id: str
) -> Optional[dict]:
    """Retrieve the stored summary for a case, or None if not found."""
    return await db[SUMMARIES].find_one({"case_id": case_id})
