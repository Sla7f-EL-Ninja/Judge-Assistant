"""
case_reasoning_service.py

Persist and retrieve Case Reasoner results in the case_reasonings collection.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import CASE_REASONINGS


async def get_case_reasoning(
    db: AsyncIOMotorDatabase, case_id: str
) -> Optional[dict]:
    """Return stored case reasoning for case_id, or None."""
    doc = await db[CASE_REASONINGS].find_one({"case_id": case_id})
    if doc:
        doc.pop("_id", None)
    return doc
