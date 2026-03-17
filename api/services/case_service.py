"""
case_service.py

CRUD operations for cases stored in MongoDB.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import CASES


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"case_{uuid.uuid4().hex[:12]}"


async def create_case(
    db: AsyncIOMotorDatabase,
    user_id: str,
    title: str,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """Insert a new case and return the document."""
    now = _now()
    doc = {
        "_id": _new_id(),
        "user_id": user_id,
        "title": title,
        "description": description,
        "status": "active",
        "metadata": metadata or {},
        "documents": [],
        "created_at": now,
        "updated_at": now,
    }
    await db[CASES].insert_one(doc)
    return doc


async def list_cases(
    db: AsyncIOMotorDatabase,
    user_id: str,
    skip: int = 0,
    limit: int = 20,
) -> Tuple[List[dict], int]:
    """Return paginated cases for a user and the total count."""
    query = {"user_id": user_id, "status": {"$ne": "deleted"}}
    total = await db[CASES].count_documents(query)
    cursor = db[CASES].find(query).sort("created_at", -1).skip(skip).limit(limit)
    cases = await cursor.to_list(length=limit)
    return cases, total


async def get_case(
    db: AsyncIOMotorDatabase, case_id: str, user_id: str
) -> Optional[dict]:
    """Fetch a single case by ID (scoped to user)."""
    return await db[CASES].find_one(
        {"_id": case_id, "user_id": user_id, "status": {"$ne": "deleted"}}
    )


async def update_case(
    db: AsyncIOMotorDatabase,
    case_id: str,
    user_id: str,
    updates: Dict[str, Any],
) -> Optional[dict]:
    """Apply partial updates to a case and return the updated document."""
    updates["updated_at"] = _now()
    result = await db[CASES].find_one_and_update(
        {"_id": case_id, "user_id": user_id, "status": {"$ne": "deleted"}},
        {"$set": updates},
        return_document=True,
    )
    return result


async def soft_delete_case(
    db: AsyncIOMotorDatabase, case_id: str, user_id: str
) -> bool:
    """Soft-delete a case by setting status to 'deleted'."""
    result = await db[CASES].update_one(
        {"_id": case_id, "user_id": user_id},
        {"$set": {"status": "deleted", "updated_at": _now()}},
    )
    return result.modified_count > 0


async def add_document_to_case(
    db: AsyncIOMotorDatabase,
    case_id: str,
    doc_ref: dict,
) -> None:
    """Push a document reference into the case's documents array."""
    await db[CASES].update_one(
        {"_id": case_id},
        {
            "$push": {"documents": doc_ref},
            "$set": {"updated_at": _now()},
        },
    )
