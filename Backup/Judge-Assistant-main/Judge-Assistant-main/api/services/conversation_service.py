"""
conversation_service.py

Conversation persistence: create, append turns, list, read, delete.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase

from api.db.collections import CONVERSATIONS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


async def create_conversation(
    db: AsyncIOMotorDatabase,
    user_id: str,
    case_id: str,
) -> dict:
    """Create a new conversation and return the document."""
    now = _now()
    doc = {
        "_id": _new_id(),
        "case_id": case_id,
        "user_id": user_id,
        "turns": [],
        "created_at": now,
        "updated_at": now,
    }
    await db[CONVERSATIONS].insert_one(doc)
    return doc


async def append_turn(
    db: AsyncIOMotorDatabase,
    conversation_id: str,
    turn: Dict[str, Any],
) -> None:
    """Append a turn to an existing conversation."""
    await db[CONVERSATIONS].update_one(
        {"_id": conversation_id},
        {
            "$push": {"turns": turn},
            "$set": {"updated_at": _now()},
        },
    )


async def get_conversation(
    db: AsyncIOMotorDatabase, conversation_id: str, user_id: str
) -> Optional[dict]:
    """Fetch a conversation by ID (scoped to user)."""
    return await db[CONVERSATIONS].find_one(
        {"_id": conversation_id, "user_id": user_id}
    )


async def list_conversations(
    db: AsyncIOMotorDatabase,
    case_id: str,
    user_id: str,
    skip: int = 0,
    limit: int = 20,
) -> Tuple[List[dict], int]:
    """Return paginated conversations for a case and the total count."""
    query = {"case_id": case_id, "user_id": user_id}
    total = await db[CONVERSATIONS].count_documents(query)
    cursor = (
        db[CONVERSATIONS].find(query).sort("created_at", -1).skip(skip).limit(limit)
    )
    convos = await cursor.to_list(length=limit)
    return convos, total


async def delete_conversation(
    db: AsyncIOMotorDatabase, conversation_id: str, user_id: str
) -> bool:
    """Delete a conversation. Returns True if a document was removed."""
    result = await db[CONVERSATIONS].delete_one(
        {"_id": conversation_id, "user_id": user_id}
    )
    return result.deleted_count > 0


async def count_conversations_for_case(
    db: AsyncIOMotorDatabase, case_id: str
) -> int:
    """Count conversations belonging to a case."""
    return await db[CONVERSATIONS].count_documents({"case_id": case_id})
