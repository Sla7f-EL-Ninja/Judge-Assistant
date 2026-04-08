"""
mongodb.py

Motor (async) MongoDB client management with production hardening.

``connect_mongo`` / ``close_mongo`` are called from the FastAPI lifespan
handler.  ``get_database`` is a FastAPI dependency that returns the
active database handle.

Production features:
- Connection pooling (configurable min/max pool size)
- Server selection timeout
- Automatic index creation on startup
"""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config.api import Settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_mongo(settings: Settings) -> None:
    """Open the Motor client with connection pooling and select the database."""
    global _client, _database
    _client = AsyncIOMotorClient(
        settings.mongo_uri,
        minPoolSize=settings.mongo_min_pool_size,
        maxPoolSize=settings.mongo_max_pool_size,
        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
    )
    _database = _client[settings.mongo_db]

    # Create indexes for production query performance
    await _ensure_indexes(_database)


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes on frequently queried fields.

    Indexes are created with ``background=True`` so they don't block
    other operations on existing collections.
    """
    from api.db.collections import CASES, CONVERSATIONS, FILES, SUMMARIES, DOCUMENTS

    try:
        # Cases: user_id + status (list queries), created_at (sorting)
        await db[CASES].create_index(
            [("user_id", 1), ("status", 1)], background=True
        )
        await db[CASES].create_index("created_at", background=True)

        # Conversations: case_id + user_id (list queries), created_at (sorting)
        await db[CONVERSATIONS].create_index(
            [("case_id", 1), ("user_id", 1)], background=True
        )
        await db[CONVERSATIONS].create_index("created_at", background=True)

        # Files: user_id (ownership queries)
        await db[FILES].create_index("user_id", background=True)

        # Summaries: case_id (lookup)
        await db[SUMMARIES].create_index("case_id", unique=True, background=True)

        # Documents: case_id (filtered retrieval)
        await db[DOCUMENTS].create_index("case_id", background=True)
        await db[DOCUMENTS].create_index(
            [("case_id", 1), ("doc_type", 1)], background=True
        )

        logger.info("MongoDB indexes ensured on all collections")
    except Exception as exc:
        logger.warning("Failed to create some MongoDB indexes: %s", exc)


async def close_mongo() -> None:
    """Close the Motor client."""
    global _client, _database
    if _client is not None:
        _client.close()
        _client = None
        _database = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database.  Raises if not connected."""
    if _database is None:
        raise RuntimeError("MongoDB is not connected. Call connect_mongo first.")
    return _database
