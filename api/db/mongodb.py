"""
mongodb.py

Motor (async) MongoDB client management.

``connect_mongo`` / ``close_mongo`` are called from the FastAPI lifespan
handler.  ``get_database`` is a FastAPI dependency that returns the
active database handle.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import Settings

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_mongo(settings: Settings) -> None:
    """Open the Motor client and select the database."""
    global _client, _database
    _client = AsyncIOMotorClient(settings.mongo_uri)
    _database = _client[settings.mongo_db]


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
