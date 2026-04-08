"""
conftest.py

Shared pytest fixtures for the API test suite.

Uses FastAPI's TestClient with dependency overrides so tests run
without a real MongoDB or valid JWT -- everything is mocked in-memory.
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from api.app import create_app
from api.config import Settings, get_settings
from api.dependencies import get_current_user, get_db, get_settings as dep_get_settings


# ---------------------------------------------------------------------------
# In-memory MongoDB mock
# ---------------------------------------------------------------------------

class FakeCursor:
    """Simulates a Motor cursor with sort/skip/limit chaining."""

    def __init__(self, docs: List[dict]):
        self._docs = list(docs)

    def sort(self, *args, **kwargs):
        return self

    def skip(self, n: int):
        self._docs = self._docs[n:]
        return self

    def limit(self, n: int):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length: int = 100):
        return self._docs[:length]


class FakeCollection:
    """In-memory collection that mimics Motor's async API."""

    def __init__(self):
        self._docs: Dict[str, dict] = {}

    async def insert_one(self, doc: dict):
        key = doc.get("_id", str(id(doc)))
        stored = dict(doc)
        stored["_id"] = key
        self._docs[key] = stored
        result = MagicMock()
        result.inserted_id = key
        return result

    async def find_one(self, query: dict):
        for doc in self._docs.values():
            if self._matches(doc, query):
                return dict(doc)
        return None

    def find(self, query: dict):
        matched = [dict(d) for d in self._docs.values() if self._matches(d, query)]
        return FakeCursor(matched)

    async def count_documents(self, query: dict):
        return sum(1 for d in self._docs.values() if self._matches(d, query))

    async def update_one(self, query: dict, update: dict, upsert: bool = False):
        for key, doc in self._docs.items():
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                if "$push" in update:
                    for field, val in update["$push"].items():
                        doc.setdefault(field, []).append(val)
                self._docs[key] = doc
                result = MagicMock()
                result.modified_count = 1
                return result
        if upsert and "$set" in update:
            doc = dict(update["$set"])
            doc.update(query)
            key = doc.get("_id", str(len(self._docs)))
            self._docs[key] = doc
            result = MagicMock()
            result.modified_count = 0
            result.upserted_id = key
            return result
        result = MagicMock()
        result.modified_count = 0
        return result

    async def find_one_and_update(self, query, update, return_document=False):
        for key, doc in self._docs.items():
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                if "$push" in update:
                    for field, val in update["$push"].items():
                        doc.setdefault(field, []).append(val)
                self._docs[key] = doc
                return dict(doc)
        return None

    async def delete_one(self, query: dict):
        to_delete = None
        for key, doc in self._docs.items():
            if self._matches(doc, query):
                to_delete = key
                break
        result = MagicMock()
        if to_delete is not None:
            del self._docs[to_delete]
            result.deleted_count = 1
        else:
            result.deleted_count = 0
        return result

    async def command(self, cmd: str):
        """Fake command for health check ping."""
        return {"ok": 1}

    def _matches(self, doc: dict, query: dict) -> bool:
        """Simple query matcher supporting equality and $ne."""
        for field, condition in query.items():
            val = doc.get(field)
            if isinstance(condition, dict):
                if "$ne" in condition and val == condition["$ne"]:
                    return False
            else:
                if val != condition:
                    return False
        return True


class FakeDatabase:
    """In-memory database that creates collections on demand."""

    def __init__(self):
        self._collections: Dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self._collections:
            self._collections[name] = FakeCollection()
        return self._collections[name]

    async def command(self, cmd: str):
        return {"ok": 1}


# ---------------------------------------------------------------------------
# Test settings
# ---------------------------------------------------------------------------

TEST_JWT_SECRET = "test-secret-key"
TEST_USER_ID = "test_user_123"


def make_test_settings() -> Settings:
    """Return settings suitable for testing."""
    return Settings(
        jwt_secret=TEST_JWT_SECRET,
        jwt_algorithm="HS256",
        mongo_uri="mongodb://localhost:27017/",
        mongo_db="test_db",
        upload_dir="/tmp/test_uploads",
        cors_origins="*",
        debug=True,
    )


def make_token(user_id: str = TEST_USER_ID) -> str:
    """Create a valid JWT for testing."""
    return jwt.encode(
        {"user_id": user_id},
        TEST_JWT_SECRET,
        algorithm="HS256",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_db():
    """Return a fresh in-memory database."""
    return FakeDatabase()


@pytest.fixture()
def test_settings():
    """Return test settings."""
    return make_test_settings()


@pytest.fixture()
def auth_headers():
    """Return Authorization headers with a valid test JWT."""
    return {"Authorization": f"Bearer {make_token()}"}


@pytest.fixture()
def client(fake_db, test_settings):
    """Return a TestClient with all dependencies overridden."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_current_user] = lambda: TEST_USER_ID
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[dep_get_settings] = lambda: test_settings

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
