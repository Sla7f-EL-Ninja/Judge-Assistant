"""
conftest.py
"""
import sys
import pathlib

_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Optional

from httpx import AsyncClient, ASGITransport
from jose import jwt
from dotenv import load_dotenv

load_dotenv()

from api.app import create_app
from config.api import get_settings
from api.db.mongodb import connect_mongo, close_mongo


def make_jwt(user_id: str = "test_user_001", extra: dict | None = None) -> str:
    settings = get_settings()
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        **(extra or {}),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def auth_headers(user_id: str = "test_user_001") -> dict:
    return {"Authorization": f"Bearer {make_jwt(user_id)}"}


# ---------------------------------------------------------------------------
# client is FUNCTION-scoped on purpose.
#
# Motor's AsyncIOMotorClient binds to the event loop that is running when
# connect_mongo() is called. pytest-asyncio gives each test function its own
# event loop. If the Motor client is created on a session loop (loop A) and
# a test runs on loop B, every Motor call crashes with "Event loop is closed".
#
# Making client function-scoped means each test gets a fresh Motor client
# bound to its own loop. MongoDB data persists across tests independently of
# the Motor client, so TestState (session-scoped) can still carry IDs.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    settings = get_settings()
    await connect_mongo(settings)

    application = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://testserver",
        timeout=120.0,
    ) as c:
        yield c

    await close_mongo()


# ---------------------------------------------------------------------------
# TestState and state fixture are still SESSION-scoped.
# IDs (case_id, file_id, conversation_id) are set once and read by later
# tests. MongoDB data persists, so later tests find the same documents even
# though they use a different Motor client.
# ---------------------------------------------------------------------------
class TestState:
    user_id: str = "test_user_001"
    file_id: Optional[str] = None
    case_id: Optional[str] = None
    conversation_id: Optional[str] = None


@pytest.fixture(scope="session")
def state() -> TestState:
    return TestState()


@pytest.fixture(scope="session")
def test_pdf_path() -> Optional[str]:
    path = os.getenv("TEST_PDF_PATH", "").strip()
    return path if path and os.path.isfile(path) else None


@pytest.fixture(scope="session")
def test_query() -> str:
    return os.getenv(
        "TEST_QUERY",
        "ما هي شروط صحة عقد البيع في القانون المدني المصري؟"
    )