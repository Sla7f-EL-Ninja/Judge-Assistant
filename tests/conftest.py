"""
conftest.py -- Root test configuration for the Hakim AI test suite.

Handles:
1. Windows event loop policy (ProactorEventLoop guard)
2. Project root on sys.path
3. Module aliasing for config namespace conflicts
4. Database fixtures with skip guards
5. test_document_id fixture (session-scoped)
6. pipeline fixture (supervisor graph)
7. app_client fixture (httpx AsyncClient with ASGITransport)
8. JWT helper for auth headers
"""

import asyncio
import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Windows event loop policy
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------------------------------------------------------------------------
# 2. Project root on sys.path
# ---------------------------------------------------------------------------
_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

load_dotenv()

# ---------------------------------------------------------------------------
# 3. Module aliasing -- ensure config package loads cleanly.
#    The codebase already handles the api.config vs RAG config conflict via
#    a shim pattern in RAG/Civil Law RAG/config.py that re-exports from
#    config.rag. We just need the project root on sys.path (done above).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JWT helper -- reuse pattern from api/tests/conftest.py
# ---------------------------------------------------------------------------
def make_jwt(user_id: str = "test_user_001", extra: Optional[dict] = None) -> str:
    """Generate a valid JWT for test authentication."""
    from jose import jwt as jose_jwt

    from config.api import get_settings

    settings = get_settings()
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        **(extra or {}),
    }
    return jose_jwt.encode(
        payload, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )


def auth_headers(user_id: str = "test_user_001") -> dict:
    """Return Authorization header dict with a valid JWT."""
    return {"Authorization": f"Bearer {make_jwt(user_id)}"}


# ---------------------------------------------------------------------------
# 4. Database fixtures with skip guards
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def motor_client():
    """Function-scoped async Motor client; skips if MongoDB is unreachable.

    Function-scoped because Motor's AsyncIOMotorClient binds to the running
    event loop. Each test function gets its own loop in pytest-asyncio.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        from config.api import get_settings

        settings = get_settings()
        client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=3000,
        )
        # Verify connectivity
        await client.admin.command("ping")
        db = client[settings.mongo_db]
        yield db
        client.close()
    except Exception as exc:
        pytest.skip(f"MongoDB unavailable: {exc}")


@pytest.fixture(scope="session")
def qdrant_client():
    """Session-scoped Qdrant client; skips if unreachable."""
    try:
        from qdrant_client import QdrantClient

        from config.api import get_settings

        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5,
        )
        # Verify connectivity
        client.get_collections()
        yield client
        client.close()
    except Exception as exc:
        pytest.skip(f"Qdrant unavailable: {exc}")


@pytest_asyncio.fixture
async def redis_client():
    """Function-scoped async Redis client; skips if unreachable."""
    try:
        import redis.asyncio as aioredis

        from config.api import get_settings

        settings = get_settings()
        client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        await client.ping()
        yield client
        await client.aclose()
    except Exception as exc:
        pytest.skip(f"Redis unavailable: {exc}")


@pytest.fixture(scope="session")
def minio_client():
    """Session-scoped MinIO client; skips if unreachable."""
    try:
        from minio import Minio

        from config.api import get_settings

        settings = get_settings()
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=getattr(settings, "minio_secure", False),
        )
        # Verify connectivity
        client.list_buckets()
        yield client
    except Exception as exc:
        pytest.skip(f"MinIO unavailable: {exc}")


# ---------------------------------------------------------------------------
# 5. test_document_id fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_pdf_bytes() -> bytes:
    """Generate a small synthetic PDF for testing."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, text="Test document for Hakim AI test suite", ln=True)
        pdf.cell(200, 10, text="This is a synthetic test PDF.", ln=True)
        return bytes(pdf.output())
    except ImportError:
        # Minimal valid PDF if fpdf2 not available
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )


# ---------------------------------------------------------------------------
# 6. pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline():
    """Initialize the supervisor graph for direct invocation tests."""
    try:
        from Supervisor.graph import build_supervisor_graph

        return build_supervisor_graph()
    except Exception as exc:
        pytest.skip(f"Supervisor graph unavailable: {exc}")


# ---------------------------------------------------------------------------
# 7. app_client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_client() -> AsyncGenerator:
    """httpx AsyncClient wrapping the FastAPI app via ASGITransport.

    Function-scoped because Motor binds to the running event loop.
    """
    from httpx import ASGITransport, AsyncClient

    from api.app import create_app
    from api.db.mongodb import close_mongo, connect_mongo
    from config.api import get_settings

    settings = get_settings()
    await connect_mongo(settings)
    application = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://testserver",
        timeout=120.0,
    ) as client:
        yield client

    await close_mongo()


# ---------------------------------------------------------------------------
# 8. Test query fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_query() -> str:
    """Default Arabic legal query for tests."""
    return os.getenv(
        "TEST_QUERY",
        "\u0645\u0627 \u0647\u064a \u0634\u0631\u0648\u0637 \u0635\u062d\u0629 "
        "\u0639\u0642\u062f \u0627\u0644\u0628\u064a\u0639 \u0641\u064a "
        "\u0627\u0644\u0642\u0627\u0646\u0648\u0646 \u0627\u0644\u0645\u062f\u0646\u064a "
        "\u0627\u0644\u0645\u0635\u0631\u064a\u061f",
    )
