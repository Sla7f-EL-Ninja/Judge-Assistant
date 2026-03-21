"""
test_query.py

Tests for POST /api/v1/query (SSE streaming endpoint).

Since the actual Supervisor graph requires LLMs and real dependencies,
we mock the streaming function to test the endpoint plumbing.
"""

from unittest.mock import patch, AsyncMock


def test_query_returns_sse_stream(client):
    """POST /api/v1/query should return a text/event-stream response."""

    async def fake_sse(*args, **kwargs):
        yield 'event: progress\ndata: {"step": "classify_intent", "status": "done"}\n\n'
        yield 'event: result\ndata: {"final_response": "Test response", "sources": [], "intent": "civil_law_rag", "agents_used": ["civil_law_rag"], "conversation_id": "conv_1"}\n\n'
        yield 'event: done\ndata: {}\n\n'

    with patch("api.routers.query.run_query_sse", side_effect=fake_sse):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is article 148?", "case_id": "case_1"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        assert "event: progress" in body
        assert "event: result" in body
        assert "event: done" in body
        assert "Test response" in body


def test_query_missing_query_field(client):
    """POST /api/v1/query without a query should return 422."""
    resp = client.post("/api/v1/query", json={"case_id": "case_1"})
    assert resp.status_code == 422


def test_query_empty_query(client):
    """POST /api/v1/query with an empty query string should return 422."""
    resp = client.post(
        "/api/v1/query",
        json={"query": "", "case_id": "case_1"},
    )
    assert resp.status_code == 422
