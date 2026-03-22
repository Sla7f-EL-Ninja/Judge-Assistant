"""
test_conversations.py

Verify that conversation turns were persisted after the query tests
and that listing / reading / deleting all work correctly.
"""

import pytest
from httpx import AsyncClient

from conftest import TestState, auth_headers

HEADERS = auth_headers()


def _assert_error_envelope(r, expected_status: int):
    """Verify response matches the standard ErrorEnvelope shape."""
    assert r.status_code == expected_status, r.text
    body = r.json()
    assert "error" in body, f"Missing 'error' key: {body}"
    err = body["error"]
    assert "code" in err and "detail" in err and "status" in err
    assert err["status"] == r.status_code


@pytest.mark.asyncio
async def test_list_conversations_for_case(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.get(
        f"/api/v1/cases/{state.case_id}/conversations",
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert "conversations" in body and "total" in body


@pytest.mark.asyncio
async def test_list_conversations_pagination(client: AsyncClient, state: TestState):
    """Verify pagination params work on the list endpoint."""
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.get(
        f"/api/v1/cases/{state.case_id}/conversations?skip=0&limit=1",
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["conversations"]) <= 1


@pytest.mark.asyncio
async def test_conversation_was_persisted_after_query(client: AsyncClient, state: TestState):
    if not state.conversation_id:
        pytest.skip("conversation_id not set -- run query tests first")

    r = await client.get(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["_id"] == state.conversation_id
    assert body["case_id"] == state.case_id
    assert len(body["turns"]) >= 1, "Expected at least one turn to be persisted"


@pytest.mark.asyncio
async def test_conversation_turn_has_correct_structure(client: AsyncClient, state: TestState):
    if not state.conversation_id:
        pytest.skip("conversation_id not set")

    r = await client.get(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=HEADERS,
    )
    turn = r.json()["turns"][0]
    for field in ("turn_number", "query", "response", "intent", "agents_used", "sources", "timestamp"):
        assert field in turn, f"Turn missing field '{field}'. Turn: {turn}"


@pytest.mark.asyncio
async def test_conversation_turn_count_matches_queries_sent(client: AsyncClient, state: TestState):
    """
    We sent 2 queries in test_query.py (the main one + the follow-up),
    so the conversation should have 2 turns.
    """
    if not state.conversation_id:
        pytest.skip("conversation_id not set")

    r = await client.get(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=HEADERS,
    )
    turns = r.json()["turns"]
    assert len(turns) == 2, (
        f"Expected 2 turns (initial query + follow-up), got {len(turns)}"
    )


@pytest.mark.asyncio
async def test_get_nonexistent_conversation_returns_404(client: AsyncClient):
    r = await client.get(
        "/api/v1/conversations/conv_doesnotexist999",
        headers=HEADERS,
    )
    _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_get_invalid_conversation_id_format(client: AsyncClient):
    """A conversation ID with invalid format should return 404."""
    r = await client.get(
        "/api/v1/conversations/!@#$%^&*()",
        headers=HEADERS,
    )
    _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_other_user_cannot_read_conversation(client: AsyncClient, state: TestState):
    if not state.conversation_id:
        pytest.skip("conversation_id not set")

    other = auth_headers("other_user_999")
    r = await client.get(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=other,
    )
    _assert_error_envelope(r, 404)


# ---------------------------------------------------------------------------
# Delete conversation -- after we finish reading it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_conversation(client: AsyncClient, state: TestState):
    if not state.conversation_id:
        pytest.skip("conversation_id not set")

    r = await client.delete(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert "deleted" in r.json()["message"].lower()


@pytest.mark.asyncio
async def test_deleted_conversation_is_gone(client: AsyncClient, state: TestState):
    if not state.conversation_id:
        pytest.skip("conversation_id not set")

    r = await client.get(
        f"/api/v1/conversations/{state.conversation_id}",
        headers=HEADERS,
    )
    _assert_error_envelope(r, 404)
