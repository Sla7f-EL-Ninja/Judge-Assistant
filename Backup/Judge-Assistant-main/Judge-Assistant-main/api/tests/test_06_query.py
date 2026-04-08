"""
test_query.py

Fire a real Arabic legal query at the Supervisor graph and verify
the full SSE stream:

  event: progress  (step=starting)
  event: progress  (step=<node_name>) × N
  event: result    (final_response, sources, intent, agents_used, conversation_id)
  event: done

Skipped if TEST_QUERY is not set.
The conversation_id is stored in TestState for the conversation tests.
"""

import json
import pytest
from httpx import AsyncClient

from conftest import TestState, auth_headers

HEADERS = auth_headers()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def parse_sse(raw: str) -> list[dict]:
    """
    Parse a raw SSE response body into a list of
    {"event": ..., "data": ...} dicts.
    """
    events = []
    current_event = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current_event["event"] = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            raw_data = line.removeprefix("data:").strip()
            try:
                current_event["data"] = json.loads(raw_data)
            except json.JSONDecodeError:
                current_event["data"] = raw_data
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}
    if current_event:
        events.append(current_event)
    return events


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_returns_sse_stream(client: AsyncClient, state: TestState, test_query: str):
    if not state.case_id:
        pytest.skip("case_id not set — run case tests first")

    r = await client.post(
        "/api/v1/query",
        json={"query": test_query, "case_id": state.case_id},
        headers=HEADERS,
        timeout=180.0,
    )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", ""), (
        f"Expected SSE content-type, got: {r.headers.get('content-type')}"
    )


@pytest.mark.asyncio
async def test_query_sse_contains_required_events(client: AsyncClient, state: TestState, test_query: str):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        "/api/v1/query",
        json={"query": test_query, "case_id": state.case_id},
        headers=HEADERS,
        timeout=180.0,
    )
    events = parse_sse(r.text)
    event_names = [e["event"] for e in events]

    assert "progress" in event_names, f"No 'progress' event found. Events: {event_names}"
    assert "result" in event_names, f"No 'result' event found. Events: {event_names}"
    assert "done" in event_names, f"No 'done' event found. Events: {event_names}"


@pytest.mark.asyncio
async def test_query_result_has_non_empty_response(client: AsyncClient, state: TestState, test_query: str):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        "/api/v1/query",
        json={"query": test_query, "case_id": state.case_id},
        headers=HEADERS,
        timeout=180.0,
    )
    events = parse_sse(r.text)
    result_events = [e for e in events if e["event"] == "result"]
    assert result_events, "No result event in SSE stream"

    result = result_events[0]["data"]
    assert result.get("final_response"), (
        f"final_response is empty. Full result payload: {result}"
    )
    assert result.get("conversation_id"), "conversation_id missing from result"

    # Store for conversation tests
    state.conversation_id = result["conversation_id"]


@pytest.mark.asyncio
async def test_query_result_has_expected_fields(client: AsyncClient, state: TestState, test_query: str):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        "/api/v1/query",
        json={"query": test_query, "case_id": state.case_id},
        headers=HEADERS,
        timeout=180.0,
    )
    events = parse_sse(r.text)
    result = next(e["data"] for e in events if e["event"] == "result")

    for field in ("final_response", "sources", "intent", "agents_used", "conversation_id"):
        assert field in result, f"Field '{field}' missing from result payload"


@pytest.mark.asyncio
async def test_query_progress_starts_with_starting_event(client: AsyncClient, state: TestState, test_query: str):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        "/api/v1/query",
        json={"query": test_query, "case_id": state.case_id},
        headers=HEADERS,
        timeout=180.0,
    )
    events = parse_sse(r.text)
    first_progress = next((e for e in events if e["event"] == "progress"), None)
    assert first_progress is not None
    assert first_progress["data"].get("step") == "starting"


@pytest.mark.asyncio
async def test_query_continues_existing_conversation(client: AsyncClient, state: TestState, test_query: str):
    """Second turn — should reuse the conversation created in the first query."""
    if not state.conversation_id:
        pytest.skip("conversation_id not set — run first query test")
    if not state.case_id:
        pytest.skip("case_id not set")

    followup = "هل يمكنك توضيح الشروط التي ذكرتها؟"
    r = await client.post(
        "/api/v1/query",
        json={
            "query": followup,
            "case_id": state.case_id,
            "conversation_id": state.conversation_id,
        },
        headers=HEADERS,
        timeout=180.0,
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    result = next((e["data"] for e in events if e["event"] == "result"), None)
    assert result is not None
    # Must return the SAME conversation_id, not create a new one
    assert result["conversation_id"] == state.conversation_id, (
        f"Expected conversation_id={state.conversation_id}, "
        f"got {result['conversation_id']}"
    )


@pytest.mark.asyncio
async def test_query_empty_string_rejected(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        "/api/v1/query",
        json={"query": "", "case_id": state.case_id},
        headers=HEADERS,
    )
    assert r.status_code == 422
