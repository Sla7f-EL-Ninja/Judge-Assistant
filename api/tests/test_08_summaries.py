"""
test_summaries.py

Verify summary retrieval.
A summary is only stored when the Supervisor uses the 'summarize' agent,
so if no query has triggered that, the 404 case is the expected outcome.
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
async def test_summary_endpoint_reachable(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.get(
        f"/api/v1/cases/{state.case_id}/summary",
        headers=HEADERS,
    )
    # Either a summary was generated (200) or it wasn't (404) -- both are valid
    assert r.status_code in (200, 404), f"Unexpected status: {r.status_code} {r.text}"
    if r.status_code == 404:
        _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_summary_structure_when_present(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.get(
        f"/api/v1/cases/{state.case_id}/summary",
        headers=HEADERS,
    )
    if r.status_code == 404:
        pytest.skip("No summary generated yet -- fire a summarize query to test this")

    body = r.json()
    for field in ("case_id", "summary", "generated_at", "sources"):
        assert field in body, f"Field '{field}' missing from summary response"
    assert body["case_id"] == state.case_id
    assert len(body["summary"]) > 0


@pytest.mark.asyncio
async def test_summary_for_nonexistent_case_returns_404(client: AsyncClient):
    r = await client.get(
        "/api/v1/cases/case_doesnotexist999/summary",
        headers=HEADERS,
    )
    _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_summary_wrong_user_returns_404(client: AsyncClient, state: TestState):
    """Another user should not be able to access the summary."""
    if not state.case_id:
        pytest.skip("case_id not set")

    other = auth_headers("other_user_999")
    r = await client.get(
        f"/api/v1/cases/{state.case_id}/summary",
        headers=other,
    )
    _assert_error_envelope(r, 404)
