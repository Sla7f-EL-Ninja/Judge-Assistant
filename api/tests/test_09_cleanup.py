"""
test_cleanup.py

Delete the test case created at the start of the session.
This file is named so pytest runs it last (alphabetically after all others).
"""

import pytest
from httpx import AsyncClient

from conftest import TestState, auth_headers

HEADERS = auth_headers()


@pytest.mark.asyncio
async def test_delete_test_case(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("No case was created — nothing to clean up")

    r = await client.delete(
        f"/api/v1/cases/{state.case_id}",
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert "deleted" in r.json()["message"].lower()


@pytest.mark.asyncio
async def test_deleted_case_is_gone(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("No case was created")

    r = await client.get(
        f"/api/v1/cases/{state.case_id}",
        headers=HEADERS,
    )
    assert r.status_code == 404
