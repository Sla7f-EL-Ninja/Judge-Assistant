"""
test_cases.py

Tests for the /api/v1/cases CRUD endpoints.
"""


def test_create_case(client):
    """POST /api/v1/cases should create a new case and return it."""
    resp = client.post(
        "/api/v1/cases",
        json={
            "title": "Test Case",
            "description": "A test case for unit testing",
            "metadata": {"court": "Test Court"},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Test Case"
    assert data["description"] == "A test case for unit testing"
    assert data["status"] == "active"
    assert data["metadata"]["court"] == "Test Court"
    assert "_id" in data


def test_create_case_minimal(client):
    """Creating a case with only the required title should work."""
    resp = client.post("/api/v1/cases", json={"title": "Minimal Case"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Minimal Case"
    assert data["description"] == ""


def test_create_case_missing_title(client):
    """Creating a case without a title should return 422."""
    resp = client.post("/api/v1/cases", json={})
    assert resp.status_code == 422


def test_list_cases_empty(client):
    """Listing cases when none exist should return an empty list."""
    resp = client.get("/api/v1/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cases"] == []
    assert data["total"] == 0


def test_list_cases_after_create(client):
    """Listing cases should include a case we just created."""
    client.post("/api/v1/cases", json={"title": "Case 1"})
    client.post("/api/v1/cases", json={"title": "Case 2"})

    resp = client.get("/api/v1/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["cases"]) == 2


def test_get_case_by_id(client):
    """GET /api/v1/cases/{id} should return the case details."""
    create_resp = client.post("/api/v1/cases", json={"title": "Get Me"})
    case_id = create_resp.json()["_id"]

    resp = client.get(f"/api/v1/cases/{case_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Get Me"


def test_get_case_not_found(client):
    """Getting a non-existent case should return 404."""
    resp = client.get("/api/v1/cases/nonexistent_id")
    assert resp.status_code == 404


def test_update_case(client):
    """PATCH /api/v1/cases/{id} should update the case."""
    create_resp = client.post("/api/v1/cases", json={"title": "Original"})
    case_id = create_resp.json()["_id"]

    resp = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"title": "Updated", "status": "archived"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Updated"
    assert data["status"] == "archived"


def test_update_case_no_fields(client):
    """PATCH with empty body should return 400."""
    create_resp = client.post("/api/v1/cases", json={"title": "No Update"})
    case_id = create_resp.json()["_id"]

    resp = client.patch(f"/api/v1/cases/{case_id}", json={})
    assert resp.status_code == 400


def test_update_case_not_found(client):
    """PATCH on a non-existent case should return 404."""
    resp = client.patch(
        "/api/v1/cases/nonexistent_id",
        json={"title": "Ghost"},
    )
    assert resp.status_code == 404


def test_delete_case(client):
    """DELETE /api/v1/cases/{id} should soft-delete the case."""
    create_resp = client.post("/api/v1/cases", json={"title": "Delete Me"})
    case_id = create_resp.json()["_id"]

    resp = client.delete(f"/api/v1/cases/{case_id}")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Case deleted"

    # Should no longer be found
    get_resp = client.get(f"/api/v1/cases/{case_id}")
    assert get_resp.status_code == 404


def test_delete_case_not_found(client):
    """Deleting a non-existent case should return 404."""
    resp = client.delete("/api/v1/cases/nonexistent_id")
    assert resp.status_code == 404


def test_list_cases_pagination(client):
    """Pagination parameters should limit results."""
    for i in range(5):
        client.post("/api/v1/cases", json={"title": f"Case {i}"})

    resp = client.get("/api/v1/cases?skip=0&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["cases"]) == 2
