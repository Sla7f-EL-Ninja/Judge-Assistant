"""
test_conversations.py

Tests for conversation history endpoints.
"""

from api.tests.conftest import TEST_USER_ID


def _seed_conversation(fake_db, case_id="case_1", conv_id="conv_1"):
    """Insert a conversation directly into the fake DB."""
    import asyncio
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    doc = {
        "_id": conv_id,
        "case_id": case_id,
        "user_id": TEST_USER_ID,
        "turns": [
            {
                "turn_number": 1,
                "query": "What is article 148?",
                "response": "Article 148 states...",
                "intent": "civil_law_rag",
                "agents_used": ["civil_law_rag"],
                "sources": ["civil_law.txt"],
                "timestamp": now.isoformat(),
            }
        ],
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    # Synchronously insert into the fake collection
    asyncio.run(
        fake_db["conversations"].insert_one(doc)
    )
    return doc


def test_list_conversations_empty(client):
    """Listing conversations for a case with none should return empty."""
    resp = client.get("/api/v1/cases/case_1/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversations"] == []
    assert data["total"] == 0


def test_list_conversations_with_data(client, fake_db):
    """Listing conversations should return seeded data."""
    _seed_conversation(fake_db, case_id="case_1", conv_id="conv_1")

    resp = client.get("/api/v1/cases/case_1/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["conversations"]) == 1
    assert data["conversations"][0]["_id"] == "conv_1"


def test_get_conversation(client, fake_db):
    """GET /api/v1/conversations/{id} should return the conversation."""
    _seed_conversation(fake_db, conv_id="conv_get")

    resp = client.get("/api/v1/conversations/conv_get")
    assert resp.status_code == 200
    data = resp.json()
    assert data["_id"] == "conv_get"
    assert len(data["turns"]) == 1
    assert data["turns"][0]["query"] == "What is article 148?"


def test_get_conversation_not_found(client):
    """Getting a non-existent conversation should return 404."""
    resp = client.get("/api/v1/conversations/nonexistent")
    assert resp.status_code == 404


def test_delete_conversation(client, fake_db):
    """DELETE /api/v1/conversations/{id} should remove the conversation."""
    _seed_conversation(fake_db, conv_id="conv_del")

    resp = client.delete("/api/v1/conversations/conv_del")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Conversation deleted"

    # Should no longer be found
    get_resp = client.get("/api/v1/conversations/conv_del")
    assert get_resp.status_code == 404


def test_delete_conversation_not_found(client):
    """Deleting a non-existent conversation should return 404."""
    resp = client.delete("/api/v1/conversations/nonexistent")
    assert resp.status_code == 404
