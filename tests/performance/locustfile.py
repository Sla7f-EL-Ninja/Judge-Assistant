"""
locustfile.py -- Load testing for the Judge Assistant API.

Run with:
    locust -f tests/performance/locustfile.py --host http://localhost:8000

Target: 20 concurrent users with weighted task distribution.

Marker: performance (not run by pytest -- standalone Locust file)
"""

import json
import os
from datetime import datetime, timedelta, timezone

from locust import HttpUser, between, task


def _make_jwt():
    """Generate a JWT for load testing."""
    try:
        from jose import jwt as jose_jwt

        secret = os.getenv("JWT_SECRET", "change-me-in-production")
        algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        payload = {
            "user_id": "locust_load_tester",
            "exp": datetime.now(timezone.utc) + timedelta(hours=2),
        }
        return jose_jwt.encode(payload, secret, algorithm=algorithm)
    except ImportError:
        return "dummy-token-for-load-testing"


class JudgeUser(HttpUser):
    """Simulated judge user for load testing."""

    wait_time = between(1, 5)

    def on_start(self):
        """Set up auth headers on user creation."""
        token = _make_jwt()
        self.headers = {"Authorization": f"Bearer {token}"}

    @task(3)
    def civil_law_query(self):
        """Query civil law RAG (highest weight)."""
        queries = [
            "\u0645\u0627 \u0647\u064a \u0634\u0631\u0648\u0637 \u0635\u062d\u0629 \u0627\u0644\u0639\u0642\u062f\u061f",
            "\u0645\u0627 \u0623\u062d\u0643\u0627\u0645 \u0627\u0644\u0645\u0633\u0624\u0648\u0644\u064a\u0629 \u0627\u0644\u062a\u0642\u0635\u064a\u0631\u064a\u0629\u061f",
            "\u0645\u0627 \u0647\u064a \u0623\u062d\u0643\u0627\u0645 \u0627\u0644\u0641\u0633\u062e\u061f",
        ]
        import random

        query = random.choice(queries)
        self.client.post(
            "/api/v1/query",
            json={"query": query, "case_id": ""},
            headers=self.headers,
            name="POST /api/v1/query [civil_law]",
        )

    @task(2)
    def summarize_query(self):
        """Request case summarization (medium weight)."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "\u0644\u062e\u0635 \u0644\u064a \u0645\u0644\u0641 \u0627\u0644\u0642\u0636\u064a\u0629",
                "case_id": "",
            },
            headers=self.headers,
            name="POST /api/v1/query [summarize]",
        )

    @task(1)
    def health_check(self):
        """Check service health (lowest weight)."""
        self.client.get("/api/v1/health", name="GET /api/v1/health")
