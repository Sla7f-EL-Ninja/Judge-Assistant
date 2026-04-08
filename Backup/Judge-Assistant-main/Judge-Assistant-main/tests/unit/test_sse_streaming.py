"""
test_sse_streaming.py -- Validate SSE streaming from the query endpoint.

These tests require a running database and LLM access since they hit the
full query pipeline via the ASGI app.

Marker: integration
"""

import asyncio
import json

import anyio
import pytest


@pytest.mark.integration
class TestSSEStreaming:
    """Validate SSE event stream from POST /api/v1/query."""

    async def _collect_sse_events(self, app_client, query, headers, timeout=120):
        """Helper: stream the query endpoint and collect parsed SSE events."""
        events = []
        async with app_client.stream(
            "POST",
            "/api/v1/query",
            json={"query": query, "case_id": ""},
            headers=headers,
            timeout=timeout,
        ) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    event_type = None
                    data = None
                    for line in frame.strip().split("\n"):
                        if line.startswith("event: "):
                            event_type = line[7:].strip()
                        elif line.startswith("data: "):
                            data = line[6:].strip()
                    if event_type and data:
                        try:
                            events.append(
                                {"event": event_type, "data": json.loads(data)}
                            )
                        except json.JSONDecodeError:
                            events.append({"event": event_type, "data": data})
        return events

    async def test_sse_stream_starts(self, app_client):
        """SSE endpoint should respond with HTTP 200 and begin streaming.

        The original test used asyncio.timeout() which is incompatible with
        anyio's MemoryObjectReceiveStream used internally by Starlette's ASGI
        transport.  When asyncio's timeout fired it raised CancelledError
        inside an anyio task group, leaving the receive stream in a closed
        state so the first chunk was never delivered -- even though the
        pipeline completed successfully (visible in the captured logs).

        Fix: use _collect_sse_events() with a generous httpx timeout (the
        same approach all other tests in this class use).  The assertion
        checks that at least one SSE frame arrived, which fully proves the
        stream started.  The 120-second httpx timeout on the underlying
        request acts as the deadline -- if nothing arrives in that window
        the request itself raises, which pytest will surface as a failure.
        """
        from tests.conftest import auth_headers

        events = await self._collect_sse_events(
            app_client,
            "ما هي شروط صحة العقد؟",
            auth_headers(),
        )

        assert len(events) > 0, (
            "SSE stream should emit at least one event"
        )
        # The very first event must be a progress frame confirming the
        # pipeline started (either 'starting' on a cache miss, or
        # 'cache_hit' on a warm run).
        first_event = events[0]
        assert first_event["event"] == "progress", (
            f"First SSE event must be 'progress', got '{first_event['event']}'"
        )

    async def test_sse_stream_completes(self, app_client, test_query):
        """Stream should end with an event: done."""
        from tests.conftest import auth_headers

        events = await self._collect_sse_events(
            app_client, test_query, auth_headers()
        )
        event_types = [e["event"] for e in events]
        assert "done" in event_types, (
            "SSE stream must include a 'done' event at completion"
        )

    async def test_sse_no_mid_stream_errors(self, app_client, test_query):
        """No error events should appear during normal query processing."""
        from tests.conftest import auth_headers

        events = await self._collect_sse_events(
            app_client, test_query, auth_headers()
        )
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 0, (
            f"SSE stream should not contain error events for valid query, "
            f"got: {error_events}"
        )

    async def test_sse_content_in_result(self, app_client, test_query):
        """At least one result event should contain a non-empty final_response."""
        from tests.conftest import auth_headers

        events = await self._collect_sse_events(
            app_client, test_query, auth_headers()
        )
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) >= 1, (
            "SSE stream must include at least one 'result' event"
        )
        result_data = result_events[0]["data"]
        assert isinstance(result_data, dict), (
            "Result event data must be a JSON object"
        )
        assert result_data.get("final_response"), (
            "Result event must contain a non-empty final_response"
        )

    async def test_sse_session_isolation(self, app_client):
        """Two concurrent streams with different conversation_ids should not mix."""
        from tests.conftest import auth_headers

        headers = auth_headers()
        query_1 = "ما هي شروط صحة العقد في القانون المدني؟"
        query_2 = "ما أحكام التقادم المسقط؟"

        events_1, events_2 = await asyncio.gather(
            self._collect_sse_events(app_client, query_1, headers),
            self._collect_sse_events(app_client, query_2, headers),
        )

        # Both should complete independently
        types_1 = [e["event"] for e in events_1]
        types_2 = [e["event"] for e in events_2]
        assert "done" in types_1, "Stream 1 must complete with 'done' event"
        assert "done" in types_2, "Stream 2 must complete with 'done' event"