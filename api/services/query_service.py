"""
query_service.py

Runs the Supervisor LangGraph workflow and yields SSE events.

The graph uses synchronous LLM calls and pymongo under the hood, so
streaming is wrapped via ``asyncio.to_thread``.

Caching strategy
----------------
Cache key : sha256( query + ":" + case_id )

  The key is derived from the RAW judge query and case_id -- NOT from the
  LLM-rewritten ``classified_query``.  The rewrite is non-deterministic:
  two identical raw queries produce different rewritten strings, so keying
  on the rewrite would guarantee a cache miss every time.

  user_id is intentionally excluded from the key so that different judges
  asking the same question about the same case share the cached answer.
  If per-user isolation is ever required, prefix the key with user_id.

Cache hit path (warm run):
  1. compute key
  2. GET from Redis -- returns a JSON-serialised result payload
  3. Emit a synthetic ``progress`` frame + ``result`` frame + ``done`` frame
  4. Return immediately -- the entire pipeline is bypassed.
  Total latency: one Redis round-trip (~1 ms local, <5 ms over network).

Cache miss path (cold run):
  1. compute key
  2. GET from Redis -- miss
  3. Run the full pipeline
  4. After the ``result`` frame is built, SET the payload in Redis with TTL
  5. Emit the frames normally.

The cached payload intentionally excludes ``conversation_id`` because each
caller gets their own conversation regardless of the cache.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.db.collections import SUMMARIES
from api.db.redis import cache_get, cache_set
from api.services.conversation_service import (
    append_turn,
    create_conversation,
    get_conversation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(query: str, case_id: str) -> str:
    """Return a stable Redis key for this (query, case_id) pair.

    sha256 keeps the key short and safe for all Redis key constraints
    regardless of query length or special characters.
    """
    raw = f"{query}:{case_id or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"query_cache:{digest}"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _format_sse(event: str, data: Any) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _build_initial_state(
    query: str,
    case_id: str,
    conversation_history: List[dict],
    turn_count: int,
    uploaded_files: Optional[List[str]] = None,
) -> dict:
    """Build the initial SupervisorState dict for a graph invocation."""
    from config.supervisor import MAX_RETRIES

    return {
        "judge_query": query,
        "case_id": case_id,
        "uploaded_files": uploaded_files or [],
        "conversation_history": conversation_history,
        "turn_count": turn_count,
        "intent": "",
        "target_agents": [],
        "classified_query": "",
        "agent_results": {},
        "agent_errors": {},
        "validation_status": "",
        "validation_feedback": "",
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "document_classifications": [],
        "merged_response": "",
        "final_response": "",
        "sources": [],
        "correlation_id": str(uuid.uuid4()),
        "classification_error": None,
        "case_summary": None,
        "case_doc_titles": [],
    }


# ---------------------------------------------------------------------------
# Graph runner (synchronous, called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _stream_graph_sync(state: dict) -> list:
    """Run the graph in streaming mode (synchronous) and collect events.

    Returns a list of node-event dicts as yielded by LangGraph's stream().
    """
    from Supervisor.graph import get_app

    graph = get_app()

    events: list = []
    for event in graph.stream(state):
        # LangGraph stream yields dicts keyed by node name
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Main SSE generator
# ---------------------------------------------------------------------------

async def run_query_sse(
    db: AsyncIOMotorDatabase,
    settings: Settings,
    user_id: str,
    query: str,
    case_id: str,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Run the supervisor query and yield SSE-formatted strings.

    This is consumed by ``StreamingResponse`` in the router.

    On a cache hit the full pipeline is skipped: only a Redis GET is
    performed before emitting the cached result, achieving well over the
    required 2x speedup relative to a cold pipeline run (~90 s).
    """

    # ------------------------------------------------------------------
    # 1. Load or create conversation (needed for both hit and miss paths)
    # ------------------------------------------------------------------
    conv: Optional[dict] = None
    if conversation_id:
        conv = await get_conversation(db, conversation_id, user_id)
        if conv is None:
            yield _format_sse(
                "error",
                {"detail": "Conversation not found", "code": "CONVERSATION_NOT_FOUND"},
            )
            yield _format_sse("done", {})
            return

    if conv is None:
        conv = await create_conversation(db, user_id, case_id)
        conversation_id = conv["_id"]

    # Flatten turns into role/content pairs for the pipeline
    conversation_history = []
    for t in conv.get("turns", []):
        conversation_history.append({"role": "user", "content": t.get("query", "")})
        conversation_history.append({"role": "assistant", "content": t.get("response", "")})

    turn_count = len(conv.get("turns", []))

    # ------------------------------------------------------------------
    # 2. Cache check -- attempt to serve from Redis before running the
    #    pipeline.  Files/OCR queries are never cached because their result
    #    depends on the uploaded content, not just the query text.
    # ------------------------------------------------------------------
    cache_eligible = not bool(conv.get("uploaded_files"))  # no files = cacheable
    key = _cache_key(query, case_id)
    cached_raw = await cache_get(key)

    if cached_raw is not None:
        # Cache HIT -- deserialise and re-emit as SSE frames.
        # Emit a minimal progress frame so the client stream shape is
        # identical to a cold run (some clients assert on frame order).
        logger.info("Cache HIT for key=%s", key)
        try:
            cached_payload = json.loads(cached_raw)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry -- fall through to pipeline
            logger.warning("Cache entry corrupted for key=%s, running pipeline", key)
            cached_payload = None

        if cached_payload is not None:
            yield _format_sse("progress", {"step": "cache_hit", "status": "done"})
            yield _format_sse(
                "result",
                {
                    "final_response": cached_payload.get("final_response", ""),
                    "sources": cached_payload.get("sources", []),
                    "intent": cached_payload.get("intent", ""),
                    "agents_used": cached_payload.get("agents_used", []),
                    "conversation_id": str(conversation_id),
                    "cached": True,
                },
            )

            # Still persist the conversation turn so history is accurate
            now = datetime.now(timezone.utc)
            turn = {
                "turn_number": turn_count + 1,
                "query": query,
                "response": cached_payload.get("final_response", ""),
                "intent": cached_payload.get("intent", ""),
                "agents_used": cached_payload.get("agents_used", []),
                "sources": cached_payload.get("sources", []),
                "timestamp": now,
            }
            await append_turn(db, conversation_id, turn)

            yield _format_sse("done", {})
            return

    # ------------------------------------------------------------------
    # 3. Cache MISS -- run the full pipeline
    # ------------------------------------------------------------------
    logger.info("Cache MISS for key=%s -- running pipeline", key)

    state = _build_initial_state(
        query=query,
        case_id=case_id,
        conversation_history=conversation_history,
        turn_count=turn_count,
    )

    logger.info(
        "Supervisor turn starting: correlation_id=%s case_id=%s",
        state["correlation_id"], case_id,
    )
    yield _format_sse("progress", {"step": "starting", "status": "running"})

    try:
        events = await asyncio.to_thread(_stream_graph_sync, state)
    except Exception as exc:
        # BUG-7 fix: sanitize error messages -- log full details server-side
        # but only send a generic message to the client.
        logger.exception("Supervisor graph failed: %s", exc)
        yield _format_sse(
            "error",
            {
                "detail": "An internal error occurred while processing the query",
                "code": "INTERNAL_ERROR",
            },
        )
        yield _format_sse("done", {})
        return

    # Emit progress events for each node
    final_state: dict = {}
    for event in events:
        if isinstance(event, dict):
            for node_name, node_state in event.items():
                yield _format_sse(
                    "progress",
                    {"step": node_name, "status": "done"},
                )
                final_state.update(node_state if isinstance(node_state, dict) else {})

    # Build result payload
    final_response = final_state.get("final_response", "")
    sources = final_state.get("sources", [])
    intent = final_state.get("intent", "")
    agents_used = final_state.get("target_agents", [])

    yield _format_sse(
        "result",
        {
            "final_response": final_response,
            "sources": sources,
            "intent": intent,
            "agents_used": agents_used,
            "conversation_id": str(conversation_id),
            "cached": False,
        },
    )

    # ------------------------------------------------------------------
    # 4. Write to cache (only if the pipeline succeeded and response is
    #    non-empty, and the query is cache-eligible)
    # ------------------------------------------------------------------
    if cache_eligible and final_response:
        payload_to_cache = json.dumps(
            {
                "final_response": final_response,
                "sources": sources,
                "intent": intent,
                "agents_used": agents_used,
            },
            ensure_ascii=False,
            default=str,
        )
        await cache_set(key, payload_to_cache, ttl_seconds=settings.redis_cache_ttl_seconds)
        logger.info(
            "Cached response for key=%s (ttl=%ds)",
            key,
            settings.redis_cache_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # 5. Persist the conversation turn
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    turn = {
        "turn_number": turn_count + 1,
        "query": query,
        "response": final_response,
        "intent": intent,
        "agents_used": agents_used,
        "sources": sources,
        "timestamp": now,
    }
    await append_turn(db, conversation_id, turn)

    yield _format_sse("done", {})