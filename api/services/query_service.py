"""
query_service.py

Runs the Supervisor LangGraph workflow and yields SSE events.

The graph uses synchronous LLM calls and pymongo under the hood, so
streaming is wrapped via ``asyncio.to_thread``.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.db.collections import SUMMARIES
from api.services.conversation_service import (
    append_turn,
    create_conversation,
    get_conversation,
)

logger = logging.getLogger(__name__)


def _format_sse(event: str, data: Any) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


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
    }


def _stream_graph_sync(state: dict) -> list:
    """Run the graph in streaming mode (synchronous) and collect events.

    Returns a list of ``(node_name, state_snapshot)`` pairs.
    """
    from Supervisor.graph import build_supervisor_graph

    graph = build_supervisor_graph()

    events: list = []
    for event in graph.stream(state):
        # LangGraph stream yields dicts keyed by node name
        events.append(event)
    return events


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
    """
    # Load or create conversation
    # BUG-11 fix: if conversation_id was explicitly provided but not found,
    # emit an error instead of silently creating a new one.
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

    # Flatten turns into role/content pairs
    conversation_history = []
    for t in conv.get("turns", []):
        conversation_history.append({"role": "user", "content": t.get("query", "")})
        conversation_history.append({"role": "assistant", "content": t.get("response", "")})

    turn_count = len(conv.get("turns", []))

    # Build initial state
    state = _build_initial_state(
        query=query,
        case_id=case_id,
        conversation_history=conversation_history,
        turn_count=turn_count,
    )

    # Stream progress
    yield _format_sse("progress", {"step": "starting", "status": "running"})

    try:
        events = await asyncio.to_thread(_stream_graph_sync, state)
    except Exception as exc:
        # BUG-7 fix: sanitize error messages -- log full details server-side
        # but only send a generic message to the client.
        logger.exception("Supervisor graph failed: %s", exc)
        yield _format_sse(
            "error",
            {"detail": "An internal error occurred while processing the query", "code": "INTERNAL_ERROR"},
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

    # Build result
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
            "conversation_id": conversation_id,
        },
    )

    # Persist the conversation turn
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

    # If summarize agent was used, store the summary
    if "summarize" in agents_used and final_response:
        await db[SUMMARIES].update_one(
            {"case_id": case_id},
            {
                "$set": {
                    "case_id": case_id,
                    "summary": final_response,
                    "generated_at": now,
                    "sources": sources,
                }
            },
            upsert=True,
        )

    yield _format_sse("done", {})
