"""
query.py

POST /api/v1/query -- supervisor query with SSE streaming.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.schemas.query import QueryRequest
from api.services.query_service import run_query_sse

router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post(
    "/query",
    summary="Run a supervisor query with SSE progress streaming",
    response_description="Server-Sent Events stream",
    status_code=status.HTTP_200_OK,
)
async def supervisor_query(
    body: QueryRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Accept a judge query, run the full Supervisor graph, and stream
    progress events via Server-Sent Events.
    """
    event_generator = run_query_sse(
        db=db,
        settings=settings,
        user_id=user_id,
        query=body.query,
        case_id=body.case_id,
        conversation_id=body.conversation_id,
    )
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
