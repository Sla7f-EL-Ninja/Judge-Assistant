"""
conversations.py

Conversation history endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.schemas.common import MessageResponse
from api.schemas.conversations import ConversationListResponse, ConversationResponse
from api.services import conversation_service

router = APIRouter(prefix="/api/v1", tags=["Conversations"])


@router.get(
    "/cases/{case_id}/conversations",
    response_model=ConversationListResponse,
    summary="List conversations for a case",
)
async def list_conversations(
    case_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    convos, total = await conversation_service.list_conversations(
        db, case_id, user_id, skip, limit
    )
    return {"conversations": convos, "total": total}


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get full conversation history",
)
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    conv = await conversation_service.get_conversation(db, conversation_id, user_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete(
    "/conversations/{conversation_id}",
    response_model=MessageResponse,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    deleted = await conversation_service.delete_conversation(
        db, conversation_id, user_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}
