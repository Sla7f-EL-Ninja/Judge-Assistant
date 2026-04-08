"""
cases.py

CRUD endpoints for case management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.schemas.cases import (
    CaseCreate,
    CaseListResponse,
    CaseResponse,
    CaseUpdate,
)
from api.schemas.common import ErrorEnvelope, MessageResponse
from api.services import case_service
from api.services.conversation_service import count_conversations_for_case

router = APIRouter(prefix="/api/v1/cases", tags=["Cases"])

# Shared error responses for endpoints that require auth
_AUTH_ERRORS = {
    401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
    422: {"model": ErrorEnvelope, "description": "Request validation error"},
}


def _enrich(doc: dict, conv_count: int = 0) -> dict:
    """Add computed fields before serialisation."""
    doc["conversation_count"] = conv_count
    return doc


@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new case",
    description=(
        "Create a new case for the authenticated user. The case groups documents, "
        "conversations, and summaries together. Requires a non-empty title."
    ),
    responses={
        **_AUTH_ERRORS,
    },
)
async def create_case(
    body: CaseCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await case_service.create_case(
        db, user_id, body.title, body.description, body.metadata
    )
    return _enrich(doc)


@router.get(
    "",
    response_model=CaseListResponse,
    summary="List cases for the authenticated user",
    description=(
        "Returns a paginated list of cases belonging to the authenticated user. "
        "Soft-deleted cases are excluded. Each case includes a `conversation_count` field."
    ),
    responses={
        **_AUTH_ERRORS,
    },
)
async def list_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    cases, total = await case_service.list_cases(db, user_id, skip, limit)
    enriched = []
    for c in cases:
        cnt = await count_conversations_for_case(db, c["_id"])
        enriched.append(_enrich(c, cnt))
    return {"cases": enriched, "total": total}


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Get case details",
    description="Retrieve a single case by ID. Returns 404 if the case does not exist or belongs to another user.",
    responses={
        **_AUTH_ERRORS,
        404: {"model": ErrorEnvelope, "description": "Case not found"},
    },
)
async def get_case(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await case_service.get_case(db, case_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Case not found")
    cnt = await count_conversations_for_case(db, case_id)
    return _enrich(doc, cnt)


@router.patch(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Update case metadata or status",
    description=(
        "Partially update a case. Only the provided fields are changed. "
        "Valid status values: active, archived, closed."
    ),
    responses={
        **_AUTH_ERRORS,
        400: {"model": ErrorEnvelope, "description": "No fields to update"},
        404: {"model": ErrorEnvelope, "description": "Case not found"},
    },
)
async def update_case(
    case_id: str,
    body: CaseUpdate,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    doc = await case_service.update_case(db, case_id, user_id, updates)
    if doc is None:
        raise HTTPException(status_code=404, detail="Case not found")
    cnt = await count_conversations_for_case(db, case_id)
    return _enrich(doc, cnt)


@router.delete(
    "/{case_id}",
    response_model=MessageResponse,
    summary="Soft-delete a case",
    description="Mark a case as deleted (soft-delete). The case data is retained but excluded from list results.",
    responses={
        **_AUTH_ERRORS,
        404: {"model": ErrorEnvelope, "description": "Case not found"},
    },
)
async def delete_case(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    deleted = await case_service.soft_delete_case(db, case_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"message": "Case deleted"}
