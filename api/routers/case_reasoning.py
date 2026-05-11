"""
case_reasoning.py

GET /api/v1/cases/{case_id}/case-reasoning
"""

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.errors import CASE_NOT_FOUND, CASE_REASONING_NOT_FOUND
from api.schemas.common import ErrorEnvelope
from api.schemas.case_reasoning import CaseReasoningResponse
from api.services import case_service
from api.services.case_reasoning_service import get_case_reasoning

router = APIRouter(prefix="/api/v1/cases", tags=["CaseReasoning"])


@router.get(
    "/{case_id}/case-reasoning",
    response_model=CaseReasoningResponse,
    summary="Get persisted case reasoning result",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def get_case_reasoning_endpoint(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
        )

    reasoning = await get_case_reasoning(db, case_id)
    if reasoning is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": CASE_REASONING_NOT_FOUND,
                "message": "No case reasoning has been generated for this case",
            },
        )

    return CaseReasoningResponse(**reasoning)
