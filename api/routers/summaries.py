"""
summaries.py

GET /api/v1/cases/{case_id}/summary -- retrieve stored case summary.
"""

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.schemas.common import ErrorEnvelope
from api.schemas.summaries import SummaryResponse
from api.services import case_service, summary_service

router = APIRouter(prefix="/api/v1/cases", tags=["Summaries"])


@router.get(
    "/{case_id}/summary",
    response_model=SummaryResponse,
    summary="Retrieve the stored summary for a case",
    description=(
        "Fetch the auto-generated summary for a case. Returns 404 if the case "
        "does not exist, belongs to another user, or no summary has been generated yet."
    ),
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        404: {"model": ErrorEnvelope, "description": "Case or summary not found"},
        422: {"model": ErrorEnvelope, "description": "Request validation error"},
    },
)
async def get_summary(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    # Verify case access
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    summary = await summary_service.get_summary(db, case_id)
    if summary is None:
        raise HTTPException(
            status_code=404, detail="No summary has been generated for this case"
        )
    return SummaryResponse(
        case_id=summary["case_id"],
        summary=summary["summary"],
        generated_at=summary["generated_at"],
        sources=summary.get("sources", []),
    )
