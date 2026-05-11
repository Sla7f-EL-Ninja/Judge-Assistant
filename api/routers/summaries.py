"""
summaries.py

Summary endpoints:
  GET  /api/v1/cases/{case_id}/summary          — retrieve stored summary
  POST /api/v1/cases/{case_id}/summary/generate — run pipeline and save summary
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.schemas.common import ErrorEnvelope
from api.schemas.summaries import GenerateSummaryResponse, SummaryResponse, CaseBriefResponse
from api.services import case_service, summary_service
from api.errors import CASE_NOT_FOUND, SUMMARY_NOT_FOUND
from api.db.collections import DOCUMENTS
from summarize.pipeline import run_summarization, SummarizationResult

logger = logging.getLogger("hakim.api.summaries")

router = APIRouter(prefix="/api/v1/cases", tags=["Summaries"])


# ---------------------------------------------------------------------------
# GET — retrieve stored summary
# ---------------------------------------------------------------------------

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
) -> SummaryResponse:
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    summary = await summary_service.get_summary(db, case_id)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="No summary has been generated for this case",
        )

    return SummaryResponse(
        case_id=summary["case_id"],
        summary=summary["summary"],
        generated_at=summary["generated_at"],
        sources=summary.get("sources", []),
    )


# ---------------------------------------------------------------------------
# POST — run pipeline and save summary
# ---------------------------------------------------------------------------

@router.post(
    "/{case_id}/summary/generate",
    response_model=GenerateSummaryResponse,
    summary="Generate and store a summary for a case",
    description=(
        "Fetches all documents for the case from MongoDB, runs the full "
        "summarization pipeline (Nodes 0-5), and upserts the result linked "
        "to this case_id. Re-running overwrites any previous summary."
    ),
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        404: {"model": ErrorEnvelope, "description": "Case or summary not found"},
        422: {"model": ErrorEnvelope, "description": "Request validation error"},
        500: {"model": ErrorEnvelope, "description": "Pipeline execution failed"},
    },
)
async def generate_summary(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> GenerateSummaryResponse:
    # 1. Verify case ownership
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
 
    # 2. Fetch and MAP this case's documents from MongoDB [FIXED]
    raw_docs = await db[DOCUMENTS].find(
        {"case_id": case_id},
        {"_id": 1, "text": 1},
    ).to_list(length=None)

    if not raw_docs:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this case. Upload documents before generating a summary.",
        )

    # Convert the database 'text' field to 'raw_text' as expected by Source 11
    documents = [
        {"doc_id": str(d["_id"]), "raw_text": d.get("text", "").strip()} 
        for d in raw_docs
    ]

    # 3. Run the pipeline in a thread
    logger.info("Generating summary  case_id='%s'  docs=%d", case_id, len(documents))
    try:
        result: SummarizationResult = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_summarization(
                documents=documents, # Uses the mapped list with 'raw_text' keys
                case_id=case_id,
                save_to_db=False,
            ),
        )
    except ValueError as exc:
        # Now catches actual empty text instead of missing keys
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Pipeline failed  case_id='%s': %s", case_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not result.rendered_brief:
        raise HTTPException(status_code=500, detail="Pipeline produced an empty brief.")

    # 4. Persist via the async Motor connection
    await summary_service.save_summary(
        db=db,
        case_id=case_id,
        rendered_brief=result.rendered_brief,
        all_sources=result.all_sources,
        case_brief=result.case_brief,
    )
    logger.info("Summary saved  case_id='%s'  sources=%d", case_id, len(result.all_sources))

    return GenerateSummaryResponse(
        case_id=case_id,
        sources_count=len(result.all_sources),
        message="Summary generated and saved successfully.",
    )


# ---------------------------------------------------------------------------
# GET — retrieve structured case brief
# ---------------------------------------------------------------------------

@router.get(
    "/{case_id}/case-brief",
    response_model=CaseBriefResponse,
    summary="Retrieve the structured case brief for a case",
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        404: {"model": ErrorEnvelope, "description": "Case or summary not found"},
    },
)
async def get_case_brief(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> CaseBriefResponse:
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
        )

    brief = await summary_service.get_case_brief(db, case_id)
    if brief is None or not brief.get("case_brief"):
        raise HTTPException(
            status_code=404,
            detail={"code": SUMMARY_NOT_FOUND, "message": "No summary has been generated for this case"},
        )

    return CaseBriefResponse(
        case_id=brief["case_id"],
        case_brief=brief["case_brief"],
        generated_at=brief["generated_at"],
    )