"""
reports.py

Async report generation endpoints:

  POST /api/v1/cases/{case_id}/reports/generate  — kick off pipeline, return job_id
  GET  /api/v1/cases/{case_id}/reports/{report_id} — poll job status / fetch result
  GET  /api/v1/cases/{case_id}/reports             — list historical report jobs
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_current_user, get_db
from api.errors import (
    CASE_NOT_FOUND, DOCUMENT_NOT_FOUND, NO_DOCUMENTS_FOR_CASE,
    REPORT_NOT_FOUND,
)
from api.schemas.common import ErrorEnvelope
from api.schemas.reports import (
    ReportJobAck, ReportJobListResponse, ReportJobStatusResponse, ReportJobSummary,
)
from api.schemas.summaries import SummaryResponse
from api.schemas.case_reasoning import CaseReasoningResponse
from api.services import case_service
from api.services import report_service
from api.services.summary_service import get_summary
from api.services.case_reasoning_service import get_case_reasoning
from api.db.collections import DOCUMENTS

logger = logging.getLogger("hakim.api.reports")

router = APIRouter(prefix="/api/v1/cases", tags=["Reports"])


@router.post(
    "/{case_id}/reports/generate",
    response_model=ReportJobAck,
    status_code=202,
    summary="Kick off async summarize + case-reasoning pipeline",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def generate_report(
    case_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ReportJobAck:
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
        )

    count = await db[DOCUMENTS].count_documents({"case_id": case_id})
    if count == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": NO_DOCUMENTS_FOR_CASE, "message": "No documents found for this case"},
        )

    job_id = await report_service.create_report_job(db, case_id)
    background_tasks.add_task(report_service.run_report_pipeline, db, case_id, job_id)

    return ReportJobAck(job_id=job_id, case_id=case_id)


@router.get(
    "/{case_id}/reports/{report_id}",
    response_model=ReportJobStatusResponse,
    summary="Poll report job status and fetch result when complete",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def get_report(
    case_id: str,
    report_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ReportJobStatusResponse:
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
        )

    job = await report_service.get_report_job(db, case_id, report_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"code": REPORT_NOT_FOUND, "message": "Report job not found"},
        )

    summary_resp = None
    reasoning_resp = None

    if job.get("status") == "completed":
        raw_summary = await get_summary(db, case_id)
        if raw_summary:
            summary_resp = SummaryResponse(
                case_id=raw_summary["case_id"],
                summary=raw_summary["summary"],
                generated_at=raw_summary["generated_at"],
                sources=raw_summary.get("sources", []),
            )

        raw_reasoning = await get_case_reasoning(db, case_id)
        if raw_reasoning:
            reasoning_resp = CaseReasoningResponse(**raw_reasoning)

    return ReportJobStatusResponse(
        report_id=job["report_id"],
        case_id=job["case_id"],
        status=job["status"],
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        summary=summary_resp,
        case_reasoning=reasoning_resp,
    )


@router.get(
    "/{case_id}/reports",
    response_model=ReportJobListResponse,
    summary="List historical report jobs for a case",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def list_reports(
    case_id: str,
    skip: int = 0,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ReportJobListResponse:
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
        )

    jobs, total = await report_service.list_report_jobs(db, case_id, skip, limit)
    return ReportJobListResponse(
        jobs=[
            ReportJobSummary(
                report_id=j["report_id"],
                status=j["status"],
                created_at=j["created_at"],
                completed_at=j.get("completed_at"),
            )
            for j in jobs
        ],
        total=total,
    )
