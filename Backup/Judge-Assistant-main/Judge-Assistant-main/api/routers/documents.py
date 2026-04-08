"""
documents.py

POST /api/v1/cases/{case_id}/documents -- document ingestion endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.schemas.common import ErrorEnvelope
from api.schemas.documents import IngestRequest, IngestResponse, DocumentListResponse
from api.services import case_service, document_service

router = APIRouter(prefix="/api/v1/cases", tags=["Documents"])


@router.post(
    "/{case_id}/documents",
    response_model=IngestResponse,
    summary="Ingest documents into a case",
    description=(
        "Run the document ingestion pipeline (OCR, classification, vector indexing) "
        "for the specified file IDs. The case must exist and belong to the authenticated user. "
        "This operation can take 10-30 seconds per file."
    ),
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        404: {"model": ErrorEnvelope, "description": "Case not found"},
        422: {"model": ErrorEnvelope, "description": "Request validation error (e.g. empty file_ids)"},
    },
)
async def ingest_documents(
    case_id: str,
    body: IngestRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Pre-load documents into a case: OCR, classify, store, and index."""
    # Verify the case exists and belongs to the user
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    result = await document_service.ingest_files(
        db=db,
        settings=settings,
        case_id=case_id,
        file_ids=body.file_ids,
    )
    return IngestResponse(**result)


@router.get(
    "/{case_id}/documents",
    response_model=DocumentListResponse,
    summary="List documents in a case",
    description=(
        "Returns all documents that have been ingested into the case. "
        "The case must exist and belong to the authenticated user."
    ),
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        404: {"model": ErrorEnvelope, "description": "Case not found"},
        422: {"model": ErrorEnvelope, "description": "Request validation error"},
    },
)
async def list_documents(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Return all documents ingested into a case."""
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    docs = await document_service.list_documents(db=db, case_id=case_id)
    return DocumentListResponse(documents=docs, total=len(docs))