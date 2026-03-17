"""
documents.py

POST /api/v1/cases/{case_id}/documents -- document ingestion endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.schemas.documents import IngestRequest, IngestResponse
from api.services import case_service, document_service

router = APIRouter(prefix="/api/v1/cases", tags=["Documents"])


@router.post(
    "/{case_id}/documents",
    response_model=IngestResponse,
    summary="Ingest documents into a case",
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
