"""
documents.py

Document endpoints:
  POST   /api/v1/cases/{case_id}/documents                     -- ingest
  GET    /api/v1/cases/{case_id}/documents                     -- list
  GET    /api/v1/cases/{case_id}/documents/{doc_id}            -- detail
  DELETE /api/v1/cases/{case_id}/documents/{doc_id}            -- delete
  GET    /api/v1/cases/{case_id}/documents/{doc_id}/ocr        -- get OCR text
  PATCH  /api/v1/cases/{case_id}/documents/{doc_id}/ocr        -- correct OCR text
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.errors import (
    CASE_NOT_FOUND, DOCUMENT_NOT_FOUND, QDRANT_REINDEX_FAILED, INTERNAL_ERROR
)
from api.schemas.common import ErrorEnvelope, MessageResponse
from api.schemas.documents import (
    IngestRequest, IngestResponse, DocumentListResponse,
    DocumentDetailResponse, ClassificationDetail,
    OCRTextResponse, OCRCorrectionRequest,
)
from api.services import case_service, document_service

router = APIRouter(prefix="/api/v1/cases", tags=["Documents"])


def _case_not_found():
    raise HTTPException(
        status_code=404,
        detail={"code": CASE_NOT_FOUND, "message": "Case not found"},
    )


def _doc_not_found():
    raise HTTPException(
        status_code=404,
        detail={"code": DOCUMENT_NOT_FOUND, "message": "Document not found"},
    )


@router.post(
    "/{case_id}/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents into a case",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope, "description": "Case not found"},
        422: {"model": ErrorEnvelope},
    },
)
async def ingest_documents(
    case_id: str,
    body: IngestRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    result = await document_service.ingest_files(
        db=db, settings=settings, case_id=case_id, file_ids=body.file_ids,
    )
    return IngestResponse(**result)


@router.get(
    "/{case_id}/documents",
    response_model=DocumentListResponse,
    summary="List documents in a case",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def list_documents(
    case_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    docs = await document_service.list_documents(db=db, case_id=case_id)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/{case_id}/documents/{doc_id}",
    response_model=DocumentDetailResponse,
    summary="Get document detail",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def get_document(
    case_id: str,
    doc_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    doc = await document_service.get_document(db, case_id, doc_id)
    if doc is None:
        _doc_not_found()

    text = doc.get("text", "")
    classification_raw = {
        "final_type": doc.get("doc_type", ""),
        "confidence": doc.get("classification_confidence", 0.0),
        "explanation": doc.get("classification_explanation", ""),
    }
    return DocumentDetailResponse(
        id=doc["id"],
        title=doc.get("title", ""),
        source_file=doc.get("source_file", ""),
        doc_type=doc.get("doc_type"),
        file_type=doc.get("file_type"),
        file_id=doc.get("file_id"),
        created_at=doc.get("created_at"),
        text_excerpt=text[:500],
        classification=ClassificationDetail(**classification_raw),
        storage_backend=doc.get("storage_backend"),
        minio_object=doc.get("minio_object"),
        qdrant_chunks=doc.get("qdrant_chunks", 0),
        corrected=doc.get("corrected", False),
        corrected_at=doc.get("corrected_at"),
    )


@router.delete(
    "/{case_id}/documents/{doc_id}",
    response_model=MessageResponse,
    summary="Delete a document",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def delete_document(
    case_id: str,
    doc_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    doc = await document_service.get_document(db, case_id, doc_id)
    if doc is None:
        _doc_not_found()

    await document_service.delete_document(db, case_id, doc_id)
    return MessageResponse(message="Document deleted")


@router.get(
    "/{case_id}/documents/{doc_id}/ocr",
    response_model=OCRTextResponse,
    summary="Get stored OCR text for a document",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
async def get_ocr_text(
    case_id: str,
    doc_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    doc = await document_service.get_document_ocr(db, case_id, doc_id)
    if doc is None:
        _doc_not_found()

    classification_raw = {
        "final_type": doc.get("doc_type", ""),
        "confidence": doc.get("classification_confidence", 0.0),
        "explanation": doc.get("classification_explanation", ""),
    }
    return OCRTextResponse(
        doc_id=doc["id"],
        file_id=doc.get("file_id"),
        file_type=doc.get("file_type"),
        source_file=doc.get("source_file", ""),
        text=doc.get("text", ""),
        classification=ClassificationDetail(**classification_raw),
        corrected=doc.get("corrected", False),
        corrected_at=doc.get("corrected_at"),
        original_text=doc.get("original_text"),
    )


@router.patch(
    "/{case_id}/documents/{doc_id}/ocr",
    response_model=OCRTextResponse,
    summary="Correct stored OCR text and re-index",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope},
    },
)
async def correct_ocr_text(
    case_id: str,
    doc_id: str,
    body: OCRCorrectionRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    case = await case_service.get_case(db, case_id, user_id)
    if case is None:
        _case_not_found()

    try:
        doc = await document_service.correct_document_ocr(
            db, case_id, doc_id, body.text, body.corrected_by
        )
    except RuntimeError as exc:
        if "QDRANT_REINDEX_FAILED" in str(exc):
            raise HTTPException(
                status_code=500,
                detail={"code": QDRANT_REINDEX_FAILED, "message": str(exc)},
            )
        raise HTTPException(
            status_code=500,
            detail={"code": INTERNAL_ERROR, "message": str(exc)},
        )

    if doc is None:
        _doc_not_found()

    classification_raw = {
        "final_type": doc.get("doc_type", ""),
        "confidence": doc.get("classification_confidence", 0.0),
        "explanation": doc.get("classification_explanation", ""),
    }
    return OCRTextResponse(
        doc_id=doc["id"],
        file_id=doc.get("file_id"),
        file_type=doc.get("file_type"),
        source_file=doc.get("source_file", ""),
        text=doc.get("text", ""),
        classification=ClassificationDetail(**classification_raw),
        corrected=doc.get("corrected", False),
        corrected_at=doc.get("corrected_at"),
        original_text=doc.get("original_text"),
    )
