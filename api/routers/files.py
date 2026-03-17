"""
files.py

POST /api/v1/files/upload -- file upload endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.schemas.files import FileUploadResponse
from api.services.file_service import save_upload

router = APIRouter(prefix="/api/v1/files", tags=["Files"])


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
)
async def upload_file(
    file: UploadFile,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Accept a file upload, validate it, store on disk, and return metadata."""
    content = await file.read()
    mime_type = file.content_type or "application/octet-stream"

    try:
        doc = await save_upload(
            db=db,
            settings=settings,
            filename=file.filename or "unnamed",
            content=content,
            mime_type=mime_type,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return FileUploadResponse(
        file_id=doc["_id"],
        filename=doc["filename"],
        size_bytes=doc["size_bytes"],
        mime_type=doc["mime_type"],
        uploaded_at=doc["uploaded_at"],
    )
