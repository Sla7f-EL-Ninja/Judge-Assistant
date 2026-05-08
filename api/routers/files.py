"""
files.py

POST /api/v1/files/upload -- file upload endpoint.
GET  /api/v1/files/{file_id} -- stream raw file for browser rendering.
DELETE /api/v1/files/{file_id} -- delete uploaded file.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.errors import FILE_NOT_FOUND
from api.schemas.common import ErrorEnvelope, MessageResponse
from api.schemas.files import FileUploadResponse
from api.services.file_service import save_upload, delete_file, open_file_stream
from urllib.parse import quote
router = APIRouter(prefix="/api/v1/files", tags=["Files"])


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description=(
        "Upload a file for later ingestion into a case. Supported MIME types: "
        "application/pdf, image/png, image/jpeg, image/tiff, image/bmp, image/webp. "
        "Maximum file size: 20 MB."
    ),
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid MIME type or file too large"},
        401: {"model": ErrorEnvelope, "description": "Missing or invalid JWT token"},
        422: {"model": ErrorEnvelope, "description": "Request validation error"},
    },
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


@router.get(
    "/{file_id}",
    summary="Stream a raw file for browser rendering",
    description=(
        "Returns the raw file bytes with the correct Content-Type so browsers can "
        "render PDFs, images, etc. inline. Pass ?download=1 to force attachment download."
    ),
    responses={
        200: {"description": "File stream"},
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope, "description": "File not found"},
    },
)
async def get_file(
    file_id: str,
    download: bool = Query(default=False, description="Force attachment download"),
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    result = await open_file_stream(db, file_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"code": FILE_NOT_FOUND, "message": "File not found"},
        )

    stream, meta = result
    disposition = "attachment" if download else "inline"
    headers = {
        "Content-Disposition": (
        f"{disposition}; "
        f'filename="{meta["filename"].encode("ascii", "replace").decode()}"; '
        f"filename*=UTF-8''{quote(meta['filename'])}"
    ),
        "Content-Length": str(meta["size_bytes"]),
    }
    return StreamingResponse(
        content=stream,
        media_type=meta["mime_type"],
        headers=headers,
    )


@router.delete(
    "/{file_id}",
    response_model=MessageResponse,
    summary="Delete an uploaded file",
    responses={
        401: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope, "description": "File not found"},
    },
)
async def delete_file_endpoint(
    file_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    found = await delete_file(db, file_id, user_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail={"code": FILE_NOT_FOUND, "message": "File not found"},
        )
    return MessageResponse(message="File deleted")
