"""
files.py

POST   /api/v1/files/upload      -- file upload endpoint.
GET    /api/v1/files/{file_id}   -- stream raw file for browser rendering.
DELETE /api/v1/files/{file_id}   -- delete uploaded file.

Browser rendering support:
  - application/pdf  → streamed inline; Accept-Ranges: bytes enables seek/page-jump.
  - image/png, image/jpeg, image/webp → streamed inline, natively supported.
  - image/tiff, image/bmp → converted to PNG on-the-fly; browsers cannot render
    these natively, so we read the full bytes, convert with Pillow, and respond
    with image/png. The Content-Disposition filename keeps the original name.
"""

import asyncio
import io
import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.dependencies import get_current_user, get_db, get_settings
from api.errors import FILE_NOT_FOUND
from api.schemas.common import ErrorEnvelope, MessageResponse
from api.schemas.files import FileUploadResponse
from api.services.file_service import (
    delete_file,
    get_file_bytes,
    open_file_stream,
    save_upload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files", tags=["Files"])

# MIME types that require server-side conversion before the browser can display them.
# Pillow handles both; output is always image/png.
_CONVERT_TO_PNG: frozenset[str] = frozenset({"image/tiff", "image/bmp"})


def _content_disposition(filename: str, disposition: str) -> str:
    """Build a RFC 5987-compliant Content-Disposition header value."""
    ascii_name = filename.encode("ascii", "replace").decode()
    utf8_name = quote(filename)
    return f'{disposition}; filename="{ascii_name}"; filename*=UTF-8\'\'{utf8_name}'


async def _convert_image_to_png(data: bytes) -> bytes:
    """Convert raw image bytes (TIFF or BMP) to PNG using Pillow."""
    from PIL import Image

    def _do_convert() -> bytes:
        img = Image.open(io.BytesIO(data))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return await asyncio.to_thread(_do_convert)


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
        "render PDFs, images, etc. inline.\n\n"
        "- **PDF**: streamed with `Accept-Ranges: bytes` so browsers can jump to pages.\n"
        "- **PNG / JPEG / WEBP**: streamed as-is.\n"
        "- **TIFF / BMP**: converted to PNG on-the-fly (browsers cannot render these natively).\n\n"
        "Pass `?download=1` to force an attachment download instead of inline rendering."
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
    disposition = "attachment" if download else "inline"

    # ------------------------------------------------------------------ #
    # TIFF / BMP — convert to PNG; browsers cannot render these natively  #
    # ------------------------------------------------------------------ #
    file_rec_check = await db["files"].find_one({"_id": file_id}, {"mime_type": 1})
    if file_rec_check and file_rec_check.get("mime_type") in _CONVERT_TO_PNG:
        result = await get_file_bytes(db, file_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": FILE_NOT_FOUND, "message": "File not found"},
            )
        raw_bytes, meta = result
        try:
            png_bytes = await _convert_image_to_png(raw_bytes)
        except Exception as exc:
            logger.exception("Image conversion failed for file %s: %s", file_id, exc)
            raise HTTPException(
                status_code=500,
                detail={"code": "IMAGE_CONVERSION_FAILED", "message": str(exc)},
            ) from exc

        # Swap extension in filename so the browser knows what it's getting
        original_name = meta["filename"]
        stem = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
        display_name = f"{stem}.png"

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": _content_disposition(display_name, disposition),
                "Content-Length": str(len(png_bytes)),
            },
        )

    # ------------------------------------------------------------------ #
    # PDF / PNG / JPEG / WEBP — stream directly                           #
    # ------------------------------------------------------------------ #
    result = await open_file_stream(db, file_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"code": FILE_NOT_FOUND, "message": "File not found"},
        )

    stream, meta = result
    headers = {
        "Content-Disposition": _content_disposition(meta["filename"], disposition),
        "Content-Length": str(meta["size_bytes"]),
    }

    # PDF: Accept-Ranges lets browsers seek to arbitrary byte offsets,
    # which is required for page-jump and embedded PDF viewer to work.
    if meta["mime_type"] == "application/pdf":
        headers["Accept-Ranges"] = "bytes"

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