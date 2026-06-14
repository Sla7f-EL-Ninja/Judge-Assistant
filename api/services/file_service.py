"""
file_service.py

File upload handling: validation, storage (MinIO or local disk), and metadata persistence.

Uses MinIO (S3-compatible) for production file storage. Falls back to local
disk if MinIO is not connected.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.db.collections import FILES

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_file_id() -> str:
    return f"file_{uuid.uuid4().hex[:12]}"


def _write_file_local(path: str, data: bytes) -> None:
    """Write file to local disk (fallback when MinIO is unavailable)."""
    with open(path, "wb") as f:
        f.write(data)


def _detect_mime(content: bytes, reported: str) -> str:
    """Detect real MIME type from magic bytes; fall back to browser-reported value.

    Browsers (and download managers on some OSes) occasionally report the wrong
    MIME type when uploading — e.g. a PDF file reported as ``image/jpeg``.
    Reading the first few bytes is fast and reliable for the formats we care about.
    """
    if content[:4] == b"%PDF":
        return "application/pdf"
    if content[:2] == b"\xff\xd8":          # JPEG SOI marker
        return "image/jpeg"
    if content[:8] == b"\x89PNG\r\n\x1a\n": # PNG signature
        return "image/png"
    if content[:4] in (b"GIF8", ):           # GIF87a / GIF89a
        return "image/gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return reported  # unknown — trust what the browser said


def _upload_to_minio(object_name: str, data: bytes, mime_type: str) -> str:
    """Upload file to MinIO. Returns the object name."""
    from api.db.minio_client import upload_file

    return upload_file(object_name, data, content_type=mime_type)


async def save_upload(
    db: AsyncIOMotorDatabase,
    settings: Settings,
    filename: str,
    content: bytes,
    mime_type: str,
    user_id: str,
) -> dict:
    """Validate, persist to storage, and record metadata in MongoDB.

    Returns the file metadata document.

    Tries MinIO first; falls back to local disk if MinIO is not available.

    Raises ``ValueError`` for invalid MIME type or oversized files.
    """
    # Detect real MIME from magic bytes — overrides unreliable browser-reported type
    mime_type = _detect_mime(content, mime_type)
    logger.info("Detected mime_type after magic-byte check: '%s'", mime_type)

    # Validate MIME type
    logger.info("Received mime_type: '%s'", mime_type)
    logger.info("Allowed types: %s", settings.allowed_mime_type_list)
    if mime_type not in settings.allowed_mime_type_list:
        raise ValueError(
            f"MIME type '{mime_type}' is not allowed. "
            f"Accepted: {settings.allowed_mime_type_list}"
        )

    # Validate size
    size = len(content)
    if size > settings.max_upload_bytes:
        raise ValueError(
            f"File size {size} bytes exceeds maximum {settings.max_upload_bytes} bytes"
        )

    file_id = _new_file_id()
    ext = os.path.splitext(filename)[1]
    disk_name = f"{file_id}{ext}"
    now = _now()

    # Try MinIO first, fall back to local disk
    storage_backend = "local"
    disk_path = ""
    minio_object = ""

    try:
        from api.db.minio_client import get_minio

        minio_client = get_minio()
        if minio_client is not None:
            object_name = f"{user_id}/{file_id}/{disk_name}"
            await asyncio.to_thread(
                _upload_to_minio, object_name, content, mime_type
            )
            minio_object = object_name
            storage_backend = "minio"
            logger.info("File '%s' uploaded to MinIO: %s", filename, object_name)
        else:
            raise RuntimeError("MinIO client not available")
    except Exception as exc:
        # Fall back to local disk
        logger.info(
            "MinIO unavailable (%s), falling back to local disk for '%s'",
            exc, filename,
        )
        os.makedirs(settings.upload_dir, exist_ok=True)
        disk_path = os.path.join(settings.upload_dir, disk_name)
        await asyncio.to_thread(_write_file_local, disk_path, content)
        storage_backend = "local"

    doc = {
        "_id": file_id,
        "user_id": user_id,
        "filename": filename,
        "disk_path": disk_path,
        "minio_object": minio_object,
        "storage_backend": storage_backend,
        "size_bytes": size,
        "mime_type": mime_type,
        "uploaded_at": now,
    }
    await db[FILES].insert_one(doc)
    return doc


async def get_file_record(
    db: AsyncIOMotorDatabase, file_id: str
) -> Optional[dict]:
    """Fetch file metadata from MongoDB."""
    return await db[FILES].find_one({"_id": file_id})


def _disk_chunk_generator(path: str, chunk_size: int = 8192):
    """Sync generator that reads a local file in chunks."""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def open_file_stream(
    db: AsyncIOMotorDatabase,
    file_id: str,
) -> Optional[Tuple[AsyncIterator[bytes], dict]]:
    """Return a streaming iterator and file metadata, or None if not found.

    Caller is responsible for setting Content-Type / Content-Length headers
    from the returned metadata dict.
    """
    from starlette.concurrency import iterate_in_threadpool

    file_rec = await db[FILES].find_one({"_id": file_id})
    if file_rec is None:
        return None

    _filename = file_rec.get("filename", "file")
    meta = {
        "mime_type": file_rec.get("mime_type", "application/octet-stream"),
        "filename": _filename,
        "size_bytes": file_rec.get("size_bytes", 0),
        # inline disposition tells IDM (and browsers) NOT to treat this as a
        # file download — critical for PDF.js to receive the bytes in-page.
        "content_disposition": f'inline; filename="{_filename}"',
    }

    if file_rec.get("storage_backend") == "minio" and file_rec.get("minio_object"):
        try:
            from api.db.minio_client import get_minio, get_bucket, stream_file

            minio_client = get_minio()
            if minio_client is not None:
                bucket = get_bucket()
                sync_gen = stream_file(file_rec["minio_object"])
                return iterate_in_threadpool(sync_gen), meta
        except Exception as exc:
            logger.warning("MinIO stream failed for %s: %s — trying local disk", file_id, exc)

    disk_path = file_rec.get("disk_path", "")
    if disk_path and os.path.exists(disk_path):
        sync_gen = _disk_chunk_generator(disk_path)
        return iterate_in_threadpool(sync_gen), meta

    return None


async def get_file_bytes(
    db: AsyncIOMotorDatabase,
    file_id: str,
) -> Optional[Tuple[bytes, dict]]:
    """Return the full file content as bytes along with metadata.

    Used when the caller needs the raw bytes before responding
    (e.g. image format conversion for browser rendering).
    Returns None if the file record or its storage object is missing.
    """
    file_rec = await db[FILES].find_one({"_id": file_id})
    if file_rec is None:
        return None

    _filename = file_rec.get("filename", "file")
    meta = {
        "mime_type": file_rec.get("mime_type", "application/octet-stream"),
        "filename": _filename,
        "size_bytes": file_rec.get("size_bytes", 0),
        "content_disposition": f'inline; filename="{_filename}"',
    }

    if file_rec.get("storage_backend") == "minio" and file_rec.get("minio_object"):
        try:
            from api.db.minio_client import get_minio, get_bucket

            minio_client = get_minio()
            if minio_client is not None:
                bucket = get_bucket()
                object_name = file_rec["minio_object"]

                def _read_minio() -> bytes:
                    response = minio_client.get_object(bucket, object_name)
                    try:
                        return response.read()
                    finally:
                        response.close()
                        response.release_conn()

                data = await asyncio.to_thread(_read_minio)
                return data, meta
        except Exception as exc:
            logger.warning(
                "MinIO get_bytes failed for %s: %s — trying local disk", file_id, exc
            )

    disk_path = file_rec.get("disk_path", "")
    if disk_path and os.path.exists(disk_path):
        def _read_disk() -> bytes:
            with open(disk_path, "rb") as f:
                return f.read()

        data = await asyncio.to_thread(_read_disk)
        return data, meta

    return None


async def delete_file(
    db: AsyncIOMotorDatabase,
    file_id: str,
    user_id: str,
) -> bool:
    """Delete a file from storage and MongoDB. Returns True if found+deleted."""
    file_rec = await db[FILES].find_one({"_id": file_id})
    if file_rec is None:
        return False

    # Delete from MinIO
    if file_rec.get("storage_backend") == "minio" and file_rec.get("minio_object"):
        try:
            from api.db.minio_client import get_minio, get_bucket
            minio_client = get_minio()
            if minio_client:
                bucket = get_bucket()
                await asyncio.to_thread(
                    minio_client.remove_object, bucket, file_rec["minio_object"]
                )
        except Exception as exc:
            logger.warning("MinIO delete failed for %s: %s", file_rec.get("minio_object"), exc)

    # Delete from local disk
    if file_rec.get("disk_path") and os.path.exists(file_rec["disk_path"]):
        try:
            await asyncio.to_thread(os.remove, file_rec["disk_path"])
        except Exception as exc:
            logger.warning("Local disk delete failed for %s: %s", file_rec.get("disk_path"), exc)

    await db[FILES].delete_one({"_id": file_id})
    return True