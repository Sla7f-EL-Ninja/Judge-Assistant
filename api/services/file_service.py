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
from typing import Optional

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
    # Validate MIME type
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
