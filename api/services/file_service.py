"""
file_service.py

File upload handling: validation, storage to disk, and metadata persistence.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional
import asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.db.collections import FILES


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_file_id() -> str:
    return f"file_{uuid.uuid4().hex[:12]}"

def _write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)

async def save_upload(
    db: AsyncIOMotorDatabase,
    settings: Settings,
    filename: str,
    content: bytes,
    mime_type: str,
    user_id: str,
) -> dict:
    """Validate, persist to disk, and record metadata in MongoDB.

    Returns the file metadata document.

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

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    file_id = _new_file_id()
    ext = os.path.splitext(filename)[1]
    disk_name = f"{file_id}{ext}"
    disk_path = os.path.join(settings.upload_dir, disk_name)

    await asyncio.to_thread(_write_file, disk_path, content)

    now = _now()
    doc = {
        "_id": file_id,
        "user_id": user_id,
        "filename": filename,
        "disk_path": disk_path,
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
