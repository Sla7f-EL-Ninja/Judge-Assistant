"""
minio_client.py

MinIO (S3-compatible) object storage client management.

Replaces local disk file storage with durable, scalable object storage.

``connect_minio`` is called from the FastAPI lifespan.
``get_minio`` returns the active client for dependency injection.
"""

import io
import logging
from typing import Optional

from minio import Minio
from minio.error import S3Error

from config.api import Settings

logger = logging.getLogger(__name__)

_client: Optional[Minio] = None
_bucket: Optional[str] = None


def connect_minio(settings: Settings) -> None:
    """Create the MinIO client and ensure the bucket exists."""
    global _client, _bucket
    _client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    _bucket = settings.minio_bucket

    # Ensure bucket exists
    try:
        if not _client.bucket_exists(_bucket):
            _client.make_bucket(_bucket)
            logger.info("Created MinIO bucket '%s'", _bucket)
        else:
            logger.info("MinIO bucket '%s' already exists", _bucket)
    except S3Error as exc:
        logger.error("MinIO bucket setup failed: %s", exc)
        raise

    logger.info("MinIO connected at %s (bucket=%s)", settings.minio_endpoint, _bucket)


def close_minio() -> None:
    """Clean up MinIO client (no persistent connection to close)."""
    global _client, _bucket
    _client = None
    _bucket = None


def get_minio() -> Optional[Minio]:
    """Return the active MinIO client, or None if not connected."""
    return _client


def get_bucket() -> str:
    """Return the configured bucket name."""
    if _bucket is None:
        raise RuntimeError("MinIO is not connected. Call connect_minio first.")
    return _bucket


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def upload_file(
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to MinIO and return the object name.

    Parameters
    ----------
    object_name : str
        The key/path for the object in the bucket.
    data : bytes
        File content to upload.
    content_type : str
        MIME type of the file.

    Returns
    -------
    str
        The object name (path) in the bucket.
    """
    if _client is None or _bucket is None:
        raise RuntimeError("MinIO is not connected. Call connect_minio first.")

    _client.put_object(
        bucket_name=_bucket,
        object_name=object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_name


def download_file(object_name: str) -> bytes:
    """Download a file from MinIO and return its contents as bytes."""
    if _client is None or _bucket is None:
        raise RuntimeError("MinIO is not connected. Call connect_minio first.")

    response = _client.get_object(_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_presigned_url(object_name: str, expires_seconds: int = 3600) -> str:
    """Generate a presigned URL for temporary file access.

    Parameters
    ----------
    object_name : str
        The key/path of the object.
    expires_seconds : int
        How long the URL should be valid (default 1 hour).

    Returns
    -------
    str
        A presigned URL that can be used to download the file.
    """
    from datetime import timedelta

    if _client is None or _bucket is None:
        raise RuntimeError("MinIO is not connected. Call connect_minio first.")

    return _client.presigned_get_object(
        _bucket, object_name, expires=timedelta(seconds=expires_seconds)
    )


def delete_file(object_name: str) -> None:
    """Delete a file from MinIO."""
    if _client is None or _bucket is None:
        raise RuntimeError("MinIO is not connected. Call connect_minio first.")

    _client.remove_object(_bucket, object_name)
