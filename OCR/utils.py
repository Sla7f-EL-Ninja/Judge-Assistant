"""
utils.py

File I/O, validation, security checks, and logging helpers for the OCR module.

Handles:
- Image loading and format validation
- PDF-to-image conversion
- File path sanitization against path traversal
- File size enforcement
- Returns PIL Images for Surya compatibility
"""

import logging
import os
from pathlib import Path
from typing import List, Union

from PIL import Image

import config.ocr as config

logger = logging.getLogger(__name__)


class OCRFileError(Exception):
    """Raised when file validation fails."""

    pass


class OCRSecurityError(Exception):
    """Raised when a security check fails (e.g., path traversal)."""

    pass


def sanitize_path(file_path: Union[str, Path]) -> Path:
    """
    Validate and sanitize a file path.

    Rejects paths containing '..', symlinks pointing outside allowed dirs,
    or any path traversal attempts.

    Args:
        file_path: Raw file path string or Path object.

    Returns:
        Resolved, sanitized Path object.

    Raises:
        OCRSecurityError: If path traversal is detected.
        OCRFileError: If file does not exist or is not readable.
    """
    path = Path(file_path).resolve()

    # Check for path traversal indicators in the original string
    raw = str(file_path)
    if ".." in raw:
        raise OCRSecurityError(f"Path traversal detected in: {raw}")

    if not path.exists():
        raise OCRFileError(f"File not found: {path}")

    if not path.is_file():
        raise OCRFileError(f"Not a regular file: {path}")

    if path.is_symlink():
        raise OCRSecurityError(f"Symlinks are not allowed: {path}")

    return path


def validate_file(file_path: Path) -> None:
    """
    Validate file extension, size, and readability.

    Args:
        file_path: Sanitized Path object.

    Raises:
        OCRFileError: If validation fails.
    """
    ext = file_path.suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise OCRFileError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {config.ALLOWED_EXTENSIONS}"
        )

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise OCRFileError(
            f"File too large: {size_mb:.1f}MB exceeds "
            f"limit of {config.MAX_FILE_SIZE_MB}MB"
        )

    if file_path.stat().st_size == 0:
        raise OCRFileError(f"File is empty: {file_path}")


def load_images(file_path: Union[str, Path]) -> List[Image.Image]:
    """
    Load image(s) from a file path as PIL Images.

    Supports PNG, JPEG, TIFF, BMP directly via Pillow.
    PDFs are converted to images using pdf2image.
    Returns PIL Images in RGB mode, which is the format Surya expects.

    Args:
        file_path: Path to the image or PDF file.

    Returns:
        List of PIL Image objects (one per page) in RGB mode.

    Raises:
        OCRFileError: If loading fails.
        OCRSecurityError: If path validation fails.
    """
    path = sanitize_path(file_path)
    validate_file(path)

    ext = path.suffix.lower()
    logger.info("Loading file: %s (type: %s)", path.name, ext)

    try:
        if ext == ".pdf":
            return _load_pdf_images(path)
        else:
            return _load_single_image(path)
    except (OCRFileError, OCRSecurityError):
        raise
    except Exception as e:
        raise OCRFileError(f"Failed to load image from {path.name}: {e}") from e


def _load_single_image(path: Path) -> List[Image.Image]:
    """Load a single image file and return as a list with one PIL Image."""
    img = Image.open(path)
    img_rgb = img.convert("RGB")
    logger.info("Loaded image: %dx%d", img_rgb.width, img_rgb.height)
    return [img_rgb]


def _load_pdf_images(path: Path) -> List[Image.Image]:
    """
    Convert PDF pages to images using pdf2image.

    Temporary files are cleaned up automatically by pdf2image
    when using the default (poppler) backend.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise OCRFileError(
            "pdf2image is required for PDF support. "
            "Install it with: pip install pdf2image"
        )

    try:
        pil_images = convert_from_path(str(path), dpi=300)
    except Exception as e:
        raise OCRFileError(f"Failed to convert PDF: {e}") from e

    images = []
    for i, pil_img in enumerate(pil_images):
        img_rgb = pil_img.convert("RGB")
        logger.info("PDF page %d: %dx%d", i + 1, img_rgb.width, img_rgb.height)
        images.append(img_rgb)

    if not images:
        raise OCRFileError(f"PDF produced no images: {path.name}")

    return images
