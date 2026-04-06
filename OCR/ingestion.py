"""
OCR.ingestion
-------------
Document loading: PDF and image ingestion.

Converts a file path into a list of PIL Images (one per page).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Set

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
}
SUPPORTED_PDF_EXTENSIONS: Set[str] = {".pdf"}


def ingest_document(
    file_path: str,
    pdf_dpi: int = 400,
    max_file_size_mb: int = 50,
    allowed_extensions: List[str] | None = None,
) -> List[Image.Image]:
    """Load a PDF or image file and return a list of PIL Images (one per page).

    Parameters
    ----------
    file_path:
        Path to the document file.
    pdf_dpi:
        DPI to use when converting PDF pages to images.
    max_file_size_mb:
        Maximum allowed file size in megabytes.
    allowed_extensions:
        Optional list of allowed file extensions. If ``None``, uses the
        built-in supported extensions.

    Returns
    -------
    list[PIL.Image.Image]
        One image per page.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is unsupported or exceeds the size limit.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Size check
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValueError(
            f"File size ({file_size_mb:.1f} MB) exceeds limit ({max_file_size_mb} MB)"
        )

    suffix = path.suffix.lower()

    # Extension validation
    if allowed_extensions is not None:
        allowed = {ext.lower() for ext in allowed_extensions}
    else:
        allowed = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

    if suffix not in allowed:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return _ingest_pdf(path, pdf_dpi)
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return _ingest_image(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _ingest_pdf(path: Path, pdf_dpi: int) -> List[Image.Image]:
    """Convert a PDF file to a list of RGB PIL Images."""
    from pdf2image import convert_from_path

    logger.info("Converting PDF to images at %d DPI: %s", pdf_dpi, path.name)
    pil_pages = convert_from_path(str(path), dpi=pdf_dpi)
    pages = [p.convert("RGB") if p.mode != "RGB" else p for p in pil_pages]
    logger.info("PDF converted: %d page(s)", len(pages))
    return pages


def _ingest_image(path: Path) -> List[Image.Image]:
    """Load a single image file and return it as a one-element list."""
    logger.info("Loading image: %s", path.name)
    img = Image.open(path)
    img.verify()
    img = Image.open(path)  # re-open after verify
    if img.mode != "RGB":
        img = img.convert("RGB")
    logger.info("Image loaded: %dx%d", img.size[0], img.size[1])
    return [img]
