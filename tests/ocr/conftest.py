# """
# tests/conftest.py
# ------------------
# Shared pytest fixtures and configuration.

# CLI flags
# ---------
# --run-integration   Enable integration tests that require Mongo, Qdrant,
#                     and all other services to be running.
#                     Without this flag, @pytest.mark.integration tests are
#                     automatically skipped.

# Sample files
# ------------
# Drop any .pdf, .png, .jpg, .jpeg, .tiff, or .bmp file into tests/test_samples/
# before running.  Tests that need a sample file are auto-skipped when the
# folder is empty.
# """

# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import List

# import pytest

# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------

# TESTS_DIR = Path(__file__).resolve().parent
# SAMPLES_DIR = TESTS_DIR / "test_samples"
# SAMPLES_DIR.mkdir(exist_ok=True)

# SAMPLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


# # ---------------------------------------------------------------------------
# # CLI option
# # ---------------------------------------------------------------------------

# def pytest_addoption(parser: pytest.Parser) -> None:
#     parser.addoption(
#         "--run-integration",
#         action="store_true",
#         default=False,
#         help="Run integration tests that require MongoDB, Qdrant, and all services.",
#     )


# def pytest_configure(config: pytest.Config) -> None:
#     config.addinivalue_line(
#         "markers",
#         "integration: end-to-end tests requiring MongoDB, Qdrant, and all services",
#     )


# def pytest_collection_modifyitems(
#     config: pytest.Config,
#     items: List[pytest.Item],
# ) -> None:
#     """Skip integration tests unless --run-integration is passed."""
#     if config.getoption("--run-integration"):
#         return  # run everything

#     skip_integration = pytest.mark.skip(
#         reason="Pass --run-integration to run end-to-end tests"
#     )
#     for item in items:
#         if "integration" in item.keywords:
#             item.add_marker(skip_integration)


# # ---------------------------------------------------------------------------
# # Fixtures
# # ---------------------------------------------------------------------------

# @pytest.fixture(scope="session")
# def sample_files() -> List[Path]:
#     """Return all supported sample files found in tests/test_samples/.

#     Tests that depend on this fixture are automatically skipped when the
#     folder is empty.
#     """
#     files = sorted(
#         p for p in SAMPLES_DIR.iterdir()
#         if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS
#     )
#     return files


# @pytest.fixture(scope="session")
# def first_sample(sample_files: List[Path]) -> Path:
#     """Return the first available sample file, or skip if none exist."""
#     if not sample_files:
#         pytest.skip(
#             f"No sample files found in {SAMPLES_DIR}. "
#             "Drop a .pdf or image file there and re-run."
#         )
#     return sample_files[0]


# @pytest.fixture(scope="session")
# def gcv_engine():
#     """Return the singleton GCVEngine — skips if GCV credentials are missing."""
#     cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
#     if not cred_path or not Path(cred_path).exists():
#         pytest.skip(
#             "GOOGLE_APPLICATION_CREDENTIALS is not set or file not found. "
#             "Set it in your .env file."
#         )
#     from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
#     return get_gcv_engine()


# @pytest.fixture(scope="session")
# def pil_first_page(first_sample: Path):
#     """Return the first PIL Image from the first sample file."""
#     from DocumentProcessor.OCR.ingestion import ingest_document
#     pages = ingest_document(str(first_sample))
#     assert pages, f"ingest_document returned no pages for {first_sample.name}"
#     return pages[0]


"""
tests/ocr/conftest.py
----------------------
Shared pytest fixtures and configuration.

Project layout assumed
----------------------
project_root/
├── Case Sample/
├── DocumentProcessor/
├── config/
├── api/
└── tests/
    └── ocr/
        ├── conftest.py          ← this file
        ├── test_ocr_pipeline.py
        └── test_samples/        ← drop your documents here

CLI flags
---------
--run-integration   Enable integration tests that require Mongo, Qdrant,
                    and all other services to be running.
                    Without this flag, @pytest.mark.integration tests are
                    automatically skipped.

Sample files
------------
Drop any .pdf, .png, .jpg, .jpeg, .tiff, .bmp, or .webp file into
tests/ocr/test_samples/ before running.  Tests that need a sample file
are auto-skipped when the folder is empty.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# tests/ocr/
OCR_TESTS_DIR = Path(__file__).resolve().parent

# tests/ocr/test_samples/
SAMPLES_DIR = OCR_TESTS_DIR / "test_samples"
SAMPLES_DIR.mkdir(exist_ok=True)

# project_root/  (two levels up: ocr → tests → project_root)
PROJECT_ROOT = OCR_TESTS_DIR.parent.parent

# Make project root importable so `DocumentProcessor`, `config`, `api` etc.
# can all be found regardless of how pytest is invoked.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (picks up GOOGLE_APPLICATION_CREDENTIALS etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed — rely on shell environment

SAMPLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require MongoDB, Qdrant, and all services.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests requiring MongoDB, Qdrant, and all services",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return  # run everything

    skip_integration = pytest.mark.skip(
        reason="Pass --run-integration to run end-to-end tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_files() -> List[Path]:
    """Return all supported sample files found in tests/ocr/test_samples/.

    Tests that depend on this fixture are automatically skipped when the
    folder is empty.
    """
    files = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS
    )
    return files


@pytest.fixture(scope="session")
def first_sample(sample_files: List[Path]) -> Path:
    """Return the first available sample file, or skip if none exist."""
    if not sample_files:
        pytest.skip(
            f"No sample files found in {SAMPLES_DIR}. "
            "Drop a .pdf or image file there and re-run."
        )
    return sample_files[0]


@pytest.fixture(scope="session")
def gcv_engine():
    """Return the singleton GCVEngine — skips if GCV credentials are missing."""
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not cred_path or not Path(cred_path).exists():
        pytest.skip(
            "GOOGLE_APPLICATION_CREDENTIALS is not set or file not found. "
            "Set it in your .env file."
        )
    from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
    return get_gcv_engine()


@pytest.fixture(scope="session")
def pil_first_page(first_sample: Path):
    """Return the first PIL Image from the first sample file."""
    from DocumentProcessor.OCR.ingestion import ingest_document
    pages = ingest_document(str(first_sample))
    assert pages, f"ingest_document returned no pages for {first_sample.name}"
    return pages[0]