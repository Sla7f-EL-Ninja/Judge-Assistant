"""
tests/ocr/conftest.py
----------------------
Shared fixtures and pytest hooks for the OCR test suite.

Directory layout
----------------
project_root/
├── DocumentProcessor/OCR/
├── config/
└── tests/ocr/
    ├── conftest.py          ← this file
    ├── test_samples/        ← real documents (optional, gitignored)
    ├── test_models.py
    ├── test_gcv.py
    ├── test_llm.py
    └── test_pipeline.py

CLI flags
---------
--run-integration   Enable tests that make live GCV / LLM API calls.
                    Without this flag those tests are auto-skipped.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths — make project root importable regardless of invocation cwd
# ---------------------------------------------------------------------------
OCR_TESTS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR   = OCR_TESTS_DIR / "test_samples"
SAMPLES_DIR.mkdir(exist_ok=True)
PROJECT_ROOT  = OCR_TESTS_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

SAMPLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# CLI options & markers
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit live GCV / LLM APIs.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests requiring live GCV/LLM API calls "
        "(enable with --run-integration)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take > 10 s (live API calls)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="Pass --run-integration to run live API tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Helpers shared across modules
# ---------------------------------------------------------------------------

def make_batch_response(n: int, text: str = "sample text") -> MagicMock:
    """Build a mock batch_annotate_images response for *n* images."""
    responses = []
    for _ in range(n):
        r = MagicMock()
        r.error.message = ""
        r.full_text_annotation.text = text
        r.full_text_annotation.pages = []
        responses.append(r)
    resp = MagicMock()
    resp.responses = responses
    return resp


# ---------------------------------------------------------------------------
# Synthetic PDF fixture  (runs without any documents in test_samples/)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a minimal 3-page PDF from PIL images — no external deps.

    Page 1: Dense black rectangles → simulates a dense printed text page.
    Page 2: Sparse irregular marks → simulates a degraded / handwritten page.
    Page 3: Blank white           → simulates an empty / cover page.

    GCV can process all three; confidence scores will vary.
    """
    W, H = 1240, 1754  # A4 at ~150 DPI
    tmp  = tmp_path_factory.mktemp("synthetic_docs")
    pdf_path = tmp / "synthetic_test.pdf"
    pages: List[Image.Image] = []

    # Page 1 — printed-document style
    p1 = Image.new("RGB", (W, H), (255, 255, 255))
    d1 = ImageDraw.Draw(p1)
    d1.rectangle([60, 60, W - 60, 130], fill=(20, 20, 20))   # header bar
    for row, y in enumerate(range(180, H - 120, 42)):
        bar_w = 980 if row % 3 != 2 else 620
        d1.rectangle([120, y, 120 + bar_w, y + 14], fill=(15, 15, 15))
    pages.append(p1)

    # Page 2 — degraded / handwriting style
    p2 = Image.new("RGB", (W, H), (245, 245, 245))
    d2 = ImageDraw.Draw(p2)
    rng = random.Random(42)
    for i in range(28):
        y0 = 120 + i * 58
        x0 = rng.randint(90, 280)
        x1 = rng.randint(750, 1100)
        d2.line(
            [(x0, y0), (x1, y0 + rng.randint(-6, 6))],
            fill=(0, 0, 0),
            width=rng.randint(1, 3),
        )
    pages.append(p2)

    # Page 3 — blank
    pages.append(Image.new("RGB", (W, H), (255, 255, 255)))

    pages[0].save(
        str(pdf_path), "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    return pdf_path


@pytest.fixture(scope="session")
def synthetic_first_page(synthetic_pdf: Path) -> Image.Image:
    """First PIL Image from the synthetic PDF (used for unit GCV tests)."""
    from DocumentProcessor.OCR.ingestion import ingest_document
    return ingest_document(str(synthetic_pdf))[0]


# ---------------------------------------------------------------------------
# Real-document fixtures  (fall back to synthetic when test_samples/ empty)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_files() -> List[Path]:
    return sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS
    )


@pytest.fixture(scope="session")
def first_sample(sample_files: List[Path], synthetic_pdf: Path) -> Path:
    """First real sample, or synthetic PDF when test_samples/ is empty."""
    return sample_files[0] if sample_files else synthetic_pdf


@pytest.fixture(scope="session")
def pil_first_page(first_sample: Path) -> Image.Image:
    from DocumentProcessor.OCR.ingestion import ingest_document
    pages = ingest_document(str(first_sample))
    assert pages, f"ingest_document returned no pages for {first_sample.name}"
    return pages[0]


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gcv_engine_unit():
    """GCVEngine with a fully mocked GCP client — no network, no credentials."""
    from DocumentProcessor.OCR.gcv_engine import GCVEngine

    engine = object.__new__(GCVEngine)   # bypass __init__, no auth needed
    engine._vision = MagicMock()
    engine._feature_type = MagicMock()
    engine._client = MagicMock()
    return engine


@pytest.fixture(scope="session")
def gcv_engine():
    """Real GCVEngine singleton — skips if credentials are missing."""
    cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not cred or not Path(cred).exists():
        pytest.skip("GOOGLE_APPLICATION_CREDENTIALS not set or file missing")
    from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
    return get_gcv_engine()
