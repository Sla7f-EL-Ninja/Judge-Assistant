"""
tests/test_ocr_pipeline.py
---------------------------
Test suite for the GCV + LLM refinement OCR pipeline.

Structure
---------
Section 1 — Unit / OCR-only (fast, no external services)
    1a  Ingestion
    1b  Image preprocessing (restore + perspective)
    1c  GCV engine — single page OCR
    1d  Word confidence model and band assignment
    1e  LLM refinement
    1f  run_ocr() — full OCR pipeline (GCV + refine, no Mongo/Qdrant)

Section 2 — Integration (requires --run-integration flag + all services up)
    2a  process_document_group() — full ingestion into Mongo + Qdrant
    2b  API schema round-trip — OCRTextResponse serialisation

How to run
----------
# Fast OCR-only tests (default):
    pytest tests/test_ocr_pipeline.py

# Everything including full pipeline:
    pytest tests/test_ocr_pipeline.py --run-integration

# One specific section:
    pytest tests/test_ocr_pipeline.py -k "ingestion"
    pytest tests/test_ocr_pipeline.py -k "gcv"
    pytest tests/test_ocr_pipeline.py -k "refinement"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_white_image(width: int = 400, height: int = 300) -> Image.Image:
    """Create a small white RGB image for tests that don't need a real scan."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


# ===========================================================================
# Section 1a — Ingestion
# ===========================================================================

class TestIngestion:
    """ingest_document() converts files to lists of PIL Images."""

    def test_image_file_returns_one_page(self, first_sample: Path):
        from DocumentProcessor.OCR.ingestion import ingest_document

        if first_sample.suffix.lower() == ".pdf":
            pytest.skip("First sample is a PDF — covered by test_pdf_returns_pages")

        pages = ingest_document(str(first_sample))
        assert isinstance(pages, list), "Expected a list"
        assert len(pages) == 1, "A single image file should produce exactly 1 page"
        assert isinstance(pages[0], Image.Image)

    def test_pdf_returns_pages(self, sample_files: List[Path]):
        from DocumentProcessor.OCR.ingestion import ingest_document

        pdf_files = [f for f in sample_files if f.suffix.lower() == ".pdf"]
        if not pdf_files:
            pytest.skip("No PDF samples available")

        pages = ingest_document(str(pdf_files[0]))
        assert len(pages) >= 1, "PDF should produce at least 1 page"
        assert all(isinstance(p, Image.Image) for p in pages)

    def test_missing_file_raises(self):
        from DocumentProcessor.OCR.ingestion import ingest_document

        with pytest.raises(FileNotFoundError):
            ingest_document("/nonexistent/path/file.pdf")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        from DocumentProcessor.OCR.ingestion import ingest_document

        bad_file = tmp_path / "document.xyz"
        bad_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(str(bad_file))

    def test_all_pages_are_rgb(self, first_sample: Path):
        from DocumentProcessor.OCR.ingestion import ingest_document

        pages = ingest_document(str(first_sample))
        for i, page in enumerate(pages):
            assert page.mode == "RGB", f"Page {i + 1} is not RGB (got {page.mode})"


# ===========================================================================
# Section 1b — Image preprocessing
# ===========================================================================

class TestPreprocessing:
    """restore_image and perspective_correct should never crash."""

    def test_restore_returns_pil_image(self, pil_first_page: Image.Image):
        from DocumentProcessor.OCR.restoration import restore_image

        result = restore_image(pil_first_page)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_restore_downsizes_large_image(self):
        from DocumentProcessor.OCR.restoration import restore_image

        big_img = Image.new("RGB", (8000, 6000), color=(200, 200, 200))
        result = restore_image(big_img, max_image_dimension=4000)
        w, h = result.size
        assert max(w, h) <= 4000, f"Image not downsized: {w}x{h}"

    def test_perspective_correct_passthrough_on_plain_image(self):
        """A plain white image has no detectable boundary — should pass through."""
        from DocumentProcessor.OCR.perspective_correction import perspective_correct

        plain = _make_white_image()
        result_img, was_corrected = perspective_correct(plain)
        assert isinstance(result_img, Image.Image)
        # White canvas has no contour — correction should not activate
        assert was_corrected is False

    def test_perspective_correct_returns_rgb(self, pil_first_page: Image.Image):
        from DocumentProcessor.OCR.perspective_correction import perspective_correct

        result_img, _ = perspective_correct(pil_first_page)
        assert result_img.mode == "RGB"


# ===========================================================================
# Section 1c — GCV engine
# ===========================================================================

class TestGCVEngine:
    """GCVEngine.ocr_page() against a real GCV API call."""

    def test_ocr_page_returns_expected_keys(
        self, gcv_engine, pil_first_page: Image.Image
    ):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)

        assert "raw_text" in result
        assert "confidence" in result
        assert "word_confidences" in result
        assert "error" in result

    def test_ocr_page_error_is_none_on_success(
        self, gcv_engine, pil_first_page: Image.Image
    ):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        assert result["error"] is None, f"GCV returned an error: {result['error']}"

    def test_ocr_page_raw_text_is_string(
        self, gcv_engine, pil_first_page: Image.Image
    ):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        assert isinstance(result["raw_text"], str)

    def test_ocr_page_confidence_in_range(
        self, gcv_engine, pil_first_page: Image.Image
    ):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_ocr_page_word_confidences_structure(
        self, gcv_engine, pil_first_page: Image.Image
    ):
        from DocumentProcessor.OCR.models import WordConfidence

        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        wcs = result["word_confidences"]

        if wcs is None:
            pytest.skip("GCV returned no word confidences (likely blank page)")

        assert isinstance(wcs, list)
        assert len(wcs) > 0, "Expected at least one word confidence entry"

        for wc in wcs:
            assert isinstance(wc, WordConfidence)
            assert wc.band in ("high", "mid", "low"), f"Invalid band: {wc.band}"
            assert 0.0 <= wc.confidence <= 1.0
            assert isinstance(wc.word, str) and wc.word.strip()
            assert wc.page_number == 1

    def test_ocr_page_blank_image_does_not_crash(self, gcv_engine):
        """GCV should handle a blank image gracefully (no exception)."""
        blank = _make_white_image()
        result = gcv_engine.ocr_page(blank, page_number=99)
        assert "error" in result  # key must exist even if blank
        assert "raw_text" in result

    def test_pil_to_bytes_produces_valid_png(self, gcv_engine):
        img = _make_white_image(100, 100)
        png_bytes = gcv_engine._pil_to_bytes(img)
        assert png_bytes[:4] == b"\x89PNG", "Output is not a valid PNG"
        assert len(png_bytes) > 0


# ===========================================================================
# Section 1d — WordConfidence model and band logic
# ===========================================================================

class TestWordConfidenceModel:
    """WordConfidence Pydantic model and band assignment thresholds."""

    @pytest.mark.parametrize("conf, expected_band", [
        (0.95, "high"),
        (0.90, "high"),   # boundary — at threshold
        (0.89, "mid"),
        (0.65, "mid"),    # boundary — at threshold
        (0.64, "low"),
        (0.00, "low"),
    ])
    def test_band_matches_threshold(self, conf: float, expected_band: str):
        from DocumentProcessor.OCR.gcv_engine import _band

        assert _band(conf) == expected_band, (
            f"conf={conf} → expected band '{expected_band}', got '{_band(conf)}'"
        )

    def test_word_confidence_model_valid(self):
        from DocumentProcessor.OCR.models import WordConfidence

        wc = WordConfidence(word="المحكمة", confidence=0.97, band="high", page_number=1)
        assert wc.word == "المحكمة"
        assert wc.band == "high"
        assert wc.page_number == 1

    def test_word_confidence_rejects_out_of_range(self):
        from pydantic import ValidationError
        from DocumentProcessor.OCR.models import WordConfidence

        with pytest.raises(ValidationError):
            WordConfidence(word="test", confidence=1.5, band="high", page_number=1)

    def test_word_confidence_rejects_invalid_band(self):
        from pydantic import ValidationError
        from DocumentProcessor.OCR.models import WordConfidence

        with pytest.raises(ValidationError):
            WordConfidence(word="test", confidence=0.8, band="excellent", page_number=1)

    def test_canonical_text_prefers_refined(self):
        from DocumentProcessor.OCR.models import OCRPageResult

        page = OCRPageResult(
            page_number=1,
            normalized_text="النص الاصلي",
            refined_text="النص الأصلي",
        )
        assert page.canonical_text == "النص الأصلي"

    def test_canonical_text_falls_back_to_normalized(self):
        from DocumentProcessor.OCR.models import OCRPageResult

        page = OCRPageResult(
            page_number=1,
            normalized_text="النص الاصلي",
            refined_text="",
        )
        assert page.canonical_text == "النص الاصلي"


# ===========================================================================
# Section 1e — LLM refinement
# ===========================================================================

class TestLLMRefinement:
    """refine_ocr_text() — tests against the real LLM and edge cases."""

    def test_disabled_returns_input_unchanged(self):
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        raw = "نص تجريبي"
        result = refine_ocr_text(raw, page_number=1, enabled=False)
        assert result == raw

    def test_empty_input_returns_empty(self):
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        result = refine_ocr_text("", page_number=1, enabled=True)
        assert result == ""

    def test_whitespace_only_returns_as_is(self):
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        result = refine_ocr_text("   \n  ", page_number=1, enabled=True)
        assert result.strip() == ""

    def test_llm_failure_falls_back_to_raw(self):
        """When the LLM raises, refine_ocr_text must return the raw text unchanged."""
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        raw = "نص تجريبي للاختبار"
        with patch(
            "DocumentProcessor.OCR.llm_refinement._call_llm",
            side_effect=RuntimeError("Simulated LLM timeout"),
        ):
            result = refine_ocr_text(raw, page_number=1, enabled=True)

        assert result == raw, "Should fall back to raw text on LLM failure"

    def test_llm_empty_response_falls_back_to_raw(self):
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        raw = "نص تجريبي للاختبار"
        with patch(
            "DocumentProcessor.OCR.llm_refinement._call_llm",
            return_value="",
        ):
            result = refine_ocr_text(raw, page_number=1, enabled=True)

        assert result == raw

    @pytest.mark.slow
    def test_real_llm_call_returns_nonempty_string(self):
        """Live LLM call — requires GOOGLE_API_KEY or GROQ_API_KEY in env.
        Skipped automatically if neither is set.
        """
        from DocumentProcessor.OCR.llm_refinement import refine_ocr_text

        has_google = bool(os.getenv("GOOGLE_API_KEY"))
        has_groq = bool(os.getenv("GROQ_API_KEY"))
        if not has_google and not has_groq:
            pytest.skip("No LLM API key set — skipping live refinement test")

        arabic_with_errors = "باسم الشعب - المحكمه الابتدائيه - الدائره المدنيه"
        result = refine_ocr_text(arabic_with_errors, page_number=1, enabled=True)

        assert isinstance(result, str)
        assert len(result) > 0
        # The model should produce Arabic output, not an error message
        assert any("\u0600" <= c <= "\u06FF" for c in result), (
            "Refined output contains no Arabic characters"
        )


# ===========================================================================
# Section 1f — run_ocr() full OCR pipeline
# ===========================================================================

class TestRunOCR:
    """run_ocr() — GCV + optional LLM refinement, no Mongo/Qdrant."""

    def test_run_ocr_returns_document_result(self, first_sample: Path, gcv_engine):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        from DocumentProcessor.OCR.models import OCRDocumentResult

        result = run_ocr(str(first_sample))
        assert isinstance(result, OCRDocumentResult)

    def test_run_ocr_metadata_fields_present(self, first_sample: Path, gcv_engine):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample))
        meta = result.metadata

        assert meta["filename"] == first_sample.name
        assert meta["total_pages"] >= 1
        assert meta["model_used"] == "google-cloud-vision"
        assert "processing_time_seconds" in meta
        assert "timestamp" in meta

    def test_run_ocr_page_count_matches(self, first_sample: Path, gcv_engine):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample))
        assert len(result.pages) == result.metadata["total_pages"]

    def test_run_ocr_pages_have_page_numbers(self, first_sample: Path, gcv_engine):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample))
        for i, page in enumerate(result.pages):
            assert page.page_number == i + 1

    def test_run_ocr_word_confidences_have_correct_page_numbers(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample))
        for page in result.pages:
            if not page.word_confidences:
                continue
            for wc in page.word_confidences:
                assert wc.page_number == page.page_number

    def test_run_ocr_no_preprocessing(self, first_sample: Path, gcv_engine):
        """Pipeline should still work when preprocessing is disabled."""
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample), config={"preprocessing_enabled": False})
        assert result.metadata["preprocessing_enabled"] is False
        assert len(result.pages) >= 1

    def test_run_ocr_no_refinement(self, first_sample: Path, gcv_engine):
        """refined_text should be empty when refinement is disabled."""
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample), config={"refine_enabled": False})
        assert result.metadata["refine_enabled"] is False
        for page in result.pages:
            assert page.refined_text == "", (
                f"Page {page.page_number} has refined_text but refinement was disabled"
            )

    def test_run_ocr_canonical_text_never_empty_when_ocr_succeeds(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        result = run_ocr(str(first_sample))
        for page in result.pages:
            if page.error:
                continue  # error pages are allowed to have no text
            assert page.canonical_text.strip(), (
                f"Page {page.page_number}: canonical_text is empty despite successful OCR"
            )

    def test_run_ocr_all_samples(self, sample_files: List[Path], gcv_engine):
        """Smoke-test every file in test_samples/ — none should raise."""
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr

        if not sample_files:
            pytest.skip("No sample files in test_samples/")

        failures = []
        for path in sample_files:
            try:
                result = run_ocr(str(path), config={"refine_enabled": False})
                if not result.pages:
                    failures.append(f"{path.name}: no pages returned")
            except Exception as exc:
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

        assert not failures, "Some samples failed OCR:\n" + "\n".join(failures)


# ===========================================================================
# Section 2a — Integration: full pipeline (Mongo + Qdrant)
# ===========================================================================

@pytest.mark.integration
class TestFullPipeline:
    """process_document_group() writes to MongoDB and Qdrant.

    Run with:  pytest tests/ --run-integration
    Requires:  MongoDB and Qdrant running at addresses in settings.yaml.
    """

    CASE_ID = "test-case-ocr-pipeline"

    def test_process_document_group_returns_expected_keys(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-001"],
        )

        assert "text" in result
        assert "file_type" in result
        assert "classification" in result
        assert "metadata" in result

    def test_process_document_group_stores_mongo_id(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-002"],
        )

        mongo_id = result["metadata"].get("mongo_id")
        assert mongo_id is not None, "mongo_id missing — MongoDB insert likely failed"
        assert isinstance(mongo_id, str) and len(mongo_id) > 0

    def test_process_document_group_word_confidences_stored(
        self, first_sample: Path, gcv_engine
    ):
        """word_confidences must be stored in MongoDB (not just in memory)."""
        from pymongo import MongoClient
        from bson import ObjectId
        from config.supervisor import MONGO_URI, MONGO_DB, MONGO_COLLECTION
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-003"],
        )

        mongo_id = result["metadata"].get("mongo_id")
        if not mongo_id:
            pytest.skip("No mongo_id — MongoDB unavailable")

        client = MongoClient(MONGO_URI)
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(mongo_id)})
        assert doc is not None, "Document not found in MongoDB"
        assert "word_confidences" in doc
        assert isinstance(doc["word_confidences"], list)

        if doc["word_confidences"]:
            first_wc = doc["word_confidences"][0]
            assert "word" in first_wc
            assert "confidence" in first_wc
            assert "band" in first_wc
            assert first_wc["band"] in ("high", "mid", "low")
            assert "page_number" in first_wc

    def test_process_document_group_raw_ocr_text_stored(
        self, first_sample: Path, gcv_engine
    ):
        """raw_ocr_text must be in MongoDB but NOT in the API response."""
        from pymongo import MongoClient
        from bson import ObjectId
        from config.supervisor import MONGO_URI, MONGO_DB, MONGO_COLLECTION
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-004"],
        )

        mongo_id = result["metadata"].get("mongo_id")
        if not mongo_id:
            pytest.skip("No mongo_id — MongoDB unavailable")

        client = MongoClient(MONGO_URI)
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(mongo_id)})
        assert "raw_ocr_text" in doc, "raw_ocr_text not stored in MongoDB"

    def test_process_document_group_qdrant_chunks_indexed(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-005"],
        )

        chunks = result["metadata"].get("qdrant_chunks", 0)
        assert chunks > 0, (
            "No Qdrant chunks indexed — Qdrant may be unavailable or text was empty"
        )

    def test_classification_returns_known_type(
        self, first_sample: Path, gcv_engine
    ):
        from DocumentProcessor.pipeline import process_document_group

        result = process_document_group(
            file_paths=[str(first_sample)],
            case_id=self.CASE_ID,
            file_ids=["test-file-id-006"],
        )

        clf = result["classification"]
        assert "final_type" in clf
        assert isinstance(clf["final_type"], str) and clf["final_type"]
        assert "confidence" in clf
        assert 0 <= clf["confidence"] <= 100


# ===========================================================================
# Section 2b — Integration: API schema round-trip
# ===========================================================================

@pytest.mark.integration
class TestAPISchemaRoundTrip:
    """Verify word_confidences serialise correctly through the Pydantic schema."""

    def test_ocr_text_response_serialises_word_confidences(self):
        from api.schemas.documents import OCRTextResponse, WordConfidenceItem

        items = [
            WordConfidenceItem(word="المحكمة", confidence=0.97, band="high", page_number=1),
            WordConfidenceItem(word="الابتدائية", confidence=0.61, band="low", page_number=1),
            WordConfidenceItem(word="المدني", confidence=0.75, band="mid", page_number=1),
        ]

        response = OCRTextResponse(
            doc_id="test-doc-001",
            text="المحكمة الابتدائية المدني",
            word_confidences=items,
        )

        data = response.model_dump()
        wcs = data["word_confidences"]
        assert len(wcs) == 3
        assert wcs[0]["band"] == "high"
        assert wcs[1]["band"] == "low"
        assert wcs[2]["band"] == "mid"

    def test_ocr_text_response_empty_confidences_is_valid(self):
        from api.schemas.documents import OCRTextResponse

        response = OCRTextResponse(
            doc_id="test-doc-002",
            text="plain text document",
            word_confidences=[],
        )
        data = response.model_dump()
        assert data["word_confidences"] == []

    def test_parse_word_confidences_skips_invalid_entries(self):
        """_parse_word_confidences must silently drop bad dicts, not crash."""
        from api.routers.documents import _parse_word_confidences

        raw = [
            {"word": "محكمة", "confidence": 0.92, "band": "high", "page_number": 1},
            {"word": "bad", "confidence": 9.99, "band": "high", "page_number": 1},  # invalid conf
            {"confidence": 0.5, "band": "mid", "page_number": 1},                   # missing word
        ]

        result = _parse_word_confidences(raw)
        # Only the first entry is fully valid
        assert len(result) == 1
        assert result[0].word == "محكمة"

    def test_parse_word_confidences_handles_none(self):
        from api.routers.documents import _parse_word_confidences

        assert _parse_word_confidences(None) == []
        assert _parse_word_confidences([]) == []
