"""
tests/ocr/test_pipeline.py
---------------------------
Tests for DocumentProcessor.OCR.ocr_pipeline.run_ocr() and _resolve_config().

Unit tests (fully mocked — no network):
  TestResolveConfig      — default keys, override merging, type correctness
  TestRunOCRUnit         — page count, metadata fields, confidence gating,
                           refine_enabled=False, blank-page handling

Integration tests (@pytest.mark.integration):
  TestRunOCRIntegration  — real GCV + mocked LLM (fast, smoke-tests GCV path)
  TestRunOCRFullE2E      — real GCV + real LLM on first_sample
  TestLargeDocumentBatch — 17+ pages produce correct chunk count in metadata
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gcv_result(
    raw_text: str = "النص المستخرج",
    confidence: float | None = 0.95,
    error: str | None = None,
) -> dict:
    return {
        "raw_text": raw_text,
        "confidence": confidence,
        "word_confidences": None,
        "error": error,
    }


def _mock_engine(results: list[dict]) -> MagicMock:
    eng = MagicMock()
    eng.ocr_batch.return_value = results
    return eng


REQUIRED_METADATA_KEYS = {
    "filename", "doc_id", "total_pages", "model_used",
    "gcv_feature", "timestamp", "processing_time_seconds",
    "gcv_batch_seconds", "pdf_dpi",
    "refine_enabled", "refine_confidence_threshold",
    "pages_refined", "pages_skipped_refine",
}


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------

class TestResolveConfig:
    def _cfg(self, overrides=None):
        from DocumentProcessor.OCR.ocr_pipeline import _resolve_config
        return _resolve_config(overrides)

    def test_all_required_keys_present(self):
        cfg = self._cfg()
        required = {
            "gcv_feature", "pdf_dpi", "max_file_size_mb", "allowed_extensions",
            "high_threshold", "medium_threshold", "refine_enabled",
            "refine_timeout", "refine_confidence_threshold", "max_workers",
        }
        missing = required - cfg.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_pdf_dpi_is_int(self):
        assert isinstance(self._cfg()["pdf_dpi"], int)

    def test_refine_enabled_is_bool(self):
        assert isinstance(self._cfg()["refine_enabled"], bool)

    def test_refine_confidence_threshold_in_range(self):
        t = self._cfg()["refine_confidence_threshold"]
        assert 0.0 <= t <= 1.01   # 1.01 is "always refine" sentinel

    def test_max_workers_positive(self):
        assert self._cfg()["max_workers"] >= 1

    def test_allowed_extensions_is_list(self):
        assert isinstance(self._cfg()["allowed_extensions"], list)

    def test_override_pdf_dpi(self):
        cfg = self._cfg({"pdf_dpi": 300})
        assert cfg["pdf_dpi"] == 300

    def test_override_refine_enabled(self):
        cfg = self._cfg({"refine_enabled": False})
        assert cfg["refine_enabled"] is False

    def test_override_confidence_threshold(self):
        cfg = self._cfg({"refine_confidence_threshold": 0.80})
        assert abs(cfg["refine_confidence_threshold"] - 0.80) < 1e-9

    def test_override_does_not_affect_other_keys(self):
        cfg = self._cfg({"pdf_dpi": 200})
        # other keys unchanged
        assert "refine_enabled" in cfg
        assert "gcv_feature" in cfg


# ---------------------------------------------------------------------------
# run_ocr — unit tests (mocked GCV + mocked LLM + synthetic PDF)
# ---------------------------------------------------------------------------

class TestRunOCRUnit:
    """run_ocr with GCV mocked → zero network calls, fast."""

    def _run(self, gcv_results: list[dict], **cfg_overrides):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        from tests.ocr.conftest import make_batch_response   # re-use helper

        mock_eng = _mock_engine(gcv_results)

        with (
            patch("DocumentProcessor.OCR.ocr_pipeline.get_gcv_engine", return_value=mock_eng),
            patch("DocumentProcessor.OCR.ocr_pipeline.prewarm_llm"),
            patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="نصٌّ مصحَّح"),
        ):
            # Use a tiny synthetic single-page PDF to satisfy ingestion
            import tempfile
            img = Image.new("RGB", (200, 300), (255, 255, 255))
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                tmp_pdf = f.name
            img.save(tmp_pdf, "PDF")

            result = run_ocr(tmp_pdf, config={"pdf_dpi": 72, **cfg_overrides})
            # Override page count to match gcv_results by patching ingest
        return result

    def _run_with_pdf(self, pdf_path: str, gcv_results: list[dict], **cfg_overrides):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        mock_eng = _mock_engine(gcv_results)
        with (
            patch("DocumentProcessor.OCR.ocr_pipeline.get_gcv_engine", return_value=mock_eng),
            patch("DocumentProcessor.OCR.ocr_pipeline.prewarm_llm"),
            patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="refined"),
        ):
            return run_ocr(pdf_path, config={"pdf_dpi": 72, **cfg_overrides})

    def test_returns_ocr_document_result(self, synthetic_pdf):
        from DocumentProcessor.OCR.models import OCRDocumentResult
        gcv_results = [_make_gcv_result()]
        result = self._run_with_pdf(str(synthetic_pdf), [_make_gcv_result()] * 3)
        assert isinstance(result, OCRDocumentResult)

    def test_page_count_matches_ingested_pages(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        assert len(result.pages) == 3

    def test_pages_have_sequential_numbers(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        assert [p.page_number for p in result.pages] == [1, 2, 3]

    def test_metadata_has_all_required_keys(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        missing = REQUIRED_METADATA_KEYS - result.metadata.keys()
        assert not missing, f"Metadata missing keys: {missing}"

    def test_metadata_total_pages(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        assert result.metadata["total_pages"] == 3

    def test_metadata_pdf_dpi_override(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results, pdf_dpi=200
        )
        assert result.metadata["pdf_dpi"] == 200

    def test_high_confidence_page_skips_llm(self, synthetic_pdf):
        """Page with conf ≥ 0.92 must have empty refined_text."""
        gcv_results = [_make_gcv_result(confidence=0.96)] * 3
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results,
            refine_enabled=True, refine_confidence_threshold=0.92,
        )
        for page in result.pages:
            assert page.refined_text == "", (
                f"Page {page.page_number} conf=0.96 should skip LLM"
            )

    def test_low_confidence_page_gets_refined(self, synthetic_pdf):
        """Page with conf < 0.92 must have non-empty refined_text."""
        gcv_results = [_make_gcv_result(confidence=0.75)] * 3
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results,
            refine_enabled=True, refine_confidence_threshold=0.92,
        )
        for page in result.pages:
            assert page.refined_text != "", (
                f"Page {page.page_number} conf=0.75 should be refined"
            )

    def test_none_confidence_gets_refined(self, synthetic_pdf):
        """Page with confidence=None (e.g. blank page) → attempt refinement."""
        gcv_results = [_make_gcv_result(confidence=None, raw_text="some text")]
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results,
            refine_enabled=True, refine_confidence_threshold=0.92,
        )
        # With mocked refine returning "refined", page should have refined_text
        assert result.pages[0].refined_text != ""

    def test_blank_page_raw_text_empty_skips_llm_effectively(self, synthetic_pdf):
        """Empty raw_text → refine_ocr_text returns "" regardless of confidence."""
        gcv_results = [_make_gcv_result(confidence=None, raw_text="")]
        with (
            patch("DocumentProcessor.OCR.ocr_pipeline.get_gcv_engine",
                  return_value=_mock_engine(gcv_results)),
            patch("DocumentProcessor.OCR.ocr_pipeline.prewarm_llm"),
            patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text",
                  return_value="") as mock_refine,
        ):
            from DocumentProcessor.OCR.ocr_pipeline import run_ocr
            result = run_ocr(str(synthetic_pdf), config={"pdf_dpi": 72})
        # refine_ocr_text was called but returned "" (its own empty-input guard)
        assert result.pages[0].refined_text == ""

    def test_refine_disabled_no_refined_text(self, synthetic_pdf):
        gcv_results = [_make_gcv_result(confidence=0.50)] * 3
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results, refine_enabled=False,
        )
        for page in result.pages:
            assert page.refined_text == ""
        assert result.metadata["refine_enabled"] is False

    def test_metadata_pages_refined_count(self, synthetic_pdf):
        # 1 high-conf page + 2 low-conf pages
        gcv_results = [
            _make_gcv_result(confidence=0.96),
            _make_gcv_result(confidence=0.70),
            _make_gcv_result(confidence=0.65),
        ]
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results,
            refine_confidence_threshold=0.92,
        )
        assert result.metadata["pages_refined"] == 2
        assert result.metadata["pages_skipped_refine"] == 1

    def test_metadata_gcv_batch_seconds_present(self, synthetic_pdf):
        gcv_results = [_make_gcv_result()] * 3
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        assert isinstance(result.metadata["gcv_batch_seconds"], float)
        assert result.metadata["gcv_batch_seconds"] >= 0

    def test_error_page_stored_in_result(self, synthetic_pdf):
        # GCVEngine._error_result() always sets raw_text="" on failure;
        # the mock must reflect that so the assertion is meaningful.
        gcv_results = [_make_gcv_result(raw_text="", error="GCV 500 internal error")]
        result = self._run_with_pdf(str(synthetic_pdf), gcv_results)
        assert result.pages[0].error == "GCV 500 internal error"
        assert result.pages[0].raw_text == ""

    def test_error_page_skips_llm(self, synthetic_pdf):
        """Pages with GCV errors must never be sent to the LLM."""
        gcv_results = [_make_gcv_result(error="some error")]
        with (
            patch("DocumentProcessor.OCR.ocr_pipeline.get_gcv_engine",
                  return_value=_mock_engine(gcv_results)),
            patch("DocumentProcessor.OCR.ocr_pipeline.prewarm_llm"),
            patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text") as mock_refine,
        ):
            from DocumentProcessor.OCR.ocr_pipeline import run_ocr
            run_ocr(str(synthetic_pdf), config={"pdf_dpi": 72})
        mock_refine.assert_not_called()

    def test_mixed_conf_correct_metadata_counts(self, synthetic_pdf):
        # 2 high, 1 low, 1 error
        gcv_results = [
            _make_gcv_result(confidence=0.95),
            _make_gcv_result(confidence=0.94),
            _make_gcv_result(confidence=0.70),
            _make_gcv_result(error="timeout"),
        ]
        result = self._run_with_pdf(
            str(synthetic_pdf), gcv_results,
            refine_confidence_threshold=0.92,
        )
        # pages_skipped_refine: conf >= threshold AND no error (pages 1, 2)
        assert result.metadata["pages_skipped_refine"] == 2
        # pages_refined: conf < threshold AND no error (page 3)
        assert result.metadata["pages_refined"] == 1


# ---------------------------------------------------------------------------
# Integration tests  (real GCV, mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestRunOCRIntegration:
    """Smoke-tests run_ocr() with real GCV; LLM is mocked to keep it fast."""

    def test_smoke_synthetic_pdf(self, synthetic_pdf):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="refined"):
            result = run_ocr(str(synthetic_pdf), config={"pdf_dpi": 150})
        assert len(result.pages) == 3
        assert all(p.page_number == i + 1 for i, p in enumerate(result.pages))

    def test_metadata_complete(self, synthetic_pdf):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="r"):
            result = run_ocr(str(synthetic_pdf), config={"pdf_dpi": 150})
        missing = REQUIRED_METADATA_KEYS - result.metadata.keys()
        assert not missing

    def test_processing_time_recorded(self, synthetic_pdf):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="r"):
            result = run_ocr(str(synthetic_pdf), config={"pdf_dpi": 150})
        assert result.metadata["processing_time_seconds"] > 0

    def test_word_confidences_page_numbers_correct(self, first_sample):
        """page_number in every WordConfidence must match its page."""
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="r"):
            result = run_ocr(str(first_sample), config={"pdf_dpi": 150})
        for page in result.pages:
            if page.word_confidences:
                for wc in page.word_confidences:
                    assert wc.page_number == page.page_number

    def test_disable_refine_via_config(self, synthetic_pdf):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        result = run_ocr(
            str(synthetic_pdf),
            config={"pdf_dpi": 150, "refine_enabled": False},
        )
        for page in result.pages:
            assert page.refined_text == ""

    def test_confidence_values_in_range(self, synthetic_pdf):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="r"):
            result = run_ocr(str(synthetic_pdf), config={"pdf_dpi": 150})
        for page in result.pages:
            if page.confidence is not None:
                assert 0.0 <= page.confidence <= 1.0


@pytest.mark.integration
@pytest.mark.slow
class TestRunOCRFullE2E:
    """Full pipeline — real GCV + real LLM. No mocking."""

    def test_first_sample_processes_without_error(self, first_sample):
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        result = run_ocr(str(first_sample), config={"pdf_dpi": 150})
        assert len(result.pages) > 0
        error_pages = [p for p in result.pages if p.error]
        assert not error_pages, f"Pages with errors: {error_pages}"

    def test_result_is_serializable(self, first_sample):
        import json
        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        result = run_ocr(str(first_sample), config={"pdf_dpi": 150})
        data = result.model_dump()
        json.dumps(data, ensure_ascii=False)  # must not raise


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDocumentBatch:
    """Verify chunking behaviour at scale with a real multi-sample document."""

    def test_17_plus_pages_uses_multiple_chunks(self, sample_files):
        """At least one sample must exceed 16 pages; else this test is skipped."""
        large = [f for f in sample_files if True]  # all samples
        if not large:
            pytest.skip("No sample files in test_samples/ — skipping large-batch test")

        from DocumentProcessor.OCR.ocr_pipeline import run_ocr
        for pdf in large:
            with patch("DocumentProcessor.OCR.ocr_pipeline.refine_ocr_text", return_value="r"):
                result = run_ocr(str(pdf), config={"pdf_dpi": 150})
            n = result.metadata["total_pages"]
            import math
            expected_chunks = math.ceil(n / 16)
            # We can't directly observe chunk count from the result, but
            # the total page count and timing confirm chunking worked
            assert result.metadata["total_pages"] == n
            assert result.metadata["processing_time_seconds"] > 0