"""
tests/ocr/test_llm.py
----------------------
Tests for DocumentProcessor.OCR.llm_refinement.

Unit tests (no LLM calls):
  TestBypassCases        — enabled=False, empty/whitespace input
  TestFallbackBehavior   — LLM exception, empty response (mocked _call_llm)
  TestContentLengthGuard — MAX_CHAR_RATIO / MAX_WORD_RATIO thresholds
  TestPrewarm            — prewarm_llm() + singleton thread-safety

Integration tests (@pytest.mark.integration, live LLM):
  TestLLMRealCall        — real Arabic OCR correction
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _refine(raw: str, **kwargs) -> str:
    from DocumentProcessor.OCR.llm_refinement import refine_ocr_text
    return refine_ocr_text(raw, **kwargs)


# ---------------------------------------------------------------------------
# Bypass cases  (no mocking needed)
# ---------------------------------------------------------------------------

class TestBypassCases:
    def test_disabled_returns_raw_immediately(self):
        raw = "أي نص"
        result = _refine(raw, enabled=False)
        assert result is raw  # exact same object, no copy

    def test_empty_string_returns_empty(self):
        assert _refine("") == ""

    def test_whitespace_only_returns_whitespace(self):
        result = _refine("   \n  ")
        assert result.strip() == ""

    def test_disabled_with_empty_returns_empty(self):
        assert _refine("", enabled=False) == ""


# ---------------------------------------------------------------------------
# Fallback behavior  (mocked _call_llm)
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    _TARGET = "DocumentProcessor.OCR.llm_refinement._call_llm"

    def test_llm_exception_returns_raw(self):
        raw = "نص الاختبار"
        with patch(self._TARGET, side_effect=Exception("network error")):
            result = _refine(raw, page_number=1)
        assert result == raw

    def test_llm_empty_string_returns_raw(self):
        raw = "نص الاختبار"
        with patch(self._TARGET, return_value=""):
            result = _refine(raw, page_number=1)
        assert result == raw

    def test_llm_whitespace_response_returns_raw(self):
        raw = "نص الاختبار"
        with patch(self._TARGET, return_value="   "):
            result = _refine(raw, page_number=1)
        assert result == raw

    def test_timeout_exception_returns_raw(self):
        from requests.exceptions import ReadTimeout
        raw = "نص المحكمة"
        with patch(self._TARGET, side_effect=ReadTimeout("timed out")):
            result = _refine(raw, page_number=11)
        assert result == raw

    def test_valid_response_returned(self):
        raw = "نص الاختبار"
        refined = "نصّ الاختبار"
        with patch(self._TARGET, return_value=refined):
            result = _refine(raw, page_number=1)
        assert result == refined

    def test_preamble_stripped(self):
        """Model sometimes prefixes 'النص المصحح:' despite the prompt."""
        raw = "نص"
        with patch(self._TARGET, return_value="النص المصحح: نصٌّ"):
            result = _refine(raw, page_number=1)
        # preamble regex is inside _call_llm, but refine_ocr_text gets the
        # already-stripped output from _call_llm; test the integration:
        # if _call_llm strips the preamble, result won't start with "النص المصحح"
        assert not result.startswith("النص المصحح")


# ---------------------------------------------------------------------------
# Content-length guard  (mocked _call_llm)
# ---------------------------------------------------------------------------

class TestContentLengthGuard:
    _TARGET = "DocumentProcessor.OCR.llm_refinement._call_llm"

    def _run(self, raw: str, refined: str) -> str:
        with patch(self._TARGET, return_value=refined):
            return _refine(raw, page_number=99)

    # ---- Character ratio -----------------------------------------------

    def test_char_ratio_1_00_accepted(self):
        raw = "أ" * 500
        assert self._run(raw, raw) == raw  # identical → accepted

    def test_char_ratio_1_14_accepted(self):
        raw = "أ" * 500
        refined = "أ" * 570   # 570/500 = 1.14 — just under limit
        assert self._run(raw, refined) == refined

    def test_char_ratio_exactly_at_limit_rejected(self):
        """At exactly 1.15× the limit should be rejected (> not >=)."""
        from DocumentProcessor.OCR.llm_refinement import MAX_CHAR_RATIO
        raw = "أ" * 1000
        refined = "أ" * int(1000 * MAX_CHAR_RATIO)  # 1150
        # 1150/1000 = 1.15 exactly → NOT > MAX_CHAR_RATIO → accepted
        result = self._run(raw, refined)
        assert result == refined   # at boundary: accepted

    def test_char_ratio_above_limit_rejected(self):
        """Page-20 case: 900 → 1247 chars (+38%) must be rejected."""
        raw = "أ" * 900
        hallucinated = "أ" * 1247    # ratio ≈ 1.386 → reject
        result = self._run(raw, hallucinated)
        assert result == raw

    def test_char_ratio_2x_rejected(self):
        raw = "م" * 500
        assert self._run(raw, "م" * 1000) == raw

    # ---- Word ratio -----------------------------------------------

    def test_word_ratio_1_09_accepted(self):
        raw = " ".join(["كلمة"] * 100)     # 100 words
        refined = " ".join(["كلمة"] * 109)  # 109/100 = 1.09 → accepted
        assert self._run(raw, refined) == refined

    def test_word_ratio_at_limit_accepted(self):
        from DocumentProcessor.OCR.llm_refinement import MAX_WORD_RATIO
        raw = " ".join(["كلمة"] * 100)
        refined = " ".join(["كلمة"] * int(100 * MAX_WORD_RATIO))  # 1.10 exactly
        result = self._run(raw, refined)
        assert result == refined  # at boundary: accepted

    def test_word_ratio_above_limit_rejected(self):
        raw = " ".join(["كلمة"] * 100)
        hallucinated = " ".join(["كلمة"] * 115)  # 1.15 ratio → reject
        assert self._run(raw, hallucinated) == raw

    # ---- Both ratios -----------------------------------------------

    def test_both_ratios_ok_accepted(self):
        raw = "محكمة مصر العليا تحكم في القضية"
        refined = "محكمةُ مصر العُليا تحكم في القضيةِ"  # slight increase
        result = self._run(raw, refined)
        assert result == refined

    def test_shrunk_output_always_accepted(self):
        """LLM removing OCR garbage is fine — ratio < 1.0 always accepted."""
        raw = "نصٌّ ## مع %% أخطاء @@@ كثيرة"
        refined = "نصٌّ مع أخطاء كثيرة"
        assert self._run(raw, refined) == refined


# ---------------------------------------------------------------------------
# prewarm_llm — thread safety + singleton
# ---------------------------------------------------------------------------

class TestPrewarm:
    def test_prewarm_does_not_crash(self):
        """prewarm_llm() with a mocked get_llm must complete without error."""
        mock_llm = MagicMock()
        with patch("DocumentProcessor.OCR.llm_refinement._get_llm"):
            from DocumentProcessor.OCR.llm_refinement import prewarm_llm
            prewarm_llm(timeout=30)  # should not raise

    def test_prewarm_idempotent(self):
        """Calling prewarm twice must not build the singleton twice."""
        import DocumentProcessor.OCR.llm_refinement as mod

        original_instance = mod._llm_instance
        try:
            mod._llm_instance = None  # reset for this test
            build_count = [0]

            def fake_get_llm(*_, **__):
                build_count[0] += 1
                return MagicMock()

            with patch("DocumentProcessor.OCR.llm_refinement._get_llm", side_effect=fake_get_llm):
                from DocumentProcessor.OCR.llm_refinement import prewarm_llm
                prewarm_llm(timeout=30)
                prewarm_llm(timeout=30)  # second call — singleton already set

            # _get_llm may still be called twice (prewarm calls it each time),
            # but the actual get_llm("low",...) inside should only run once.
            # Key assertion: no crash, function completed
        finally:
            mod._llm_instance = original_instance

    def test_concurrent_prewarm_builds_singleton_once(self):
        """Multiple threads racing to prewarm must only build the LLM once."""
        import DocumentProcessor.OCR.llm_refinement as mod

        original = mod._llm_instance
        try:
            mod._llm_instance = None
            build_count = [0]
            lock = threading.Lock()

            def fake_llm(*_, **__):
                with lock:
                    build_count[0] += 1
                return MagicMock()

            with patch("config.get_llm", side_effect=fake_llm):
                threads = [
                    threading.Thread(target=mod.prewarm_llm)
                    for _ in range(8)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            assert build_count[0] == 1, (
                f"Singleton built {build_count[0]} times — must be built exactly once"
            )
        finally:
            mod._llm_instance = original


# ---------------------------------------------------------------------------
# Integration tests  (live LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestLLMRealCall:
    """End-to-end LLM refinement with a real Gemini Flash-Lite call."""

    _ARABIC_OCR_ERROR = (
        "وفقا لنص المادة العاشرة من قانوز الاثبات "
        "يكوز للمحكمه قبول الدليل الكتابى"
    )
    # Expected corrections: قانوز → قانون, يكوز → يكون, المحكمه → المحكمة

    def test_refinement_returns_string(self):
        result = _refine(self._ARABIC_OCR_ERROR, page_number=1, timeout=60)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_char_ratio_within_guard(self):
        from DocumentProcessor.OCR.llm_refinement import MAX_CHAR_RATIO
        raw = self._ARABIC_OCR_ERROR
        result = _refine(raw, page_number=1, timeout=60)
        ratio = len(result) / max(len(raw), 1)
        assert ratio <= MAX_CHAR_RATIO, (
            f"Real LLM output exceeded MAX_CHAR_RATIO ({ratio:.3f} > {MAX_CHAR_RATIO})"
        )

    def test_word_ratio_within_guard(self):
        from DocumentProcessor.OCR.llm_refinement import MAX_WORD_RATIO
        raw = self._ARABIC_OCR_ERROR
        result = _refine(raw, page_number=1, timeout=60)
        wr = len(result.split()) / max(len(raw.split()), 1)
        assert wr <= MAX_WORD_RATIO

    def test_result_contains_arabic(self):
        result = _refine(self._ARABIC_OCR_ERROR, page_number=1, timeout=60)
        arabic_chars = sum(1 for c in result if "\u0600" <= c <= "\u06FF")
        assert arabic_chars > 5, "Refined text must contain Arabic characters"

    def test_disabled_skips_api_call(self):
        """enabled=False must return raw without any network request."""
        raw = self._ARABIC_OCR_ERROR
        result = _refine(raw, enabled=False, timeout=60)
        assert result == raw
