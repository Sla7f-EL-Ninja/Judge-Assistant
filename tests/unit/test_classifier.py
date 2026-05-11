"""
Unit tests for DocumentProcessor.classifier and supporting modules.

All tests are LLM-free: the LLM path is exercised via mocks only.
Run: pytest tests/unit/test_classifier.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from DocumentProcessor.arabic_norm import normalize
from DocumentProcessor.classifier import (
    _is_unambiguous,
    _prepare,
    _score_candidates,
    classify_document,
)
from config.taxonomy import get_doc_types, get_taxonomy, get_unknown_label


# ---------------------------------------------------------------------------
# arabic_norm
# ---------------------------------------------------------------------------

class TestArabicNorm:
    def test_tatweel_stripped(self):
        assert normalize("مـحـكـمـة") == "محكمة"

    def test_alef_forms_unified(self):
        assert normalize("أإآٱ") == "اااا"

    def test_alef_maqsura_unified(self):
        assert normalize("على") == "علي"

    def test_whitespace_collapsed(self):
        assert normalize("كلمة   أخرى\n\nثالثة") == "كلمة اخري ثالثة"

    def test_plain_text_unchanged_structure(self):
        result = normalize("باسم الشعب")
        assert "باسم الشعب" in result

    def test_alef_in_keyword_normalizes(self):
        # "إنه" has hamza-below alef; should normalize to "انه"
        assert normalize("إنه في يوم") == "انه في يوم"


# ---------------------------------------------------------------------------
# Taxonomy loader
# ---------------------------------------------------------------------------

class TestTaxonomy:
    def test_all_expected_types_present(self):
        types = get_doc_types()
        expected = [
            "حكم", "صحيفة دعوى", "مذكرة بدفاع", "محضر جلسة",
            "إعلان", "أمر أداء", "أمر على عريضة", "تقرير خبير",
            "محضر إثبات حالة",
        ]
        for t in expected:
            assert t in types, f"Missing doc type: {t}"

    def test_unknown_label_present(self):
        assert get_unknown_label() == "مستند غير معروف"

    def test_each_type_has_keyword_tiers(self):
        taxonomy = get_taxonomy()
        for dtype, entry in taxonomy["doc_types"].items():
            assert "strong" in entry, f"{dtype} missing strong"
            assert "weak" in entry, f"{dtype} missing weak"
            assert "anti" in entry, f"{dtype} missing anti"

    def test_keywords_are_normalized(self):
        # "باسم الشعب" has no alef variants, so survives normalization intact
        taxonomy = get_taxonomy()
        hukm_strong = taxonomy["doc_types"]["حكم"]["strong"]
        assert "باسم الشعب" in hukm_strong


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

class TestHeuristicScoring:
    def test_strong_marker_scores_highest(self):
        ranked = _score_candidates("باسم الشعب فلهذه الأسباب")
        top_type, top_score, strong_hits, _ = ranked[0]
        assert top_type == "حكم"
        assert strong_hits == 2
        assert top_score == 120  # 2 * 60

    def test_no_match_all_zero(self):
        ranked = _score_candidates("نص عشوائي لا يحتوي على كلمات دلالية")
        assert all(score == 0 for _, score, _, _ in ranked)

    def test_anti_keyword_reduces_score(self):
        # "صحيفة دعوى" has "باسم الشعب" as anti; حكم has it as strong
        text = "صحيفة دعوى - باسم الشعب"
        ranked = _score_candidates(text)
        scores = {t: s for t, s, _, _ in ranked}
        # صحيفة دعوى should have strong(60) + anti(-30) = 30
        assert scores["صحيفة دعوى"] == 30
        # حكم should score higher (60 from "باسم الشعب")
        assert scores["حكم"] >= 60

    def test_multiple_weak_hits_accumulate(self):
        text = "المدعي يطلب - المدعى عليه يرفض - الطلبات محددة - الوقائع كالآتي"
        ranked = _score_candidates(text)
        scores = {t: s for t, s, _, _ in ranked}
        # 4 weak hits = 4 * 15 = 60
        assert scores["صحيفة دعوى"] == 60


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

class TestAmbiguityDetection:
    def test_unambiguous_when_strong_hit_and_clear_margin(self):
        ranked = [
            ("حكم", 120, 2, ["باسم الشعب", "فلهذه الأسباب"]),
            ("صحيفة دعوى", 30, 0, []),
        ]
        assert _is_unambiguous(ranked) is True

    def test_ambiguous_when_no_strong_hit(self):
        ranked = [
            ("حكم", 75, 0, ["قضت المحكمة", "وحيث إن", "حضوريا", "غيابيا", "الدائرة المدنية"]),
            ("صحيفة دعوى", 60, 0, []),
        ]
        assert _is_unambiguous(ranked) is False

    def test_ambiguous_when_margin_too_small(self):
        ranked = [
            ("حكم", 90, 1, ["باسم الشعب"]),
            ("صحيفة دعوى", 75, 0, []),
        ]
        # margin = 15 < 30
        assert _is_unambiguous(ranked) is False

    def test_ambiguous_when_scores_all_zero(self):
        ranked = [("حكم", 0, 0, []), ("صحيفة دعوى", 0, 0, [])]
        assert _is_unambiguous(ranked) is False

    def test_empty_ranked_not_unambiguous(self):
        assert _is_unambiguous([]) is False


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_header_limited_to_10_lines(self):
        text = "\n".join(f"سطر {i}" for i in range(20))
        header, _ = _prepare(text)
        assert len(header.split("\n")) == 10

    def test_body_limited_to_500_words(self):
        text = " ".join(["كلمة"] * 1000)
        _, body = _prepare(text)
        assert len(body.split()) == 500

    def test_short_text_returned_intact(self):
        text = "نص قصير جداً"
        header, body = _prepare(text)
        assert "نص قصير جداً" in header or "نص قصير جدا" in header  # normalized

    def test_normalization_applied(self):
        text = "إنه في يوم"  # إ has hamza-below
        header, _ = _prepare(text)
        assert "انه" in header  # normalized form


# ---------------------------------------------------------------------------
# classify_document — heuristic path
# ---------------------------------------------------------------------------

class TestClassifyDocumentHeuristic:
    def test_empty_text_returns_unknown(self):
        result = classify_document("")
        assert result["final_type"] == "مستند غير معروف"
        assert result["confidence"] == 0

    def test_whitespace_only_returns_unknown(self):
        result = classify_document("   \n\n  ")
        assert result["final_type"] == "مستند غير معروف"

    def test_strong_judgment_markers_classify_without_llm(self):
        text = "باسم الشعب\nفلهذه الأسباب قضت المحكمة بإلزام المدعى عليه"
        result = classify_document(text)
        assert result["final_type"] == "حكم"
        assert result["confidence"] > 0
        assert "باسم الشعب" in result["explanation"] or "فلهذه الأسباب" in result["explanation"]

    def test_tatweel_variant_still_matches(self):
        # tatweel in "مـحضر إعلان" should normalize and still detect إعلان markers
        text = "إنه في يوم مـحضر إعلان - أعلنت"
        result = classify_document(text)
        # Strong hit on "انه في يوم" (normalized "إنه في يوم") → إعلان, no LLM needed
        assert result["final_type"] == "إعلان"

    def test_alef_variant_normalizes_and_matches(self):
        # "أمر أداء" with alef variants should still match after normalization
        text = "أمر أداء صادر في الدعوى رقم"
        result = classify_document(text)
        assert result["final_type"] == "أمر أداء"


# ---------------------------------------------------------------------------
# classify_document — LLM path (mocked)
# ---------------------------------------------------------------------------

class TestClassifyDocumentLLM:
    def _ambiguous_text(self):
        # Mix of weak markers from multiple types → not unambiguous → LLM path
        return "الطلبات والوقائع والدفاع أولاً وثانياً - مستند مختلط"

    def test_llm_result_used_when_heuristic_ambiguous(self):
        from DocumentProcessor.classifier import _ClassificationResult

        mock_result = _ClassificationResult(
            doc_type="مذكرة بدفاع", confidence=85, reasons="يحتوي على دفوع متعددة"
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            result = classify_document(self._ambiguous_text())

        assert result["final_type"] == "مذكرة بدفاع"
        assert result["confidence"] == 85
        assert result["explanation"] == "يحتوي على دفوع متعددة"

    def test_out_of_taxonomy_llm_result_falls_back_to_heuristic(self):
        from DocumentProcessor.classifier import _ClassificationResult

        mock_result = _ClassificationResult(
            doc_type="عقد بيع", confidence=70, reasons="نوع جديد غير موجود في التصنيف"
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            result = classify_document(self._ambiguous_text())

        # "عقد بيع" not in taxonomy → must fall back
        assert result["final_type"] != "عقد بيع"

    def test_llm_exception_falls_back_to_heuristic(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("timeout")

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            result = classify_document(self._ambiguous_text())

        # Should not raise; should return a valid dict
        assert "final_type" in result
        assert "confidence" in result
        assert "explanation" in result

    def test_llm_returns_unknown_label_is_accepted(self):
        from DocumentProcessor.classifier import _ClassificationResult

        unknown = get_unknown_label()
        mock_result = _ClassificationResult(
            doc_type=unknown, confidence=30, reasons="لا يطابق أي نوع محدد"
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            result = classify_document(self._ambiguous_text())

        assert result["final_type"] == unknown

    def test_confidence_is_integer(self):
        from DocumentProcessor.classifier import _ClassificationResult

        mock_result = _ClassificationResult(
            doc_type="حكم", confidence=88, reasons="..."
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            result = classify_document(self._ambiguous_text())

        assert isinstance(result["confidence"], int)

    def test_return_shape_always_complete(self):
        """classify_document must always return all three keys."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("down")

        with patch("DocumentProcessor.classifier.get_llm", return_value=mock_llm):
            for text in ["", "  ", "random text", "باسم الشعب"]:
                result = classify_document(text)
                assert set(result.keys()) >= {"final_type", "confidence", "explanation"}
