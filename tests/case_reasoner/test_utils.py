"""
test_utils.py — Pure utility function tests for Case Reasoner.

Tests cover all helper functions that are imported directly from node modules:
    - validation.py: _normalize, _extract_cited_article_numbers, _available_article_numbers
    - retrieval.py: _parse_articles
    - aggregation.py: _extract_entities
    - application.py: _format_elements_with_classification
    - evidence.py: _format_elements
    - counterargument.py: _format_classifications
    - confidence.py: _compute_issue_signals, _level_from_score
"""

import pathlib
import sys

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.validation import (
    _normalize,
    _extract_cited_article_numbers,
    _available_article_numbers,
)
from nodes.retrieval import _parse_articles
from nodes.aggregation import _extract_entities
from nodes.application import _format_elements_with_classification
from nodes.confidence import _compute_issue_signals, _level_from_score


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNormalize:
    """T-UTIL-01: _normalize strips diacritics and converts Arabic digits."""

    def test_arabic_digits_converted(self):
        result = _normalize("المادة ١٤٨")
        assert "148" in result

    def test_all_arabic_digits(self):
        result = _normalize("٠١٢٣٤٥٦٧٨٩")
        assert result == "0123456789"

    def test_diacritics_stripped(self):
        result = _normalize("العَقْدُ")
        assert "َ" not in result
        assert "ْ" not in result
        assert "ُ" not in result
        assert "العقد" in result

    def test_latin_digits_unchanged(self):
        result = _normalize("148")
        assert "148" in result

    def test_mixed_digits(self):
        result = _normalize("من المادة ١٤٨ إلى ١٥٢")
        assert "148" in result
        assert "152" in result


# ---------------------------------------------------------------------------
# _extract_cited_article_numbers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractCitedArticleNumbers:
    """T-UTIL-02: _extract_cited_article_numbers covers all 4 citation patterns."""

    def test_pattern1_madda_form(self):
        """المادة ١٤٨ or مادة 176."""
        nums = _extract_cited_article_numbers("وفقاً للمادة ١٤٨ من القانون المدني")
        assert 148 in nums

    def test_pattern1_latin_digits(self):
        nums = _extract_cited_article_numbers("وفقاً لمادة 176 من القانون")
        assert 176 in nums

    def test_pattern1_abbreviated_m(self):
        """م. 221 or م 221."""
        nums = _extract_cited_article_numbers("م. ٢٢١ يُلزم المدين")
        assert 221 in nums

    def test_pattern1_parenthesized(self):
        """المادة (248)."""
        nums = _extract_cited_article_numbers("المادة (٢٤٨) من القانون")
        assert 248 in nums

    def test_pattern2_raqm_form(self):
        """المادة رقم ١٤٨."""
        nums = _extract_cited_article_numbers("تنص المادة رقم ١٤٨ على أن")
        assert 148 in nums

    def test_pattern3_range_ila(self):
        """المواد من 148 إلى 150."""
        nums = _extract_cited_article_numbers("المواد من ١٤٨ إلى ١٥٠")
        assert 148 in nums
        assert 150 in nums

    def test_pattern3_range_hata(self):
        """المواد 221 حتى 225."""
        nums = _extract_cited_article_numbers("المواد ٢٢١ حتى ٢٢٥")
        assert 221 in nums
        assert 225 in nums

    def test_pattern4_paragraph_form(self):
        """الفقرة ٢ من المادة ١٤٨."""
        nums = _extract_cited_article_numbers("الفقرة ٢ من المادة ١٤٨ تشترط")
        assert 148 in nums

    def test_pattern4_latin_digits(self):
        nums = _extract_cited_article_numbers("الفقرة 1 من مادة 176")
        assert 176 in nums

    def test_multiple_citations_in_text(self):
        nums = _extract_cited_article_numbers(
            "استناداً للمادة ١٤٨ والمادة ١٧٦ وطبقاً للمادة رقم ٢٢١"
        )
        assert 148 in nums
        assert 176 in nums
        assert 221 in nums

    def test_no_citations_returns_empty(self):
        nums = _extract_cited_article_numbers("نص قانوني بدون أرقام مواد")
        assert len(nums) == 0

    def test_returns_set_type(self):
        nums = _extract_cited_article_numbers("المادة ١٤٨")
        assert isinstance(nums, set)


# ---------------------------------------------------------------------------
# _available_article_numbers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAvailableArticleNumbers:
    """T-UTIL-03: _available_article_numbers extracts integer article_number fields."""

    def test_extracts_integers(self):
        articles = [
            {"article_number": 148, "article_text": "نص"},
            {"article_number": 176, "article_text": "نص"},
        ]
        nums = _available_article_numbers(articles)
        assert nums == {148, 176}

    def test_skips_non_integer(self):
        articles = [
            {"article_number": "148", "article_text": "نص"},  # string, not int
            {"article_number": 176, "article_text": "نص"},
        ]
        nums = _available_article_numbers(articles)
        assert 148 not in nums
        assert 176 in nums

    def test_skips_missing_article_number(self):
        articles = [{"article_text": "نص بدون رقم"}]
        nums = _available_article_numbers(articles)
        assert len(nums) == 0

    def test_empty_list(self):
        nums = _available_article_numbers([])
        assert nums == set()


# ---------------------------------------------------------------------------
# _parse_articles
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseArticles:
    """T-UTIL-04: _parse_articles normalizes source dicts into flat article list."""

    def test_parses_article_key(self):
        sources = [{"article": 148, "content": "نص المادة", "title": "عقد"}]
        articles = _parse_articles(sources)
        assert len(articles) == 1
        assert articles[0]["article_number"] == 148

    def test_parses_article_number_key(self):
        sources = [{"article_number": 176, "article_text": "نص المادة", "title": "مسؤولية"}]
        articles = _parse_articles(sources)
        assert len(articles) == 1
        assert articles[0]["article_number"] == 176

    def test_arabic_digit_normalization(self):
        sources = [{"article": "٢٤٨", "content": "نص"}]
        articles = _parse_articles(sources)
        assert articles[0]["article_number"] == 248

    def test_skips_missing_article_field(self):
        sources = [{"content": "نص بدون رقم"}]
        articles = _parse_articles(sources)
        assert len(articles) == 0

    def test_skips_non_numeric_article(self):
        sources = [{"article": "غير رقمي", "content": "نص"}]
        articles = _parse_articles(sources)
        assert len(articles) == 0

    def test_output_has_required_keys(self):
        sources = [{"article": 148, "content": "نص", "title": "عقد", "book": "كتاب", "part": "باب", "chapter": "فصل"}]
        articles = _parse_articles(sources)
        assert "article_number" in articles[0]
        assert "article_text" in articles[0]
        assert "title" in articles[0]
        assert "book" in articles[0]

    def test_multiple_sources(self):
        sources = [
            {"article": 148, "content": "نص 148"},
            {"article": 176, "content": "نص 176"},
        ]
        articles = _parse_articles(sources)
        assert len(articles) == 2
        nums = {a["article_number"] for a in articles}
        assert nums == {148, 176}

    def test_empty_sources(self):
        articles = _parse_articles([])
        assert articles == []


# ---------------------------------------------------------------------------
# _extract_entities (aggregation)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractEntities:
    """T-UTIL-05: _extract_entities extracts dates, amounts, and named parties."""

    def test_date_entity_extracted(self):
        entities = _extract_entities("تم إبرام العقد في 25/03/2024")
        assert "25/03/2024" in entities

    def test_date_with_dash_separator(self):
        entities = _extract_entities("بتاريخ 01-07-2023")
        assert "01-07-2023" in entities

    def test_monetary_amount_extracted(self):
        entities = _extract_entities("بمبلغ 100,000 جنيه كتعويض")
        assert any("جنيه" in e for e in entities)

    def test_named_party_extracted(self):
        entities = _extract_entities("السيد أحمد محمد")
        assert any("أحمد" in e for e in entities)

    def test_no_entities_empty_set(self):
        entities = _extract_entities("نص قانوني مجرد بدون كيانات")
        assert isinstance(entities, set)

    def test_empty_string(self):
        entities = _extract_entities("")
        assert isinstance(entities, set)

    def test_multiple_entities(self):
        text = "السيد أحمد محمد دفع 50,000 جنيه بتاريخ 01/01/2024"
        entities = _extract_entities(text)
        assert len(entities) >= 2


# ---------------------------------------------------------------------------
# _format_elements_with_classification (application)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatElementsWithClassification:
    """T-UTIL-06: _format_elements_with_classification excludes insufficient_evidence."""

    def test_includes_active_elements(self):
        elements = [{"element_id": "E1", "description": "وجود عقد", "element_type": "legal"}]
        classifications = [{"element_id": "E1", "status": "established"}]
        text = _format_elements_with_classification(elements, classifications)
        assert "E1" in text

    def test_excludes_insufficient_evidence(self):
        elements = [
            {"element_id": "E1", "description": "وجود عقد", "element_type": "legal"},
            {"element_id": "E2", "description": "وقوع ضرر", "element_type": "factual"},
        ]
        classifications = [
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "insufficient_evidence"},
        ]
        text = _format_elements_with_classification(elements, classifications)
        assert "E1" in text
        assert "E2" not in text

    def test_all_insufficient_returns_fallback(self):
        elements = [{"element_id": "E1", "description": "وجود عقد", "element_type": "legal"}]
        classifications = [{"element_id": "E1", "status": "insufficient_evidence"}]
        text = _format_elements_with_classification(elements, classifications)
        assert text == "لا عناصر قابلة للتحليل"

    def test_unknown_status_defaults_to_disputed(self):
        elements = [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}]
        classifications = []  # no classification → defaults to "disputed"
        text = _format_elements_with_classification(elements, classifications)
        assert "E1" in text


# ---------------------------------------------------------------------------
# _compute_issue_signals (confidence)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeIssueSignals:
    """T-UTIL-07: _compute_issue_signals produces correct ratio signals."""

    def _make_clean_analysis(self):
        return {
            "required_elements": [{"element_id": "E1"}, {"element_id": "E2"}],
            "element_classifications": [
                {"element_id": "E1", "status": "established"},
                {"element_id": "E2", "status": "established"},
            ],
            "applied_elements": [
                {"element_id": "E1", "cited_articles": [148]},
                {"element_id": "E2", "cited_articles": [176]},
            ],
            "citation_check": {
                "total_citations": 2,
                "unsupported_conclusions": [],
                "missing_citations": [],
            },
            "logical_consistency_check": {"severity": "none"},
            "completeness_check": {"coverage_ratio": 1.0},
        }

    def test_clean_issue_all_signals_zero(self):
        analysis = self._make_clean_analysis()
        signals = _compute_issue_signals(analysis, set())
        assert signals["unsupported_ratio"] == 0.0
        assert signals["disputed_ratio"] == 0.0
        assert signals["insufficient_ratio"] == 0.0
        assert signals["citation_failure_ratio"] == 0.0
        assert signals["logical_issues"] == 0.0
        assert signals["completeness_gap"] == 0.0
        assert signals["reconciliation_triggered"] == 0.0

    def test_disputed_ratio_computed(self):
        analysis = self._make_clean_analysis()
        analysis["element_classifications"] = [
            {"element_id": "E1", "status": "disputed"},
            {"element_id": "E2", "status": "established"},
        ]
        signals = _compute_issue_signals(analysis, set())
        assert signals["disputed_ratio"] == pytest.approx(0.5)

    def test_insufficient_ratio_computed(self):
        analysis = self._make_clean_analysis()
        analysis["element_classifications"] = [
            {"element_id": "E1", "status": "insufficient_evidence"},
            {"element_id": "E2", "status": "established"},
        ]
        signals = _compute_issue_signals(analysis, set())
        assert signals["insufficient_ratio"] == pytest.approx(0.5)

    def test_reconciliation_flag_set_when_conflict(self):
        analysis = self._make_clean_analysis()
        analysis["issue_id"] = 1
        signals = _compute_issue_signals(analysis, {1})  # issue 1 in conflict
        assert signals["reconciliation_triggered"] == 1.0

    def test_reconciliation_flag_clear_when_no_conflict(self):
        analysis = self._make_clean_analysis()
        analysis["issue_id"] = 1
        signals = _compute_issue_signals(analysis, {2})  # different issue in conflict
        assert signals["reconciliation_triggered"] == 0.0

    def test_major_logical_issue_penalty(self):
        analysis = self._make_clean_analysis()
        analysis["logical_consistency_check"] = {"severity": "major"}
        signals = _compute_issue_signals(analysis, set())
        assert signals["logical_issues"] == 1.0

    def test_minor_logical_issue_penalty(self):
        analysis = self._make_clean_analysis()
        analysis["logical_consistency_check"] = {"severity": "minor"}
        signals = _compute_issue_signals(analysis, set())
        assert signals["logical_issues"] == 0.5

    def test_completeness_gap_computed(self):
        analysis = self._make_clean_analysis()
        analysis["completeness_check"] = {"coverage_ratio": 0.6}
        signals = _compute_issue_signals(analysis, set())
        assert signals["completeness_gap"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _level_from_score (confidence)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLevelFromScore:
    """T-UTIL-08: _level_from_score maps scores to confidence levels correctly."""

    _THRESHOLDS = {"high": 0.75, "medium": 0.45}

    def test_score_above_high_threshold(self):
        assert _level_from_score(0.80, self._THRESHOLDS) == "high"

    def test_score_exactly_at_high_threshold(self):
        assert _level_from_score(0.75, self._THRESHOLDS) == "high"

    def test_score_in_medium_range(self):
        assert _level_from_score(0.60, self._THRESHOLDS) == "medium"

    def test_score_exactly_at_medium_threshold(self):
        assert _level_from_score(0.45, self._THRESHOLDS) == "medium"

    def test_score_below_medium_threshold(self):
        assert _level_from_score(0.30, self._THRESHOLDS) == "low"

    def test_score_zero(self):
        assert _level_from_score(0.0, self._THRESHOLDS) == "low"

    def test_score_one(self):
        assert _level_from_score(1.0, self._THRESHOLDS) == "high"
