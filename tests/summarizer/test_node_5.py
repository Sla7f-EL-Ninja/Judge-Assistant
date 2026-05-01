"""
test_node_5.py — Unit tests for Summerize/node_5.py (Brief Generator)

Tests:
    T-NODE5-01: validate_brief() fails when any section is empty
    T-NODE5-02: validate_brief() fails for each bias keyword
    T-NODE5-03: build_fallback_brief() produces all 7 non-empty sections
    T-NODE5-04: build_fallback_brief() uncontested_facts only uses themes without key_disputes
    T-NODE5-05: compile_key_disputes() deduplication
    T-NODE5-06: collect_all_sources() deduplication
    T-NODE5-07: render_brief() contains all 7 Arabic section headings
    T-NODE5-08: process() empty input → all 7 fields = "لا تتوفر معلومات كافية"
    T-NODE5-09: process() LLM exception → fallback brief with all 7 sections
    T-NODE5-10: build_fallback_brief() legal_questions from الأساس القانوني key_disputes
    T-NODE5-11: build_fallback_brief() legal_questions falls back to all key_disputes
"""

import pathlib
import sys
from unittest.mock import MagicMock

import pytest


from summarize.nodes.brief import BIAS_KEYWORDS, Node5_BriefGenerator
from summarize.schemas import CaseBrief


def make_case_brief(**overrides) -> CaseBrief:
    defaults = dict(
        dispute_summary="ملخص النزاع",
        uncontested_facts="الوقائع الثابتة",
        key_disputes="نقاط الخلاف",
        party_requests="طلبات الخصوم",
        party_defenses="دفوع الخصوم",
        submitted_documents="المستندات المقدمة",
        legal_questions="الأسئلة القانونية",
    )
    defaults.update(overrides)
    return CaseBrief(**defaults)


def make_theme_summary(
    theme="موضوع",
    summary="ملخص",
    key_disputes=None,
    sources=None,
) -> dict:
    return {
        "theme": theme,
        "summary": summary,
        "key_disputes": key_disputes or [],
        "sources": sources or ["doc1 ص1 ف1"],
    }


def make_role_theme_summaries(role="الوقائع", theme_summaries=None) -> dict:
    return {"role": role, "theme_summaries": theme_summaries or [make_theme_summary()]}


def make_node5(parser_result=None, raises=None) -> Node5_BriefGenerator:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node5_BriefGenerator(llm)


@pytest.mark.unit
class TestValidateBrief:
    def setup_method(self):
        self.node = make_node5()

    def test_valid_brief_passes(self):
        """All 7 fields non-empty with no bias → True."""
        brief = make_case_brief()
        assert self.node.validate_brief(brief) is True

    def test_empty_dispute_summary_fails(self):
        """T-NODE5-01: Empty dispute_summary → False."""
        brief = make_case_brief(dispute_summary="")
        assert self.node.validate_brief(brief) is False

    def test_empty_uncontested_facts_fails(self):
        """T-NODE5-01: Empty uncontested_facts → False."""
        brief = make_case_brief(uncontested_facts="")
        assert self.node.validate_brief(brief) is False

    def test_empty_key_disputes_fails(self):
        """T-NODE5-01: Empty key_disputes → False."""
        brief = make_case_brief(key_disputes="")
        assert self.node.validate_brief(brief) is False

    def test_empty_party_requests_fails(self):
        """T-NODE5-01: Empty party_requests → False."""
        brief = make_case_brief(party_requests="")
        assert self.node.validate_brief(brief) is False

    def test_empty_party_defenses_fails(self):
        """T-NODE5-01: Empty party_defenses → False."""
        brief = make_case_brief(party_defenses="")
        assert self.node.validate_brief(brief) is False

    def test_empty_submitted_documents_fails(self):
        """T-NODE5-01: Empty submitted_documents → False."""
        brief = make_case_brief(submitted_documents="")
        assert self.node.validate_brief(brief) is False

    def test_empty_legal_questions_fails(self):
        """T-NODE5-01: Empty legal_questions → False."""
        brief = make_case_brief(legal_questions="")
        assert self.node.validate_brief(brief) is False

    def test_whitespace_only_field_fails(self):
        """T-NODE5-01: Whitespace-only field → False."""
        brief = make_case_brief(dispute_summary="   ")
        assert self.node.validate_brief(brief) is False

    @pytest.mark.parametrize("keyword", BIAS_KEYWORDS)
    def test_bias_keyword_fails(self, keyword):
        """T-NODE5-02: Each bias keyword in any section → False."""
        brief = make_case_brief(dispute_summary=f"نص طبيعي ثم {keyword} ثم نهاية")
        assert self.node.validate_brief(brief) is False


@pytest.mark.unit
class TestBuildFallbackBrief:
    def setup_method(self):
        self.node = make_node5()

    def _make_full_role_map(self) -> dict:
        return {
            "الوقائع": [
                make_theme_summary("وقائع 1", "ملخص 1", key_disputes=[], sources=["s1"]),
                make_theme_summary("وقائع 2", "ملخص 2", key_disputes=["خلاف"], sources=["s2"]),
            ],
            "الطلبات": [make_theme_summary("طلبات", "طلبات المدعي")],
            "الدفوع": [make_theme_summary("دفوع", "دفوع المدعى عليه")],
            "المستندات": [make_theme_summary("مستندات", "قائمة المستندات")],
            "الأساس القانوني": [
                make_theme_summary("أساس", "المواد القانونية", key_disputes=["مسألة قانونية"])
            ],
            "الإجراءات": [make_theme_summary("إجراءات", "سير الدعوى")],
        }

    def test_all_7_sections_non_empty(self):
        """T-NODE5-03: Fallback brief has all 7 non-empty sections."""
        role_map = self._make_full_role_map()
        brief = self.node.build_fallback_brief(role_map, ["خلاف رئيسي"])
        assert brief.dispute_summary.strip()
        assert brief.uncontested_facts.strip()
        assert brief.key_disputes.strip()
        assert brief.party_requests.strip()
        assert brief.party_defenses.strip()
        assert brief.submitted_documents.strip()
        assert brief.legal_questions.strip()

    def test_uncontested_facts_excludes_themes_with_disputes(self):
        """T-NODE5-04: uncontested_facts only uses themes with no key_disputes."""
        role_map = {
            "الوقائع": [
                make_theme_summary("وقائع متفق عليها", "ملخص غير متنازع", key_disputes=[]),
                make_theme_summary("وقائع متنازع عليها", "ملخص متنازع", key_disputes=["خلاف"]),
            ]
        }
        brief = self.node.build_fallback_brief(role_map, [])
        # Should include "ملخص غير متنازع" but NOT "ملخص متنازع"
        assert "ملخص غير متنازع" in brief.uncontested_facts
        assert "ملخص متنازع" not in brief.uncontested_facts

    def test_legal_questions_from_legal_basis_role(self):
        """T-NODE5-10: legal_questions populated from الأساس القانوني key_disputes."""
        role_map = {
            "الأساس القانوني": [
                make_theme_summary("أساس قانوني", "نص", key_disputes=["مسألة عقدية", "مسألة تعويض"])
            ]
        }
        brief = self.node.build_fallback_brief(role_map, [])
        assert "مسألة عقدية" in brief.legal_questions
        assert "مسألة تعويض" in brief.legal_questions

    def test_legal_questions_fallback_to_all_disputes(self):
        """T-NODE5-11: No الأساس القانوني → legal_questions from key_disputes list."""
        role_map = {
            "الوقائع": [make_theme_summary("وقائع", "نص")]
        }
        key_disputes = ["مسألة قانونية عامة"]
        brief = self.node.build_fallback_brief(role_map, key_disputes)
        assert "مسألة قانونية عامة" in brief.legal_questions

    def test_no_info_when_role_missing(self):
        """When الطلبات is missing, party_requests contains the no-info string."""
        role_map = {}  # no roles at all
        brief = self.node.build_fallback_brief(role_map, [])
        assert "لا تتوفر" in brief.party_requests


@pytest.mark.unit
class TestCompileKeyDisputes:
    def setup_method(self):
        self.node = make_node5()

    def test_deduplication(self):
        """T-NODE5-05: Same dispute string from 2 roles appears once."""
        inputs = {
            "role_theme_summaries": [
                make_role_theme_summaries(
                    "الوقائع",
                    [make_theme_summary(key_disputes=["فسخ العقد", "التعويض"])]
                ),
                make_role_theme_summaries(
                    "الطلبات",
                    [make_theme_summary(key_disputes=["فسخ العقد", "إلزام المدعى عليه"])]
                ),
            ]
        }
        result = self.node.compile_key_disputes(inputs)
        assert result.count("فسخ العقد") == 1

    def test_empty_input_returns_empty(self):
        assert self.node.compile_key_disputes({"role_theme_summaries": []}) == []


@pytest.mark.unit
class TestCollectAllSources:
    def setup_method(self):
        self.node = make_node5()

    def test_deduplication(self):
        """T-NODE5-06: Same source from 2 themes appears once."""
        inputs = {
            "role_theme_summaries": [
                make_role_theme_summaries(
                    "الوقائع",
                    [
                        make_theme_summary(sources=["doc1 ص1 ف1", "doc2 ص2 ف2"]),
                        make_theme_summary(sources=["doc1 ص1 ف1", "doc3 ص3 ف3"]),
                    ]
                ),
            ]
        }
        sources = self.node.collect_all_sources(inputs)
        assert sources.count("doc1 ص1 ف1") == 1
        assert "doc2 ص2 ف2" in sources
        assert "doc3 ص3 ف3" in sources

    def test_empty_input_returns_empty(self):
        assert self.node.collect_all_sources({"role_theme_summaries": []}) == []


@pytest.mark.unit
class TestRenderBrief:
    def setup_method(self):
        self.node = make_node5()

    def test_all_7_arabic_headings_present(self):
        """T-NODE5-07: Rendered text contains all 7 Arabic section headings."""
        brief = make_case_brief()
        rendered = self.node.render_brief(brief, ["doc1 ص1 ف1"])
        assert "أولاً: ملخص النزاع" in rendered
        assert "ثانياً" in rendered
        assert "ثالثاً" in rendered
        assert "رابعاً" in rendered
        assert "خامساً" in rendered
        assert "سادساً" in rendered
        assert "سابعاً" in rendered

    def test_sources_in_rendered_output(self):
        """All sources appear in the rendered brief footer."""
        brief = make_case_brief()
        sources = ["doc1 ص1 ف1", "doc2 ص3 ف5"]
        rendered = self.node.render_brief(brief, sources)
        assert "doc1 ص1 ف1" in rendered
        assert "doc2 ص3 ف5" in rendered

    def test_empty_sources_message(self):
        """Empty sources list produces no-sources message."""
        brief = make_case_brief()
        rendered = self.node.render_brief(brief, [])
        assert "لا توجد مصادر" in rendered


@pytest.mark.unit
class TestProcess:
    def test_empty_input_returns_no_info_brief(self):
        """T-NODE5-08: Empty role_theme_summaries → all 7 fields = 'لا تتوفر معلومات كافية'."""
        node = make_node5()
        result = node.process({"role_theme_summaries": []})
        brief_dict = result["case_brief"]
        for field in ("dispute_summary", "uncontested_facts", "key_disputes",
                      "party_requests", "party_defenses", "submitted_documents", "legal_questions"):
            assert brief_dict[field] == "لا تتوفر معلومات كافية"

    def test_empty_input_has_rendered_brief(self):
        """Empty input still produces a rendered_brief string."""
        node = make_node5()
        result = node.process({"role_theme_summaries": []})
        assert isinstance(result["rendered_brief"], str)
        assert len(result["rendered_brief"]) > 0

    def test_llm_exception_uses_fallback(self):
        """T-NODE5-09: LLM exception → fallback brief with all 7 sections rendered."""
        node = make_node5(raises=RuntimeError("LLM unavailable"))
        inputs = {
            "role_theme_summaries": [
                make_role_theme_summaries(
                    "الوقائع",
                    [make_theme_summary("موضوع", "ملخص تجريبي")]
                )
            ]
        }
        result = node.process(inputs)
        rendered = result["rendered_brief"]
        # All 7 headings should still appear in fallback
        for heading in ["أولاً", "ثانياً", "ثالثاً", "رابعاً", "خامساً", "سادساً", "سابعاً"]:
            assert heading in rendered

    def test_output_keys_complete(self):
        """process() output has case_brief, all_sources, rendered_brief keys."""
        node = make_node5()
        result = node.process({"role_theme_summaries": []})
        assert "case_brief" in result
        assert "all_sources" in result
        assert "rendered_brief" in result

    def test_sources_deduped_in_output(self):
        """all_sources in output are deduplicated."""
        node = make_node5(raises=RuntimeError("use fallback"))
        inputs = {
            "role_theme_summaries": [
                make_role_theme_summaries(
                    "الوقائع",
                    [
                        make_theme_summary(sources=["doc1 ص1 ف1", "doc2 ص2 ف2"]),
                        make_theme_summary(sources=["doc1 ص1 ف1", "doc3 ص3 ف3"]),
                    ]
                )
            ]
        }
        result = node.process(inputs)
        sources = result["all_sources"]
        assert sources.count("doc1 ص1 ف1") == 1
