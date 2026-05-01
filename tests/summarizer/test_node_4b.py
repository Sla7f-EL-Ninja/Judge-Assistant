"""
test_node_4b.py — Unit tests for Summerize/node_4b.py (Theme Synthesis)

Tests:
    T-NODE4B-01: collect_sources() deduplicates across agreed/disputed/party_specific
    T-NODE4B-02: synthesize_theme() returns LLM summary when non-empty
    T-NODE4B-03: synthesize_theme() uses fallback when LLM returns empty summary
    T-NODE4B-04: synthesize_theme() extracts dispute subjects when key_disputes empty
    T-NODE4B-05: synthesize_theme() exception → fallback summary with sources preserved
    T-NODE4B-06: process_role() preserves original theme order (concurrency safety)
    T-NODE4B-07: process_role() one theme failure doesn't cascade to others
    T-NODE4B-08: process_role() empty themes returns empty theme_summaries
    T-NODE4B-09: build_fallback_summary() starts with [ملخص خام - يحتاج مراجعة]
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest


from summarize.nodes.synthesis import Node4B_ThemeSynthesis, SynthesisResultLLM


def make_theme_cluster(
    theme_name="موضوع تجريبي",
    agreed=None,
    disputed=None,
    party_specific=None,
    bullet_count=1,
) -> dict:
    return {
        "theme_name": theme_name,
        "agreed": agreed or [],
        "disputed": disputed or [],
        "party_specific": party_specific or [],
        "bullet_count": bullet_count,
    }


def make_agreed(text="نص متفق", sources=None):
    return {"text": text, "sources": sources or ["doc1 ص1 ف1"]}


def make_disputed(subject="موضوع خلاف", positions=None):
    return {"subject": subject, "positions": positions or []}


def make_party_specific(party="المدعي", text="نص خاص", sources=None):
    return {"party": party, "text": text, "sources": sources or ["doc2 ص2 ف2"]}


def make_node4b(parser_result=None, raises=None) -> Node4B_ThemeSynthesis:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node4B_ThemeSynthesis(llm)


@pytest.mark.unit
class TestCollectSources:
    def setup_method(self):
        self.node = make_node4b()

    def test_deduplication_across_types(self):
        """T-NODE4B-01: Source appearing in both agreed and disputed appears once."""
        cluster = make_theme_cluster(
            agreed=[make_agreed(sources=["doc1 ص1 ف1", "doc2 ص2 ف2"])],
            disputed=[make_disputed(positions=[
                {"party": "المدعي", "sources": ["doc1 ص1 ف1", "doc3 ص3 ف3"], "bullets": []}
            ])],
        )
        sources = self.node.collect_sources(cluster)
        assert sources.count("doc1 ص1 ف1") == 1

    def test_all_types_included(self):
        """Sources from agreed, disputed, and party_specific all collected."""
        cluster = make_theme_cluster(
            agreed=[make_agreed(sources=["src-a"])],
            disputed=[make_disputed(positions=[
                {"party": "المدعي", "sources": ["src-b"], "bullets": []}
            ])],
            party_specific=[make_party_specific(sources=["src-c"])],
        )
        sources = self.node.collect_sources(cluster)
        assert "src-a" in sources
        assert "src-b" in sources
        assert "src-c" in sources

    def test_empty_cluster_returns_empty(self):
        cluster = make_theme_cluster()
        assert self.node.collect_sources(cluster) == []

    def test_order_preserved_first_occurrence(self):
        """Sources appear in order of first occurrence."""
        cluster = make_theme_cluster(
            agreed=[make_agreed(sources=["src-1", "src-2"])],
            party_specific=[make_party_specific(sources=["src-2", "src-3"])],
        )
        sources = self.node.collect_sources(cluster)
        assert sources.index("src-1") < sources.index("src-2")
        assert sources.index("src-2") < sources.index("src-3")


@pytest.mark.unit
class TestSynthesizeTheme:
    def test_returns_llm_summary_when_non_empty(self):
        """T-NODE4B-02: LLM returns non-empty summary → used as-is."""
        llm_result = SynthesisResultLLM(
            summary="ملخص قانوني وافٍ بالموضوع",
            key_disputes=["نقطة خلاف أولى"],
            sentences=[],
        )
        node = make_node4b(parser_result=llm_result)
        cluster = make_theme_cluster(theme_name="موضوع العقد")
        result = node.synthesize_theme(cluster, "الوقائع")
        assert result["summary"] == "ملخص قانوني وافٍ بالموضوع"
        assert result["key_disputes"] == ["نقطة خلاف أولى"]
        assert result["theme"] == "موضوع العقد"

    def test_empty_summary_triggers_fallback(self):
        """T-NODE4B-03: LLM returns empty string → fallback summary used."""
        llm_result = SynthesisResultLLM(summary="", key_disputes=[], sentences=[])
        node = make_node4b(parser_result=llm_result)
        cluster = make_theme_cluster(
            theme_name="موضوع",
            agreed=[make_agreed(text="نقطة متفق عليها")],
        )
        result = node.synthesize_theme(cluster, "الوقائع")
        assert "[ملخص خام - يحتاج مراجعة]" in result["summary"]

    def test_extracts_dispute_subjects_when_key_disputes_empty(self):
        """T-NODE4B-04: LLM returns empty key_disputes but disputed items exist."""
        llm_result = SynthesisResultLLM(
            summary="ملخص",
            key_disputes=[],  # empty
            sentences=[]
        )
        node = make_node4b(parser_result=llm_result)
        cluster = make_theme_cluster(
            disputed=[
                make_disputed(subject="مسألة الملكية"),
                make_disputed(subject="مسألة التسليم"),
            ]
        )
        result = node.synthesize_theme(cluster, "الوقائع")
        assert "مسألة الملكية" in result["key_disputes"]
        assert "مسألة التسليم" in result["key_disputes"]

    def test_exception_uses_fallback_with_sources(self):
        """T-NODE4B-05: LLM exception → fallback summary, original sources preserved."""
        node = make_node4b(raises=RuntimeError("LLM error"))
        cluster = make_theme_cluster(
            theme_name="موضوع",
            agreed=[make_agreed(sources=["doc1 ص1 ف1"])],
            disputed=[make_disputed(subject="موضوع النزاع")],
        )
        result = node.synthesize_theme(cluster, "الوقائع")
        assert "[ملخص خام - يحتاج مراجعة]" in result["summary"]
        assert "doc1 ص1 ف1" in result["sources"]
        assert "موضوع النزاع" in result["key_disputes"]

    def test_output_has_required_keys(self):
        """Output dict has: theme, summary, key_disputes, sources."""
        llm_result = SynthesisResultLLM(summary="ملخص", key_disputes=[], sentences=[])
        node = make_node4b(parser_result=llm_result)
        result = node.synthesize_theme(make_theme_cluster(), "الوقائع")
        assert "theme" in result
        assert "summary" in result
        assert "key_disputes" in result
        assert "sources" in result


@pytest.mark.unit
class TestProcessRole:
    def test_empty_themes_returns_empty(self):
        """T-NODE4B-08: Themed role with no themes → empty theme_summaries."""
        node = make_node4b()
        themed_role = {"role": "الوقائع", "themes": []}
        result = node.process_role(themed_role)
        assert result == {"role": "الوقائع", "theme_summaries": []}

    def test_output_order_matches_input_order(self):
        """T-NODE4B-06: Output theme_summaries[i] corresponds to input themes[i]."""
        theme_names = ["موضوع أول", "موضوع ثاني", "موضوع ثالث", "موضوع رابع", "موضوع خامس"]

        def make_result(theme_cluster, role):
            return {
                "theme": theme_cluster["theme_name"],
                "summary": f"ملخص {theme_cluster['theme_name']}",
                "key_disputes": [],
                "sources": [],
            }

        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node4B_ThemeSynthesis(llm)

        # Patch synthesize_theme to return a result based on theme name
        with patch.object(node, "synthesize_theme", side_effect=make_result):
            themed_role = {
                "role": "الوقائع",
                "themes": [make_theme_cluster(tn) for tn in theme_names],
            }
            result = node.process_role(themed_role)

        output_themes = [ts["theme"] for ts in result["theme_summaries"]]
        assert output_themes == theme_names

    def test_one_theme_failure_isolated(self):
        """T-NODE4B-07: Exception for one theme doesn't affect others."""
        call_count = [0]

        def synthesize_side_effect(theme_cluster, role):
            call_count[0] += 1
            if theme_cluster["theme_name"] == "موضوع_يفشل":
                raise RuntimeError("Simulated failure")
            return {
                "theme": theme_cluster["theme_name"],
                "summary": "ملخص",
                "key_disputes": [],
                "sources": [],
            }

        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node4B_ThemeSynthesis(llm)

        themes = [
            make_theme_cluster("موضوع_أول"),
            make_theme_cluster("موضوع_يفشل"),
            make_theme_cluster("موضوع_ثالث"),
        ]
        themed_role = {"role": "الوقائع", "themes": themes}

        with patch.object(node, "synthesize_theme", side_effect=synthesize_side_effect):
            result = node.process_role(themed_role)

        assert len(result["theme_summaries"]) == 3
        # Failed theme should still be in output (as fallback)
        themes_in_output = [ts["theme"] for ts in result["theme_summaries"]]
        assert "موضوع_يفشل" in themes_in_output
        assert "موضوع_أول" in themes_in_output
        assert "موضوع_ثالث" in themes_in_output


@pytest.mark.unit
class TestBuildFallbackSummary:
    def setup_method(self):
        self.node = make_node4b()

    def test_starts_with_raw_prefix(self):
        """T-NODE4B-09: Fallback summary starts with [ملخص خام - يحتاج مراجعة]."""
        cluster = make_theme_cluster(agreed=[make_agreed()])
        result = self.node.build_fallback_summary(cluster)
        assert result.startswith("[ملخص خام - يحتاج مراجعة]")

    def test_includes_agreed_text(self):
        """Agreed items appear in fallback."""
        cluster = make_theme_cluster(agreed=[make_agreed(text="وقائع ثابتة")])
        result = self.node.build_fallback_summary(cluster)
        assert "وقائع ثابتة" in result

    def test_includes_disputed_subject(self):
        """Disputed subjects appear in fallback."""
        cluster = make_theme_cluster(disputed=[make_disputed(subject="مسألة الملكية")])
        result = self.node.build_fallback_summary(cluster)
        assert "مسألة الملكية" in result

    def test_includes_party_specific_text(self):
        """Party-specific text appears in fallback."""
        cluster = make_theme_cluster(party_specific=[make_party_specific(text="موقف المدعي")])
        result = self.node.build_fallback_summary(cluster)
        assert "موقف المدعي" in result

    def test_empty_cluster_returns_just_prefix(self):
        """Empty cluster produces only the prefix line."""
        cluster = make_theme_cluster()
        result = self.node.build_fallback_summary(cluster)
        assert result.startswith("[ملخص خام - يحتاج مراجعة]")
