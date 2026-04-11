"""
test_node_3.py — Unit tests for Summerize/node_3.py (Aggregator)

Tests:
    T-NODE3-01: validate_coverage() with complete coverage — no IDs added
    T-NODE3-02: validate_coverage() duplicate ID removed from second bucket
    T-NODE3-03: validate_coverage() missing IDs added to party_specific
    T-NODE3-04: has_multiple_parties() True for 2 parties, False for 1
    T-NODE3-05: process_role() single-party shortcut (no LLM call)
    T-NODE3-06: process_role() batching for > MAX_BULLETS_PER_CALL
    T-NODE3-07: process_role() LLM fallback when exception raised
    T-NODE3-08: resolve_sources() deduplication
    T-NODE3-09: build_role_aggregation() output dict structure
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_SUMMARIZE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "Summerize"
if str(_SUMMARIZE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUMMARIZE_DIR))

from node_3 import (
    AgreedItemLLM,
    DisputedItemLLM,
    DisputeSideLLM,
    Node3_Aggregator,
    PartySpecificItemLLM,
    RoleAggregationLLM,
)


def make_bullet(
    bullet_id="b1",
    role="الوقائع",
    bullet="نص النقطة",
    source=None,
    party="المدعي",
    chunk_id="c1",
) -> dict:
    return {
        "bullet_id": bullet_id,
        "role": role,
        "bullet": bullet,
        "source": source or ["doc1 ص1 ف1"],
        "party": party,
        "chunk_id": chunk_id,
    }


def make_node3(parser_result=None, raises=None) -> Node3_Aggregator:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node3_Aggregator(llm)


@pytest.mark.unit
class TestHasMultipleParties:
    def setup_method(self):
        self.node = make_node3()

    def test_single_party_returns_false(self):
        """T-NODE3-04: Single party → False."""
        bullets = [
            make_bullet("b1", party="المدعي"),
            make_bullet("b2", party="المدعي"),
        ]
        assert self.node.has_multiple_parties(bullets) is False

    def test_two_parties_returns_true(self):
        """T-NODE3-04: Two parties → True."""
        bullets = [
            make_bullet("b1", party="المدعي"),
            make_bullet("b2", party="المدعى عليه"),
        ]
        assert self.node.has_multiple_parties(bullets) is True

    def test_three_parties_returns_true(self):
        bullets = [
            make_bullet("b1", party="المدعي"),
            make_bullet("b2", party="المدعى عليه"),
            make_bullet("b3", party="خبير"),
        ]
        assert self.node.has_multiple_parties(bullets) is True


@pytest.mark.unit
class TestValidateCoverage:
    def setup_method(self):
        self.node = make_node3()

    def _make_bullets(self, ids, party="المدعي"):
        return [make_bullet(bid, party=party) for bid in ids]

    def test_complete_coverage_no_additions(self):
        """T-NODE3-01: All IDs present, no missing IDs added."""
        bullets = self._make_bullets(["b1", "b2", "b3"])
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص", bullet_ids=["b1", "b2"])],
            disputed=[],
            party_specific=[PartySpecificItemLLM(party="المدعي", bullet_ids=["b3"], text="نص")],
        )
        result = self.node.validate_coverage(llm_result, {"b1", "b2", "b3"}, bullets)
        # No extra items added
        all_ids_out = (
            [bid for item in result.agreed for bid in item.bullet_ids]
            + [bid for item in result.party_specific for bid in item.bullet_ids]
        )
        assert set(all_ids_out) == {"b1", "b2", "b3"}

    def test_duplicate_id_removed_from_second_bucket(self):
        """T-NODE3-02: 'b1' in both agreed and party_specific → kept only in agreed."""
        bullets = self._make_bullets(["b1", "b2"])
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص", bullet_ids=["b1"])],
            disputed=[],
            party_specific=[PartySpecificItemLLM(party="المدعي", bullet_ids=["b1", "b2"], text="نص")],
        )
        result = self.node.validate_coverage(llm_result, {"b1", "b2"}, bullets)
        agreed_ids = [bid for item in result.agreed for bid in item.bullet_ids]
        party_ids = [bid for item in result.party_specific for bid in item.bullet_ids]
        assert "b1" in agreed_ids
        assert "b1" not in party_ids  # duplicate removed

    def test_missing_id_added_to_party_specific(self):
        """T-NODE3-03: Missing 'b3' added to party_specific with original bullet text."""
        bullets = [
            make_bullet("b1", party="المدعي", bullet="نقطة أولى"),
            make_bullet("b2", party="المدعي", bullet="نقطة ثانية"),
            make_bullet("b3", party="المدعى عليه", bullet="نقطة ثالثة مفقودة"),
        ]
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص", bullet_ids=["b1"])],
            disputed=[],
            party_specific=[PartySpecificItemLLM(party="المدعي", bullet_ids=["b2"], text="نص")],
        )
        result = self.node.validate_coverage(llm_result, {"b1", "b2", "b3"}, bullets)
        all_party_specific_ids = [bid for item in result.party_specific for bid in item.bullet_ids]
        assert "b3" in all_party_specific_ids

    def test_unknown_id_from_llm_is_dropped(self):
        """LLM returns an unknown bullet_id → it's dropped."""
        bullets = [make_bullet("b1")]
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص", bullet_ids=["GHOST_ID"])],
            disputed=[],
            party_specific=[PartySpecificItemLLM(party="المدعي", bullet_ids=["b1"], text="نص")],
        )
        result = self.node.validate_coverage(llm_result, {"b1"}, bullets)
        agreed_ids = [bid for item in result.agreed for bid in item.bullet_ids]
        assert "GHOST_ID" not in agreed_ids


@pytest.mark.unit
class TestProcessRole:
    def test_single_party_no_llm_call(self):
        """T-NODE3-05: Single party → all bullets in party_specific, no LLM."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node3_Aggregator(llm)

        bullets = [
            make_bullet("b1", party="المدعي", bullet="نقطة 1"),
            make_bullet("b2", party="المدعي", bullet="نقطة 2"),
            make_bullet("b3", party="المدعي", bullet="نقطة 3"),
        ]
        lookup = {b["bullet_id"]: b for b in bullets}
        result = node.process_role("الوقائع", bullets, lookup)

        # No LLM call
        assert not parser.invoke.called

        # All bullets in party_specific
        assert result["role"] == "الوقائع"
        assert result["agreed"] == []
        assert result["disputed"] == []
        assert len(result["party_specific"]) == 3

    def test_batching_for_large_role(self):
        """T-NODE3-06: > MAX_BULLETS_PER_CALL bullets triggers 2 LLM calls."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        node = Node3_Aggregator(llm)
        max_b = node.MAX_BULLETS_PER_CALL  # 50

        # 75 bullets from 2 parties (so LLM is called)
        bullets = []
        for i in range(max_b + 25):
            party = "المدعي" if i % 2 == 0 else "المدعى عليه"
            bullets.append(make_bullet(f"b{i}", party=party, bullet=f"نقطة {i}"))

        # Return a valid (empty-but-coverage-complete) result per batch
        def make_fallback_result(call_bullets):
            return RoleAggregationLLM(
                agreed=[],
                disputed=[],
                party_specific=[
                    PartySpecificItemLLM(
                        party=b["party"], bullet_ids=[b["bullet_id"]], text=b["bullet"]
                    )
                    for b in call_bullets
                ],
            )

        # Track what bullets each call receives
        call_bullets_tracker = []
        def side_effect(messages):
            # We can't easily introspect the exact bullets here, return empty agg
            return RoleAggregationLLM(agreed=[], disputed=[], party_specific=[])

        parser.invoke.side_effect = side_effect

        lookup = {b["bullet_id"]: b for b in bullets}
        node.process_role("الوقائع", bullets, lookup)

        # Should have been called twice: ceil(75/50) = 2
        assert parser.invoke.call_count == 2

    def test_llm_exception_uses_fallback(self):
        """T-NODE3-07: LLM raises → all bullets go to party_specific."""
        node = make_node3(raises=RuntimeError("LLM error"))
        bullets = [
            make_bullet("b1", party="المدعي"),
            make_bullet("b2", party="المدعى عليه"),
        ]
        lookup = {b["bullet_id"]: b for b in bullets}
        result = node.process_role("الوقائع", bullets, lookup)

        assert result["agreed"] == []
        assert result["disputed"] == []
        assert len(result["party_specific"]) == 2

    def test_output_has_required_structure(self):
        """T-NODE3-09: Output dict has role, agreed, disputed, party_specific."""
        bullets = [make_bullet("b1", party="المدعي")]
        lookup = {b["bullet_id"]: b for b in bullets}
        node = make_node3()
        result = node.process_role("الوقائع", bullets, lookup)
        assert "role" in result
        assert "agreed" in result
        assert "disputed" in result
        assert "party_specific" in result


@pytest.mark.unit
class TestResolveSources:
    def setup_method(self):
        self.node = make_node3()

    def test_deduplication(self):
        """T-NODE3-08: Sources from shared bullets deduplicated."""
        lookup = {
            "b1": {"source": ["doc1 ص1 ف1", "doc2 ص2 ف3"]},
            "b2": {"source": ["doc1 ص1 ف1", "doc3 ص3 ف5"]},  # doc1 shared
            "b3": {"source": ["doc4 ص4 ف7"]},
        }
        sources = self.node.resolve_sources(["b1", "b2", "b3"], lookup)
        # doc1 should appear only once
        assert sources.count("doc1 ص1 ف1") == 1
        assert len(sources) == 4  # doc1, doc2, doc3, doc4

    def test_missing_bullet_id_skipped(self):
        """bullet_id not in lookup is silently skipped."""
        lookup = {"b1": {"source": ["doc1 ص1 ف1"]}}
        sources = self.node.resolve_sources(["b1", "MISSING"], lookup)
        assert sources == ["doc1 ص1 ف1"]

    def test_empty_ids_returns_empty(self):
        assert self.node.resolve_sources([], {}) == []


@pytest.mark.unit
class TestBuildRoleAggregation:
    def setup_method(self):
        self.node = make_node3()

    def test_output_structure_keys(self):
        """T-NODE3-09: build_role_aggregation output has role, agreed, disputed, party_specific."""
        bullets = [make_bullet("b1", bullet="نقطة"), make_bullet("b2", bullet="نقطة أخرى")]
        lookup = {b["bullet_id"]: b for b in bullets}
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص متفق عليه", bullet_ids=["b1"])],
            disputed=[],
            party_specific=[PartySpecificItemLLM(party="المدعي", bullet_ids=["b2"], text="نص")],
        )
        result = self.node.build_role_aggregation("الوقائع", llm_result, lookup)
        assert "role" in result
        assert "agreed" in result
        assert "disputed" in result
        assert "party_specific" in result

    def test_agreed_items_have_text_and_sources(self):
        """Agreed items have 'text' and 'sources' keys."""
        bullet = make_bullet("b1", bullet="نقطة", source=["doc1 ص1 ف1"])
        lookup = {"b1": bullet}
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص متفق عليه", bullet_ids=["b1"])],
            disputed=[],
            party_specific=[],
        )
        result = self.node.build_role_aggregation("الوقائع", llm_result, lookup)
        assert len(result["agreed"]) == 1
        assert "text" in result["agreed"][0]
        assert "sources" in result["agreed"][0]

    def test_empty_bullet_ids_item_excluded(self):
        """Items with empty bullet_ids (after validation) are excluded from output."""
        lookup = {}
        llm_result = RoleAggregationLLM(
            agreed=[AgreedItemLLM(text="نص", bullet_ids=[])],  # empty
            disputed=[],
            party_specific=[],
        )
        result = self.node.build_role_aggregation("الوقائع", llm_result, lookup)
        assert result["agreed"] == []
