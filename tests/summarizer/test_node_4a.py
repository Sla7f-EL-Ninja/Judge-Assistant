"""
test_node_4a.py — Unit tests for Summerize/node_4a.py (Thematic Clustering)

Tests:
    T-NODE4A-01: assign_item_ids() produces unique IDs with correct type prefixes
    T-NODE4A-02: validate_coverage() missing IDs added to 'أخرى'
    T-NODE4A-03: validate_coverage() duplicate ID kept only in first theme
    T-NODE4A-04: process_role() skips LLM for < MIN_ITEMS_FOR_CLUSTERING items
    T-NODE4A-05: process_role() multi-batch passes existing_theme_names to later batches
    T-NODE4A-06: process_role() LLM exception → single-theme fallback
    T-NODE4A-07: reconstruct_themed_role() routes items to correct type lists
"""

import pathlib
import sys
from unittest.mock import MagicMock

import pytest

_SUMMARIZE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "Summerize"
if str(_SUMMARIZE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUMMARIZE_DIR))

from node_4a import ClusteringResultLLM, Node4A_ThematicClustering, ThemeAssignmentLLM


def make_role_agg(
    role="الوقائع",
    agreed=None,
    disputed=None,
    party_specific=None,
) -> dict:
    return {
        "role": role,
        "agreed": agreed or [],
        "disputed": disputed or [],
        "party_specific": party_specific or [],
    }


def make_agreed(text="نص متفق عليه", sources=None):
    return {"text": text, "sources": sources or ["doc1 ص1 ف1"]}


def make_disputed(subject="موضوع خلاف", positions=None):
    return {"subject": subject, "positions": positions or []}


def make_party_specific(party="المدعي", text="نص خاص", sources=None):
    return {"party": party, "text": text, "sources": sources or ["doc1 ص1 ف1"]}


def make_node4a(parser_result=None, raises=None) -> Node4A_ThematicClustering:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node4A_ThematicClustering(llm)


@pytest.mark.unit
class TestAssignItemIds:
    def setup_method(self):
        self.node = make_node4a()

    def test_agreed_ids_have_agreed_prefix(self):
        """T-NODE4A-01: Agreed items get 'agreed-NNN-hex' IDs."""
        agg = make_role_agg(agreed=[make_agreed(), make_agreed()])
        lookup, items = self.node.assign_item_ids(agg)
        agreed_ids = [iid for iid in lookup if lookup[iid]["type"] == "agreed"]
        for iid in agreed_ids:
            assert iid.startswith("agreed-")

    def test_disputed_ids_have_disputed_prefix(self):
        """T-NODE4A-01: Disputed items get 'disputed-NNN-hex' IDs."""
        agg = make_role_agg(disputed=[make_disputed()])
        lookup, items = self.node.assign_item_ids(agg)
        disputed_ids = [iid for iid in lookup if lookup[iid]["type"] == "disputed"]
        for iid in disputed_ids:
            assert iid.startswith("disputed-")

    def test_party_specific_ids_have_party_prefix(self):
        """T-NODE4A-01: Party-specific items get 'party-NNN-hex' IDs."""
        agg = make_role_agg(party_specific=[make_party_specific()])
        lookup, items = self.node.assign_item_ids(agg)
        party_ids = [iid for iid in lookup if lookup[iid]["type"] == "party_specific"]
        for iid in party_ids:
            assert iid.startswith("party-")

    def test_all_ids_unique(self):
        """T-NODE4A-01: All generated IDs are unique across types."""
        agg = make_role_agg(
            agreed=[make_agreed(), make_agreed(), make_agreed()],
            disputed=[make_disputed(), make_disputed()],
            party_specific=[make_party_specific(), make_party_specific(), make_party_specific(), make_party_specific()],
        )
        lookup, items = self.node.assign_item_ids(agg)
        ids = list(lookup.keys())
        assert len(set(ids)) == len(ids) == 9

    def test_items_with_ids_count_matches_total(self):
        """items_with_ids count matches total number of items across all types."""
        agg = make_role_agg(
            agreed=[make_agreed()],
            disputed=[make_disputed()],
            party_specific=[make_party_specific(), make_party_specific()],
        )
        lookup, items = self.node.assign_item_ids(agg)
        assert len(items) == 4
        assert len(lookup) == 4

    def test_lookup_entries_have_type_and_data(self):
        """Each lookup entry has 'type' and 'data' keys."""
        agg = make_role_agg(agreed=[make_agreed()])
        lookup, _ = self.node.assign_item_ids(agg)
        for entry in lookup.values():
            assert "type" in entry
            assert "data" in entry


@pytest.mark.unit
class TestValidateCoverage:
    def setup_method(self):
        self.node = make_node4a()

    def test_missing_ids_added_to_okhra(self):
        """T-NODE4A-02: Missing item_ids added to 'أخرى' fallback theme."""
        all_ids = {"id1", "id2", "id3"}
        merged = {"theme_A": ["id1", "id2"]}  # id3 missing
        result = self.node.validate_coverage(merged, all_ids)
        assert "id3" in result.get("أخرى", [])

    def test_duplicate_id_kept_in_first_theme(self):
        """T-NODE4A-03: Duplicate id kept only in the first theme that claims it."""
        all_ids = {"id1", "id2"}
        merged = {
            "theme_A": ["id1", "id2"],
            "theme_B": ["id1"],  # id1 duplicated
        }
        result = self.node.validate_coverage(merged, all_ids)
        all_theme_a = result.get("theme_A", [])
        all_theme_b = result.get("theme_B", [])
        assert "id1" in all_theme_a
        assert "id1" not in all_theme_b

    def test_unknown_id_from_llm_dropped(self):
        """Unknown item_id from LLM is dropped."""
        all_ids = {"id1"}
        merged = {"theme_A": ["id1", "PHANTOM"]}
        result = self.node.validate_coverage(merged, all_ids)
        assert "PHANTOM" not in result.get("theme_A", [])

    def test_complete_coverage_no_okhra(self):
        """When all IDs covered, 'أخرى' theme not created."""
        all_ids = {"id1", "id2"}
        merged = {"theme_A": ["id1"], "theme_B": ["id2"]}
        result = self.node.validate_coverage(merged, all_ids)
        assert "أخرى" not in result

    def test_empty_theme_after_cleanup_excluded(self):
        """Theme with all IDs dropped (unknown) produces no entry."""
        all_ids = {"id1"}
        merged = {"ghost_theme": ["PHANTOM1", "PHANTOM2"]}
        result = self.node.validate_coverage(merged, all_ids)
        assert "ghost_theme" not in result


@pytest.mark.unit
class TestProcessRole:
    def test_small_role_skips_clustering(self):
        """T-NODE4A-04: < MIN_ITEMS_FOR_CLUSTERING items → single theme, no LLM."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node4A_ThematicClustering(llm)

        min_items = node.MIN_ITEMS_FOR_CLUSTERING  # 6
        # 4 items total (< 6)
        agg = make_role_agg(
            role="الطلبات",
            agreed=[make_agreed() for _ in range(2)],
            party_specific=[make_party_specific() for _ in range(2)],
        )
        result = node.process_role(agg)

        assert not parser.invoke.called
        assert result["role"] == "الطلبات"
        assert len(result["themes"]) == 1

    def test_llm_exception_returns_single_theme(self):
        """T-NODE4A-06: LLM exception → all items in single theme named after role."""
        node = make_node4a(raises=RuntimeError("LLM failed"))
        agg = make_role_agg(
            role="الدفوع",
            agreed=[make_agreed() for _ in range(3)],
            disputed=[make_disputed() for _ in range(2)],
            party_specific=[make_party_specific() for _ in range(3)],
        )
        result = node.process_role(agg)
        assert result["role"] == "الدفوع"
        assert len(result["themes"]) == 1
        assert result["themes"][0]["theme_name"] == "الدفوع"

    def test_output_has_role_and_themes_keys(self):
        """Output dict has 'role' and 'themes' keys."""
        parser = MagicMock()
        parser.invoke.return_value = ClusteringResultLLM(
            themes=[ThemeAssignmentLLM(theme_name="موضوع", item_ids=[])]
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node4A_ThematicClustering(llm)

        agg = make_role_agg(role="الوقائع", party_specific=[make_party_specific()])
        result = node.process_role(agg)
        assert "role" in result
        assert "themes" in result

    def test_multi_batch_calls_cluster_twice(self):
        """T-NODE4A-05: > MAX_ITEMS_PER_CALL items → 2 LLM calls."""
        parser = MagicMock()
        parser.invoke.return_value = ClusteringResultLLM(themes=[])
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node4A_ThematicClustering(llm)

        max_items = node.MAX_ITEMS_PER_CALL  # 50
        agg = make_role_agg(
            role="الوقائع",
            agreed=[make_agreed() for _ in range(max_items + 10)],
        )
        node.process_role(agg)
        assert parser.invoke.call_count == 2


@pytest.mark.unit
class TestReconstructThemedRole:
    def setup_method(self):
        self.node = make_node4a()

    def test_agreed_item_goes_to_agreed_list(self):
        """T-NODE4A-07: Items with type='agreed' placed in theme.agreed list."""
        agreed_data = make_agreed(text="نص متفق عليه")
        id_lookup = {"agreed-001-abc": {"type": "agreed", "data": agreed_data}}
        merged = {"موضوع أول": ["agreed-001-abc"]}
        result = self.node.reconstruct_themed_role("الوقائع", merged, id_lookup)
        theme = result["themes"][0]
        assert len(theme["agreed"]) == 1
        assert theme["agreed"][0]["text"] == "نص متفق عليه"

    def test_disputed_item_goes_to_disputed_list(self):
        """T-NODE4A-07: Items with type='disputed' placed in theme.disputed list."""
        disputed_data = make_disputed(subject="موضوع النزاع")
        id_lookup = {"disputed-001-abc": {"type": "disputed", "data": disputed_data}}
        merged = {"موضوع ثانٍ": ["disputed-001-abc"]}
        result = self.node.reconstruct_themed_role("الوقائع", merged, id_lookup)
        theme = result["themes"][0]
        assert len(theme["disputed"]) == 1
        assert theme["disputed"][0]["subject"] == "موضوع النزاع"

    def test_party_specific_item_goes_to_party_specific_list(self):
        """T-NODE4A-07: Items with type='party_specific' in party_specific list."""
        ps_data = make_party_specific(text="نص خاص")
        id_lookup = {"party-001-abc": {"type": "party_specific", "data": ps_data}}
        merged = {"موضوع ثالث": ["party-001-abc"]}
        result = self.node.reconstruct_themed_role("الوقائع", merged, id_lookup)
        theme = result["themes"][0]
        assert len(theme["party_specific"]) == 1
        assert theme["party_specific"][0]["text"] == "نص خاص"

    def test_bullet_count_in_theme(self):
        """bullet_count equals number of items in theme."""
        id_lookup = {
            "agreed-001-aaa": {"type": "agreed", "data": make_agreed()},
            "party-001-bbb": {"type": "party_specific", "data": make_party_specific()},
        }
        merged = {"موضوع": ["agreed-001-aaa", "party-001-bbb"]}
        result = self.node.reconstruct_themed_role("الوقائع", merged, id_lookup)
        assert result["themes"][0]["bullet_count"] == 2
