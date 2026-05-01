"""
test_data_contracts.py — Integration tests for inter-node data contracts.

Tests:
    T-CONTRACT-01: Node 0 output chunks contain all keys Node 1 accesses
    T-CONTRACT-02: Node 2 output bullets contain all keys Node 3 accesses
    T-CONTRACT-03: Source strings match citation pattern from build_citation()
"""

import pathlib
import re
import sys
from unittest.mock import MagicMock

import pytest


from summarize.nodes.intake import Node0_DocumentIntake
from summarize.nodes.classifier import BatchClassificationResult, ClassificationItem, Node1_RoleClassifier
from summarize.nodes.extractor import BatchBulletResult, ChunkBullets, Node2_BulletExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node0_with_heuristic() -> Node0_DocumentIntake:
    """Node 0 that can classify صحيفة دعوى via heuristic (no LLM call)."""
    parser = MagicMock()
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node0_DocumentIntake(llm)


def make_node2_with_result(chunk_ids, bullets_per_chunk=1) -> Node2_BulletExtractor:
    batch_result = BatchBulletResult(
        extractions=[
            ChunkBullets(chunk_id=cid, bullets=[f"نقطة قانونية من {cid}"])
            for cid in chunk_ids
        ]
    )
    parser = MagicMock()
    parser.invoke.return_value = batch_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node2_BulletExtractor(llm)


# ---------------------------------------------------------------------------
# T-CONTRACT-01: Node 0 → Node 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNode0ToNode1Contract:
    """T-CONTRACT-01: Node 0 output satisfies Node 1 input contract."""

    NODE1_REQUIRED_KEYS = {"chunk_id", "doc_id", "page_number", "paragraph_number", "clean_text", "doc_type", "party"}

    def test_chunks_contain_chunk_id(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص القضية", "doc_id": "test"})
        for chunk in result["chunks"]:
            assert "chunk_id" in chunk

    def test_chunks_contain_doc_id(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص", "doc_id": "contract-test"})
        for chunk in result["chunks"]:
            assert "doc_id" in chunk
            assert chunk["doc_id"] == "contract-test"

    def test_chunks_contain_clean_text(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nمحتوى مهم", "doc_id": "t"})
        for chunk in result["chunks"]:
            assert "clean_text" in chunk
            assert isinstance(chunk["clean_text"], str)

    def test_chunks_contain_doc_type(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص قانوني", "doc_id": "t"})
        for chunk in result["chunks"]:
            assert "doc_type" in chunk

    def test_chunks_contain_party(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص قانوني", "doc_id": "t"})
        for chunk in result["chunks"]:
            assert "party" in chunk

    def test_all_required_keys_present(self):
        """T-CONTRACT-01: All 7 keys that Node 1 accesses are in each chunk."""
        node = make_node0_with_heuristic()
        result = node.process({
            "raw_text": "صحيفة دعوى مقدمة من المدعي\n\nيدعي المدعي بأن العقد أخل به المدعى عليه",
            "doc_id": "full-contract"
        })
        for chunk in result["chunks"]:
            assert self.NODE1_REQUIRED_KEYS.issubset(chunk.keys()), (
                f"Missing keys: {self.NODE1_REQUIRED_KEYS - chunk.keys()}"
            )

    def test_chunk_id_is_str(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص", "doc_id": "t"})
        for chunk in result["chunks"]:
            assert isinstance(chunk["chunk_id"], str)

    def test_page_number_is_positive_int(self):
        node = make_node0_with_heuristic()
        result = node.process({"raw_text": "صحيفة دعوى من المدعي\n\nنص قانوني مهم", "doc_id": "t"})
        for chunk in result["chunks"]:
            assert isinstance(chunk["page_number"], int)
            assert chunk["page_number"] >= 1


# ---------------------------------------------------------------------------
# T-CONTRACT-02: Node 2 → Node 3
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNode2ToNode3Contract:
    """T-CONTRACT-02: Node 2 output satisfies Node 3 input contract."""

    NODE3_REQUIRED_KEYS = {"bullet_id", "role", "party", "bullet", "source", "chunk_id"}

    def _make_classified_chunks(self, n=3):
        return [
            {
                "chunk_id": f"c{i}",
                "doc_id": "doc1",
                "page_number": 1,
                "paragraph_number": i + 1,
                "clean_text": f"نص قانوني رقم {i}",
                "doc_type": "صحيفة دعوى",
                "party": "المدعي",
                "role": "الوقائع",
                "confidence": 1.0,
            }
            for i in range(n)
        ]

    def test_bullets_contain_bullet_id(self):
        chunks = self._make_classified_chunks(3)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert "bullet_id" in b

    def test_bullets_contain_role(self):
        chunks = self._make_classified_chunks(3)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert "role" in b

    def test_bullets_contain_party(self):
        chunks = self._make_classified_chunks(3)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert "party" in b

    def test_bullets_contain_bullet_text(self):
        chunks = self._make_classified_chunks(3)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert "bullet" in b
            assert isinstance(b["bullet"], str)

    def test_bullets_contain_source_list(self):
        chunks = self._make_classified_chunks(3)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert "source" in b
            assert isinstance(b["source"], list)

    def test_all_required_keys_present(self):
        """T-CONTRACT-02: All 6 keys that Node 3 accesses are in each bullet."""
        chunks = self._make_classified_chunks(5)
        node = make_node2_with_result([c["chunk_id"] for c in chunks])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            assert self.NODE3_REQUIRED_KEYS.issubset(b.keys()), (
                f"Missing keys: {self.NODE3_REQUIRED_KEYS - b.keys()}"
            )


# ---------------------------------------------------------------------------
# T-CONTRACT-03: Source format consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSourceFormat:
    """T-CONTRACT-03: Source citation format matches '{doc_id} ص{N} ف{N}'."""

    CITATION_PATTERN = re.compile(r".+ ص\d+ ف\d+$")

    def _make_classified_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "doc_id": "صحيفة_دعوى",
                "page_number": 3,
                "paragraph_number": 7,
                "clean_text": "نص قانوني",
                "doc_type": "صحيفة دعوى",
                "party": "المدعي",
                "role": "الوقائع",
                "confidence": 1.0,
            }
        ]

    def test_source_matches_citation_pattern(self):
        """T-CONTRACT-03: Source strings match '{doc_id} ص{N} ف{N}' pattern."""
        chunks = self._make_classified_chunks()
        node = make_node2_with_result(["c1"])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            for src in b["source"]:
                assert self.CITATION_PATTERN.match(src), (
                    f"Source '{src}' does not match citation pattern"
                )

    def test_source_contains_arabic_page_marker(self):
        """Source strings contain Arabic page marker 'ص'."""
        chunks = self._make_classified_chunks()
        node = make_node2_with_result(["c1"])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            for src in b["source"]:
                assert "ص" in src

    def test_source_contains_arabic_para_marker(self):
        """Source strings contain Arabic paragraph marker 'ف'."""
        chunks = self._make_classified_chunks()
        node = make_node2_with_result(["c1"])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            for src in b["source"]:
                assert "ف" in src

    def test_source_contains_correct_doc_id(self):
        """Source string begins with the chunk's doc_id."""
        chunks = self._make_classified_chunks()
        node = make_node2_with_result(["c1"])
        result = node.process({"classified_chunks": chunks})
        for b in result["bullets"]:
            for src in b["source"]:
                assert src.startswith("صحيفة_دعوى")
