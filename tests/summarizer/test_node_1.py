"""
test_node_1.py — Unit tests for Summerize/node_1.py (Role Classifier)

Tests:
    T-NODE1-01: _build_messages() returns [SystemMessage, HumanMessage]
    T-NODE1-02: _build_messages() handles curly braces in chunk text
    T-NODE1-03: process_batch() maps all chunk_ids to classified output
    T-NODE1-04: process_batch() fallback on LLM exception
    T-NODE1-05: process() creates separate batches per (doc_type, party) group
    T-NODE1-06: process() all 7 legal roles can appear in output
    T-NODE1-07: process() empty input returns {"classified_chunks": []}
"""

import pathlib
import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage


from summarize.nodes.classifier import BatchClassificationResult, ClassificationItem, Node1_RoleClassifier


def make_chunk(
    chunk_id="c1",
    clean_text="نص قانوني",
    doc_type="صحيفة دعوى",
    party="المدعي",
    **kwargs,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": kwargs.get("doc_id", "doc1"),
        "page_number": kwargs.get("page_number", 1),
        "paragraph_number": kwargs.get("paragraph_number", 1),
        "clean_text": clean_text,
        "doc_type": doc_type,
        "party": party,
    }


def make_node1(parser_result=None, raises=None) -> Node1_RoleClassifier:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node1_RoleClassifier(llm)


@pytest.mark.unit
class TestBuildMessages:
    def setup_method(self):
        self.node = make_node1()

    def test_returns_two_messages(self):
        """T-NODE1-01: Returns list of length 2."""
        chunks = [make_chunk("c1"), make_chunk("c2")]
        doc_meta = {"doc_type": "صحيفة دعوى", "party": "المدعي"}
        messages = self.node._build_messages(chunks, doc_meta)
        assert len(messages) == 2

    def test_first_message_is_system(self):
        """T-NODE1-01: First message is SystemMessage."""
        messages = self.node._build_messages([make_chunk()], {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert isinstance(messages[0], SystemMessage)

    def test_second_message_is_human(self):
        """T-NODE1-01: Second message is HumanMessage."""
        messages = self.node._build_messages([make_chunk()], {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert isinstance(messages[1], HumanMessage)

    def test_system_message_contains_doc_type(self):
        """T-NODE1-01: SystemMessage content includes doc_type."""
        messages = self.node._build_messages([make_chunk()], {"doc_type": "محضر جلسة", "party": "المحكمة"})
        assert "محضر جلسة" in messages[0].content

    def test_system_message_contains_party(self):
        """T-NODE1-01: SystemMessage content includes party."""
        messages = self.node._build_messages([make_chunk()], {"doc_type": "صحيفة دعوى", "party": "خبير"})
        assert "خبير" in messages[0].content

    def test_human_message_contains_chunk_ids(self):
        """T-NODE1-01: HumanMessage contains chunk IDs."""
        chunks = [make_chunk("chunk-001"), make_chunk("chunk-002")]
        messages = self.node._build_messages(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert "chunk-001" in messages[1].content
        assert "chunk-002" in messages[1].content

    def test_curly_braces_in_text_no_error(self):
        """T-NODE1-02: Chunk text with { and } does not raise."""
        chunk = make_chunk(clean_text="النص {متغير} والكود {code}")
        doc_meta = {"doc_type": "صحيفة دعوى", "party": "المدعي"}
        # Should not raise KeyError or any formatting exception
        messages = self.node._build_messages([chunk], doc_meta)
        assert isinstance(messages[1], HumanMessage)


@pytest.mark.unit
class TestProcessBatch:
    def test_all_chunk_ids_in_output(self):
        """T-NODE1-03: Every input chunk_id appears in output."""
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        batch_result = BatchClassificationResult(
            classifications=[
                ClassificationItem(chunk_id=f"c{i}", role="الوقائع")
                for i in range(5)
            ]
        )
        node = make_node1(parser_result=batch_result)
        results = node.process_batch(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        output_ids = {r["chunk_id"] for r in results}
        assert output_ids == {f"c{i}" for i in range(5)}

    def test_output_has_role_field(self):
        """T-NODE1-03: Each output dict has 'role' key."""
        chunks = [make_chunk("c1")]
        batch_result = BatchClassificationResult(
            classifications=[ClassificationItem(chunk_id="c1", role="الطلبات")]
        )
        node = make_node1(parser_result=batch_result)
        results = node.process_batch(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert results[0]["role"] == "الطلبات"

    def test_output_has_confidence_field(self):
        """T-NODE1-03: Each output dict has 'confidence' field = 1.0."""
        chunks = [make_chunk("c1")]
        batch_result = BatchClassificationResult(
            classifications=[ClassificationItem(chunk_id="c1", role="الدفوع")]
        )
        node = make_node1(parser_result=batch_result)
        results = node.process_batch(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert results[0]["confidence"] == 1.0

    def test_fallback_on_llm_exception(self):
        """T-NODE1-04: LLM exception → all chunks marked غير محدد with confidence=0."""
        chunks = [make_chunk(f"c{i}") for i in range(3)]
        node = make_node1(raises=RuntimeError("LLM error"))
        results = node.process_batch(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        assert len(results) == 3
        for r in results:
            assert r["role"] == "غير محدد"
            assert r["confidence"] == 0.0

    def test_unknown_chunk_id_from_llm_gets_default_role(self):
        """Unknown chunk_id from LLM is handled; original chunk gets غير محدد."""
        chunks = [make_chunk("c1")]
        # LLM returns unknown ID
        batch_result = BatchClassificationResult(
            classifications=[ClassificationItem(chunk_id="PHANTOM", role="الوقائع")]
        )
        node = make_node1(parser_result=batch_result)
        results = node.process_batch(chunks, {"doc_type": "صحيفة دعوى", "party": "المدعي"})
        # c1 should still appear with default role
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["role"] == "غير محدد"  # role_map.get with default

    def test_original_chunk_data_preserved(self):
        """T-NODE1-03: Original chunk fields preserved in output (non-mutating)."""
        chunk = make_chunk("c1", clean_text="نص مهم", doc_type="محضر جلسة", party="المحكمة")
        batch_result = BatchClassificationResult(
            classifications=[ClassificationItem(chunk_id="c1", role="الإجراءات")]
        )
        node = make_node1(parser_result=batch_result)
        results = node.process_batch([chunk], {"doc_type": "محضر جلسة", "party": "المحكمة"})
        assert results[0]["clean_text"] == "نص مهم"
        assert results[0]["doc_type"] == "محضر جلسة"


@pytest.mark.unit
class TestProcess:
    def test_empty_input_returns_empty(self):
        """T-NODE1-07: Empty chunks input returns empty classified_chunks."""
        node = make_node1()
        result = node.process({"chunks": []})
        assert result == {"classified_chunks": []}

    def test_groups_by_doc_type_and_party(self):
        """T-NODE1-05: process() groups by (doc_type, party) and calls batch per group."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        chunks_group1 = [
            make_chunk(f"c{i}", doc_type="صحيفة دعوى", party="المدعي") for i in range(3)
        ]
        chunks_group2 = [
            make_chunk(f"d{i}", doc_type="مذكرة دفاع", party="المدعى عليه") for i in range(3)
        ]

        def side_effect(messages):
            return BatchClassificationResult(classifications=[
                ClassificationItem(chunk_id=c["chunk_id"], role="الوقائع")
                for c in (chunks_group1 + chunks_group2)
                if c["chunk_id"] in messages[1].content
            ])

        parser.invoke.return_value = BatchClassificationResult(classifications=[])
        node = Node1_RoleClassifier(llm)

        node.process({"chunks": chunks_group1 + chunks_group2})

        # Parser called at least twice (once per group)
        assert parser.invoke.call_count >= 2

    def test_all_roles_can_appear_in_output(self):
        """T-NODE1-06: All 7 LegalRoleEnum values can appear as output roles."""
        from typing import get_args
        from summarize.schemas import LegalRoleEnum
        all_roles = list(get_args(LegalRoleEnum))

        # Build 7 chunks, one per role
        chunks = [make_chunk(f"c{i}") for i in range(7)]
        batch_result = BatchClassificationResult(
            classifications=[
                ClassificationItem(chunk_id=f"c{i}", role=all_roles[i])
                for i in range(7)
            ]
        )
        node = make_node1(parser_result=batch_result)
        result = node.process({"chunks": chunks})
        output_roles = {c["role"] for c in result["classified_chunks"]}
        # All 7 roles should appear
        assert output_roles == set(all_roles)

    def test_output_count_matches_input(self):
        """Total classified chunks equals total input chunks."""
        chunks = [make_chunk(f"c{i}") for i in range(12)]
        batch_result = BatchClassificationResult(
            classifications=[
                ClassificationItem(chunk_id=f"c{i}", role="الوقائع")
                for i in range(12)
            ]
        )
        node = make_node1(parser_result=batch_result)
        result = node.process({"chunks": chunks})
        assert len(result["classified_chunks"]) == 12
