"""
test_node_2.py — Unit tests for Summerize/node_2.py (Bullet Extractor)

Tests:
    T-NODE2-01: build_citation() format "{doc_id} ص{page} ف{para}"
    T-NODE2-02: process_batch() produces bullets for all input chunk_ids
    T-NODE2-03: process_batch() drops bullets with unknown chunk_ids
    T-NODE2-04: process_batch() creates fallback bullet for missed chunk_id
    T-NODE2-05: process_batch() full exception fallback wraps all chunks
    T-NODE2-06: process() filters out chunks with empty clean_text
    T-NODE2-07: process() bullet_id uniqueness across all output bullets
"""

import pathlib
import sys
from unittest.mock import MagicMock

import pytest


from summarize.nodes.extractor import BatchBulletResult, ChunkBullets, Node2_BulletExtractor


def make_classified_chunk(
    chunk_id="c1",
    doc_id="doc1",
    page_number=1,
    paragraph_number=1,
    clean_text="نص قانوني تجريبي",
    doc_type="صحيفة دعوى",
    party="المدعي",
    role="الوقائع",
    confidence=1.0,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "page_number": page_number,
        "paragraph_number": paragraph_number,
        "clean_text": clean_text,
        "doc_type": doc_type,
        "party": party,
        "role": role,
        "confidence": confidence,
    }


def make_node2(parser_result=None, raises=None) -> Node2_BulletExtractor:
    parser = MagicMock()
    if raises:
        parser.invoke.side_effect = raises
    elif parser_result is not None:
        parser.invoke.return_value = parser_result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return Node2_BulletExtractor(llm)


@pytest.mark.unit
class TestBuildCitation:
    def setup_method(self):
        self.node = make_node2()

    def test_format_correct(self):
        """T-NODE2-01: Citation format is '{doc_id} ص{page} ف{para}'."""
        chunk = {"doc_id": "doc1", "page_number": 3, "paragraph_number": 7}
        assert self.node.build_citation(chunk) == "doc1 ص3 ف7"

    def test_page_1_para_1(self):
        chunk = {"doc_id": "صحيفة_دعوى", "page_number": 1, "paragraph_number": 1}
        assert self.node.build_citation(chunk) == "صحيفة_دعوى ص1 ف1"

    def test_arabic_doc_id(self):
        chunk = {"doc_id": "تقرير_الخبير", "page_number": 5, "paragraph_number": 12}
        result = self.node.build_citation(chunk)
        assert result == "تقرير_الخبير ص5 ف12"
        assert "ص" in result
        assert "ف" in result


@pytest.mark.unit
class TestProcessBatch:
    def test_bullets_produced_for_all_chunks(self):
        """T-NODE2-02: At least one bullet per input chunk_id."""
        chunks = [make_classified_chunk(f"c{i}") for i in range(3)]
        batch_result = BatchBulletResult(
            extractions=[
                ChunkBullets(chunk_id=f"c{i}", bullets=["نقطة قانونية مهمة"])
                for i in range(3)
            ]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        output_chunk_ids = {r["chunk_id"] for r in results}
        assert output_chunk_ids == {"c0", "c1", "c2"}

    def test_unknown_chunk_id_from_llm_is_dropped(self):
        """T-NODE2-03: LLM-returned unknown chunk_id not included in output."""
        chunks = [make_classified_chunk("c1"), make_classified_chunk("c2")]
        batch_result = BatchBulletResult(
            extractions=[
                ChunkBullets(chunk_id="c1", bullets=["نقطة من c1"]),
                ChunkBullets(chunk_id="PHANTOM_ID", bullets=["نقطة من مجهول"]),
                ChunkBullets(chunk_id="c2", bullets=["نقطة من c2"]),
            ]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        for r in results:
            assert r["chunk_id"] != "PHANTOM_ID"

    def test_missed_chunk_gets_fallback_bullet(self):
        """T-NODE2-04: LLM misses a chunk_id → fallback bullet with clean_text."""
        chunks = [
            make_classified_chunk("c1", clean_text="النص الأول"),
            make_classified_chunk("c2", clean_text="النص الثاني"),
            make_classified_chunk("c3", clean_text="النص الثالث"),
        ]
        # LLM only returns 2 of 3
        batch_result = BatchBulletResult(
            extractions=[
                ChunkBullets(chunk_id="c1", bullets=["نقطة من c1"]),
                ChunkBullets(chunk_id="c2", bullets=["نقطة من c2"]),
            ]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        c3_bullets = [r for r in results if r["chunk_id"] == "c3"]
        assert len(c3_bullets) == 1
        assert c3_bullets[0]["bullet"] == "النص الثالث"

    def test_full_exception_fallback(self):
        """T-NODE2-05: LLM exception → each chunk becomes one bullet with clean_text."""
        chunks = [
            make_classified_chunk("c1", clean_text="النص الأول"),
            make_classified_chunk("c2", clean_text="النص الثاني"),
            make_classified_chunk("c3", clean_text="النص الثالث"),
        ]
        node = make_node2(raises=RuntimeError("LLM down"))
        results = node.process_batch(chunks, "الوقائع")
        assert len(results) == 3
        texts = {r["bullet"] for r in results}
        assert "النص الأول" in texts
        assert "النص الثاني" in texts
        assert "النص الثالث" in texts

    def test_bullet_dict_has_required_keys(self):
        """Output bullet dicts have all required keys for Node 3."""
        chunks = [make_classified_chunk("c1")]
        batch_result = BatchBulletResult(
            extractions=[ChunkBullets(chunk_id="c1", bullets=["نقطة"])]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        required = {"bullet_id", "role", "bullet", "source", "party", "chunk_id"}
        for r in results:
            assert required.issubset(r.keys())

    def test_source_is_list_with_citation(self):
        """Source field is a list with one citation string."""
        chunks = [make_classified_chunk("c1", doc_id="doc-A", page_number=2, paragraph_number=5)]
        batch_result = BatchBulletResult(
            extractions=[ChunkBullets(chunk_id="c1", bullets=["نقطة"])]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        assert isinstance(results[0]["source"], list)
        assert results[0]["source"][0] == "doc-A ص2 ف5"

    def test_empty_bullet_text_skipped(self):
        """LLM returning empty string in bullets list is not included."""
        chunks = [make_classified_chunk("c1")]
        batch_result = BatchBulletResult(
            extractions=[ChunkBullets(chunk_id="c1", bullets=["", "نقطة حقيقية", "  "])]
        )
        node = make_node2(parser_result=batch_result)
        results = node.process_batch(chunks, "الوقائع")
        assert all(r["bullet"].strip() for r in results)


@pytest.mark.unit
class TestProcess:
    def test_empty_chunks_returns_empty_bullets(self):
        """process() with empty classified_chunks returns {'bullets': []}."""
        node = make_node2()
        result = node.process({"classified_chunks": []})
        assert result == {"bullets": []}

    def test_empty_clean_text_chunks_filtered(self):
        """T-NODE2-06: Chunks with empty clean_text are not sent to LLM."""
        parser = MagicMock()
        parser.invoke.return_value = BatchBulletResult(extractions=[])
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node2_BulletExtractor(llm)

        chunks = [
            make_classified_chunk("c1", clean_text="نص حقيقي"),
            make_classified_chunk("c2", clean_text=""),       # empty — should be filtered
            make_classified_chunk("c3", clean_text="   "),   # whitespace — should be filtered
        ]
        node.process({"classified_chunks": chunks})
        # LLM called with only the non-empty chunk
        assert parser.invoke.call_count >= 1
        # Verify the empty chunk IDs don't appear in any LLM call's message
        for call_args in parser.invoke.call_args_list:
            messages = call_args[0][0]
            content = " ".join(m.content for m in messages)
            assert "c2" not in content
            assert "c3" not in content

    def test_bullet_id_uniqueness(self):
        """T-NODE2-07: All bullet_ids in output are unique."""
        chunks = [make_classified_chunk(f"c{i}", clean_text=f"نص {i}") for i in range(10)]
        batch_result = BatchBulletResult(
            extractions=[
                ChunkBullets(chunk_id=f"c{i}", bullets=["نقطة أ", "نقطة ب"])
                for i in range(10)
            ]
        )
        node = make_node2(parser_result=batch_result)
        result = node.process({"classified_chunks": chunks})
        ids = [b["bullet_id"] for b in result["bullets"]]
        assert len(set(ids)) == len(ids)

    def test_party_propagated_from_chunk(self):
        """party field in bullet inherited from source chunk."""
        chunks = [make_classified_chunk("c1", party="خبير")]
        batch_result = BatchBulletResult(
            extractions=[ChunkBullets(chunk_id="c1", bullets=["نقطة"])]
        )
        node = make_node2(parser_result=batch_result)
        result = node.process({"classified_chunks": chunks})
        assert result["bullets"][0]["party"] == "خبير"
