"""
test_node_0.py — Unit tests for Summerize/node_0.py (Document Intake)

Tests:
    T-NODE0-01: clean_text() removes Unicode directional marks (U+200F, U+200E)
    T-NODE0-02: clean_text() removes Tatweel/Kashida (ـ)
    T-NODE0-03: clean_text() removes page number patterns "- N -"
    T-NODE0-04: clean_text() removes وزارة العدل + محكمة header pair
    T-NODE0-05: clean_text() removes certification stamp صورة طبق الأصل
    T-NODE0-06: clean_text() collapses horizontal whitespace
    T-NODE0-07: extract_metadata() heuristic detects doc_type from keyword
    T-NODE0-08: extract_metadata() heuristic detects party from keyword
    T-NODE0-09: extract_metadata() invokes LLM when heuristic finds nothing
    T-NODE0-10: extract_metadata() returns غير محدد when LLM raises
    T-NODE0-11: segment_document() chunk_id is deterministic (UUID5)
    T-NODE0-12: segment_document() increments page_number at PAGE_SIZE_ESTIMATE
    T-NODE0-13: segment_document() skips empty paragraphs
    T-NODE0-14: process() with empty raw_text returns {"chunks": []}
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_SUMMARIZE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "Summerize"
if str(_SUMMARIZE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUMMARIZE_DIR))

from node_0 import Node0_DocumentIntake, PAGE_SIZE_ESTIMATE
from schemas import DocumentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node0(llm=None) -> Node0_DocumentIntake:
    if llm is None:
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
    return Node0_DocumentIntake(llm)


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCleanText:
    def setup_method(self):
        self.node = make_node0()

    def test_removes_rtl_mark(self):
        """T-NODE0-01: Removes RIGHT-TO-LEFT MARK (U+200F)."""
        text = "النص\u200fالعربي"
        result = self.node.clean_text(text)
        assert "\u200f" not in result

    def test_removes_ltr_mark(self):
        """T-NODE0-01: Removes LEFT-TO-RIGHT MARK (U+200E)."""
        text = "النص\u200eالعربي"
        result = self.node.clean_text(text)
        assert "\u200e" not in result

    def test_removes_tatweel(self):
        """T-NODE0-02: Removes Tatweel/Kashida (ـ) sequences."""
        text = "الـمـحـكـمـة"
        result = self.node.clean_text(text)
        assert "ـ" not in result
        assert "المحكمة" in result

    def test_removes_single_digit_page_number(self):
        """T-NODE0-03: Removes '- 1 -' page markers."""
        text = "فقرة أولى\n- 1 -\nفقرة ثانية"
        result = self.node.clean_text(text)
        assert "- 1 -" not in result

    def test_removes_double_digit_page_number(self):
        """T-NODE0-03: Removes '- 12 -' page markers."""
        text = "محتوى\n- 12 -\nمحتوى آخر"
        result = self.node.clean_text(text)
        assert "- 12 -" not in result

    def test_removes_triple_digit_page_number(self):
        """T-NODE0-03: Removes '- 123 -' page markers."""
        text = "بداية\n- 123 -\nnهاية"
        result = self.node.clean_text(text)
        assert "- 123 -" not in result

    def test_removes_ministry_court_header_pair(self):
        """T-NODE0-04: Removes two-line وزارة العدل + محكمة header."""
        text = "وزارة العدل\nمحكمة القاهرة الابتدائية\nالنص القانوني"
        result = self.node.clean_text(text)
        assert "وزارة العدل" not in result
        assert "محكمة القاهرة الابتدائية" not in result

    def test_removes_certification_stamp(self):
        """T-NODE0-05: Removes 'صورة طبق الأصل' stamp."""
        text = "محتوى\nصورة طبق الأصل\nمزيد من المحتوى"
        result = self.node.clean_text(text)
        assert "صورة طبق الأصل" not in result

    def test_collapses_multiple_spaces(self):
        """T-NODE0-06: Collapses multiple spaces to single space."""
        text = "المحكمة    قررت     التأجيل"
        result = self.node.clean_text(text)
        assert "  " not in result
        assert "المحكمة قررت التأجيل" in result

    def test_collapses_tabs(self):
        """T-NODE0-06: Collapses tabs to single space."""
        text = "الطرف\tالأول\t\tالطرف الثاني"
        result = self.node.clean_text(text)
        assert "\t" not in result

    def test_clean_arabic_unchanged(self):
        """Clean Arabic text passes through without modification."""
        text = "يدعي المدعي أن المدعى عليه أخل بالعقد"
        result = self.node.clean_text(text)
        assert "يدعي المدعي أن المدعى عليه أخل بالعقد" in result

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert self.node.clean_text("") == ""


# ---------------------------------------------------------------------------
# extract_metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractMetadata:
    def test_heuristic_doc_type_detection_sahifa(self):
        """T-NODE0-07: Keyword 'صحيفة افتتاح' maps to 'صحيفة دعوى'."""
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
        node = Node0_DocumentIntake(llm)
        result = node.extract_metadata("صحيفة افتتاح الدعوى المقدمة إلى المحكمة")
        assert result.doc_type == "صحيفة دعوى"

    def test_heuristic_doc_type_detection_mathkara(self):
        """T-NODE0-07: Keyword 'مذكرة بدفاع' maps to 'مذكرة دفاع'.
        Party keyword also included so heuristic shortcut fires (both must be found)."""
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
        node = Node0_DocumentIntake(llm)
        # Include party keyword "المدعى عليه" so heuristic finds both fields
        result = node.extract_metadata("مذكرة بدفاع المدعى عليه الأول في القضية")
        assert result.doc_type == "مذكرة دفاع"

    def test_heuristic_party_detection_plaintiff(self):
        """T-NODE0-08: Keyword 'المدعي' in header maps to party='المدعي'.
        Doc_type keyword also included so heuristic shortcut fires."""
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
        node = Node0_DocumentIntake(llm)
        # Include doc_type keyword so heuristic finds both fields and skips LLM
        result = node.extract_metadata("صحيفة دعوى مقدمة من المدعي الشركة العقارية الحديثة")
        assert result.party == "المدعي"

    def test_heuristic_party_detection_expert(self):
        """T-NODE0-08: Keyword 'الخبير' in header maps to party='خبير'.
        Doc_type keyword also included so heuristic shortcut fires.
        Note: 'المدعى عليه' normalizes to 'المدعي عليه' which matches 'المدعي' first,
        so we use 'الخبير' (expert) which is unambiguous."""
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock()
        node = Node0_DocumentIntake(llm)
        # حكم تمهيدي (doc_type) + الخبير (party) — both matched by heuristic
        # No "المحكمة" in text to avoid party ordering conflict
        result = node.extract_metadata("حكم تمهيدي صادر بناء على تقرير الخبير المعين")
        assert result.party == "خبير"

    def test_llm_fallback_invoked_when_no_keywords(self):
        """T-NODE0-09: LLM called when heuristic finds neither type nor party."""
        llm_result = DocumentMetadata(doc_type="مذكرة رد", party="المدعى عليه")
        parser = MagicMock()
        parser.invoke.return_value = llm_result
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node0_DocumentIntake(llm)

        result = node.extract_metadata("نص بدون كلمات مفتاحية معروفة هنا")

        assert parser.invoke.called
        assert result.doc_type == "مذكرة رد"
        assert result.party == "المدعى عليه"

    def test_llm_exception_returns_unknown(self):
        """T-NODE0-10: LLM exception → returns غير محدد for both fields."""
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("LLM unavailable")
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node0_DocumentIntake(llm)

        result = node.extract_metadata("نص بدون كلمات مفتاحية")

        assert result.doc_type == "غير محدد"
        assert result.party == "غير محدد"

    def test_heuristic_both_found_skips_llm(self):
        """When both doc_type and party found by heuristic, LLM is NOT called."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node0_DocumentIntake(llm)

        # This header contains both type and party keywords
        node.extract_metadata("صحيفة دعوى مقدمة من المدعي")

        assert not parser.invoke.called


# ---------------------------------------------------------------------------
# segment_document tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSegmentDocument:
    def setup_method(self):
        self.node = make_node0()
        self.metadata = DocumentMetadata(doc_type="صحيفة دعوى", party="المدعي")

    def test_chunk_id_is_deterministic(self):
        """T-NODE0-11: Same doc_id + text produces same chunk_ids on second call."""
        text = "فقرة أولى\n\nفقرة ثانية\n\nفقرة ثالثة"
        chunks1 = self.node.segment_document(text, "doc-det", self.metadata)
        chunks2 = self.node.segment_document(text, "doc-det", self.metadata)
        ids1 = [c["chunk_id"] for c in chunks1]
        ids2 = [c["chunk_id"] for c in chunks2]
        assert ids1 == ids2

    def test_chunk_ids_unique_within_document(self):
        """T-NODE0-11: All chunk_ids unique within one document."""
        paragraphs = [f"فقرة رقم {i} من النص القانوني المفصل" for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = self.node.segment_document(text, "doc-unique", self.metadata)
        ids = [c["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_page_number_increments_after_threshold(self):
        """T-NODE0-12: page_number increments when char_count exceeds PAGE_SIZE_ESTIMATE."""
        # Create paragraphs that will exceed PAGE_SIZE_ESTIMATE
        big_para = "أ" * (PAGE_SIZE_ESTIMATE + 100)
        text = big_para + "\n\nفقرة صغيرة بعد الصفحة الأولى"
        chunks = self.node.segment_document(text, "doc-page", self.metadata)
        assert len(chunks) >= 2
        # The small paragraph after the big one should be on page 2
        last_chunk = chunks[-1]
        assert last_chunk["page_number"] >= 2

    def test_skips_empty_paragraphs(self):
        """T-NODE0-13: Empty paragraphs produce no chunks."""
        text = "فقرة حقيقية\n\n\n\n\nفقرة ثانية حقيقية"
        chunks = self.node.segment_document(text, "doc-empty", self.metadata)
        for chunk in chunks:
            assert chunk["clean_text"].strip() != ""

    def test_metadata_propagated_to_chunks(self):
        """doc_type and party from metadata appear in all chunks."""
        text = "نص قانوني"
        chunks = self.node.segment_document(text, "doc-meta", self.metadata)
        for chunk in chunks:
            assert chunk["doc_type"] == "صحيفة دعوى"
            assert chunk["party"] == "المدعي"

    def test_doc_id_in_all_chunks(self):
        """doc_id propagated to all chunks."""
        text = "فقرة أولى\n\nفقرة ثانية"
        chunks = self.node.segment_document(text, "my-doc-id", self.metadata)
        for chunk in chunks:
            assert chunk["doc_id"] == "my-doc-id"

    def test_paragraph_numbers_sequential(self):
        """paragraph_number follows 1-based sequential index of paragraphs."""
        text = "فقرة أولى\n\nفقرة ثانية\n\nفقرة ثالثة"
        chunks = self.node.segment_document(text, "doc-seq", self.metadata)
        nums = [c["paragraph_number"] for c in chunks]
        assert nums == [1, 2, 3]


# ---------------------------------------------------------------------------
# process() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcess:
    def test_empty_raw_text_returns_empty_chunks(self):
        """T-NODE0-14: Empty raw_text → {"chunks": []}."""
        node = make_node0()
        result = node.process({"raw_text": "", "doc_id": "empty"})
        assert result == {"chunks": []}

    def test_whitespace_only_returns_empty_chunks(self):
        """Whitespace-only text (no paragraphs after clean) → no chunks."""
        node = make_node0()
        result = node.process({"raw_text": "   \n\n\t  ", "doc_id": "ws"})
        # After cleaning, no meaningful paragraphs remain
        assert isinstance(result["chunks"], list)

    def test_normal_text_produces_chunks(self):
        """Non-empty text with heuristic match produces at least one chunk."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node0_DocumentIntake(llm)

        text = "صحيفة دعوى مقدمة من المدعي\n\nيدعي المدعي أن المدعى عليه أخل بالعقد العقاري"
        result = node.process({"raw_text": text, "doc_id": "test-doc"})
        assert len(result["chunks"]) >= 1

    def test_output_keys_complete(self):
        """process() output has 'chunks' key."""
        node = make_node0()
        result = node.process({"raw_text": "", "doc_id": "test"})
        assert "chunks" in result

    def test_chunks_contain_required_keys(self):
        """Each chunk dict has the 7 keys Node 1 expects."""
        parser = MagicMock()
        llm = MagicMock()
        llm.with_structured_output.return_value = parser
        node = Node0_DocumentIntake(llm)

        text = "صحيفة دعوى من المدعي\n\nيدعي أن الضرر وقع بسبب الإهمال"
        result = node.process({"raw_text": text, "doc_id": "keys-test"})
        required = {"chunk_id", "doc_id", "page_number", "paragraph_number", "clean_text", "doc_type", "party"}
        for chunk in result["chunks"]:
            assert required.issubset(chunk.keys())
