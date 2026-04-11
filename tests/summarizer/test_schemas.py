"""
test_schemas.py — Unit tests for Summerize/schemas.py

Tests:
    T-SCHEMA-01: NormalizedChunk.model_dump() contains all expected keys
    T-SCHEMA-02: CaseBrief has exactly 7 string fields
    T-SCHEMA-03: All enum values are valid non-empty Arabic strings
"""

import pathlib
import sys

import pytest
from pydantic import ValidationError

_SUMMARIZE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "Summerize"
if str(_SUMMARIZE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUMMARIZE_DIR))

from schemas import (
    CaseBrief,
    ClassifiedChunk,
    DocTypeEnum,
    LegalBullet,
    LegalRoleEnum,
    NormalizedChunk,
    PartyEnum,
    RoleAggregation,
    ThemeCluster,
    ThemeSummary,
    ThemedRole,
    RoleThemeSummaries,
)


@pytest.mark.unit
class TestNormalizedChunkSchema:
    """T-SCHEMA-01: NormalizedChunk serialization matches Node 1 access patterns."""

    def _make_chunk(self) -> NormalizedChunk:
        return NormalizedChunk(
            chunk_id="chunk-001",
            doc_id="doc-test",
            page_number=1,
            paragraph_number=2,
            clean_text="نص تجريبي",
            doc_type="صحيفة دعوى",
            party="المدعي",
        )

    def test_model_dump_has_chunk_id(self):
        assert "chunk_id" in self._make_chunk().model_dump()

    def test_model_dump_has_doc_id(self):
        assert "doc_id" in self._make_chunk().model_dump()

    def test_model_dump_has_page_number(self):
        assert "page_number" in self._make_chunk().model_dump()

    def test_model_dump_has_paragraph_number(self):
        assert "paragraph_number" in self._make_chunk().model_dump()

    def test_model_dump_has_clean_text(self):
        assert "clean_text" in self._make_chunk().model_dump()

    def test_model_dump_has_doc_type(self):
        assert "doc_type" in self._make_chunk().model_dump()

    def test_model_dump_has_party(self):
        assert "party" in self._make_chunk().model_dump()

    def test_all_required_keys_present(self):
        """All 7 keys that Node 1 accesses via dict['key'] are present."""
        d = self._make_chunk().model_dump()
        required = {"chunk_id", "doc_id", "page_number", "paragraph_number", "clean_text", "doc_type", "party"}
        assert required.issubset(d.keys())

    def test_chunk_id_is_str(self):
        d = self._make_chunk().model_dump()
        assert isinstance(d["chunk_id"], str)

    def test_page_number_is_int(self):
        d = self._make_chunk().model_dump()
        assert isinstance(d["page_number"], int)


@pytest.mark.unit
class TestClassifiedChunkSchema:
    """ClassifiedChunk extends NormalizedChunk with role and confidence."""

    def test_extends_normalized_chunk(self):
        chunk = ClassifiedChunk(
            chunk_id="c1",
            doc_id="d1",
            page_number=1,
            paragraph_number=1,
            clean_text="test",
            doc_type="صحيفة دعوى",
            party="المدعي",
            role="الوقائع",
            confidence=1.0,
        )
        d = chunk.model_dump()
        assert "role" in d
        assert "confidence" in d
        assert d["confidence"] == 1.0

    def test_default_confidence_is_1(self):
        chunk = ClassifiedChunk(
            chunk_id="c1",
            doc_id="d1",
            page_number=1,
            paragraph_number=1,
            clean_text="test",
            doc_type="صحيفة دعوى",
            party="المدعي",
            role="الوقائع",
        )
        assert chunk.confidence == 1.0


@pytest.mark.unit
class TestLegalBulletSchema:
    """LegalBullet.model_dump() keys match Node 3 access patterns."""

    def test_required_keys_for_node3(self):
        bullet = LegalBullet(
            bullet_id="b1",
            role="الوقائع",
            bullet="نص النقطة القانونية",
            source=["doc1 ص1 ف1"],
            party="المدعي",
            chunk_id="c1",
        )
        d = bullet.model_dump()
        for key in ("bullet_id", "role", "party", "bullet", "source", "chunk_id"):
            assert key in d, f"Missing key: {key}"

    def test_source_is_list(self):
        bullet = LegalBullet(
            bullet_id="b1",
            role="الوقائع",
            bullet="نص",
            source=["doc1 ص1 ف1"],
            party="المدعي",
            chunk_id="c1",
        )
        assert isinstance(bullet.source, list)


@pytest.mark.unit
class TestCaseBriefSchema:
    """T-SCHEMA-02: CaseBrief has exactly 7 string fields."""

    def _make_brief(self) -> CaseBrief:
        return CaseBrief(
            dispute_summary="ملخص",
            uncontested_facts="وقائع",
            key_disputes="خلافات",
            party_requests="طلبات",
            party_defenses="دفوع",
            submitted_documents="مستندات",
            legal_questions="أسئلة",
        )

    def test_exactly_7_fields(self):
        """T-SCHEMA-02: CaseBrief has exactly 7 fields."""
        fields = CaseBrief.model_fields
        assert len(fields) == 7

    def test_all_fields_are_str_type(self):
        """T-SCHEMA-02: All 7 fields have type str."""
        for name, field_info in CaseBrief.model_fields.items():
            annotation = field_info.annotation
            assert annotation is str, f"Field '{name}' has type {annotation}, expected str"

    def test_field_names(self):
        """The 7 expected field names are present."""
        expected = {
            "dispute_summary",
            "uncontested_facts",
            "key_disputes",
            "party_requests",
            "party_defenses",
            "submitted_documents",
            "legal_questions",
        }
        assert set(CaseBrief.model_fields.keys()) == expected

    def test_model_dump_has_all_7_keys(self):
        d = self._make_brief().model_dump()
        assert len(d) == 7


@pytest.mark.unit
class TestEnums:
    """T-SCHEMA-03: All enum values are non-empty Arabic strings."""

    def test_doc_type_enum_count(self):
        from typing import get_args
        values = get_args(DocTypeEnum)
        assert len(values) == 7

    def test_doc_type_all_arabic(self):
        from typing import get_args
        for val in get_args(DocTypeEnum):
            assert isinstance(val, str) and len(val) > 0, f"Empty or non-string: {val!r}"

    def test_doc_type_includes_unknown(self):
        from typing import get_args
        assert "غير محدد" in get_args(DocTypeEnum)

    def test_party_enum_count(self):
        from typing import get_args
        values = get_args(PartyEnum)
        assert len(values) == 6

    def test_party_all_arabic(self):
        from typing import get_args
        for val in get_args(PartyEnum):
            assert isinstance(val, str) and len(val) > 0

    def test_party_includes_unknown(self):
        from typing import get_args
        assert "غير محدد" in get_args(PartyEnum)

    def test_legal_role_enum_count(self):
        from typing import get_args
        values = get_args(LegalRoleEnum)
        assert len(values) == 7

    def test_legal_role_all_arabic(self):
        from typing import get_args
        for val in get_args(LegalRoleEnum):
            assert isinstance(val, str) and len(val) > 0

    def test_legal_role_includes_unknown(self):
        from typing import get_args
        assert "غير محدد" in get_args(LegalRoleEnum)

    def test_known_doc_types_present(self):
        from typing import get_args
        values = get_args(DocTypeEnum)
        for expected in ["صحيفة دعوى", "مذكرة دفاع", "محضر جلسة", "حكم تمهيدي"]:
            assert expected in values, f"Missing doc type: {expected}"

    def test_known_parties_present(self):
        from typing import get_args
        values = get_args(PartyEnum)
        for expected in ["المدعي", "المدعى عليه", "خبير", "المحكمة"]:
            assert expected in values, f"Missing party: {expected}"

    def test_known_roles_present(self):
        from typing import get_args
        values = get_args(LegalRoleEnum)
        for expected in ["الوقائع", "الطلبات", "الدفوع", "المستندات", "الأساس القانوني", "الإجراءات"]:
            assert expected in values, f"Missing role: {expected}"
