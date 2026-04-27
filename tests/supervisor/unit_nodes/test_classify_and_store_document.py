"""
test_classify_and_store_document.py — unit tests for classify_and_store_document_node.

Uses real Qdrant + Mongo for storage paths.
"""

import uuid
from pathlib import Path

import pytest

from Supervisor.nodes.classify_and_store_document import classify_and_store_document_node
from tests.supervisor.helpers.state_factory import make_state, make_agent_result

FIXTURES_DIR = Path(__file__).parent.parent.parent / "CASE_RAG" / "fixtures"


@pytest.fixture
def txt_fixture():
    f = FIXTURES_DIR / "صحيفة_دعوى.txt"
    if not f.exists():
        pytest.skip(f"Fixture not found: {f}")
    return str(f)


class TestClassifyAndStoreNoOp:
    def test_no_ocr_no_files_returns_empty(self):
        state = make_state(agent_results={}, uploaded_files=[])
        result = classify_and_store_document_node(state)
        assert result == {} or result.get("document_classifications") == []


class TestClassifyAndStoreDocument:
    def test_uploaded_txt_file_classified(self, txt_fixture):
        case_id = f"test-case-{uuid.uuid4()}"
        state = make_state(
            case_id=case_id,
            uploaded_files=[txt_fixture],
            agent_results={},
        )
        result = classify_and_store_document_node(state)
        classifications = result.get("document_classifications", [])
        assert isinstance(classifications, list)

    def test_per_file_result_has_status(self, txt_fixture):
        case_id = f"test-case-{uuid.uuid4()}"
        state = make_state(
            case_id=case_id,
            uploaded_files=[txt_fixture],
            agent_results={},
        )
        result = classify_and_store_document_node(state)
        for item in result.get("document_classifications", []):
            assert "status" in item

    def test_ocr_raw_texts_used_when_no_files(self):
        """OCR raw_texts path — classify from OCR results directly."""
        case_id = f"test-case-{uuid.uuid4()}"
        state = make_state(
            case_id=case_id,
            uploaded_files=[],
            agent_results={
                "ocr": make_agent_result(
                    response="نص مستخرج من مستند",
                    raw_output={"raw_texts": ["نص مستخرج من مستند قانوني مهم"]},
                )
            },
        )
        result = classify_and_store_document_node(state)
        # Should produce document_classifications or empty dict — must not raise
        assert isinstance(result, dict)
