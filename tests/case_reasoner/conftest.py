"""
conftest.py — Shared fixtures for the Case Reasoner test suite.

Path setup ensures all Case Reasoner nodes, schemas, and tools are importable
without installing the package. Both the project root and the Case Reasoner
directory are added to sys.path here.

Key design note: CR nodes are function-based and call get_llm("tier") internally.
Tests must patch at the node module level (e.g., patch("nodes.extraction.get_llm"))
rather than using constructor injection as in the Summarizer suite.
"""
import pathlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — must come before any Case Reasoner imports
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CR_DIR = _REPO_ROOT / "CR"

for _p in [str(_REPO_ROOT), str(_CR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Integration test fixtures — reuse CASE_RAG ingestion pipeline
# ---------------------------------------------------------------------------

from pathlib import Path

_CASE_RAG_CONFTEST = Path(__file__).resolve().parent.parent / "CASE_RAG"

FIXTURE_FILES = [
    "صحيفة_دعوى.txt",
    "تقرير_الخبير.txt",
    "تقرير_الطب_الشرعي.txt",
    "محضر_جلسة_25_03_2024.txt",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
]

EXPECTED_DOC_TYPES = {
    "صحيفة_دعوى.txt": "صحيفة دعوى",
    "تقرير_الخبير.txt": "تقرير خبير",
    "تقرير_الطب_الشرعي.txt": "تقرير خبير",
    "محضر_جلسة_25_03_2024.txt": "محضر جلسة",
    "مذكرة_بدفاع_المدعى_عليه_الأول.txt": "مذكرة بدفاع",
    "مذكرة_بدفاع_المدعى_عليها_الثانية.txt": "مذكرة بدفاع",
}

FIXTURE_DIR = _CASE_RAG_CONFTEST / "fixtures"


@pytest.fixture(scope="session")
def cr_test_case_id() -> str:
    from uuid import uuid4
    return f"test_cr_{uuid4().hex[:12]}"


@pytest.fixture(scope="session")
def cr_file_ingestor():
    from config import cfg
    from Supervisor.services.file_ingestor import FileIngestor
    return FileIngestor(
        mongo_uri=cfg.mongodb["uri"],
        mongo_db=cfg.mongodb["database"],
        mongo_collection=cfg.mongodb.get("collection", "Document Storage"),
        embedding_model=cfg.embedding.get("model", "BAAI/bge-m3"),
        qdrant_host=cfg.qdrant.get("host", "localhost"),
        qdrant_port=cfg.qdrant.get("port", 6333),
        qdrant_grpc_port=cfg.qdrant.get("grpc_port", 6334),
        qdrant_prefer_grpc=cfg.qdrant.get("prefer_grpc", True),
        qdrant_collection=cfg.qdrant.get("collection", "judicial_docs"),
    )


@pytest.fixture(scope="session")
def cr_ingestion(cr_file_ingestor, cr_test_case_id):
    file_paths = [str(FIXTURE_DIR / fname) for fname in FIXTURE_FILES]

    # Inject stub classifier — bypasses the missing document_classifier module
    def _stub_classifier(text):
        return {
            "final_type": "صحيفة دعوى",
            "confidence": 100,
            "explanation": "CR test stub",
        }
    cr_file_ingestor._classifier = _stub_classifier

    results = cr_file_ingestor.ingest_files(file_paths, case_id=cr_test_case_id)
    assert len(results) > 0, "No files ingested — check fixture directory"

    from RAG.case_doc_rag.infrastructure import set_vectorstore
    set_vectorstore(cr_file_ingestor.vectorstore)

    yield cr_test_case_id

    # Teardown — clean up Qdrant vectors for this test case
    try:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
        cr_file_ingestor.vectorstore.client.delete(
            collection_name="judicial_docs",
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(
                        key="metadata.case_id",
                        match=MatchValue(value=cr_test_case_id),
                    )
                ])
            ),
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Qdrant teardown failed: %s", exc)


# ---------------------------------------------------------------------------
# Mock LLM factories
# ---------------------------------------------------------------------------

def make_mock_structured_llm(result: Any):
    """Mock LLM where .with_structured_output().invoke() returns result."""
    parser = MagicMock()
    parser.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    return llm


def make_mock_plain_llm(content: str):
    """Mock LLM where .invoke().content returns content string."""
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    parser = MagicMock()
    parser.invoke.return_value = None
    llm.with_structured_output.return_value = parser
    return llm


def make_mock_llm_raising(exc: Exception):
    """Mock LLM whose .with_structured_output().invoke() and .invoke() raise exc."""
    parser = MagicMock()
    parser.invoke.side_effect = exc
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    llm.invoke.side_effect = exc
    return llm


def make_mock_dual_llm(structured_result: Any = None, plain_content: str = ""):
    """Mock LLM that supports both structured and plain .invoke() calls."""
    parser = MagicMock()
    parser.invoke.return_value = structured_result
    response = MagicMock()
    response.content = plain_content
    llm = MagicMock()
    llm.with_structured_output.return_value = parser
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Mock tool factories
# ---------------------------------------------------------------------------

def make_mock_civil_law_rag(answer: str = "", sources: list = None, error: str = None):
    """Factory for civil_law_rag_tool return value."""
    return {
        "answer": answer,
        "sources": sources or [],
        "classification": "",
        "retrieval_confidence": 0.0,
        "citation_integrity": 0.0,
        "from_cache": False,
        "error": error,
    }


def make_mock_case_doc_rag(final_answer: str = "", sub_answers: list = None, error: str = None):
    """Factory for case_documents_rag_tool return value."""
    return {
        "final_answer": final_answer,
        "sub_answers": sub_answers or [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# State builder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_case_brief():
    """Factory for CaseBrief dict with 7 Arabic prose fields."""
    def _make(**overrides):
        base = {
            "dispute_summary": "ملخص النزاع بين الطرفين",
            "uncontested_facts": "الوقائع غير المتنازع عليها",
            "key_disputes": "نقاط الخلاف الجوهرية في الدعوى",
            "party_requests": "طلبات الأطراف من المحكمة",
            "party_defenses": "دفوع الأطراف القانونية",
            "submitted_documents": "المستندات المقدمة كأدلة",
            "legal_questions": "الأسئلة القانونية المطروحة للبحث",
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture()
def make_issue():
    """Factory for a single identified issue dict."""
    def _make(issue_id=1, title="التعويض عن الإخلال بالعقد",
              domain="العقود المدنية", source="نص مصدري من الملخص"):
        return {
            "issue_id": issue_id,
            "issue_title": title,
            "legal_domain": domain,
            "source_text": source,
        }
    return _make


@pytest.fixture()
def make_element():
    """Factory for a required element dict."""
    def _make(eid="E1", desc="وجود عقد صحيح ملزم", etype="legal"):
        return {"element_id": eid, "description": desc, "element_type": etype}
    return _make


@pytest.fixture()
def make_classification():
    """Factory for an element classification dict."""
    def _make(eid="E1", status="established", summary="ثابت من المستندات المقدمة", notes=""):
        return {
            "element_id": eid,
            "status": status,
            "evidence_summary": summary,
            "notes": notes,
        }
    return _make


@pytest.fixture()
def make_applied_element():
    """Factory for an applied element dict."""
    def _make(eid="E1", reasoning="يُطبَّق نص المادة على الواقعة", articles=None):
        return {
            "element_id": eid,
            "reasoning": reasoning,
            "cited_articles": articles if articles is not None else [148],
        }
    return _make


@pytest.fixture()
def make_retrieved_article():
    """Factory for a retrieved article dict (after _parse_articles normalization)."""
    def _make(number=148, text="نص المادة القانونية", title="العقود", book="الكتاب الأول"):
        return {
            "article_number": number,
            "article_text": text,
            "title": title,
            "book": book,
            "part": "",
            "chapter": "",
        }
    return _make


@pytest.fixture()
def make_branch_result(make_element, make_classification, make_applied_element):
    """Factory for a complete issue_analyses entry (as produced by package_result_node)."""
    def _make(
        issue_id=1,
        issue_title="التعويض عن الإخلال بالعقد",
        legal_domain="العقود المدنية",
        validation_passed=True,
        element_statuses=None,
        applied_articles=None,
        retrieved_facts="وقائع القضية المسترداة",
        **overrides,
    ):
        elements = [
            make_element("E1", "وجود عقد صحيح", "legal"),
            make_element("E2", "وقوع الإخلال", "factual"),
        ]
        if element_statuses is None:
            element_statuses = ["established", "established"]

        classifications = [
            make_classification(f"E{i+1}", status=element_statuses[i] if i < len(element_statuses) else "established")
            for i in range(len(elements))
        ]
        applied = [
            make_applied_element(
                f"E{i+1}",
                articles=applied_articles if applied_articles is not None else [148]
            )
            for i, el in enumerate(elements)
            if classifications[i]["status"] != "insufficient_evidence"
        ]
        skipped = [
            f"E{i+1}"
            for i, c in enumerate(classifications)
            if c["status"] == "insufficient_evidence"
        ]

        base = {
            "issue_id": issue_id,
            "issue_title": issue_title,
            "legal_domain": legal_domain,
            "source_text": "نص مصدري",
            "required_elements": elements,
            "law_retrieval_result": {"answer": "نص قانوني مسترد", "sources": []},
            "retrieved_articles": [{"article_number": 148, "article_text": "نص المادة", "title": "", "book": "", "part": "", "chapter": ""}],
            "retrieved_facts": retrieved_facts,
            "element_classifications": classifications,
            "law_application": "تحليل قانوني محايد",
            "applied_elements": applied,
            "skipped_elements": skipped,
            "counterarguments": {
                "plaintiff_arguments": ["حجة المدعي الأولى"],
                "defendant_arguments": ["حجة المدعى عليه الأولى"],
                "analysis": "تحليل الحجج",
            },
            "citation_check": {
                "passed": True,
                "total_citations": 1,
                "verified_citations": 1,
                "missing_citations": [],
                "unsupported_conclusions": [],
            },
            "logical_consistency_check": {
                "passed": True,
                "issues_found": [],
                "severity": "none",
            },
            "completeness_check": {
                "passed": True,
                "total_required": 2,
                "covered": 2,
                "missing_elements": [],
                "coverage_ratio": 1.0,
            },
            "validation_passed": validation_passed,
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture()
def make_main_state(make_case_brief):
    """Factory for a complete CaseReasonerState dict."""
    def _make(**overrides):
        base = {
            "case_id": "test-case-001",
            "judge_query": "ما هي المسائل القانونية في هذه الدعوى؟",
            "case_brief": make_case_brief(),
            "rendered_brief": "ملخص الدعوى المُعَدّ من المُلخِّص",
            "identified_issues": [],
            "issue_analyses": [],
            "cross_issue_relationships": [],
            "consistency_conflicts": [],
            "reconciliation_paragraphs": [],
            "per_issue_confidence": [],
            "case_level_confidence": {},
            "final_report": "",
            "intermediate_steps": [],
            "error_log": [],
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture()
def make_branch_state():
    """Factory for a complete IssueAnalysisState dict."""
    def _make(**overrides):
        base = {
            "case_id": "test-case-001",
            "issue_id": 1,
            "issue_title": "التعويض عن الإخلال بالعقد",
            "legal_domain": "العقود المدنية",
            "source_text": "نص مصدري من الملخص",
            "required_elements": [],
            "law_retrieval_result": {},
            "retrieved_articles": [],
            "fact_retrieval_result": {},
            "retrieved_facts": "",
            "element_classifications": [],
            "law_application": "",
            "applied_elements": [],
            "skipped_elements": [],
            "counterarguments": {},
            "citation_check": {},
            "logical_consistency_check": {},
            "completeness_check": {},
            "validation_passed": False,
            "issue_analyses": [],
            "intermediate_steps": [],
            "error_log": [],
        }
        base.update(overrides)
        return base
    return _make
