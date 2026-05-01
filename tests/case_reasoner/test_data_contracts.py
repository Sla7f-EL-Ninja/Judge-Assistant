# """
# test_data_contracts.py — Inter-node state key contracts for the Case Reasoner.

# Verifies that the output of node N contains all keys that node N+1 reads from state.
# Tests use mocked LLMs/tools so no external services are needed.
# """

# import pathlib
# import sys
# from unittest.mock import MagicMock, patch

# import pytest

# _CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
# if str(_CR_DIR) not in sys.path:
#     sys.path.insert(0, str(_CR_DIR))

# from nodes.extraction import extract_issues_node
# from nodes.decomposition import decompose_issue_node
# from nodes.retrieval import retrieve_law_node, retrieve_facts_node
# from nodes.evidence import classify_evidence_node
# from nodes.application import apply_law_node
# from nodes.counterargument import counterargument_node
# from nodes.validation import validate_analysis_node
# from nodes.package import package_result_node
# from nodes.aggregation import aggregate_issues_node
# from nodes.consistency import check_global_consistency_node
# from nodes.confidence import compute_confidence_node
# from schemas import (
#     ExtractedIssues, LegalIssue, DecomposedIssue, RequiredElement,
#     EvidenceSufficiencyResult, ElementClassification,
#     LawApplicationResult, AppliedElement,
#     Counterarguments, LogicalConsistencyResult,
#     IssueDependencies, ConsistencyCheckResult,
# )


# # ---------------------------------------------------------------------------
# # Branch chain contracts
# # ---------------------------------------------------------------------------

# @pytest.mark.unit
# class TestExtractionOutputContract:
#     """T-CONTRACT-01: extract_issues_node output satisfies router input."""

#     def test_extraction_output_issues_have_router_keys(self):
#         router_required_keys = {"issue_id", "issue_title", "legal_domain", "source_text"}
#         mock_result = ExtractedIssues(issues=[
#             LegalIssue(issue_id=1, issue_title="مسألة", legal_domain="عقود", source_text="نص"),
#         ])
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.extraction.get_llm", return_value=llm):
#             output = extract_issues_node({"case_brief": {"legal_questions": "أ", "key_disputes": "ب"}})

#         for issue in output["identified_issues"]:
#             assert router_required_keys.issubset(issue.keys())


# @pytest.mark.unit
# class TestDecompositionOutputContract:
#     """T-CONTRACT-02: decompose_issue_node output satisfies downstream nodes."""

#     def test_decomposition_output_has_required_elements_key(self):
#         mock_result = DecomposedIssue(elements=[
#             RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
#         ])
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.decomposition.get_llm", return_value=llm):
#             output = decompose_issue_node({
#                 "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
#             })

#         assert "required_elements" in output
#         el = output["required_elements"][0]
#         assert {"element_id", "description", "element_type"}.issubset(el.keys())


# @pytest.mark.unit
# class TestRetrievalOutputContract:
#     """T-CONTRACT-03: retrieval node outputs satisfy evidence node input."""

#     # Evidence node reads: required_elements (from branch state), retrieved_facts, law_retrieval_result["answer"]

#     def test_retrieve_law_output_satisfies_evidence_input(self):
#         evidence_reads = {"law_retrieval_result", "retrieved_articles"}
#         with patch("tools.civil_law_rag_tool", return_value={
#             "answer": "نص قانوني", "sources": [{"article": 148, "content": "نص"}],
#             "classification": "", "retrieval_confidence": 0, "citation_integrity": 0,
#             "from_cache": False, "error": None,
#         }):
#             output = retrieve_law_node({"issue_title": "مسألة", "source_text": "نص", "case_id": "001"})

#         assert evidence_reads.issubset(output.keys())

#     def test_retrieve_facts_output_satisfies_evidence_input(self):
#         evidence_reads = {"fact_retrieval_result", "retrieved_facts"}
#         with patch("tools.case_documents_rag_tool", return_value={
#             "final_answer": "وقائع", "sub_answers": [], "error": None,
#         }):
#             output = retrieve_facts_node({"issue_title": "مسألة", "source_text": "نص", "case_id": "001"})

#         assert evidence_reads.issubset(output.keys())

#     def test_law_retrieval_result_has_answer_key(self):
#         """Evidence node accesses law_retrieval_result['answer'] specifically."""
#         with patch("tools.civil_law_rag_tool", return_value={
#             "answer": "نص قانوني", "sources": [], "classification": "",
#             "retrieval_confidence": 0, "citation_integrity": 0, "from_cache": False, "error": None,
#         }):
#             output = retrieve_law_node({"issue_title": "مسألة", "source_text": "نص", "case_id": "001"})

#         assert "answer" in output["law_retrieval_result"]


# @pytest.mark.unit
# class TestEvidenceOutputContract:
#     """T-CONTRACT-04: classify_evidence_node output satisfies apply_law_node input."""

#     def test_classifications_have_element_id_and_status(self):
#         application_required_keys = {"element_id", "status"}
#         mock_result = EvidenceSufficiencyResult(classifications=[
#             ElementClassification(element_id="E1", status="established", evidence_summary="ثابت"),
#         ])
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.evidence.get_llm", return_value=llm):
#             output = classify_evidence_node({
#                 "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
#                 "retrieved_facts": "وقائع",
#                 "law_retrieval_result": {"answer": "نص"},
#             })

#         assert "element_classifications" in output
#         for c in output["element_classifications"]:
#             assert application_required_keys.issubset(c.keys())


# @pytest.mark.unit
# class TestApplicationOutputContract:
#     """T-CONTRACT-05: apply_law_node output satisfies counterargument_node input."""

#     def test_application_output_has_counterargument_required_keys(self):
#         counterarg_reads = {"law_application", "applied_elements", "skipped_elements"}
#         mock_result = LawApplicationResult(
#             elements=[AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148])],
#             synthesis="تحليل إجمالي",
#         )
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.application.get_llm", return_value=llm):
#             output = apply_law_node({
#                 "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
#                 "element_classifications": [{"element_id": "E1", "status": "established"}],
#                 "retrieved_facts": "وقائع",
#                 "law_retrieval_result": {"answer": "نص"},
#             })

#         assert counterarg_reads.issubset(output.keys())


# @pytest.mark.unit
# class TestCounterargumentOutputContract:
#     """T-CONTRACT-06: counterargument_node output satisfies validate_analysis_node input."""

#     def test_counterarguments_has_required_structure(self):
#         validation_reads_from_counterargs = {"plaintiff_arguments", "defendant_arguments"}
#         mock_result = Counterarguments(
#             plaintiff_arguments=["حجة"],
#             defendant_arguments=["دفع"],
#             analysis="تحليل",
#         )
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.counterargument.get_llm", return_value=llm):
#             output = counterargument_node({
#                 "law_application": "تحليل", "element_classifications": [],
#                 "retrieved_facts": "وقائع", "issue_title": "مسألة",
#             })

#         assert "counterarguments" in output
#         assert validation_reads_from_counterargs.issubset(output["counterarguments"].keys())


# @pytest.mark.unit
# class TestValidationOutputContract:
#     """T-CONTRACT-07: validate_analysis_node output satisfies package_result_node input."""

#     def test_validation_output_has_all_check_keys(self):
#         package_required_keys = {
#             "citation_check", "logical_consistency_check", "completeness_check", "validation_passed"
#         }
#         mock_consistency = LogicalConsistencyResult(passed=True, issues_found=[], severity="none")
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_consistency

#         with patch("nodes.validation.get_llm", return_value=llm), \
#              patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}):
#             output = validate_analysis_node({
#                 "issue_title": "مسألة",
#                 "law_application": "وفقاً للمادة 148",
#                 "applied_elements": [{"element_id": "E1", "cited_articles": [148]}],
#                 "retrieved_articles": [{"article_number": 148, "article_text": "نص"}],
#                 "element_classifications": [{"element_id": "E1", "status": "established"}],
#                 "counterarguments": {"plaintiff_arguments": [], "defendant_arguments": []},
#                 "required_elements": [{"element_id": "E1"}],
#                 "skipped_elements": [],
#             })

#         assert package_required_keys.issubset(output.keys())


# @pytest.mark.unit
# class TestPackageOutputContract:
#     """T-CONTRACT-08: package_result_node output satisfies aggregate_issues_node input."""

#     _AGGREGATION_READS_FROM_PACKAGE = {
#         "issue_id", "applied_elements", "retrieved_facts",
#         "required_elements", "issue_title",
#     }

#     def test_package_output_has_aggregation_required_keys(self):
#         state = {
#             "issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود",
#             "source_text": "نص",
#             "required_elements": [{"element_id": "E1"}],
#             "law_retrieval_result": {"answer": ""},
#             "retrieved_articles": [], "retrieved_facts": "وقائع",
#             "element_classifications": [],
#             "law_application": "", "applied_elements": [], "skipped_elements": [],
#             "counterarguments": {},
#             "citation_check": {}, "logical_consistency_check": {},
#             "completeness_check": {}, "validation_passed": True,
#             "issue_analyses": [], "intermediate_steps": [], "error_log": [],
#         }
#         output = package_result_node(state)
#         wrapped = output["issue_analyses"][0]
#         assert self._AGGREGATION_READS_FROM_PACKAGE.issubset(wrapped.keys())


# # ---------------------------------------------------------------------------
# # Main graph contracts
# # ---------------------------------------------------------------------------

# @pytest.mark.unit
# class TestBranchToAggregationContract:
#     """T-CONTRACT-09: Branch results satisfy aggregation input contract."""

#     def test_issue_analyses_entry_has_aggregation_keys(self, make_branch_result):
#         analysis = make_branch_result()
#         assert "issue_id" in analysis
#         assert "applied_elements" in analysis
#         assert all("cited_articles" in el for el in analysis["applied_elements"])
#         assert "retrieved_facts" in analysis
#         assert "required_elements" in analysis


# @pytest.mark.unit
# class TestAggregationToConsistencyContract:
#     """T-CONTRACT-10: aggregate output satisfies consistency input."""

#     def test_aggregation_output_has_cross_issue_relationships(self):
#         mock_result = IssueDependencies(dependencies=[])
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = mock_result

#         with patch("nodes.aggregation.get_llm", return_value=llm):
#             output = aggregate_issues_node({
#                 "issue_analyses": [{"issue_id": 1, "applied_elements": [], "retrieved_facts": ""}],
#                 "identified_issues": [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود"}],
#             })

#         assert "cross_issue_relationships" in output
#         assert isinstance(output["cross_issue_relationships"], list)


# @pytest.mark.unit
# class TestConsistencyToConfidenceContract:
#     """T-CONTRACT-11: consistency output satisfies confidence input."""

#     def test_consistency_output_has_conflicts_key(self):
#         no_conflict = ConsistencyCheckResult(conflicts=[], has_conflicts=False)
#         llm = MagicMock()
#         llm.with_structured_output.return_value.invoke.return_value = no_conflict

#         with patch("nodes.consistency.get_llm", return_value=llm):
#             output = check_global_consistency_node({
#                 "issue_analyses": [
#                     {"issue_id": 1, "issue_title": "مسألة", "law_application": "تحليل",
#                      "applied_elements": []},
#                     {"issue_id": 2, "issue_title": "مسألة 2", "law_application": "تحليل 2",
#                      "applied_elements": []},
#                 ],
#                 "cross_issue_relationships": [],
#             })

#         assert "consistency_conflicts" in output
#         assert "reconciliation_paragraphs" in output


# @pytest.mark.unit
# class TestConfidenceToReportContract:
#     """T-CONTRACT-12: confidence output satisfies report input."""

#     def test_confidence_output_has_report_required_keys(self):
#         analysis = {
#             "issue_id": 1, "issue_title": "مسألة",
#             "required_elements": [{"element_id": "E1"}],
#             "element_classifications": [{"element_id": "E1", "status": "established"}],
#             "applied_elements": [{"element_id": "E1", "cited_articles": [148]}],
#             "citation_check": {"total_citations": 1, "unsupported_conclusions": [], "missing_citations": []},
#             "logical_consistency_check": {"severity": "none"},
#             "completeness_check": {"coverage_ratio": 1.0},
#         }
#         response = MagicMock()
#         response.content = "مبرر"
#         llm = MagicMock()
#         llm.invoke.return_value = response

#         with patch("nodes.confidence.get_llm", return_value=llm):
#             output = compute_confidence_node({
#                 "issue_analyses": [analysis],
#                 "consistency_conflicts": [],
#             })

#         assert "per_issue_confidence" in output
#         assert "case_level_confidence" in output
#         pc = output["per_issue_confidence"][0]
#         assert {"issue_id", "level", "raw_score", "justification"}.issubset(pc.keys())
#         cc = output["case_level_confidence"]
#         assert {"level", "raw_score", "justification"}.issubset(cc.keys())

"""
test_data_contracts.py — Inter-node state key contracts for the Case Reasoner.

Verifies that the output of node N contains all keys that node N+1 reads from state.
Tests use mocked LLMs/tools so no external services are needed.
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.extraction import extract_issues_node
from nodes.decomposition import decompose_issue_node
from nodes.query_generation import generate_retrieval_queries_node
from nodes.retrieval import retrieve_law_node, retrieve_facts_node
from nodes.evidence import classify_evidence_node
from nodes.application import apply_law_node
from nodes.counterargument import counterargument_node
from nodes.validation import validate_analysis_node
from nodes.package import package_result_node
from nodes.aggregation import aggregate_issues_node
from nodes.consistency import check_global_consistency_node
from nodes.confidence import compute_confidence_node
from schemas import (
    ExtractedIssues, LegalIssue, DecomposedIssue, RequiredElement,
    ElementQuery, RetrievalQueries,
    EvidenceSufficiencyResult, ElementClassification,
    LawApplicationResult, AppliedElement,
    Counterarguments, LogicalConsistencyResult,
    IssueDependencies, ConsistencyCheckResult,
)


# ---------------------------------------------------------------------------
# Branch chain contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractionOutputContract:
    """T-CONTRACT-01: extract_issues_node output satisfies router input."""

    def test_extraction_output_issues_have_router_keys(self):
        router_required_keys = {"issue_id", "issue_title", "legal_domain", "source_text"}
        mock_result = ExtractedIssues(issues=[
            LegalIssue(issue_id=1, issue_title="مسألة", legal_domain="عقود", source_text="نص"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node({"case_brief": {"legal_questions": "أ", "key_disputes": "ب"}})

        for issue in output["identified_issues"]:
            assert router_required_keys.issubset(issue.keys())


@pytest.mark.unit
class TestDecompositionOutputContract:
    """T-CONTRACT-02: decompose_issue_node output satisfies query_generation_node input."""

    def test_decomposition_output_has_required_elements_key(self):
        mock_result = DecomposedIssue(elements=[
            RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
            })

        assert "required_elements" in output
        el = output["required_elements"][0]
        assert {"element_id", "description", "element_type"}.issubset(el.keys())

    def test_decomposition_output_satisfies_query_generation_input(self):
        """T-CONTRACT-02b: required_elements from decomposition consumed by query generation."""
        query_generation_reads = {"required_elements"}
        mock_result = DecomposedIssue(elements=[
            RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
            })

        assert query_generation_reads.issubset(output.keys())


@pytest.mark.unit
class TestQueryGenerationOutputContract:
    """T-CONTRACT-03: generate_retrieval_queries_node output satisfies retrieval nodes."""

    def test_output_has_law_queries_key(self):
        mock_result = RetrievalQueries(queries=[
            ElementQuery(element_id="E1", law_query="سؤال قانوني", fact_query="سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
            })

        assert "law_queries" in output

    def test_output_has_fact_queries_key(self):
        mock_result = RetrievalQueries(queries=[
            ElementQuery(element_id="E1", law_query="سؤال قانوني", fact_query="سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
            })

        assert "fact_queries" in output

    def test_law_queries_is_list_of_dicts_with_element_id_and_query(self):
        """retrieve_law_node reads law_queries as list of {element_id, query}."""
        mock_result = RetrievalQueries(queries=[
            ElementQuery(element_id="E1", law_query="سؤال قانوني", fact_query="سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
            })

        assert isinstance(output["law_queries"], list)
        for lq in output["law_queries"]:
            assert "element_id" in lq
            assert "query" in lq

    def test_fact_queries_is_list_of_dicts_with_element_id_and_query(self):
        """retrieve_facts_node reads fact_queries as list of {element_id, query}."""
        mock_result = RetrievalQueries(queries=[
            ElementQuery(element_id="E1", law_query="سؤال قانوني", fact_query="سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
            })

        assert isinstance(output["fact_queries"], list)
        for fq in output["fact_queries"]:
            assert "element_id" in fq
            assert "query" in fq


@pytest.mark.unit
class TestRetrievalOutputContract:
    """T-CONTRACT-04: retrieval node outputs satisfy evidence node input."""

    def test_retrieve_law_output_satisfies_evidence_input(self):
        evidence_reads = {"law_retrieval_result", "retrieved_articles"}
        with patch("tools.civil_law_rag_tool", return_value={
            "answer": "نص قانوني", "sources": [{"article": 148, "content": "نص"}],
            "classification": "", "retrieval_confidence": 0, "citation_integrity": 0,
            "from_cache": False, "error": None,
        }):
            output = retrieve_law_node({
                "issue_title": "مسألة", "source_text": "نص", "case_id": "001",
                "law_queries": [{"element_id": "E1", "query": "سؤال قانوني"}],
            })

        assert evidence_reads.issubset(output.keys())

    def test_retrieve_facts_output_satisfies_evidence_input(self):
        evidence_reads = {"fact_retrieval_result", "retrieved_facts"}
        with patch("tools.case_documents_rag_tool", return_value={
            "final_answer": "وقائع", "sub_answers": [], "error": None,
        }):
            output = retrieve_facts_node({
                "issue_title": "مسألة", "source_text": "نص", "case_id": "001",
                "fact_queries": [{"element_id": "E1", "query": "سؤال وقائعي"}],
            })

        assert evidence_reads.issubset(output.keys())

    def test_law_retrieval_result_has_answer_key(self):
        with patch("tools.civil_law_rag_tool", return_value={
            "answer": "نص قانوني", "sources": [], "classification": "",
            "retrieval_confidence": 0, "citation_integrity": 0, "from_cache": False, "error": None,
        }):
            output = retrieve_law_node({
                "issue_title": "مسألة", "source_text": "نص", "case_id": "001",
                "law_queries": [{"element_id": "E1", "query": "سؤال"}],
            })

        assert "answer" in output["law_retrieval_result"]

    def test_retrieved_facts_is_string(self):
        """Evidence node reads retrieved_facts as a plain string."""
        with patch("tools.case_documents_rag_tool", return_value={
            "final_answer": "وقائع", "sub_answers": [], "error": None,
        }):
            output = retrieve_facts_node({
                "issue_title": "مسألة", "source_text": "نص", "case_id": "001",
                "fact_queries": [{"element_id": "E1", "query": "سؤال"}],
            })

        assert isinstance(output["retrieved_facts"], str)


@pytest.mark.unit
class TestEvidenceOutputContract:
    """T-CONTRACT-05: classify_evidence_node output satisfies apply_law_node input."""

    def test_classifications_have_element_id_and_status(self):
        application_required_keys = {"element_id", "status"}
        mock_result = EvidenceSufficiencyResult(classifications=[
            ElementClassification(element_id="E1", status="established", evidence_summary="ثابت"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node({
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
                "retrieved_facts": "وقائع",
                "law_retrieval_result": {"answer": "نص"},
            })

        assert "element_classifications" in output
        for c in output["element_classifications"]:
            assert application_required_keys.issubset(c.keys())


@pytest.mark.unit
class TestApplicationOutputContract:
    """T-CONTRACT-06: apply_law_node output satisfies counterargument_node input."""

    def test_application_output_has_counterargument_required_keys(self):
        counterarg_reads = {"law_application", "applied_elements", "skipped_elements"}
        mock_result = LawApplicationResult(
            elements=[AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148])],
            synthesis="تحليل إجمالي",
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node({
                "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
                "element_classifications": [{"element_id": "E1", "status": "established"}],
                "retrieved_facts": "وقائع",
                "law_retrieval_result": {"answer": "نص"},
            })

        assert counterarg_reads.issubset(output.keys())


@pytest.mark.unit
class TestCounterargumentOutputContract:
    """T-CONTRACT-07: counterargument_node output satisfies validate_analysis_node input."""

    def test_counterarguments_has_required_structure(self):
        validation_reads_from_counterargs = {"plaintiff_arguments", "defendant_arguments"}
        mock_result = Counterarguments(
            plaintiff_arguments=["حجة"],
            defendant_arguments=["دفع"],
            analysis="تحليل",
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node({
                "law_application": "تحليل", "element_classifications": [],
                "retrieved_facts": "وقائع", "issue_title": "مسألة",
            })

        assert "counterarguments" in output
        assert validation_reads_from_counterargs.issubset(output["counterarguments"].keys())


@pytest.mark.unit
class TestValidationOutputContract:
    """T-CONTRACT-08: validate_analysis_node output satisfies package_result_node input."""

    def test_validation_output_has_all_check_keys(self):
        package_required_keys = {
            "citation_check", "logical_consistency_check", "completeness_check", "validation_passed"
        }
        mock_consistency = LogicalConsistencyResult(passed=True, issues_found=[], severity="none")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_consistency

        with patch("nodes.validation.get_llm", return_value=llm), \
             patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}):
            output = validate_analysis_node({
                "issue_title": "مسألة",
                "law_application": "وفقاً للمادة 148",
                "applied_elements": [{"element_id": "E1", "cited_articles": [148]}],
                "retrieved_articles": [{"article_number": 148, "article_text": "نص"}],
                "element_classifications": [{"element_id": "E1", "status": "established"}],
                "counterarguments": {"plaintiff_arguments": [], "defendant_arguments": []},
                "required_elements": [{"element_id": "E1"}],
                "skipped_elements": [],
            })

        assert package_required_keys.issubset(output.keys())


@pytest.mark.unit
class TestPackageOutputContract:
    """T-CONTRACT-09: package_result_node output satisfies aggregate_issues_node input."""

    _AGGREGATION_READS_FROM_PACKAGE = {
        "issue_id", "applied_elements", "retrieved_facts",
        "required_elements", "issue_title",
    }

    def test_package_output_has_aggregation_required_keys(self):
        state = {
            "issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود",
            "source_text": "نص",
            "required_elements": [{"element_id": "E1"}],
            "law_retrieval_result": {"answer": ""},
            "retrieved_articles": [], "retrieved_facts": "وقائع",
            "element_classifications": [],
            "law_application": "", "applied_elements": [], "skipped_elements": [],
            "counterarguments": {},
            "citation_check": {}, "logical_consistency_check": {},
            "completeness_check": {}, "validation_passed": True,
            "issue_analyses": [], "intermediate_steps": [], "error_log": [],
        }
        output = package_result_node(state)
        wrapped = output["issue_analyses"][0]
        assert self._AGGREGATION_READS_FROM_PACKAGE.issubset(wrapped.keys())


# ---------------------------------------------------------------------------
# Main graph contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBranchToAggregationContract:
    """T-CONTRACT-10: Branch results satisfy aggregation input contract."""

    def test_issue_analyses_entry_has_aggregation_keys(self, make_branch_result):
        analysis = make_branch_result()
        assert "issue_id" in analysis
        assert "applied_elements" in analysis
        assert all("cited_articles" in el for el in analysis["applied_elements"])
        assert "retrieved_facts" in analysis
        assert "required_elements" in analysis


@pytest.mark.unit
class TestAggregationToConsistencyContract:
    """T-CONTRACT-11: aggregate output satisfies consistency input."""

    def test_aggregation_output_has_cross_issue_relationships(self):
        mock_result = IssueDependencies(dependencies=[])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node({
                "issue_analyses": [{"issue_id": 1, "applied_elements": [], "retrieved_facts": ""}],
                "identified_issues": [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود"}],
            })

        assert "cross_issue_relationships" in output
        assert isinstance(output["cross_issue_relationships"], list)


@pytest.mark.unit
class TestConsistencyToConfidenceContract:
    """T-CONTRACT-12: consistency output satisfies confidence input."""

    def test_consistency_output_has_conflicts_key(self):
        no_conflict = ConsistencyCheckResult(conflicts=[], has_conflicts=False)
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = no_conflict

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node({
                "issue_analyses": [
                    {"issue_id": 1, "issue_title": "مسألة", "law_application": "تحليل",
                     "applied_elements": []},
                    {"issue_id": 2, "issue_title": "مسألة 2", "law_application": "تحليل 2",
                     "applied_elements": []},
                ],
                "cross_issue_relationships": [],
            })

        assert "consistency_conflicts" in output
        assert "reconciliation_paragraphs" in output


@pytest.mark.unit
class TestConfidenceToReportContract:
    """T-CONTRACT-13: confidence output satisfies report input."""

    def test_confidence_output_has_report_required_keys(self):
        analysis = {
            "issue_id": 1, "issue_title": "مسألة",
            "required_elements": [{"element_id": "E1"}],
            "element_classifications": [{"element_id": "E1", "status": "established"}],
            "applied_elements": [{"element_id": "E1", "cited_articles": [148]}],
            "citation_check": {"total_citations": 1, "unsupported_conclusions": [], "missing_citations": []},
            "logical_consistency_check": {"severity": "none"},
            "completeness_check": {"coverage_ratio": 1.0},
        }
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node({
                "issue_analyses": [analysis],
                "consistency_conflicts": [],
            })

        assert "per_issue_confidence" in output
        assert "case_level_confidence" in output
        pc = output["per_issue_confidence"][0]
        assert {"issue_id", "level", "raw_score", "justification"}.issubset(pc.keys())
        cc = output["case_level_confidence"]
        assert {"level", "raw_score", "justification"}.issubset(cc.keys())


# ---------------------------------------------------------------------------
# Full chain: decomposition → query generation → retrieval
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDecompositionToRetrievalChain:
    """T-CONTRACT-14: decomposition → query generation → retrieval full chain."""

    def test_decomposition_output_feeds_query_generation_feeds_retrieval(self):
        """End-to-end data flow: required_elements → law_queries → civil_law_rag_tool."""
        # Step 1: decomposition
        decomp_result = DecomposedIssue(elements=[
            RequiredElement(element_id="E1", description="وجود عقد صحيح", element_type="legal"),
        ])
        decomp_llm = MagicMock()
        decomp_llm.with_structured_output.return_value.invoke.return_value = decomp_result

        with patch("nodes.decomposition.get_llm", return_value=decomp_llm):
            decomp_output = decompose_issue_node({
                "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
            })

        # Step 2: query generation consumes required_elements from decomposition
        qgen_result = RetrievalQueries(queries=[
            ElementQuery(
                element_id="E1",
                law_query="ما هي أحكام العقد الصحيح في القانون المدني المصري؟",
                fact_query="ما الوقائع المتعلقة بالعقد في ملف القضية؟",
            ),
        ])
        qgen_llm = MagicMock()
        qgen_llm.with_structured_output.return_value.invoke.return_value = qgen_result

        with patch("nodes.query_generation.get_llm", return_value=qgen_llm):
            qgen_output = generate_retrieval_queries_node({
                "issue_title": "مسألة",
                "legal_domain": "عقود",
                "source_text": "نص",
                "required_elements": decomp_output["required_elements"],
            })

        # Step 3: retrieval consumes law_queries from query generation
        civil_law_result = {
            "answer": "نص المادة 89", "sources": [{"article": 89, "content": "نص"}],
            "classification": "", "retrieval_confidence": 0.9,
            "citation_integrity": 0.9, "from_cache": False, "error": None,
        }
        with patch("tools.civil_law_rag_tool", return_value=civil_law_result) as mock_tool:
            retrieval_output = retrieve_law_node({
                "issue_title": "مسألة",
                "source_text": "نص",
                "case_id": "test-001",
                "law_queries": qgen_output["law_queries"],
            })

        # Verify full chain
        assert len(decomp_output["required_elements"]) == 1
        assert len(qgen_output["law_queries"]) == 1
        assert qgen_output["law_queries"][0]["query"] == "ما هي أحكام العقد الصحيح في القانون المدني المصري؟"
        assert mock_tool.call_count == 1
        assert 89 in {a["article_number"] for a in retrieval_output["retrieved_articles"]}
