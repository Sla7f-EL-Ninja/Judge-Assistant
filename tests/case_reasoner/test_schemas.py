"""
test_schemas.py — Pydantic schema validation for all 9 Case Reasoner models.

Tests:
    T-SCHEMA-01: LegalIssue / ExtractedIssues
    T-SCHEMA-02: RequiredElement / DecomposedIssue
    T-SCHEMA-03: ElementClassification / EvidenceSufficiencyResult
    T-SCHEMA-04: AppliedElement / LawApplicationResult
    T-SCHEMA-05: Counterarguments
    T-SCHEMA-06: LogicalConsistencyResult
    T-SCHEMA-07: IssueDependency / IssueDependencies
    T-SCHEMA-08: ConsistencyConflict / ConsistencyCheckResult
"""

import pathlib
import sys

import pytest
from pydantic import ValidationError

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from schemas import (
    AppliedElement,
    Counterarguments,
    ConsistencyCheckResult,
    ConsistencyConflict,
    DecomposedIssue,
    ElementClassification,
    EvidenceSufficiencyResult,
    ExtractedIssues,
    IssueDependencies,
    IssueDependency,
    LawApplicationResult,
    LegalIssue,
    LogicalConsistencyResult,
    RequiredElement,
)


# ---------------------------------------------------------------------------
# T-SCHEMA-01: LegalIssue / ExtractedIssues
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLegalIssueSchema:
    """T-SCHEMA-01: LegalIssue validates correctly."""

    def test_valid_legal_issue(self):
        issue = LegalIssue(
            issue_id=1,
            issue_title="التعويض عن الإخلال بالعقد",
            legal_domain="العقود المدنية",
            source_text="نص مقتبس من الملخص",
        )
        assert issue.issue_id == 1
        assert issue.issue_title == "التعويض عن الإخلال بالعقد"

    def test_extracted_issues_with_multiple(self):
        issues = ExtractedIssues(issues=[
            LegalIssue(issue_id=1, issue_title="مسألة أولى", legal_domain="عقود", source_text="نص"),
            LegalIssue(issue_id=2, issue_title="مسألة ثانية", legal_domain="مسؤولية", source_text="نص"),
        ])
        assert len(issues.issues) == 2

    def test_extracted_issues_empty(self):
        issues = ExtractedIssues(issues=[])
        assert issues.issues == []

    def test_legal_issue_requires_all_fields(self):
        with pytest.raises(ValidationError):
            LegalIssue(issue_id=1, issue_title="عنوان")  # missing fields


# ---------------------------------------------------------------------------
# T-SCHEMA-02: RequiredElement / DecomposedIssue
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRequiredElementSchema:
    """T-SCHEMA-02: RequiredElement validates element_type literal."""

    def test_valid_legal_element(self):
        el = RequiredElement(element_id="E1", description="وجود عقد صحيح", element_type="legal")
        assert el.element_type == "legal"

    def test_valid_factual_element(self):
        el = RequiredElement(element_id="E2", description="وقوع الضرر", element_type="factual")
        assert el.element_type == "factual"

    def test_invalid_element_type_rejected(self):
        with pytest.raises(ValidationError):
            RequiredElement(element_id="E1", description="وصف", element_type="unknown")

    def test_decomposed_issue_with_elements(self):
        di = DecomposedIssue(elements=[
            RequiredElement(element_id="E1", description="عنصر قانوني", element_type="legal"),
            RequiredElement(element_id="E2", description="عنصر واقعي", element_type="factual"),
        ])
        assert len(di.elements) == 2

    def test_decomposed_issue_empty(self):
        di = DecomposedIssue(elements=[])
        assert di.elements == []


# ---------------------------------------------------------------------------
# T-SCHEMA-03: ElementClassification / EvidenceSufficiencyResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestElementClassificationSchema:
    """T-SCHEMA-03: ElementClassification validates all 4 status literals."""

    @pytest.mark.parametrize("status", [
        "established", "not_established", "disputed", "insufficient_evidence"
    ])
    def test_valid_status(self, status):
        c = ElementClassification(
            element_id="E1",
            status=status,
            evidence_summary="ملخص الدليل",
        )
        assert c.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            ElementClassification(element_id="E1", status="unknown", evidence_summary="")

    def test_notes_defaults_to_empty(self):
        c = ElementClassification(element_id="E1", status="disputed", evidence_summary="ملخص")
        assert c.notes == ""

    def test_evidence_sufficiency_result_wraps_list(self):
        result = EvidenceSufficiencyResult(classifications=[
            ElementClassification(element_id="E1", status="established", evidence_summary="ثابت"),
        ])
        assert len(result.classifications) == 1


# ---------------------------------------------------------------------------
# T-SCHEMA-04: AppliedElement / LawApplicationResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAppliedElementSchema:
    """T-SCHEMA-04: AppliedElement and LawApplicationResult."""

    def test_applied_element_with_articles(self):
        el = AppliedElement(
            element_id="E1",
            reasoning="تحليل قانوني",
            cited_articles=[148, 176],
        )
        assert 148 in el.cited_articles

    def test_applied_element_empty_articles(self):
        el = AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[])
        assert el.cited_articles == []

    def test_law_application_result(self):
        result = LawApplicationResult(
            elements=[AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148])],
            synthesis="تحليل إجمالي محايد",
        )
        assert result.synthesis == "تحليل إجمالي محايد"
        assert len(result.elements) == 1


# ---------------------------------------------------------------------------
# T-SCHEMA-05: Counterarguments
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterargumentsSchema:
    """T-SCHEMA-05: Counterarguments schema."""

    def test_valid_counterarguments(self):
        ca = Counterarguments(
            plaintiff_arguments=["حجة المدعي الأولى", "حجة المدعي الثانية"],
            defendant_arguments=["دفع المدعى عليه"],
            analysis="مقارنة محايدة للحجج",
        )
        assert len(ca.plaintiff_arguments) == 2
        assert len(ca.defendant_arguments) == 1

    def test_empty_arguments_allowed(self):
        ca = Counterarguments(
            plaintiff_arguments=[],
            defendant_arguments=[],
            analysis="لا حجج",
        )
        assert ca.plaintiff_arguments == []


# ---------------------------------------------------------------------------
# T-SCHEMA-06: LogicalConsistencyResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLogicalConsistencyResultSchema:
    """T-SCHEMA-06: LogicalConsistencyResult severity literals."""

    @pytest.mark.parametrize("severity", ["none", "minor", "major"])
    def test_valid_severity(self, severity):
        result = LogicalConsistencyResult(passed=True, issues_found=[], severity=severity)
        assert result.severity == severity

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            LogicalConsistencyResult(passed=True, issues_found=[], severity="critical")

    def test_failed_with_issues(self):
        result = LogicalConsistencyResult(
            passed=False,
            issues_found=["تناقض في تقييم العنصر E1"],
            severity="major",
        )
        assert not result.passed
        assert len(result.issues_found) == 1


# ---------------------------------------------------------------------------
# T-SCHEMA-07: IssueDependency / IssueDependencies
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIssueDependencySchema:
    """T-SCHEMA-07: IssueDependency and IssueDependencies."""

    def test_valid_dependency(self):
        dep = IssueDependency(
            upstream_issue_id=1,
            downstream_issue_id=2,
            dependency_type="شرطية",
            explanation="يجب إثبات المسألة الأولى قبل الثانية",
        )
        assert dep.upstream_issue_id == 1
        assert dep.downstream_issue_id == 2

    def test_issue_dependencies_empty(self):
        deps = IssueDependencies(dependencies=[])
        assert deps.dependencies == []

    def test_issue_dependencies_with_items(self):
        deps = IssueDependencies(dependencies=[
            IssueDependency(upstream_issue_id=1, downstream_issue_id=2,
                            dependency_type="فرعية", explanation="تابعة"),
        ])
        assert len(deps.dependencies) == 1


# ---------------------------------------------------------------------------
# T-SCHEMA-08: ConsistencyConflict / ConsistencyCheckResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConsistencyCheckResultSchema:
    """T-SCHEMA-08: ConsistencyConflict and ConsistencyCheckResult."""

    def test_valid_conflict(self):
        conflict = ConsistencyConflict(
            issue_ids=[1, 2],
            conflict_type="contradictory_article_application",
            description="تطبيق متناقض للمادة 148",
        )
        assert 1 in conflict.issue_ids
        assert 2 in conflict.issue_ids

    def test_consistency_check_result_with_conflicts(self):
        result = ConsistencyCheckResult(
            conflicts=[
                ConsistencyConflict(
                    issue_ids=[1, 2],
                    conflict_type="contradictory_fact_evaluation",
                    description="تقييم متناقض للوقائع",
                )
            ],
            has_conflicts=True,
        )
        assert result.has_conflicts
        assert len(result.conflicts) == 1

    def test_no_conflicts(self):
        result = ConsistencyCheckResult(conflicts=[], has_conflicts=False)
        assert not result.has_conflicts
        assert result.conflicts == []
