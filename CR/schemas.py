from typing import List, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Issue Extraction
# ---------------------------------------------------------------------------

class LegalIssue(BaseModel):
    issue_id: int
    issue_title: str
    legal_domain: str
    source_text: str = Field(description="Exact excerpt from the brief that raises this issue")

class ExtractedIssues(BaseModel):
    issues: List[LegalIssue]


# ---------------------------------------------------------------------------
# Issue Decomposition
# ---------------------------------------------------------------------------

class RequiredElement(BaseModel):
    element_id: str                              # e.g. "E1", "E2"
    description: str                             # Arabic description of what must be proven
    element_type: Literal["factual", "legal"]

class DecomposedIssue(BaseModel):
    elements: List[RequiredElement]


# ---------------------------------------------------------------------------
# Evidence Sufficiency
# ---------------------------------------------------------------------------

class ElementClassification(BaseModel):
    element_id: str
    status: Literal["established", "not_established", "disputed", "insufficient_evidence"]
    evidence_summary: str
    notes: str = ""

class EvidenceSufficiencyResult(BaseModel):
    classifications: List[ElementClassification]


# ---------------------------------------------------------------------------
# Law Application
# ---------------------------------------------------------------------------

class AppliedElement(BaseModel):
    element_id: str
    reasoning: str
    cited_articles: List[int]

class LawApplicationResult(BaseModel):
    elements: List[AppliedElement]
    synthesis: str                               # Neutral per-issue Arabic analysis, NO ruling


# ---------------------------------------------------------------------------
# Counterarguments
# ---------------------------------------------------------------------------

class Counterarguments(BaseModel):
    plaintiff_arguments: List[str]
    defendant_arguments: List[str]
    analysis: str                                # Neutral comparison — NOT a conclusion


# ---------------------------------------------------------------------------
# Logical Consistency (Validation sub-step 2)
# ---------------------------------------------------------------------------

class LogicalConsistencyResult(BaseModel):
    passed: bool
    issues_found: List[str]
    severity: Literal["none", "minor", "major"]


# ---------------------------------------------------------------------------
# Aggregation — Issue Dependencies
# ---------------------------------------------------------------------------

class IssueDependency(BaseModel):
    upstream_issue_id: int
    downstream_issue_id: int
    dependency_type: str    # e.g. "شرطية" (conditional), "فرعية" (subsidiary)
    explanation: str

class IssueDependencies(BaseModel):
    dependencies: List[IssueDependency]


# ---------------------------------------------------------------------------
# Global Consistency
# ---------------------------------------------------------------------------

class ConsistencyConflict(BaseModel):
    issue_ids: List[int]
    conflict_type: str    # e.g. "contradictory_article_application", "contradictory_fact_evaluation"
    description: str

class ConsistencyCheckResult(BaseModel):
    conflicts: List[ConsistencyConflict]
    has_conflicts: bool
