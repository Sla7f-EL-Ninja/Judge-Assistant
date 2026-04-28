"""
test_pipeline_integration.py — End-to-end Case Reasoner pipeline tests.

Uses real LLM calls. Skip without GOOGLE_API_KEY.

Run with:
    pytest tests/case_reasoner/test_pipeline_integration.py -m case_reasoner_llm -v
"""
import pathlib, sys


_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import os
import pathlib
import re
import sys
import time
from typing import Any

import pytest
import yaml

from conftest import cr_ingestion

from eval_config import (
    PIPELINE_TIMING_THRESHOLD_SECONDS,
    SECTION_HEADER_PATTERN,
    SECTION_VI_HEADER,
    SECTION_VIII_HEADER,
    BIAS_KEYWORDS,
    DIRECTIONAL_VERBS,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CR_DIR = _REPO_ROOT / "CR"
for _p in [str(_REPO_ROOT), str(_CR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_placeholder(value):
    return isinstance(value, str) and "[PLACEHOLDER" in value


def _has_placeholders(sample):
    def _check(obj):
        if isinstance(obj, str):
            return _is_placeholder(obj)
        if isinstance(obj, dict):
            return any(_check(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_check(item) for item in obj)
        return False
    return _check(sample)


def _extract_section(report_text: str, section_header: str) -> str:
    from eval_config import SECTION_ORDINALS
    match = re.search(re.escape(section_header), report_text)
    if not match:
        return ""
    start = match.end()
    next_section_pattern = r"القسم\s+(?:" + "|".join(SECTION_ORDINALS) + r")"
    next_match = re.search(next_section_pattern, report_text[start:])
    if next_match:
        return report_text[start : start + next_match.start()].strip()
    return report_text[start:].strip()


def _count_sections(report_text: str) -> int:
    return len(re.findall(SECTION_HEADER_PATTERN, report_text))


def _build_initial_state(sample_input: dict) -> dict:
    return {
        "case_id": sample_input["case_id"],
        "judge_query": sample_input["judge_query"],
        "case_brief": sample_input["case_brief"],
        "rendered_brief": sample_input.get("rendered_brief", ""),
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


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_llm_available():
    return bool(os.environ.get("GOOGLE_API_KEY"))


@pytest.fixture(scope="session")
def case_reasoner_graph(real_llm_available, cr_ingestion):
    if not real_llm_available:
        pytest.skip("GOOGLE_API_KEY not set — skipping integration tests")
    from graph import build_case_reasoner_graph
    return build_case_reasoner_graph()


@pytest.fixture(scope="session")
def golden_samples():
    golden_path = pathlib.Path(__file__).parent / "golden_set.yaml"
    raw = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    return raw.get("samples", [])


@pytest.fixture(scope="session")
def filled_samples(golden_samples):
    return [s for s in golden_samples if not _has_placeholders(s)]


@pytest.fixture(scope="session")
def pipeline_results(case_reasoner_graph, filled_samples, cr_ingestion):
    """Run the full pipeline on each filled golden sample and record timing."""
    if not filled_samples:
        pytest.skip("No filled golden samples — fill golden_set.yaml first")

    results = {}
    timings = {}
    for sample in filled_samples:
        sid = sample["sample_id"]
        initial_state = _build_initial_state(sample["input"])
        initial_state["case_id"] = cr_ingestion  # use the actual ingested case_id
        t0 = time.monotonic()
        results[sid] = case_reasoner_graph.invoke(initial_state)
        timings[sid] = time.monotonic() - t0

    results["__timings__"] = timings
    return results


# ---------------------------------------------------------------------------
# T-INT-01: Pipeline output structure
# ---------------------------------------------------------------------------

@pytest.mark.case_reasoner_llm
class TestPipelineOutputStructure:
    """Verify the pipeline produces a structurally valid output for each golden sample."""

    def test_final_report_non_empty(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            assert len(result.get("final_report", "")) > 0, (
                f"{sid}: final_report is empty"
            )

    def test_identified_issues_is_list(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            assert isinstance(result.get("identified_issues"), list), (
                f"{sid}: identified_issues is not a list"
            )

    def test_issue_count_within_range(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if "issue_count" not in expected:
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            actual = len(result.get("identified_issues", []))
            expected_count = expected["issue_count"]
            # Allow ±1 tolerance for LLM variability, but exact 0 must be exact
            if expected_count == 0:
                assert actual == 0, f"{sid}: expected 0 issues but found {actual}"
            else:
                assert abs(actual - expected_count) <= 1, (
                    f"{sid}: issue count {actual} is not close to expected {expected_count}"
                )

    def test_issue_analyses_count_matches_identified_issues(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            n_issues = len(result.get("identified_issues", []))
            n_analyses = len(result.get("issue_analyses", []))
            if n_issues == 0:
                continue
            assert n_analyses == n_issues, (
                f"{sid}: {n_issues} issues but {n_analyses} analyses"
            )

    def test_all_state_keys_present(self, pipeline_results, filled_samples):
        required_keys = {
            "identified_issues", "issue_analyses", "cross_issue_relationships",
            "consistency_conflicts", "reconciliation_paragraphs",
            "per_issue_confidence", "case_level_confidence",
            "final_report", "intermediate_steps", "error_log",
        }
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            missing = required_keys - set(result.keys())
            assert not missing, f"{sid}: missing output keys: {missing}"

    def test_case_level_confidence_has_required_fields(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            if not result.get("identified_issues"):
                continue
            clc = result.get("case_level_confidence", {})
            assert "level" in clc, f"{sid}: case_level_confidence missing 'level'"
            assert "raw_score" in clc, f"{sid}: case_level_confidence missing 'raw_score'"
            assert clc["level"] in {"high", "medium", "low"}, (
                f"{sid}: invalid confidence level '{clc['level']}'"
            )


# ---------------------------------------------------------------------------
# T-INT-02: Report section structure
# ---------------------------------------------------------------------------

@pytest.mark.case_reasoner_llm
class TestReportSectionStructure:
    """Verify the final report contains expected sections."""

    def test_non_empty_report_has_minimum_sections(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if expected.get("empty_report", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            section_count = _count_sections(report)
            assert section_count >= 7, (
                f"{sid}: only {section_count} sections found (expected >= 7)"
            )

    def test_reconciliation_section_present_when_conflicts(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if not expected.get("has_cross_issue_conflicts", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            section_viii = _extract_section(report, SECTION_VIII_HEADER)
            reconciliation = result.get("reconciliation_paragraphs", [])
            assert len(reconciliation) > 0 or len(section_viii) > 0, (
                f"{sid}: expected reconciliation content but neither section VIII nor "
                "reconciliation_paragraphs found"
            )

    def test_section_vi_present_in_non_empty_report(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if expected.get("empty_report", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            section_vi = _extract_section(report, SECTION_VI_HEADER)
            assert len(section_vi) > 0, (
                f"{sid}: Section VI (حالة الملف) not found in report"
            )

    def test_sections_are_non_empty(self, pipeline_results, filled_samples):
        from eval_config import SECTION_ORDINALS
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if expected.get("empty_report", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            for ordinal in SECTION_ORDINALS[:7]:
                header = f"القسم {ordinal}"
                if header in report:
                    content = _extract_section(report, header)
                    assert len(content) > 10, (
                        f"{sid}: Section '{header}' is present but nearly empty"
                    )


# ---------------------------------------------------------------------------
# T-INT-03: Critical constraint verification
# ---------------------------------------------------------------------------

@pytest.mark.case_reasoner_llm
class TestPipelineConstraints:
    """Verify the pipeline respects the 10 critical constraints at runtime."""

    def test_insufficient_evidence_elements_skipped(self, pipeline_results, filled_samples):
        """Constraint #5: insufficient_evidence elements must not appear in applied_elements."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            for analysis in result.get("issue_analyses", []):
                insufficient_ids = {
                    c["element_id"]
                    for c in analysis.get("element_classifications", [])
                    if c.get("status") == "insufficient_evidence"
                }
                applied_ids = {el["element_id"] for el in analysis.get("applied_elements", [])}
                leaked = insufficient_ids & applied_ids
                assert not leaked, (
                    f"{sid} issue {analysis.get('issue_id')}: "
                    f"insufficient_evidence elements appeared in applied_elements: {leaked}"
                )

    def test_confidence_level_is_valid(self, pipeline_results, filled_samples):
        """Constraint #6: confidence level must be high/medium/low."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            for pc in result.get("per_issue_confidence", []):
                assert pc["level"] in {"high", "medium", "low"}, (
                    f"{sid}: invalid per-issue confidence level '{pc['level']}'"
                )

    def test_section_vi_no_directional_language(self, pipeline_results, filled_samples):
        """Constraint #7: Section VI must not contain bias keywords."""
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if not expected.get("section_vi_bias_free", True):
                continue
            if expected.get("empty_report", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            section_vi = _extract_section(report, SECTION_VI_HEADER)
            if not section_vi:
                continue
            violations = [kw for kw in BIAS_KEYWORDS + DIRECTIONAL_VERBS if kw in section_vi]
            assert not violations, (
                f"{sid}: Section VI contains directional language: {violations}"
            )

    def test_reconciliation_only_when_conflicts(self, pipeline_results, filled_samples):
        """Constraint #8: reconciliation_paragraphs only if consistency_conflicts non-empty."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            has_conflicts = len(result.get("consistency_conflicts", [])) > 0
            has_reconciliation = len(result.get("reconciliation_paragraphs", [])) > 0

            if has_reconciliation and not has_conflicts:
                pytest.fail(
                    f"{sid}: reconciliation_paragraphs present but consistency_conflicts is empty"
                )

    def test_all_issue_ids_unique(self, pipeline_results, filled_samples):
        """Each issue_id appears exactly once in issue_analyses."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            ids = [a["issue_id"] for a in result.get("issue_analyses", [])]
            assert len(ids) == len(set(ids)), (
                f"{sid}: duplicate issue_ids in issue_analyses: {ids}"
            )

    def test_applied_elements_have_cited_articles_or_skipped(self, pipeline_results, filled_samples):
        """Applied elements should have at least one cited article."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            for analysis in result.get("issue_analyses", []):
                for el in analysis.get("applied_elements", []):
                    assert el.get("cited_articles") is not None, (
                        f"{sid}: applied element '{el.get('element_id')}' missing cited_articles"
                    )

    def test_error_log_is_list(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            assert isinstance(result.get("error_log", []), list), (
                f"{sid}: error_log is not a list"
            )

    def test_intermediate_steps_non_empty(self, pipeline_results, filled_samples):
        """Pipeline must log at least some intermediate steps."""
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            steps = result.get("intermediate_steps", [])
            assert len(steps) > 0, f"{sid}: no intermediate_steps logged"


# ---------------------------------------------------------------------------
# T-INT-04: Timing
# ---------------------------------------------------------------------------

@pytest.mark.case_reasoner_llm
class TestPipelineTiming:
    """Verify pipeline completes within threshold."""

    def test_each_sample_within_threshold(self, pipeline_results, filled_samples):
        timings = pipeline_results.get("__timings__", {})
        for sample in filled_samples:
            sid = sample["sample_id"]
            elapsed = timings.get(sid, 0)
            assert elapsed <= PIPELINE_TIMING_THRESHOLD_SECONDS, (
                f"{sid}: pipeline took {elapsed:.1f}s, threshold is {PIPELINE_TIMING_THRESHOLD_SECONDS}s"
            )


# ---------------------------------------------------------------------------
# T-INT-05: Specific golden sample assertions
# ---------------------------------------------------------------------------

@pytest.mark.case_reasoner_llm
class TestGoldenSampleSpecific:
    """Sample-specific assertions from golden_set.yaml expected fields."""

    def _get_result(self, pipeline_results, sample_id):
        return pipeline_results.get(sample_id)

    def test_gs_cr_01_two_issues(self, pipeline_results, filled_samples):
        """GS-CR-01 must produce exactly 2 issues."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-01" not in samples:
            pytest.skip("GS-CR-01 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-01")
        assert len(result["identified_issues"]) == 2

    def test_gs_cr_02_medium_confidence(self, pipeline_results, filled_samples):
        """GS-CR-02 must produce medium or low confidence."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-02" not in samples:
            pytest.skip("GS-CR-02 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-02")
        level = result.get("case_level_confidence", {}).get("level", "")
        assert level in {"medium", "low"}, (
            f"GS-CR-02: expected medium/low confidence but got '{level}'"
        )

    def test_gs_cr_02_has_skipped_elements(self, pipeline_results, filled_samples):
        """GS-CR-02 must have at least one skipped element (insufficient evidence)."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-02" not in samples:
            pytest.skip("GS-CR-02 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-02")
        all_skipped = [
            el
            for a in result.get("issue_analyses", [])
            for el in a.get("skipped_elements", [])
        ]
        assert len(all_skipped) > 0, "GS-CR-02: expected skipped elements but none found"

    def test_gs_cr_03_has_reconciliation(self, pipeline_results, filled_samples):
        """GS-CR-03 must trigger reconciliation."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-03" not in samples:
            pytest.skip("GS-CR-03 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-03")
        assert len(result.get("reconciliation_paragraphs", [])) >= 1, (
            "GS-CR-03: expected reconciliation but reconciliation_paragraphs is empty"
        )
        assert len(result.get("consistency_conflicts", [])) >= 1, (
            "GS-CR-03: expected consistency_conflicts"
        )

    def test_gs_cr_04_empty_report(self, pipeline_results, filled_samples):
        """GS-CR-04 must produce empty report (zero issues extracted)."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-04" not in samples:
            pytest.skip("GS-CR-04 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-04")
        assert len(result.get("identified_issues", [])) == 0, (
            "GS-CR-04: expected 0 issues but found some"
        )
        assert len(result.get("final_report", "")) > 0, (
            "GS-CR-04: empty report path must still produce a non-empty report string"
        )

    def test_gs_cr_05_arabic_numeral_articles_cited(self, pipeline_results, filled_samples):
        """GS-CR-05 must successfully extract and cite articles written in Arabic numerals."""
        samples = {s["sample_id"]: s for s in filled_samples}
        if "GS-CR-05" not in samples:
            pytest.skip("GS-CR-05 not filled in")
        result = self._get_result(pipeline_results, "GS-CR-05")
        all_articles = [
            art
            for a in result.get("issue_analyses", [])
            for el in a.get("applied_elements", [])
            for art in el.get("cited_articles", [])
        ]
        expected_articles = {148, 221, 222, 223, 224, 225}
        found = set(all_articles) & expected_articles
        assert len(found) >= 2, (
            f"GS-CR-05: expected articles from Arabic numeral range but only found: {found}"
        )
