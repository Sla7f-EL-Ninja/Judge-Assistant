"""
test_eval_quality.py — LLM-as-judge quality evaluation harness for Case Reasoner.

Run with:
    pytest tests/case_reasoner/test_eval_quality.py -m llm_eval -v

Requires:
    - GOOGLE_API_KEY environment variable
    - golden_set.yaml filled in by a legal expert (currently contains placeholders)
    - A running Case Reasoner pipeline
"""
import pathlib, sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import json
import os
import pathlib
import re
import sys
from typing import Any

import pytest
import yaml

from eval_config import (
    EVAL_DIMENSIONS,
    BIAS_KEYWORDS,
    DIRECTIONAL_VERBS,
    SECTION_HEADER_PATTERN,
    SECTION_VI_HEADER,
    SECTION_VIII_HEADER,
    SECTION_ORDINALS,
    CONFIDENCE_LEVEL_ARABIC,
    ARABIC_REGISTER_PROMPT,
    NEUTRALITY_PROMPT,
    FAITHFULNESS_PROMPT,
    COUNTERARGUMENT_BALANCE_PROMPT,
    RECONCILIATION_QUALITY_PROMPT,
    PIPELINE_TIMING_THRESHOLD_SECONDS,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CR_DIR = _REPO_ROOT / "CR"
for _p in [str(_REPO_ROOT), str(_CR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden_set():
    golden_path = pathlib.Path(__file__).parent / "golden_set.yaml"
    raw = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    return raw.get("samples", [])


def _is_placeholder(value):
    if not isinstance(value, str):
        return False
    return "[PLACEHOLDER" in value


def _has_placeholders(sample):
    """Recursively check if any string value in sample is a placeholder."""
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
    """Extract section content from report text."""
    pattern = re.escape(section_header)
    next_section_pattern = r"القسم\s+(?:" + "|".join(SECTION_ORDINALS) + r")"
    match = re.search(pattern, report_text)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(next_section_pattern, report_text[start:])
    if next_match:
        return report_text[start : start + next_match.start()].strip()
    return report_text[start:].strip()


def _count_sections(report_text: str) -> int:
    return len(re.findall(SECTION_HEADER_PATTERN, report_text))


def _call_llm_judge(prompt: str) -> dict:
    """Call the real LLM for judge evaluation. Returns parsed JSON dict."""
    from config import get_llm
    llm = get_llm("high")
    response = llm.invoke(prompt)
    content = response.content.strip()
    content = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_llm_available():
    return bool(os.environ.get("GOOGLE_API_KEY"))


@pytest.fixture(scope="session")
def golden_samples():
    return _load_golden_set()


@pytest.fixture(scope="session")
def filled_samples(golden_samples):
    """Only samples with no placeholders are used in eval."""
    return [s for s in golden_samples if not _has_placeholders(s)]


@pytest.fixture(scope="session")
def pipeline_results(filled_samples, real_llm_available):
    """Run the full pipeline on each filled golden sample. Cached session-wide."""
    if not real_llm_available:
        pytest.skip("GOOGLE_API_KEY not set — skipping eval tests")
    if not filled_samples:
        pytest.skip("No filled golden samples available — fill golden_set.yaml first")

    from graph import build_case_reasoner_graph
    graph = build_case_reasoner_graph()

    results = {}
    for sample in filled_samples:
        inp = sample["input"]
        initial_state = {
            "case_id": inp["case_id"],
            "judge_query": inp["judge_query"],
            "case_brief": inp["case_brief"],
            "rendered_brief": inp.get("rendered_brief", ""),
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
        results[sample["sample_id"]] = graph.invoke(initial_state)

    return results


# ---------------------------------------------------------------------------
# Rule-Based Metrics
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
class TestBranchCoverageRate:
    """CR-EV-01: All extracted issues must complete with validation_passed=True."""

    def test_branch_coverage_100_percent(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            issues = result.get("identified_issues", [])
            analyses = result.get("issue_analyses", [])

            if not issues:
                continue  # empty report path — covered by CR-EV-05

            passed = sum(1 for a in analyses if a.get("validation_passed", False))
            rate = passed / len(issues) * 100 if issues else 100
            assert rate == 100, (
                f"{sid}: branch coverage {rate:.0f}% < 100% — "
                f"{len(issues) - passed} issue(s) failed validation"
            )


@pytest.mark.llm_eval
class TestCitationPresenceRate:
    """CR-EV-02: >=80% of applied elements must cite at least one article."""

    def test_citation_rate_above_threshold(self, pipeline_results, filled_samples):
        dim = EVAL_DIMENSIONS["CR-EV-02"]
        for sample in filled_samples:
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            all_applied = [
                el
                for a in result.get("issue_analyses", [])
                for el in a.get("applied_elements", [])
            ]
            if not all_applied:
                continue
            cited = sum(1 for el in all_applied if el.get("cited_articles"))
            rate = cited / len(all_applied) * 100
            assert rate >= dim["pass_threshold"], (
                f"{sid}: citation rate {rate:.1f}% < {dim['pass_threshold']}%"
            )


@pytest.mark.llm_eval
class TestConfidenceSignalAccuracy:
    """CR-EV-04: Computed confidence level matches golden set expected level."""

    def test_confidence_level_matches_expected(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if "expected_confidence_level" not in expected:
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            actual_level = result.get("case_level_confidence", {}).get("level", "")
            expected_level = expected["expected_confidence_level"]
            assert actual_level == expected_level, (
                f"{sid}: confidence level '{actual_level}' != expected '{expected_level}'"
            )


@pytest.mark.llm_eval
class TestEmptyReportAccuracy:
    """CR-EV-05: Empty report IFF expected issue_count == 0."""

    def test_empty_report_when_expected(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            issues = result.get("identified_issues", [])
            report = result.get("final_report", "")

            if expected.get("empty_report", False):
                assert len(issues) == 0, f"{sid}: expected empty report but {len(issues)} issues found"
                assert len(report) > 0, f"{sid}: empty report path must still produce a report string"
            else:
                expected_count = expected.get("issue_count", 0)
                if expected_count > 0:
                    assert len(issues) > 0, f"{sid}: expected {expected_count} issues but got empty report"


@pytest.mark.llm_eval
class TestReconciliationTriggerAccuracy:
    """CR-EV-06: Reconciliation fires IFF has_cross_issue_conflicts=True."""

    def test_reconciliation_fires_iff_conflicts_expected(self, pipeline_results, filled_samples):
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if "has_cross_issue_conflicts" not in expected:
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            reconciliation = result.get("reconciliation_paragraphs", [])
            has_conflicts_expected = expected["has_cross_issue_conflicts"]

            if has_conflicts_expected:
                assert len(reconciliation) > 0, (
                    f"{sid}: expected reconciliation but reconciliation_paragraphs is empty"
                )
            else:
                assert len(reconciliation) == 0, (
                    f"{sid}: expected no reconciliation but found {len(reconciliation)} paragraphs"
                )


@pytest.mark.llm_eval
class TestSectionCompleteness:
    """CR-EV-07: Expected number of sections present in final report."""

    def test_section_count_matches_expected(self, pipeline_results, filled_samples):
        dim = EVAL_DIMENSIONS["CR-EV-07"]
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if "report_sections_present" not in expected:
                continue
            if expected.get("empty_report", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            actual_count = _count_sections(report)
            expected_count = expected["report_sections_present"]

            assert actual_count >= dim["pass_threshold"], (
                f"{sid}: only {actual_count} sections found, expected {expected_count} "
                f"(min threshold: {dim['pass_threshold']})"
            )


# ---------------------------------------------------------------------------
# Rule-based Section VI neutrality pre-check (fast, no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
class TestSectionVINeutralityRuleBased:
    """Pre-check: Section VI must not contain any BIAS_KEYWORDS or DIRECTIONAL_VERBS."""

    def test_section_vi_no_bias_keywords(self, pipeline_results, filled_samples):
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

            violations = []
            for keyword in BIAS_KEYWORDS + DIRECTIONAL_VERBS:
                if keyword in section_vi:
                    violations.append(keyword)

            assert not violations, (
                f"{sid}: Section VI contains directional language: {violations}"
            )


# ---------------------------------------------------------------------------
# LLM-as-Judge Metrics (require real LLM)
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
class TestArabicRegisterQuality:
    """CR-EV-08: Formal legal Arabic throughout (LLM judge, max_score=10, pass=7)."""

    def test_arabic_register_score_passes(self, pipeline_results, filled_samples, real_llm_available):
        if not real_llm_available:
            pytest.skip("GOOGLE_API_KEY not set")
        dim = EVAL_DIMENSIONS["CR-EV-08"]
        for sample in filled_samples:
            if sample.get("llm_judge_expectations", {}).get("arabic_register") == "n/a":
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")

            prompt = ARABIC_REGISTER_PROMPT.format(report_text=report[:3000])
            scores = _call_llm_judge(prompt)

            assert scores["total"] >= dim["pass_threshold"], (
                f"{sid}: Arabic register score {scores['total']} < {dim['pass_threshold']}. "
                f"Feedback: {scores.get('feedback', '')}"
            )


@pytest.mark.llm_eval
class TestNeutralityLLMJudge:
    """CR-EV-09: Section VI must be fully neutral per LLM judge."""

    def test_section_vi_neutral(self, pipeline_results, filled_samples, real_llm_available):
        if not real_llm_available:
            pytest.skip("GOOGLE_API_KEY not set")
        for sample in filled_samples:
            if sample.get("llm_judge_expectations", {}).get("neutrality") == "n/a":
                continue
            if not sample.get("expected", {}).get("section_vi_bias_free", True):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")

            prompt = NEUTRALITY_PROMPT.format(report_text=report[:4000])
            judgment = _call_llm_judge(prompt)

            assert judgment["neutral"] is True, (
                f"{sid}: Section VI neutrality failed. "
                f"Violations: {judgment.get('violations', [])}. "
                f"Severity: {judgment.get('severity', '')}. "
                f"Feedback: {judgment.get('feedback', '')}"
            )


@pytest.mark.llm_eval
class TestFactualFaithfulness:
    """CR-EV-10: All reasoning traceable to retrieved facts (LLM judge, max=15, pass=11)."""

    def test_faithfulness_score_passes(self, pipeline_results, filled_samples, real_llm_available):
        if not real_llm_available:
            pytest.skip("GOOGLE_API_KEY not set")
        dim = EVAL_DIMENSIONS["CR-EV-10"]
        for sample in filled_samples:
            if sample.get("llm_judge_expectations", {}).get("faithfulness") == "n/a":
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")
            facts_summary = "\n".join(
                a.get("retrieved_facts", "")[:500]
                for a in result.get("issue_analyses", [])
            )[:2000]

            prompt = FAITHFULNESS_PROMPT.format(
                report_text=report[:3000],
                retrieved_facts_summary=facts_summary,
            )
            scores = _call_llm_judge(prompt)

            assert scores["total"] >= dim["pass_threshold"], (
                f"{sid}: faithfulness score {scores['total']} < {dim['pass_threshold']}. "
                f"Hallucinated: {scores.get('hallucinated_items', [])}. "
                f"Feedback: {scores.get('feedback', '')}"
            )


@pytest.mark.llm_eval
class TestCounterargumentBalance:
    """CR-EV-11: Both parties' arguments present and substantive (LLM judge, max=10, pass=7)."""

    def test_counterargument_balance_passes(self, pipeline_results, filled_samples, real_llm_available):
        if not real_llm_available:
            pytest.skip("GOOGLE_API_KEY not set")
        dim = EVAL_DIMENSIONS["CR-EV-11"]
        for sample in filled_samples:
            judge_exp = sample.get("llm_judge_expectations", {})
            if judge_exp.get("counterargument_balance") == "n/a":
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            report = result.get("final_report", "")

            prompt = COUNTERARGUMENT_BALANCE_PROMPT.format(report_text=report[:4000])
            scores = _call_llm_judge(prompt)

            assert scores["total"] >= dim["pass_threshold"], (
                f"{sid}: counterargument balance score {scores['total']} < {dim['pass_threshold']}. "
                f"Feedback: {scores.get('feedback', '')}"
            )


@pytest.mark.llm_eval
class TestReconciliationQuality:
    """CR-EV-12: Reconciliation section neutral and clear (LLM judge, max=10, pass=7)."""

    def test_reconciliation_quality_passes(self, pipeline_results, filled_samples, real_llm_available):
        if not real_llm_available:
            pytest.skip("GOOGLE_API_KEY not set")
        dim = EVAL_DIMENSIONS["CR-EV-12"]
        for sample in filled_samples:
            expected = sample.get("expected", {})
            if not expected.get("has_cross_issue_conflicts", False):
                continue
            sid = sample["sample_id"]
            result = pipeline_results[sid]
            reconciliation = result.get("reconciliation_paragraphs", [])
            if not reconciliation:
                pytest.fail(f"{sid}: expected reconciliation paragraphs but none found")

            reconciliation_text = "\n\n".join(reconciliation)
            prompt = RECONCILIATION_QUALITY_PROMPT.format(reconciliation_text=reconciliation_text)
            scores = _call_llm_judge(prompt)

            assert scores["total"] >= dim["pass_threshold"], (
                f"{sid}: reconciliation quality {scores['total']} < {dim['pass_threshold']}. "
                f"Feedback: {scores.get('feedback', '')}"
            )


# ---------------------------------------------------------------------------
# Golden set integrity checks (always run — no LLM needed)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGoldenSetIntegrity:
    """Verify golden_set.yaml structure is valid (before expert fills it in)."""

    def test_golden_set_has_samples(self):
        samples = _load_golden_set()
        assert len(samples) >= 5, "Expected at least 5 golden samples"

    def test_all_samples_have_sample_id(self):
        for sample in _load_golden_set():
            assert "sample_id" in sample, f"Missing sample_id in: {sample}"

    def test_all_samples_have_input(self):
        for sample in _load_golden_set():
            assert "input" in sample, f"{sample['sample_id']}: missing 'input'"

    def test_all_samples_have_expected(self):
        for sample in _load_golden_set():
            assert "expected" in sample, f"{sample['sample_id']}: missing 'expected'"

    def test_all_samples_have_llm_judge_expectations(self):
        for sample in _load_golden_set():
            assert "llm_judge_expectations" in sample, (
                f"{sample['sample_id']}: missing 'llm_judge_expectations'"
            )

    def test_sample_ids_are_unique(self):
        samples = _load_golden_set()
        ids = [s["sample_id"] for s in samples]
        assert len(ids) == len(set(ids)), f"Duplicate sample_ids: {ids}"

    def test_expected_confidence_levels_are_valid(self):
        valid_levels = {"high", "medium", "low"}
        for sample in _load_golden_set():
            level = sample.get("expected", {}).get("expected_confidence_level")
            if level is not None and not _is_placeholder(level):
                assert level in valid_levels, (
                    f"{sample['sample_id']}: invalid confidence level '{level}'"
                )

    def test_gs_cr_04_has_empty_report_flag(self):
        samples = {s["sample_id"]: s for s in _load_golden_set()}
        assert "GS-CR-04" in samples
        assert samples["GS-CR-04"]["expected"].get("empty_report") is True

    def test_gs_cr_03_has_conflict_flags(self):
        samples = {s["sample_id"]: s for s in _load_golden_set()}
        assert "GS-CR-03" in samples
        exp = samples["GS-CR-03"]["expected"]
        assert exp.get("has_cross_issue_conflicts") is True
        assert exp.get("has_shared_articles") is True
        assert exp.get("expected_reconciliation_count", 0) >= 1


# ---------------------------------------------------------------------------
# Eval dimension config integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvalConfigIntegrity:
    """Verify eval_config.py structure is complete."""

    def test_all_dimensions_present(self):
        expected_ids = {f"CR-EV-{i:02d}" for i in range(1, 13)}
        assert set(EVAL_DIMENSIONS.keys()) == expected_ids

    def test_all_dimensions_have_required_keys(self):
        required = {"name", "description", "max_score", "pass_threshold", "requires_llm"}
        for dim_id, dim in EVAL_DIMENSIONS.items():
            assert required.issubset(dim.keys()), (
                f"{dim_id}: missing keys {required - dim.keys()}"
            )

    def test_llm_dimensions_flagged_correctly(self):
        llm_dims = {"CR-EV-08", "CR-EV-09", "CR-EV-10", "CR-EV-11", "CR-EV-12"}
        for dim_id, dim in EVAL_DIMENSIONS.items():
            if dim_id in llm_dims:
                assert dim["requires_llm"] is True, f"{dim_id} should require LLM"
            else:
                assert dim["requires_llm"] is False, f"{dim_id} should not require LLM"

    def test_bias_keyword_list_nonempty(self):
        assert len(BIAS_KEYWORDS) >= 10
        assert len(DIRECTIONAL_VERBS) >= 5

    def test_all_prompts_are_strings(self):
        prompts = [
            ARABIC_REGISTER_PROMPT,
            NEUTRALITY_PROMPT,
            FAITHFULNESS_PROMPT,
            COUNTERARGUMENT_BALANCE_PROMPT,
            RECONCILIATION_QUALITY_PROMPT,
        ]
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 100

    def test_prompts_request_json_output(self):
        prompts = [
            ARABIC_REGISTER_PROMPT,
            NEUTRALITY_PROMPT,
            FAITHFULNESS_PROMPT,
            COUNTERARGUMENT_BALANCE_PROMPT,
            RECONCILIATION_QUALITY_PROMPT,
        ]
        for prompt in prompts:
            assert "JSON" in prompt, f"Prompt does not request JSON output: {prompt[:80]}..."
