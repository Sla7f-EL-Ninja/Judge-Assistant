"""
legal_rag_report_plugin.py
--------------------------
Pytest plugin for the legal_rag test suite.

Registered automatically by conftest.py — no pytest.ini entry needed.
Can also be invoked standalone:
    pytest -p legal_rag_report_plugin --legal-rag-report=/path/to/report.json

Report sections
---------------
  report_meta           Timestamp, Python version, exit code, total wall time.

  overall_summary       Aggregate counts and pass rate (skipped excluded from denominator).

  accuracy_metrics      Four-way split:
                        · unit_tests        — node/router/service code correctness
                        · integration_tests — compiled-graph path verification
                        · golden_set_tests  — RAG answer quality (live services, -m golden)
                        · e2e_tests         — live smoke tests (-m e2e)

  rag_pipeline_health   Traffic-light per pipeline stage derived from unit test outcomes.

  node_coverage         Per RAG node: pass rate + per-behavior breakdown.

  behavior_accuracy     Cross-cutting view: which behavior patterns are green vs red.

  golden_evaluation     Per-sample × per-metric matrix from golden test outcomes.
                        Handles both the preprocessor golden set (S/R/N/O/E IDs) and
                        the corpus-routing golden set (C/EV/PR/MIX/O IDs).

  graph_path_coverage   Which named LangGraph end-to-end paths were verified.

  module_summaries      Per-file pass rates and per-test breakdown.

  failures              Full tracebacks for all failed/errored tests.

  skipped_tests         All skipped tests with reasons.

  slowest_tests         Top-15 by wall-clock duration.

  all_tests             Complete flat list with all metadata attached.

  collection_errors     Any pytest collection failures.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Module → category string
# ---------------------------------------------------------------------------
MODULE_CATEGORIES: Dict[str, str] = {
    "test_routers":           "unit/routers",
    "test_fallback":          "unit/fallback",
    "test_service":           "unit/service",
    "test_corpus_router":     "unit/corpus_router",
    "test_preprocessor":      "unit/preprocessor",
    "test_graders":           "unit/graders",
    "test_refine":            "unit/refine",
    "test_generate":          "unit/generate",
    "test_scope_classifier":  "unit/scope_classifier",
    "test_retrieve":          "unit/retrieve",
    "test_textual":           "unit/textual",
    "test_graph_flow":        "integration/graph_flow",
    "test_e2e":               "e2e/live_pipeline",
    "test_golden_set":        "golden/evaluation",
}

# ---------------------------------------------------------------------------
# Test class → RAG pipeline node / layer
# ---------------------------------------------------------------------------
RAG_NODE_MAP: Dict[str, str] = {
    # Routing functions (pure logic, no I/O)
    "TestPostPreprocessorRouter":   "router/post_preprocessor",
    "TestCorpusClassifierRouter":   "router/corpus_classifier",
    "TestRuleGraderRouter":         "router/rule_grader",
    "TestLLMGraderRouter":          "router/llm_grader",
    # Fallback terminal nodes
    "TestOffTopicNode":             "node/off_topic",
    "TestCannotAnswerNode":         "node/cannot_answer",
    # Service layer
    "TestValidateQuery":            "service/validate_query",
    "TestExtractSources":           "service/extract_sources",
    "TestAskQuestion":              "service/ask_question",
    # Pipeline nodes
    "TestCorpusRouterNode":         "node/corpus_router",
    "TestPreprocessorNode":         "node/preprocessor",
    "TestRuleGraderNode":           "node/rule_grader",
    "TestLLMGraderNode":            "node/llm_grader",
    "TestRefineNode":               "node/refine",
    "TestVerifyCitations":          "node/generate/citations",
    "TestRetrievedArticleIndices":  "node/generate/article_indices",
    "TestGenerateAnswerNode":       "node/generate",
    "TestScopeClassifierNode":      "node/scope_classifier",
    "TestRetrieveNode":             "node/retrieve",
    "TestTextualNode":              "node/textual",
    # Integration
    "TestGraphFlow":                "integration/graph_flow",
}

# ---------------------------------------------------------------------------
# Behavior patterns — detected from test function names.
# First match per keyword group wins; a test may match multiple groups.
# ---------------------------------------------------------------------------
BEHAVIOR_PATTERNS: List[Tuple[List[str], str]] = [
    (["budget", "exhausted"],                                   "budget_handling"),
    (["malformed", "invalid_json", "not_valid", "not_json"],    "malformed_input"),
    (["exception", "crash", "down", "raises"],                  "error_handling"),
    (["off_topic", "offtopic", "non_arabic",
      "non_string", "empty_query",
      "empty_string", "whitespace", "short_"],                  "off_topic_and_rejection"),
    (["route", "routing", "routes_to"],                         "routing_logic"),
    (["incremented", "llm_call_count"],                         "llm_call_counting"),
    (["law_name", "corpus_config", "injected_into_prompt"],     "prompt_building"),
    (["confidence", "threshold", "borderline",
      "exactly_min", "exactly_pass", "exactly_max"],            "confidence_thresholds"),
    (["duplicate", "dedup", "deduped"],                         "deduplication"),
    (["filter", "scope_filter", "scope"],                       "filter_and_scope"),
    (["score", "scores", "routing_scores"],                     "score_tracking"),
    (["history", "query_history"],                              "state_history"),
    (["arabic", "arabic_ratio", "language"],                    "language_validation"),
    (["length", "too_short", "too_long",
      "max_length", "min_length"],                              "length_validation"),
    (["citation", "integrity", "invented", "stripped"],         "citation_integrity"),
    (["corpus_name", "highest_confidence",
      "winner", "unknown_corpus"],                              "corpus_selection"),
    (["refined", "fallback_to_original"],                       "query_refinement"),
    (["rerank", "reranked"],                                    "reranking"),
    (["semantic", "fallback_semantic", "semantic_fallback"],    "semantic_fallback"),
    (["sorted", "sort_", "order"],                              "result_ordering"),
    (["range_", "article_range", "exact_article",
      "arabic_bein"],                                           "article_lookup"),
    (["section", "chapter", "chapter_and"],                     "scope_narrowing"),
    (["classification", "classified",
      "textual_class", "analytical_class"],                     "query_classification"),
    (["defaults_to", "default_"],                               "graceful_defaults"),
    (["does_not_erase", "same_state"],                          "state_mutation"),
    (["preferred", "priority", "takes_priority",
      "used_when", "used_over"],                                "priority_logic"),
    (["singleton", "sequential"],                               "graph_lifecycle"),
    (["hyde", "expansion", "parallel"],                         "hyde_retrieval"),
    (["correct_collection", "collection_name"],                 "collection_routing"),
    (["preferred_over", "refined_query_used"],                  "query_priority"),
]

# ---------------------------------------------------------------------------
# Golden set: test function base name → metric label
# ---------------------------------------------------------------------------
GOLDEN_METRIC_MAP: Dict[str, str] = {
    "test_corpus_routing":           "corpus_routing",
    "test_classification":           "query_classification",
    "test_article_coverage":         "article_coverage",
    "test_answer_keywords":          "answer_keywords",
    "test_off_topic_rejection":      "off_topic_rejection",
    "test_answer_not_empty":         "answer_completeness",
    "test_cross_law_corpus_routing": "cross_law_routing",
    "test_golden_set_loaded":        "schema_validation",
}

# ---------------------------------------------------------------------------
# Golden set: sample-ID prefix → category label.
# Covers both the preprocessor golden set (S/R/N/O/E) and the
# corpus-routing golden set (C/EV/PR/MIX/O) so the plugin works with
# either YAML — or both in future.
# Longer prefixes are matched first (e.g. "EV" before "E", "MIX" before "M").
# ---------------------------------------------------------------------------
GOLDEN_SAMPLE_CATEGORIES: Dict[str, str] = {
    # Preprocessor golden set (currently active YAML)
    "S":   "straightforward",
    "R":   "rewrite",
    "N":   "number_conversion",
    "E":   "edge_case",
    # Corpus-routing golden set (in YAML but commented out)
    "C":   "civil_law",
    "EV":  "evidence_law",
    "PR":  "procedures_law",
    "MIX": "cross_law",
    # Shared across both sets
    "O":   "off_topic",
}

# ---------------------------------------------------------------------------
# Integration: test function name → named LangGraph path
# ---------------------------------------------------------------------------
GRAPH_PATH_MAP: Dict[str, str] = {
    "test_happy_path_analytical_produces_answer":
        "analytical_happy_path",
    "test_corpus_router_off_topic_terminates_at_off_topic_node":
        "off_topic_via_corpus_router",
    "test_preprocessor_off_topic_terminates_at_off_topic_node":
        "off_topic_via_preprocessor",
    "test_textual_query_routes_to_textual_node":
        "textual_happy_path",
    "test_rule_grader_refine_then_pass_produces_answer":
        "refine_then_pass",
    "test_max_retries_exhausted_routes_to_cannot_answer":
        "max_retries_exhausted",
    "test_llm_grader_pass_produces_answer":
        "llm_grader_borderline_path",
    "test_evidence_corpus_resolved_correctly":
        "evidence_corpus_routing",
    "test_build_graph_singleton":
        "graph_singleton",
    "test_two_sequential_invocations_have_independent_state":
        "state_isolation",
}

# ---------------------------------------------------------------------------
# Pipeline stages → the RAG nodes that belong to each stage.
# Used for the traffic-light pipeline health rollup.
# ---------------------------------------------------------------------------
PIPELINE_STAGES: Dict[str, List[str]] = {
    "input_validation":     ["service/validate_query"],
    "preprocessing":        ["node/preprocessor"],
    "corpus_routing":       ["node/corpus_router", "router/corpus_classifier"],
    "scope_classification": ["node/scope_classifier"],
    "retrieval":            ["node/retrieve", "node/textual"],
    "grading":              ["node/rule_grader",  "node/llm_grader",
                             "router/rule_grader", "router/llm_grader"],
    "query_refinement":     ["node/refine"],
    "generation":           ["node/generate",
                             "node/generate/citations",
                             "node/generate/article_indices"],
    "fallback":             ["node/off_topic", "node/cannot_answer"],
    "service_layer":        ["service/extract_sources", "service/ask_question"],
    "routing_logic":        ["router/post_preprocessor"],
}


# ===========================================================================
# Pure helpers
# ===========================================================================

def _module_key(nodeid: str) -> str:
    stem = Path(nodeid.split("::")[0]).stem
    return MODULE_CATEGORIES.get(stem, f"other/{stem}")


def _rag_node(class_name: Optional[str]) -> Optional[str]:
    return RAG_NODE_MAP.get(class_name) if class_name else None


def _behavior_tags(fn_lower: str) -> List[str]:
    """
    Match a test function name (already lower-cased) against BEHAVIOR_PATTERNS.
    Returns every matching behavior label; falls back to ["core_functionality"].
    """
    matched: List[str] = []
    for keywords, label in BEHAVIOR_PATTERNS:
        if any(kw in fn_lower for kw in keywords):
            if label not in matched:
                matched.append(label)
    return matched or ["core_functionality"]


def _golden_metric(base_fn: str) -> Optional[str]:
    return GOLDEN_METRIC_MAP.get(base_fn)


def _golden_sample_id(nodeid: str) -> Optional[str]:
    m = re.search(r"\[([^\]]+)\]$", nodeid)
    return m.group(1) if m else None


def _golden_sample_category(sample_id: str) -> str:
    if not sample_id:
        return "unknown"
    # Try longest-prefix first so "EV" beats "E", "MIX" beats "M", etc.
    for prefix in sorted(GOLDEN_SAMPLE_CATEGORIES, key=len, reverse=True):
        if sample_id.upper().startswith(prefix):
            return GOLDEN_SAMPLE_CATEGORIES[prefix]
    return "unknown"


def _graph_path(base_fn: str) -> Optional[str]:
    return GRAPH_PATH_MAP.get(base_fn)


def _safe_duration(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is not None and end is not None:
        return round(end - start, 4)
    return None


def _format_longrepr(longrepr) -> Optional[str]:
    if longrepr is None:
        return None
    if isinstance(longrepr, str):
        return longrepr
    try:
        return str(longrepr)
    except Exception:
        return "<unserializable traceback>"


def _extract_skip_reason(report) -> Optional[str]:
    if report.longrepr is None:
        return None
    try:
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            return str(report.longrepr[2])
        return str(report.longrepr)
    except Exception:
        return "<unknown skip reason>"


def _parse_marks(item) -> List[str]:
    try:
        return [m.name for m in item.iter_markers()]
    except Exception:
        return []


def _outcome_icon(outcome: str) -> str:
    return {
        "passed":  "✅",
        "failed":  "❌",
        "skipped": "⏭️",
        "xfailed": "⚠️",
        "xpassed": "🔁",
        "error":   "💥",
        "pending": "⏳",
    }.get(outcome, "❓")


def _rates(subset: List[dict]) -> dict:
    """
    Compute outcome counts and honest pass rate for a subset of tests.
    Denominator = passed + failed + error  (excludes skipped / pending).
    """
    total   = len(subset)
    passed  = sum(1 for t in subset if t["outcome"] == "passed")
    failed  = sum(1 for t in subset if t["outcome"] == "failed")
    errored = sum(1 for t in subset if t["outcome"] == "error")
    skipped = sum(1 for t in subset if t["outcome"] == "skipped")
    xfailed = sum(1 for t in subset if t["outcome"] == "xfailed")
    ran     = passed + failed + errored

    return {
        "total":          total,
        "passed":         passed,
        "failed":         failed,
        "error":          errored,
        "skipped":        skipped,
        "xfailed":        xfailed,
        "ran":            ran,
        "pass_rate_pct":  round(passed / ran * 100, 1) if ran else None,
        "fail_rate_pct":  round((failed + errored) / ran * 100, 1) if ran else None,
        "skip_rate_pct":  round(skipped / total * 100, 1) if total else None,
    }


def _node_status(r: dict) -> str:
    if r["failed"] > 0 or r["error"] > 0:
        return "FAIL"
    if r["ran"] > 0:
        return "PASS"
    if r["skipped"] > 0:
        return "SKIP"
    return "UNTESTED"


# ===========================================================================
# Plugin class
# ===========================================================================

class LegalRAGReportPlugin:
    """
    Pytest plugin that shadows every test in the legal_rag suite and emits a
    comprehensive JSON report at session end.

    Registered by conftest.py — see conftest.pytest_configure().
    """

    def __init__(self, report_path: Path) -> None:
        self.report_path       = report_path
        self._session_start    = time.perf_counter()
        self._session_start_dt = datetime.now(timezone.utc).isoformat()
        self._tests: Dict[str, dict]  = {}
        self._call_start: Dict[str, float] = {}
        self._collection_errors: List[dict] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def pytest_collection_modifyitems(self, items):
        """Pre-register every collected item so skipped tests still appear."""
        for item in items:
            self._register_item(item)

    def _register_item(self, item) -> dict:
        nid    = item.nodeid
        marks  = _parse_marks(item)
        cls    = item.cls.__name__ if item.cls else None
        fn     = item.name
        base   = fn.split("[")[0]
        fn_low = base.lower()

        entry = {
            "nodeid":           nid,
            "module":           _module_key(nid),
            "class":            cls,
            "function":         fn,
            "base_function":    base,
            "marks":            marks,
            "outcome":          "pending",
            "duration_s":       None,
            "longrepr":         None,
            "skip_reason":      None,
            "xfail_reason":     None,
            "warnings":         [],
            # Derived metadata
            "rag_node":         _rag_node(cls),
            "behaviors":        _behavior_tags(fn_low),
            "is_golden":        "golden" in marks,
            "is_e2e":           "e2e" in marks,
            "is_slow":          "slow" in marks,
            "golden_metric":    _golden_metric(base),
            "golden_sample_id": _golden_sample_id(nid),
            "graph_path":       _graph_path(base),
        }
        self._tests[nid] = entry
        return entry

    # ------------------------------------------------------------------
    # Test lifecycle
    # ------------------------------------------------------------------

    def pytest_runtest_logstart(self, nodeid, location):
        self._call_start[nodeid] = time.perf_counter()

    def pytest_runtest_logreport(self, report):
        nid = report.nodeid

        # ── Collection-phase failures (import errors, syntax errors, etc.) ──
        # In pytest 9+, pytest_collecterror was removed; these failures now
        # arrive here with when="collect" and failed=True.  The nodeid is a
        # file path (e.g. "tests/test_foo.py"), not a test-item nodeid, so
        # we record them in _collection_errors and return early rather than
        # trying to look them up in self._tests.
        if report.when == "collect" and report.failed:
            self._collection_errors.append({
                "nodeid":   nid,
                "longrepr": _format_longrepr(report.longrepr),
            })
            return

        # Dynamically created tests (late parametrize, etc.)
        if nid not in self._tests:
            parts = nid.split("::")
            cls   = parts[-2] if len(parts) >= 3 and not parts[-2].endswith(".py") else None
            mock_item = type("_Item", (), {
                "nodeid": nid, "cls": type(cls, (), {}) if cls else None,
                "name": parts[-1],
            })()
            if cls:
                mock_item.cls.__name__ = cls
            entry = self._register_item(mock_item)
            entry["marks"] = []

        entry = self._tests[nid]

        if report.when == "call":
            start = self._call_start.get(nid)
            entry["duration_s"] = _safe_duration(start, time.perf_counter())

            if hasattr(report, "wasxfail") and report.passed:
                entry["outcome"] = "xpassed"
            elif hasattr(report, "wasxfail") and report.skipped:
                entry["outcome"] = "xfailed"
                entry["xfail_reason"] = str(report.wasxfail)
            elif report.passed:
                entry["outcome"] = "passed"
            elif report.failed:
                entry["outcome"] = "failed"
                entry["longrepr"] = _format_longrepr(report.longrepr)
            elif report.skipped:
                entry["outcome"] = "skipped"
                entry["skip_reason"] = _extract_skip_reason(report)

        elif report.when in ("setup", "teardown"):
            if report.failed:
                entry["outcome"] = "error"
                entry["longrepr"] = _format_longrepr(report.longrepr)
            elif report.skipped and entry["outcome"] == "pending":
                entry["outcome"] = "skipped"
                entry["skip_reason"] = _extract_skip_reason(report)

    def pytest_warning_recorded(self, warning_message, nodeid):
        if nodeid and nodeid in self._tests:
            self._tests[nodeid]["warnings"].append({
                "category": warning_message.category.__name__,
                "message":  str(warning_message.message),
                "filename": warning_message.filename,
                "lineno":   warning_message.lineno,
            })

    # ------------------------------------------------------------------
    # Session finish → write report
    # ------------------------------------------------------------------

    def pytest_sessionfinish(self, session, exitstatus):
        elapsed = round(time.perf_counter() - self._session_start, 3)
        report  = self._build_report(elapsed, int(exitstatus))
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        total  = report["overall_summary"]["total"]
        passed = report["overall_summary"]["passed"]
        rate   = report["overall_summary"]["pass_rate_pct"]
        print(
            f"\n✅ [LegalRAG Report] {passed}/{total} passed ({rate}%) — "
            f"report written to:\n   {self.report_path.absolute()}\n"
        )

    # ------------------------------------------------------------------
    # Main report builder
    # ------------------------------------------------------------------

    def _build_report(self, elapsed: float, exitstatus: int) -> dict:
        tests = list(self._tests.values())

        unit_tests   = [t for t in tests if t["module"].startswith("unit/")]
        integ_tests  = [t for t in tests if t["module"].startswith("integration/")]
        golden_tests = [t for t in tests if t["module"].startswith("golden/")]
        e2e_tests    = [t for t in tests if t["module"].startswith("e2e/")]

        by_module: Dict[str, List[dict]] = defaultdict(list)
        for t in tests:
            by_module[t["module"]].append(t)

        counts = _rates(tests)

        return {
            "report_meta": self._meta(elapsed, exitstatus, counts),
            "overall_summary": {
                **counts,
                "collection_errors": len(self._collection_errors),
                "test_files_run":    len(by_module),
                "pass_rate_note": (
                    "pass_rate_pct denominator = passed + failed + error. "
                    "Skipped tests are excluded."
                ),
            },
            "accuracy_metrics": self._accuracy_metrics(
                unit_tests, integ_tests, golden_tests, e2e_tests
            ),
            "rag_pipeline_health": self._pipeline_health(unit_tests),
            "node_coverage":       self._node_coverage(unit_tests + integ_tests),
            "behavior_accuracy":   self._behavior_accuracy(unit_tests),
            "golden_evaluation":   self._golden_evaluation(golden_tests),
            "graph_path_coverage": self._graph_path_coverage(integ_tests),
            "module_summaries": {
                mod: self._module_summary(mod_tests)
                for mod, mod_tests in sorted(by_module.items())
            },
            "failures": [
                {
                    "nodeid":     t["nodeid"],
                    "module":     t["module"],
                    "class":      t["class"],
                    "function":   t["function"],
                    "rag_node":   t["rag_node"],
                    "behaviors":  t["behaviors"],
                    "duration_s": t["duration_s"],
                    "longrepr":   t["longrepr"],
                }
                for t in tests if t["outcome"] in ("failed", "error")
            ],
            "skipped_tests": [
                {
                    "nodeid":      t["nodeid"],
                    "module":      t["module"],
                    "skip_reason": t["skip_reason"],
                    "marks":       t["marks"],
                    "is_golden":   t["is_golden"],
                    "is_e2e":      t["is_e2e"],
                }
                for t in tests if t["outcome"] == "skipped"
            ],
            "slowest_tests": sorted(
                [
                    {
                        "nodeid":     t["nodeid"],
                        "module":     t["module"],
                        "duration_s": t["duration_s"],
                        "outcome":    t["outcome"],
                        "rag_node":   t["rag_node"],
                    }
                    for t in tests if t["duration_s"] is not None
                ],
                key=lambda x: x["duration_s"],
                reverse=True,
            )[:15],
            "all_tests": [
                {
                    "nodeid":           t["nodeid"],
                    "module":           t["module"],
                    "class":            t["class"],
                    "function":         t["function"],
                    "outcome":          t["outcome"],
                    "icon":             _outcome_icon(t["outcome"]),
                    "duration_s":       t["duration_s"],
                    "marks":            t["marks"],
                    "rag_node":         t["rag_node"],
                    "behaviors":        t["behaviors"],
                    "is_golden":        t["is_golden"],
                    "is_e2e":           t["is_e2e"],
                    "is_slow":          t["is_slow"],
                    "golden_metric":    t["golden_metric"],
                    "golden_sample_id": t["golden_sample_id"],
                    "graph_path":       t["graph_path"],
                    "skip_reason":      t["skip_reason"],
                    "xfail_reason":     t["xfail_reason"],
                    "warning_count":    len(t["warnings"]),
                    "longrepr":         t["longrepr"],
                }
                for t in tests
            ],
            "collection_errors": self._collection_errors,
        }

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _meta(self, elapsed: float, exitstatus: int, counts: dict) -> dict:
        total   = counts["total"]
        passed  = counts["passed"]
        skipped = counts["skipped"]
        failed  = counts["failed"] + counts["error"]

        return {
            "title":             "legal_rag — Comprehensive Test & Quality Report",
            "generated_at":      self._session_start_dt,
            "report_version":    "2.0.0",
            "python_version":    sys.version,
            "platform":          platform.platform(),
            "pytest_exit_code":  exitstatus,
            "total_duration_s":  elapsed,
            "test_count_summary": (
                f"{passed} passed / {failed} failed / {skipped} skipped / {total} total"
            ),
        }

    def _module_summary(self, tests: List[dict]) -> dict:
        r         = _rates(tests)
        durations = [t["duration_s"] for t in tests if t["duration_s"] is not None]
        return {
            **r,
            "status":           _node_status(r),
            "total_duration_s": round(sum(durations), 4) if durations else None,
            "avg_duration_s":   round(sum(durations) / len(durations), 4) if durations else None,
            "slowest_test":     max(tests, key=lambda t: t["duration_s"] or 0)["function"]
                                if durations else None,
            "tests": [
                {
                    "function":    t["function"],
                    "outcome":     t["outcome"],
                    "icon":        _outcome_icon(t["outcome"]),
                    "duration_s":  t["duration_s"],
                    "rag_node":    t["rag_node"],
                    "behaviors":   t["behaviors"],
                    "skip_reason": t["skip_reason"],
                    "warnings":    len(t["warnings"]),
                }
                for t in tests
            ],
        }

    def _accuracy_metrics(
        self,
        unit_tests:   List[dict],
        integ_tests:  List[dict],
        golden_tests: List[dict],
        e2e_tests:    List[dict],
    ) -> dict:
        """
        High-level accuracy table.
        · unit/integration rates → code correctness of the RAG implementation.
        · golden rate → RAG answer quality (live services required).
        · e2e rate    → live-system smoke results (live services required).
        """
        unit_r   = _rates(unit_tests)
        integ_r  = _rates(integ_tests)
        golden_r = _rates(golden_tests)
        e2e_r    = _rates(e2e_tests)

        # Per-class breakdown for unit tests so we can spot which node broke.
        by_class: Dict[str, List[dict]] = defaultdict(list)
        for t in unit_tests:
            by_class[t["class"] or "_module_level"].append(t)

        per_class_accuracy = {}
        for cls, cls_tests in sorted(by_class.items()):
            r = _rates(cls_tests)
            per_class_accuracy[cls] = {
                **r,
                "rag_node": _rag_node(cls),
                "status":   _node_status(r),
            }

        return {
            "unit_tests": {
                **unit_r,
                "status": _node_status(unit_r),
                "note": (
                    "All node/router/service functions tested with mocked LLM+Qdrant. "
                    "Pass rate = code correctness."
                ),
                "per_class": per_class_accuracy,
            },
            "integration_tests": {
                **integ_r,
                "status": _node_status(integ_r),
                "note": (
                    "Full compiled LangGraph invocations with all I/O mocked. "
                    "Validates end-to-end routing paths and state transitions."
                ),
            },
            "golden_set_tests": {
                **golden_r,
                "ran":    golden_r["ran"] > 0,
                "status": _node_status(golden_r) if golden_r["ran"] > 0 else "NOT_RUN",
                "skipped_details": [
                    {"nodeid": t["nodeid"], "reason": t["skip_reason"]}
                    for t in golden_tests if t["outcome"] == "skipped"
                ],
                "note": (
                    "RAG answer-quality evaluation against curated samples. "
                    "Requires live LLM + Qdrant (run with -m golden -m e2e). "
                    "Pass rate here = the RAG system's answer accuracy, NOT code correctness."
                ),
            },
            "e2e_tests": {
                **e2e_r,
                "ran":    e2e_r["ran"] > 0,
                "status": _node_status(e2e_r) if e2e_r["ran"] > 0 else "NOT_RUN",
                "skipped_details": [
                    {"nodeid": t["nodeid"], "reason": t["skip_reason"]}
                    for t in e2e_tests if t["outcome"] == "skipped"
                ],
                "note": (
                    "Live pipeline smoke tests. "
                    "Skipped automatically unless API keys and Qdrant are configured."
                ),
            },
        }

    def _pipeline_health(self, unit_tests: List[dict]) -> dict:
        """
        Roll up unit test outcomes by pipeline stage.
        Each stage gets PASS / FAIL / SKIP / UNTESTED based on whether any
        of its constituent node tests failed.
        """
        by_node: Dict[str, List[dict]] = defaultdict(list)
        for t in unit_tests:
            node = t.get("rag_node")
            if node:
                by_node[node].append(t)

        stages: Dict[str, dict] = {}
        for stage_name, stage_nodes in PIPELINE_STAGES.items():
            stage_tests: List[dict] = []
            for node in stage_nodes:
                stage_tests.extend(by_node.get(node, []))

            if not stage_tests:
                stages[stage_name] = {
                    "nodes":         stage_nodes,
                    "total":         0,
                    "passed":        0,
                    "failed":        0,
                    "pass_rate_pct": None,
                    "status":        "UNTESTED",
                }
                continue

            r = _rates(stage_tests)
            stages[stage_name] = {
                "nodes":         stage_nodes,
                **r,
                "status":        _node_status(r),
            }

        all_statuses   = [s["status"] for s in stages.values()]
        failing_stages = [n for n, s in stages.items() if s["status"] == "FAIL"]
        pipeline_ok    = not failing_stages

        return {
            "overall_status":  "HEALTHY" if pipeline_ok else "DEGRADED",
            "failing_stages":  failing_stages,
            "stages":          stages,
            "interpretation": (
                "HEALTHY = every tested pipeline stage has 100% unit-test pass rate. "
                "DEGRADED = ≥1 stage has failing tests — see 'failing_stages'."
            ),
        }

    def _node_coverage(self, tests: List[dict]) -> dict:
        """
        Per RAG node: total/passed/failed rates + per-behavior breakdown.
        Includes integration tests so the graph-flow node appears here too.
        """
        by_node: Dict[str, List[dict]] = defaultdict(list)
        for t in tests:
            node = t.get("rag_node")
            if node:
                by_node[node].append(t)

        coverage: Dict[str, dict] = {}
        for node, node_tests in sorted(by_node.items()):
            r = _rates(node_tests)

            # Per-behavior drill-down within this node
            by_beh: Dict[str, List[dict]] = defaultdict(list)
            for t in node_tests:
                for beh in t["behaviors"]:
                    by_beh[beh].append(t)

            beh_details: Dict[str, dict] = {}
            for beh, beh_tests in sorted(by_beh.items()):
                b = _rates(beh_tests)
                beh_details[beh] = {
                    "total":         b["total"],
                    "passed":        b["passed"],
                    "failed":        b["failed"],
                    "pass_rate_pct": b["pass_rate_pct"],
                    "status":        _node_status(b),
                }

            coverage[node] = {
                **r,
                "status":            _node_status(r),
                "behaviors_covered": sorted(beh_details.keys()),
                "behavior_details":  beh_details,
                "passing_behaviors": [b for b, d in beh_details.items() if d["status"] == "PASS"],
                "failing_behaviors": [b for b, d in beh_details.items() if d["status"] == "FAIL"],
            }

        return coverage

    def _behavior_accuracy(self, tests: List[dict]) -> dict:
        """
        Cross-cutting view: for each detected behavior pattern, how many tests
        (across ALL unit-test nodes) passed vs failed.
        Surfaces systemic weaknesses (e.g. budget_handling always fails).
        """
        by_beh: Dict[str, List[dict]] = defaultdict(list)
        for t in tests:
            for beh in t["behaviors"]:
                by_beh[beh].append(t)

        behavior_table: Dict[str, dict] = {}
        for beh, beh_tests in sorted(by_beh.items()):
            r = _rates(beh_tests)
            behavior_table[beh] = {
                **r,
                "status":         _node_status(r),
                "affected_nodes": sorted({t["rag_node"] for t in beh_tests if t["rag_node"]}),
            }

        failing = [b for b, d in behavior_table.items() if d["status"] == "FAIL"]
        passing = [b for b, d in behavior_table.items() if d["status"] == "PASS"]

        return {
            "summary": {
                "total_behaviors_detected": len(behavior_table),
                "fully_passing":            len(passing),
                "has_failures":             len(failing),
                "critical_failures":        failing,
                "interpretation": (
                    "A 'critical failure' means a behavior pattern is broken across ≥1 node. "
                    "Check 'per_behavior[<name>].affected_nodes' to see which nodes are impacted."
                ),
            },
            "per_behavior": behavior_table,
        }

    def _golden_evaluation(self, golden_tests: List[dict]) -> dict:
        """
        Reconstruct the golden-set evaluation matrix purely from test outcomes.

        Works with whatever golden_set.yaml is present:
        - Preprocessor set (S/R/N/O/E IDs): maps to straightforward/rewrite/…
        - Corpus-routing set (C/EV/PR/MIX/O IDs): maps to civil_law/evidence_law/…

        Populated only when golden tests actually ran (not skipped due to no
        live services).
        """
        ran_tests = [t for t in golden_tests if t["outcome"] in ("passed", "failed", "error")]

        if not ran_tests:
            return {
                "ran":                False,
                "total_run":          0,
                "overall_accuracy":   None,
                "per_metric":         {},
                "per_category":       {},
                "per_sample":         {},
                "note": (
                    "Golden tests were skipped (no live services). "
                    "Run with: pytest -m 'golden and e2e'  "
                    "(requires API keys + Qdrant)."
                    if any(t["outcome"] == "skipped" for t in golden_tests)
                    else "No golden tests were collected."
                ),
            }

        # ── Per-metric accuracy ───────────────────────────────────────
        by_metric: Dict[str, List[dict]] = defaultdict(list)
        for t in ran_tests:
            metric = t.get("golden_metric")
            if metric:
                by_metric[metric].append(t)

        per_metric: Dict[str, dict] = {}
        for metric, m_tests in sorted(by_metric.items()):
            passed = sum(1 for t in m_tests if t["outcome"] == "passed")
            failed = sum(1 for t in m_tests if t["outcome"] in ("failed", "error"))
            total  = len(m_tests)
            per_metric[metric] = {
                "total":          total,
                "passed":         passed,
                "failed":         failed,
                "accuracy_pct":   round(passed / total * 100, 1) if total else None,
                "status":         "PASS" if failed == 0 and passed > 0
                                  else "FAIL" if failed > 0
                                  else "SKIP",
                "failed_samples": [
                    t["golden_sample_id"] for t in m_tests
                    if t["outcome"] in ("failed", "error") and t["golden_sample_id"]
                ],
            }

        # ── Per-sample outcome matrix ─────────────────────────────────
        per_sample: Dict[str, dict] = {}
        for t in ran_tests:
            sid    = t.get("golden_sample_id")
            metric = t.get("golden_metric")
            if not sid or not metric:
                continue
            if sid not in per_sample:
                per_sample[sid] = {
                    "sample_id":  sid,
                    "category":   _golden_sample_category(sid),
                    "metrics":    {},
                    "all_passed": True,
                }
            ok = t["outcome"] == "passed"
            per_sample[sid]["metrics"][metric] = ok
            if not ok:
                per_sample[sid]["all_passed"] = False

        # ── Per-category accuracy ─────────────────────────────────────
        by_cat: Dict[str, List[dict]] = defaultdict(list)
        for s in per_sample.values():
            by_cat[s["category"]].append(s)

        per_category: Dict[str, dict] = {}
        for cat, cat_samples in sorted(by_cat.items()):
            total      = len(cat_samples)
            passed_cnt = sum(1 for s in cat_samples if s["all_passed"])
            per_category[cat] = {
                "total":              total,
                "fully_passed":       passed_cnt,
                "partially_failed":   total - passed_cnt,
                "accuracy_pct":       round(passed_cnt / total * 100, 1) if total else None,
                "sample_ids":         sorted(s["sample_id"] for s in cat_samples),
            }

        # ── Overall golden accuracy ───────────────────────────────────
        all_samples       = list(per_sample.values())
        total_samples     = len(all_samples)
        fully_passed      = sum(1 for s in all_samples if s["all_passed"])
        overall_acc       = (
            round(fully_passed / total_samples * 100, 1) if total_samples else None
        )

        # ── Metric-level summary for quick scanning ───────────────────
        worst_metrics = sorted(
            [(m, d["accuracy_pct"] or 0) for m, d in per_metric.items()],
            key=lambda x: x[1],
        )

        return {
            "ran":                      True,
            "total_run":                len(ran_tests),
            "unique_samples_evaluated": total_samples,
            "samples_fully_passed":     fully_passed,
            "samples_partially_failed": total_samples - fully_passed,
            "overall_accuracy_pct":     overall_acc,
            "worst_metrics":            [m for m, _ in worst_metrics[:3]],
            "note": (
                "overall_accuracy_pct counts samples where EVERY metric passed. "
                "A sample is 'partially failed' if even one metric check failed."
            ),
            "per_metric":   per_metric,
            "per_category": per_category,
            "per_sample": {
                sid: {
                    "category":   d["category"],
                    "all_passed": d["all_passed"],
                    "metrics":    d["metrics"],
                }
                for sid, d in sorted(per_sample.items())
            },
        }

    def _graph_path_coverage(self, integ_tests: List[dict]) -> dict:
        """
        For integration tests (TestGraphFlow), report which named end-to-end
        LangGraph paths were exercised and whether they passed.
        """
        path_results: Dict[str, dict] = {}

        for t in integ_tests:
            path = t.get("graph_path")
            if not path:
                continue
            path_results[path] = {
                "path":       path,
                "test_fn":    t["function"],
                "outcome":    t["outcome"],
                "status":     (
                    "PASS"    if t["outcome"] == "passed"
                    else "FAIL"    if t["outcome"] in ("failed", "error")
                    else "SKIP"    if t["outcome"] == "skipped"
                    else "PENDING"
                ),
                "duration_s": t["duration_s"],
            }

        # Add defined-but-not-collected paths
        for path in GRAPH_PATH_MAP.values():
            if path not in path_results:
                path_results[path] = {
                    "path":       path,
                    "test_fn":    None,
                    "outcome":    "not_collected",
                    "status":     "NOT_RUN",
                    "duration_s": None,
                }

        verified  = [p for p in path_results.values() if p["status"] == "PASS"]
        failed    = [p for p in path_results.values() if p["status"] == "FAIL"]
        total_def = len(GRAPH_PATH_MAP)

        return {
            "summary": {
                "total_paths_defined": total_def,
                "paths_verified":      len(verified),
                "paths_failed":        len(failed),
                "paths_not_run":       sum(
                    1 for p in path_results.values()
                    if p["status"] in ("NOT_RUN", "SKIP", "PENDING")
                ),
                "coverage_pct":        (
                    round(len(verified) / total_def * 100, 1) if total_def else None
                ),
            },
            "paths":             path_results,
            "failed_path_names": [p["path"] for p in failed],
        }