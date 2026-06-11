"""
hakim_report_plugin.py — Pytest plugin for the Hakim Summarizer test suite.

Generates a comprehensive JSON report alongside conftest.py after every run.

Usage (place this file in the same directory as conftest.py, then add to
pytest.ini / pyproject.toml / setup.cfg):

    # pytest.ini
    [pytest]
    addopts = -p hakim_report_plugin

Or invoke once with:
    pytest -p hakim_report_plugin

The report is written to  <conftest_dir>/hakim_test_report.json
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Constants — kept in sync with eval_config.py
# ---------------------------------------------------------------------------

# The 8 evaluation dimensions tracked by test_eval_quality.py
EVAL_DIMENSIONS = {
    "EV-01": {"name": "Structural Completeness",         "max_score": 7,   "pass_threshold": 7,   "requires_llm": False},
    "EV-02": {"name": "Bullet Coverage Preservation",    "max_score": 100, "pass_threshold": 80,  "requires_llm": True},
    "EV-03": {"name": "Source Traceability",             "max_score": 100, "pass_threshold": 100, "requires_llm": False},
    "EV-04": {"name": "Neutrality / Bias Detection",     "max_score": 1,   "pass_threshold": 1,   "requires_llm": False},
    "EV-05": {"name": "Linguistic Quality",              "max_score": 10,  "pass_threshold": 7,   "requires_llm": True},
    "EV-06": {"name": "Factual Faithfulness",            "max_score": 15,  "pass_threshold": 11,  "requires_llm": True},
    "EV-07": {"name": "Multi-Party Balance",             "max_score": 100, "pass_threshold": 100, "requires_llm": False},
    "EV-08": {"name": "Pipeline Timing",                 "max_score": 200, "pass_threshold": 200, "requires_llm": False},
}

# Map pytest node-id file prefixes → test module categories
MODULE_CATEGORIES = {
    "test_node_0":             "unit/node0_document_intake",
    "test_node_1":             "unit/node1_role_classifier",
    "test_node_2":             "unit/node2_bullet_extractor",
    "test_node_3":             "unit/node3_aggregator",
    "test_node_4a":            "unit/node4a_thematic_clustering",
    "test_node_4b":            "unit/node4b_theme_synthesis",
    "test_node_5":             "unit/node5_brief_generator",
    "test_schemas":            "unit/schemas",
    "test_utils":              "unit/utils",
    "test_data_contracts":     "unit/data_contracts",
    "test_graph":              "unit/graph",
    "test_pipeline_integration": "integration/pipeline",
    "test_eval_quality":       "eval/quality_dimensions",
}

# Test ID prefixes defined in source docstrings
KNOWN_TEST_IDS = {
    "T-NODE0", "T-NODE1", "T-NODE2", "T-NODE3",
    "T-NODE4A", "T-NODE4B", "T-NODE5",
    "T-GRAPH", "T-SCHEMA", "T-UTILS", "T-CONTRACT",
    "EV-01", "EV-02", "EV-03", "EV-04",
    "EV-05", "EV-06", "EV-07", "EV-08",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _module_key(nodeid: str) -> str:
    """Extract the test file stem from a pytest nodeid."""
    filename = nodeid.split("::")[0]
    stem = Path(filename).stem
    return MODULE_CATEGORIES.get(stem, f"other/{stem}")


def _safe_duration(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is not None and end is not None:
        return round(end - start, 4)
    return None


def _extract_test_id(nodeid: str) -> Optional[str]:
    """Try to extract a structured test ID (e.g. T-NODE0-01) from the node id."""
    parts = nodeid.split("::")
    for part in reversed(parts):
        for prefix in KNOWN_TEST_IDS:
            if prefix in part.upper().replace("-", "").replace("_", ""):
                return part
    return None


def _parse_marks(item) -> List[str]:
    """Return list of mark names on a test item."""
    try:
        return [m.name for m in item.iter_markers()]
    except Exception:
        return []


def _outcome_emoji(outcome: str) -> str:
    return {"passed": "✅", "failed": "❌", "skipped": "⏭", "xfailed": "⚠️", "xpassed": "🔁", "error": "💥"}.get(outcome, "❓")


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class HakimReportPlugin:
    """Collects test results and writes hakim_test_report.json on session end."""

    def __init__(self, report_path: Path):
        self.report_path = report_path
        self._session_start: float = time.perf_counter()
        self._session_start_dt: str = datetime.now(timezone.utc).isoformat()

        # Per-test data — keyed by nodeid
        self._tests: Dict[str, dict] = {}
        self._call_times: Dict[str, float] = {}

        # Eval dimension accumulator — populated by reading eval_report fixture
        self._eval_scores: Dict[str, dict] = {}

        # Rendered brief — set by save_evaluation_results / pipeline fixtures
        self._rendered_brief: str = ""

        # Error/warning accumulator
        self._collection_errors: List[dict] = []

    # ------------------------------------------------------------------
    # Collection hooks
    # ------------------------------------------------------------------

    def pytest_collectstart(self, collector):
        pass  # reserved for future use

    def pytest_collection_modifyitems(self, session, config, items):
        """After collection, pre-register all items so skipped ones appear too."""
        for item in items:
            nid = item.nodeid
            marks = _parse_marks(item)
            self._tests[nid] = {
                "nodeid":        nid,
                "module":        _module_key(nid),
                "class":         item.cls.__name__ if item.cls else None,
                "function":      item.name,
                "marks":         marks,
                "outcome":       "pending",
                "duration_s":    None,
                "longrepr":      None,
                "skip_reason":   None,
                "xfail_reason":  None,
                "warnings":      [],
                "is_llm_eval":   "llm_eval" in marks,
                "is_integration": "summarizer_llm" in marks,
            }



    # ------------------------------------------------------------------
    # Test lifecycle hooks
    # ------------------------------------------------------------------

    def pytest_runtest_logstart(self, nodeid, location):
        self._call_times[nodeid] = time.perf_counter()

    def pytest_runtest_logreport(self, report):
        nid = report.nodeid
        if nid not in self._tests:
            # Item appeared outside collection (e.g. dynamic parametrize)
            self._tests[nid] = {
                "nodeid": nid, "module": _module_key(nid),
                "class": None, "function": nid.split("::")[-1],
                "marks": [], "outcome": "pending", "duration_s": None,
                "longrepr": None, "skip_reason": None, "xfail_reason": None,
                "warnings": [], "is_llm_eval": False, "is_integration": False,
            }

        entry = self._tests[nid]

        # Only overwrite outcome from the *call* phase (not setup/teardown)
        # unless setup/teardown themselves failed.
        if report.when == "call":
            start = self._call_times.get(nid)
            entry["duration_s"] = _safe_duration(start, time.perf_counter())

            if report.passed:
                entry["outcome"] = "passed"
            elif report.failed:
                if hasattr(report, "wasxfail"):
                    entry["outcome"] = "xpassed"
                else:
                    entry["outcome"] = "failed"
                    entry["longrepr"] = self._format_longrepr(report.longrepr)
            elif report.skipped:
                entry["outcome"] = "skipped"
                entry["skip_reason"] = self._extract_skip_reason(report)

        elif report.when in ("setup", "teardown"):
            if report.failed:
                entry["outcome"] = "error"
                entry["longrepr"] = self._format_longrepr(report.longrepr)
            elif report.skipped and entry["outcome"] == "pending":
                entry["outcome"] = "skipped"
                entry["skip_reason"] = self._extract_skip_reason(report)

        # xfail
        if hasattr(report, "wasxfail") and report.skipped:
            entry["outcome"] = "xfailed"
            entry["xfail_reason"] = report.wasxfail

        # Warnings
        if hasattr(report, "sections"):
            for title, content in report.sections:
                if "warning" in title.lower():
                    entry["warnings"].append({"section": title, "content": content})

    def pytest_warning_recorded(self, warning_message, nodeid):
        if nodeid and nodeid in self._tests:
            self._tests[nodeid]["warnings"].append({
                "category": warning_message.category.__name__,
                "message": str(warning_message.message),
                "filename": warning_message.filename,
                "lineno": warning_message.lineno,
            })

    def record_eval_scores(self, scores: dict) -> None:
        """Called by the save_evaluation_results fixture to push EV scores in."""
        self._eval_scores.update(scores)

    def record_rendered_brief(self, brief: str) -> None:
        """Store the pipeline's rendered brief; written to .md on session end."""
        if brief and brief.strip():
            self._rendered_brief = brief

    # ------------------------------------------------------------------
    # Eval fixture hook — intercept eval_report fixture content
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Session end — build and write the report
    # ------------------------------------------------------------------

    def pytest_sessionfinish(self, session, exitstatus):
        elapsed_total = round(time.perf_counter() - self._session_start, 3)
        report = self._build_report(elapsed_total, exitstatus)
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Write rendered brief as a standalone Markdown file if captured.
        if self._rendered_brief:
            brief_path = self.report_path.with_name("hakim_rendered_brief.md")
            brief_path.write_text(self._rendered_brief, encoding="utf-8")
            print(
                f"\n✅ [Hakim Report] JSON test report successfully generated at:\n"
                f"   {self.report_path.absolute()}\n"
                f"   Rendered brief → {brief_path.absolute()}\n"
            )
        else:
            print(
                f"\n✅ [Hakim Report] JSON test report successfully generated at:\n"
                f"   {self.report_path.absolute()}\n"
                f"   (No rendered brief captured — pipeline tests may not have run.)\n"
            )

    # ------------------------------------------------------------------
    # Report construction
    # ------------------------------------------------------------------

    def _build_report(self, elapsed_total: float, exitstatus: int) -> dict:
        tests = list(self._tests.values())

        # ── per-module breakdown ──────────────────────────────────────
        by_module: Dict[str, List[dict]] = defaultdict(list)
        for t in tests:
            by_module[t["module"]].append(t)

        module_summaries = {}
        for mod, mod_tests in sorted(by_module.items()):
            module_summaries[mod] = self._module_summary(mod_tests)

        # ── overall counts ────────────────────────────────────────────
        counts = self._count_outcomes(tests)

        # ── accuracy metrics ─────────────────────────────────────────
        accuracy = self._accuracy_metrics(tests)

        # ── eval quality dimensions ───────────────────────────────────
        eval_section = self._build_eval_section()

        # ── failed test details ───────────────────────────────────────
        failures = [
            {
                "nodeid":      t["nodeid"],
                "module":      t["module"],
                "class":       t["class"],
                "function":    t["function"],
                "duration_s":  t["duration_s"],
                "longrepr":    t["longrepr"],
                "marks":       t["marks"],
            }
            for t in tests if t["outcome"] in ("failed", "error")
        ]

        # ── skipped test details ──────────────────────────────────────
        skipped = [
            {
                "nodeid":      t["nodeid"],
                "module":      t["module"],
                "skip_reason": t["skip_reason"],
                "marks":       t["marks"],
            }
            for t in tests if t["outcome"] == "skipped"
        ]

        # ── slowest tests ─────────────────────────────────────────────
        slowest = sorted(
            [t for t in tests if t["duration_s"] is not None],
            key=lambda x: x["duration_s"],
            reverse=True,
        )[:10]

        return {
            "report_meta": {
                "title":           "Hakim Summarizer — Full Test & Quality Report",
                "generated_at":    self._session_start_dt,
                "report_version":  "1.0.0",
                "python_version":  sys.version,
                "platform":        platform.platform(),
                "pytest_exit_code": int(exitstatus),
                "total_duration_s": elapsed_total,
            },

            "overall_summary": {
                **counts,
                "pass_rate_pct": round(
                    counts["passed"] / counts["total"] * 100, 1
                ) if counts["total"] else 0.0,
                "collection_errors": len(self._collection_errors),
            },

            "accuracy_metrics": accuracy,

            "module_summaries": module_summaries,

            "quality_evaluation": eval_section,

            "failures": failures,

            "skipped_tests": skipped,

            "slowest_tests": [
                {
                    "nodeid":     t["nodeid"],
                    "module":     t["module"],
                    "duration_s": t["duration_s"],
                    "outcome":    t["outcome"],
                }
                for t in slowest
            ],

            "all_tests": [
                {
                    "nodeid":       t["nodeid"],
                    "module":       t["module"],
                    "class":        t["class"],
                    "function":     t["function"],
                    "outcome":      t["outcome"],
                    "outcome_icon": _outcome_emoji(t["outcome"]),
                    "duration_s":   t["duration_s"],
                    "marks":        t["marks"],
                    "skip_reason":  t["skip_reason"],
                    "xfail_reason": t["xfail_reason"],
                    "warnings":     t["warnings"],
                    "is_llm_eval":  t["is_llm_eval"],
                    "is_integration": t["is_integration"],
                    "longrepr":     t["longrepr"],
                }
                for t in tests
            ],

            "collection_errors": self._collection_errors,
        }

    # ------------------------------------------------------------------
    # Sub-builders
    # ------------------------------------------------------------------

    def _count_outcomes(self, tests: List[dict]) -> dict:
        counts: Dict[str, int] = defaultdict(int)
        for t in tests:
            counts[t["outcome"]] += 1
        return {
            "total":    len(tests),
            "passed":   counts["passed"],
            "failed":   counts["failed"],
            "skipped":  counts["skipped"],
            "xfailed":  counts["xfailed"],
            "xpassed":  counts["xpassed"],
            "error":    counts["error"],
            "pending":  counts["pending"],
        }

    def _module_summary(self, tests: List[dict]) -> dict:
        counts = self._count_outcomes(tests)
        durations = [t["duration_s"] for t in tests if t["duration_s"] is not None]
        has_failures = counts["failed"] > 0 or counts["error"] > 0
        return {
            **counts,
            "pass_rate_pct": round(
                counts["passed"] / counts["total"] * 100, 1
            ) if counts["total"] else 0.0,
            "status": "FAIL" if has_failures else ("SKIP" if counts["passed"] == 0 else "PASS"),
            "total_duration_s": round(sum(durations), 4) if durations else None,
            "avg_duration_s":   round(sum(durations) / len(durations), 4) if durations else None,
            "slowest_test": max(tests, key=lambda t: t["duration_s"] or 0)["function"] if durations else None,
            "tests": [
                {
                    "function":    t["function"],
                    "outcome":     t["outcome"],
                    "icon":        _outcome_emoji(t["outcome"]),
                    "duration_s":  t["duration_s"],
                    "skip_reason": t["skip_reason"],
                    "warnings":    len(t["warnings"]),
                }
                for t in tests
            ],
        }

    def _accuracy_metrics(self, tests: List[dict]) -> dict:
        """
        Compute accuracy metrics for each test category:
          - Unit tests:        pass rate, failure rate, skip rate
          - Integration tests: same + SKIPPED-with-reason breakdown
          - Eval tests:        pass rate + dimension-level detail from eval_scores

        All rates are computed over *collected* tests (pending = not yet run
        counts as neither pass nor fail but is reported separately).
        """
        unit_tests  = [t for t in tests if t["module"].startswith("unit/")]
        integ_tests = [t for t in tests if t["module"].startswith("integration/")]
        eval_tests  = [t for t in tests if t["module"].startswith("eval/")]

        def _rates(subset):
            n = len(subset)
            if n == 0:
                return {"total": 0, "passed": 0, "failed": 0, "skipped": 0,
                        "error": 0, "pass_rate_pct": None, "fail_rate_pct": None}
            passed  = sum(1 for t in subset if t["outcome"] == "passed")
            failed  = sum(1 for t in subset if t["outcome"] in ("failed", "error"))
            skipped = sum(1 for t in subset if t["outcome"] == "skipped")
            xfailed = sum(1 for t in subset if t["outcome"] == "xfailed")
            ran     = passed + failed  # xfailed are expected failures, not counted in denominator
            return {
                "total":         n,
                "passed":        passed,
                "failed":        failed,
                "skipped":       skipped,
                "xfailed":       xfailed,
                "ran":           ran,
                "pass_rate_pct": round(passed / ran * 100, 1) if ran else None,
                "fail_rate_pct": round(failed / ran * 100, 1) if ran else None,
                "skip_rate_pct": round(skipped / n * 100, 1),
            }

        unit_rates  = _rates(unit_tests)
        integ_rates = _rates(integ_tests)
        eval_rates  = _rates(eval_tests)

        # Per-class accuracy for unit tests (most granular)
        unit_by_class: Dict[str, List[dict]] = defaultdict(list)
        for t in unit_tests:
            cls = t["class"] or "module_level"
            unit_by_class[cls].append(t)

        class_accuracy = {}
        for cls, cls_tests in sorted(unit_by_class.items()):
            r = _rates(cls_tests)
            class_accuracy[cls] = {
                "total":         r["total"],
                "passed":        r["passed"],
                "failed":        r["failed"],
                "skipped":       r["skipped"],
                "pass_rate_pct": r["pass_rate_pct"],
                "status":        "PASS" if r["failed"] == 0 and r["passed"] > 0 else
                                 "FAIL" if r["failed"] > 0 else "SKIP",
            }

        # Integration skip reasons
        integ_skips = [
            {"nodeid": t["nodeid"], "reason": t["skip_reason"]}
            for t in integ_tests if t["outcome"] == "skipped"
        ]

        # Eval skip reasons
        eval_skips = [
            {"nodeid": t["nodeid"], "reason": t["skip_reason"]}
            for t in eval_tests if t["outcome"] == "skipped"
        ]

        return {
            "unit_tests": {
                **unit_rates,
                "note": "Pass rate over tests that actually ran (excludes skipped/pending).",
                "per_class": class_accuracy,
            },
            "integration_tests": {
                **integ_rates,
                "skipped_details": integ_skips,
                "note": "Integration tests (marked @summarizer_llm) are skipped when LLM is unavailable.",
            },
            "eval_tests": {
                **eval_rates,
                "skipped_details": eval_skips,
                "note": "Eval tests (marked @llm_eval) require a real LLM + GOOGLE_API_KEY.",
            },
        }

    def _build_eval_section(self) -> dict:
        """
        Build the quality_evaluation section.

        Merges data from:
          1. EVAL_DIMENSIONS (static config — thresholds, max scores)
          2. self._eval_scores (live scores captured from eval_report fixture)

        If eval tests were skipped the dimension shows status=SKIPPED.
        """
        dimensions = {}
        for dim_id, cfg in EVAL_DIMENSIONS.items():
            live = self._eval_scores.get(dim_id)
            if live:
                raw_score   = live.get("score")
                max_score   = live.get("max_score", cfg["max_score"])
                passed      = live.get("passed", False)
                pct         = round(raw_score / max_score * 100, 1) if (raw_score is not None and max_score) else None
                details     = live.get("details")
                extra: dict = {}
                for key in ("elapsed_seconds", "threshold_seconds",
                            "fabricated_sources", "bias_keywords_found",
                            "parties_present", "parties_missing"):
                    if key in live:
                        extra[key] = live[key]

                dimensions[dim_id] = {
                    "name":            cfg["name"],
                    "description":     self._dim_description(dim_id),
                    "requires_llm":    cfg["requires_llm"],
                    "status":          "PASS" if passed else "FAIL",
                    "raw_score":       raw_score,
                    "max_score":       max_score,
                    "score_pct":       pct,
                    "pass_threshold":  cfg["pass_threshold"],
                    "pass_threshold_pct": round(cfg["pass_threshold"] / max_score * 100, 1) if max_score else None,
                    "details":         details,
                    **extra,
                }
            else:
                # Either not run or skipped
                dimensions[dim_id] = {
                    "name":           cfg["name"],
                    "description":    self._dim_description(dim_id),
                    "requires_llm":   cfg["requires_llm"],
                    "status":         "SKIPPED",
                    "raw_score":      None,
                    "max_score":      cfg["max_score"],
                    "score_pct":      None,
                    "pass_threshold": cfg["pass_threshold"],
                    "pass_threshold_pct": round(cfg["pass_threshold"] / cfg["max_score"] * 100, 1) if cfg["max_score"] else None,
                    "details":        None,
                    "note":           "Eval tests were not executed (LLM unavailable or not run).",
                }

        # Overall eval rollup
        ran_dims  = [d for d in dimensions.values() if d["status"] != "SKIPPED"]
        pass_dims = [d for d in ran_dims if d["status"] == "PASS"]
        return {
            "summary": {
                "total_dimensions":   len(dimensions),
                "evaluated":          len(ran_dims),
                "passed":             len(pass_dims),
                "failed":             len(ran_dims) - len(pass_dims),
                "skipped":            len(dimensions) - len(ran_dims),
                "overall_pass_rate":  round(len(pass_dims) / len(ran_dims) * 100, 1) if ran_dims else None,
                "note": (
                    "Dimensions EV-05 and EV-06 require an external LLM judge (Gemini). "
                    "EV-02 (Bullet Coverage) also uses an LLM judge via GOOGLE_API_KEY."
                ),
            },
            "dimensions": dimensions,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dim_description(dim_id: str) -> str:
        descriptions = {
            "EV-01": "All 7 brief sections (dispute_summary, uncontested_facts, key_disputes, party_requests, party_defenses, submitted_documents, legal_questions) must be non-empty.",
            "EV-02": "≥80% of sampled bullets (up to 15) must be semantically represented in the rendered brief (LLM judge).",
            "EV-03": "100% of citations in all_sources must reference one of the 7 fixture doc_ids. No fabricated doc_ids allowed.",
            "EV-04": "Rendered brief must contain zero bias keywords (نوصي, يجب على المحكمة, etc.) and must mention both المدعي and المدعى عليه.",
            "EV-05": "Arabic legal register quality scored by Gemini judge on 4 sub-criteria: legal_terminology (0-3), formal_register (0-3), coherence (0-2), conciseness (0-2). Threshold: 7/10.",
            "EV-06": "Factual faithfulness scored by Gemini judge: fact_recall (0-5), fact_precision (0-5), party_attribution (0-5). Threshold: 11/15.",
            "EV-07": "All 4 fixture parties (المدعي, المدعى عليه الأول, المدعى عليها الثانية, خبير) must appear in rendered brief by canonical name or alias.",
            "EV-08": f"Total pipeline execution time for 7 documents must be < 200 seconds.",
        }
        return descriptions.get(dim_id, "")

    @staticmethod
    def _format_longrepr(longrepr) -> Optional[str]:
        if longrepr is None:
            return None
        if isinstance(longrepr, str):
            return longrepr
        try:
            return str(longrepr)
        except Exception:
            return "<unserializable longrepr>"

    @staticmethod
    def _extract_skip_reason(report) -> Optional[str]:
        if report.longrepr is None:
            return None
        try:
            if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
                return str(report.longrepr[2])
            return str(report.longrepr)
        except Exception:
            return "<unknown skip reason>"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register the plugin when loaded via -p or addopts."""
    # Find conftest.py directory so the report lands next to it
    rootdir = Path(config.rootdir)

    # Try to place the report next to the conftest.py that lives with the tests
    # Walk up from rootdir to find tests/summarizer or wherever conftest.py lives
    candidate_dirs = [
        rootdir / "tests" / "summarizer",
        rootdir / "tests",
        rootdir,
    ]
    report_dir = rootdir  # fallback
    for d in candidate_dirs:
        if (d / "conftest.py").exists():
            report_dir = d
            break

    # Allow override via --hakim-report option
    report_path_str = getattr(config.option, "hakim_report", None) if hasattr(config, "option") else None
    if report_path_str:
        report_path = Path(report_path_str)
    else:
        report_path = report_dir / "hakim_test_report.json"

    plugin = HakimReportPlugin(report_path)
    # Appended '_instance' to avoid name collision with the loaded module
    config.pluginmanager.register(plugin, "hakim_report_plugin_instance")


def pytest_addoption(parser):
    """Add --hakim-report CLI option to override output path."""
    try:
        parser.addoption(
            "--hakim-report",
            action="store",
            default=None,
            metavar="PATH",
            help="Path for the Hakim JSON test report (default: <conftest_dir>/hakim_test_report.json)",
        )
    except ValueError:
        # Option already registered (e.g. plugin loaded twice)
        pass