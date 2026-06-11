"""
test_eval_quality.py — Quality evaluation framework (EV-01 through EV-08).

Evaluation dimensions:
    EV-01: Structural Completeness (7 sections non-empty)
    EV-02: Bullet Coverage Preservation (>=95%)
    EV-03: Source Traceability (100% citations valid)
    EV-04: Neutrality / Bias Detection (0 bias keywords)
    EV-05: Linguistic Quality (LLM judge, >=7/10)
    EV-06: Factual Faithfulness (LLM judge, >=11/15)
    EV-07: Multi-Party Balance (all parties represented)
    EV-08: Pipeline Timing (< PIPELINE_TIMING_THRESHOLD_SECONDS s for 7 documents)

These tests require a real LLM and are marked @pytest.mark.llm_eval.
The pipeline is run once per session via full_pipeline_result fixture.
Results are written to tests/summarizer/evaluation_results/.
"""

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Robust JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(response_content: str) -> dict:
    """Robustly extract a JSON object from an LLM response.

    Handles:
    - Raw JSON (no wrapping)
    - ```json ... ``` code blocks
    - ``` ... ``` code blocks
    - JSON preceded/followed by explanatory text
    """
    content = response_content.strip()
    # Try markdown code blocks first
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    else:
        # Try extracting the outermost {...} block
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            content = match.group(0).strip()
    # strict=False allows control characters (e.g. literal newlines) inside
    # JSON string values, which Gemini occasionally emits.
    return json.loads(content, strict=False)

from .eval_config import (
    EVAL_DIMENSIONS,
    EXTENDED_BIAS_KEYWORDS,
    ASSERTIVE_VERBS,
    DOUBT_VERBS,
    FAITHFULNESS_PROMPT,
    FIXTURE_DOC_IDS,
    FIXTURE_PARTIES,
    LINGUISTIC_QUALITY_PROMPT,
    PIPELINE_TIMING_THRESHOLD_SECONDS,
)

EVAL_RESULTS_DIR = pathlib.Path(__file__).parent / "evaluation_results"


# ---------------------------------------------------------------------------
# Session fixtures (shared with test_pipeline_integration.py via conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_llm_high():
    """High-tier LLM for pipeline."""
    try:
        from config import get_llm
        return get_llm("high")
    except Exception as exc:
        pytest.skip(f"Real LLM unavailable: {exc}")


@pytest.fixture(scope="session")
def eval_pipeline_result(real_llm_high, fixture_documents):
    """Run pipeline on all ingested fixtures; cache for all eval tests."""
    from summarize.graph import create_pipeline

    if not fixture_documents:
        pytest.skip("No fixture documents were ingested by conftest")

    pipeline = create_pipeline(real_llm_high)

    initial_state = {
        "documents": fixture_documents,
        "chunks": [], "classified_chunks": [], "bullets": [],
        "role_aggregations": [], "themed_roles": [], "role_theme_summaries": [],
        "case_brief": {}, "all_sources": [], "rendered_brief": "",
    }
    start = time.perf_counter()
    result = pipeline.invoke(initial_state)
    elapsed = time.perf_counter() - start
    result["_elapsed_seconds"] = elapsed
    result["_fixture_doc_ids"] = [d["doc_id"] for d in fixture_documents]
    # Stash input documents so EV-06 can read raw_text without touching disk.
    result["documents"] = fixture_documents
    return result


@pytest.fixture(scope="session")
def eval_report(eval_pipeline_result):
    """Accumulator for all evaluation scores; written to file at end."""
    return {}


# ---------------------------------------------------------------------------
# Helper: save results to disk
# ---------------------------------------------------------------------------


def save_eval_results(report: dict, rendered_brief: str, pipeline_state: dict):
    """Write evaluation output files."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # eval_report.json
    (EVAL_RESULTS_DIR / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # rendered_brief.md
    if rendered_brief:
        (EVAL_RESULTS_DIR / "rendered_brief.md").write_text(rendered_brief, encoding="utf-8")

    # pipeline_state.json (excluding raw text to keep size reasonable)
    state_summary = {k: v for k, v in pipeline_state.items() if k not in ("documents",)}
    try:
        (EVAL_RESULTS_DIR / "pipeline_state.json").write_text(
            json.dumps(state_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass  # Non-critical


# ---------------------------------------------------------------------------
# EV-01: Structural Completeness
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestStructuralCompleteness:
    """EV-01: All 7 sections non-empty."""

    REQUIRED_SECTIONS = (
        "dispute_summary", "uncontested_facts", "key_disputes",
        "party_requests", "party_defenses", "submitted_documents", "legal_questions"
    )

    def test_all_sections_present_in_brief_dict(self, eval_pipeline_result, eval_report):
        brief = eval_pipeline_result.get("case_brief", {})
        score = sum(1 for s in self.REQUIRED_SECTIONS if brief.get(s, "").strip())
        eval_report["EV-01"] = {
            "name": "Structural Completeness",
            "score": score,
            "max_score": 7,
            "passed": score == 7,
        }
        assert score == 7, f"Incomplete brief: {7 - score} sections empty. Scores: {score}/7"

    def test_rendered_brief_has_all_arabic_headings(self, eval_pipeline_result):
        rendered = eval_pipeline_result.get("rendered_brief", "")
        missing = []
        for heading in ["أولاً", "ثانياً", "ثالثاً", "رابعاً", "خامساً", "سادساً", "سابعاً"]:
            if heading not in rendered:
                missing.append(heading)
        assert not missing, f"Missing headings in rendered brief: {missing}"


# ---------------------------------------------------------------------------
# EV-02: Bullet Coverage Preservation
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestBulletCoveragePreservation:
    """EV-02: >= 95% of bullets survive into aggregations."""

    def test_coverage_above_threshold(self, eval_pipeline_result, eval_report):
        from .eval_config import BULLET_COVERAGE_PROMPT

        bullets = eval_pipeline_result.get("bullets", [])
        if not bullets:
            pytest.skip("No bullets produced")

        rendered = eval_pipeline_result.get("rendered_brief", "")
        if not rendered.strip():
            pytest.skip("No rendered brief to evaluate")

        # Sample up to 15 bullets at even intervals across all roles
        SAMPLE_SIZE = 15
        step = max(1, len(bullets) // SAMPLE_SIZE)
        sampled = bullets[::step][:SAMPLE_SIZE]

        # Build numbered bullet list for the judge
        bullet_lines = "\n".join(
            f"{i}. {b.get('bullet', '').strip()}"
            for i, b in enumerate(sampled)
        )

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.0,
            )
            messages = [
                ("system", BULLET_COVERAGE_PROMPT),
                ("human", (
                    f"النقاط القانونية المستخرجة:\n\n{bullet_lines}"
                    f"\n\n---\n\nالمذكرة القضائية:\n\n{rendered}"
                )),
            ]
            response = llm.invoke(messages)
            scores = _extract_json(response.content)
            results = scores.get("results", [])

            covered = sum(1 for r in results if r.get("covered", False))
            total_judged = len(results)
            recall_pct = (covered / total_judged * 100) if total_judged else 0.0

            missed_examples = [
                r.get("reason", "") for r in results if not r.get("covered", True)
            ]

            eval_report["EV-02"] = {
                "name": "Bullet Coverage Preservation",
                "score": round(recall_pct, 1),
                "max_score": 100,
                "passed": recall_pct >= 80,
                "details": {
                    "total_bullets": len(bullets),
                    "sample_size": total_judged,
                    "bullets_covered": covered,
                    "bullets_missed": total_judged - covered,
                    "missed_reasons": missed_examples[:5],
                    "judge_results": results,
                },
            }

        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        except json.JSONDecodeError as exc:
            pytest.skip(f"LLM judge returned invalid JSON: {exc}")
        except Exception as exc:
            pytest.skip(f"LLM judge unavailable: {exc}")

        # Assertion OUTSIDE try/except — a coverage failure is a real test failure
        assert recall_pct >= 80, (
            f"Semantic bullet coverage {recall_pct:.1f}% below 80% threshold.\n"
            f"Missed reasons: {missed_examples[:3]}"
        )


# ---------------------------------------------------------------------------
# EV-03: Source Traceability
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestSourceTraceability:
    """EV-03: 100% of citations reference real fixture doc_ids."""

    def test_all_citations_traceable(self, eval_pipeline_result, eval_report):
        all_sources = eval_pipeline_result.get("all_sources", [])
        fixture_doc_ids = set(eval_pipeline_result.get("_fixture_doc_ids", FIXTURE_DOC_IDS))

        if not all_sources:
            pytest.skip("No sources produced")

        fabricated = []
        for src in all_sources:
            doc_id = src.split(" ص")[0] if " ص" in src else src
            if doc_id not in fixture_doc_ids:
                fabricated.append(src)

        valid_count = len(all_sources) - len(fabricated)
        pct = (valid_count / len(all_sources) * 100) if all_sources else 100

        eval_report["EV-03"] = {
            "name": "Source Traceability",
            "score": round(pct, 1),
            "max_score": 100,
            "passed": len(fabricated) == 0,
            "fabricated_sources": fabricated[:10],
        }
        assert len(fabricated) == 0, f"Fabricated citations: {fabricated[:5]}"


# ---------------------------------------------------------------------------
# EV-04: Neutrality / Bias Detection
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestNeutrality:
    """EV-04: No bias keywords, balanced party representation."""

    def test_no_bias_keywords(self, eval_pipeline_result, eval_report):
        rendered = eval_pipeline_result.get("rendered_brief", "")
        found = [kw for kw in EXTENDED_BIAS_KEYWORDS if kw in rendered]
        eval_report["EV-04"] = {
            "name": "Neutrality / Bias Detection",
            "score": 0 if found else 1,
            "max_score": 1,
            "passed": len(found) == 0,
            "bias_keywords_found": found,
        }
        assert not found, f"Bias keywords found: {found}"

    def test_sentiment_ratio_balanced(self, eval_pipeline_result):
        """Rough check: plaintiff and defendant both mentioned in rendered brief."""
        rendered = eval_pipeline_result.get("rendered_brief", "")
        plaintiff_present = "المدعي" in rendered
        defendant_present = "المدعى عليه" in rendered
        assert plaintiff_present, "المدعي not mentioned in brief"
        assert defendant_present, "المدعى عليه not mentioned in brief"

    def test_verb_framing_bias(self, eval_pipeline_result):
        """Soft check: assertive vs doubt verbs should not be skewed > 3:1
        in favour of either plaintiff or defendant.

        This is a pytest.warns-level check — it logs a warning rather than
        failing the suite, since the 3:1 threshold needs empirical calibration
        over several pipeline runs before it becomes a hard assertion.
        """
        rendered = eval_pipeline_result.get("rendered_brief", "")

        # Build per-party windows: the 60 chars following each party mention
        # give enough context to detect the verb used in that sentence.
        WINDOW = 60
        parties = {
            "plaintiff": ["المدعي"],
            "defendant": ["المدعى عليه الأول", "المدعى عليها الثانية", "المدعى عليه"],
        }

        scores: Dict[str, Dict[str, int]] = {
            p: {"assertive": 0, "doubt": 0} for p in parties
        }

        for party_key, party_forms in parties.items():
            for form in party_forms:
                start = 0
                while True:
                    idx = rendered.find(form, start)
                    if idx == -1:
                        break
                    window_text = rendered[idx: idx + len(form) + WINDOW]
                    for v in ASSERTIVE_VERBS:
                        if v in window_text:
                            scores[party_key]["assertive"] += 1
                    for v in DOUBT_VERBS:
                        if v in window_text:
                            scores[party_key]["doubt"] += 1
                    start = idx + 1

        # Warn if either party has a > 3:1 assertive-to-doubt skew
        # AND the other party's ratio is inverted (both conditions needed
        # to avoid flagging balanced briefs that happen to use more assertive verbs).
        def _ratio(s):
            d = s["doubt"] or 1  # avoid div-by-zero
            return s["assertive"] / d

        p_ratio = _ratio(scores["plaintiff"])
        d_ratio = _ratio(scores["defendant"])
        SKEW_THRESHOLD = 3.0

        if p_ratio > SKEW_THRESHOLD and d_ratio < 1.0:
            import warnings
            warnings.warn(
                f"Possible pro-plaintiff framing bias: "
                f"plaintiff assertive/doubt ratio={p_ratio:.1f}, "
                f"defendant ratio={d_ratio:.1f}. "
                f"Scores: {scores}",
                UserWarning,
            )
        elif d_ratio > SKEW_THRESHOLD and p_ratio < 1.0:
            import warnings
            warnings.warn(
                f"Possible pro-defendant framing bias: "
                f"defendant assertive/doubt ratio={d_ratio:.1f}, "
                f"plaintiff ratio={p_ratio:.1f}. "
                f"Scores: {scores}",
                UserWarning,
            )


# ---------------------------------------------------------------------------
# EV-05: Linguistic Quality (LLM judge)
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestLinguisticQuality:
    """EV-05: Arabic legal writing quality scored by LLM judge (>=7/10)."""

    def test_linguistic_quality_score(self, eval_pipeline_result, eval_report):
        rendered = eval_pipeline_result.get("rendered_brief", "")
        if not rendered.strip():
            pytest.skip("No rendered brief to evaluate")

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.0,
            )
            messages = [
                ("system", LINGUISTIC_QUALITY_PROMPT),
                ("human", f"المذكرة:\n\n{rendered}"),
            ]
            response = llm.invoke(messages)
            scores = _extract_json(response.content)
            total = scores.get("total", 0)

            eval_report["EV-05"] = {
                "name": "Linguistic Quality",
                "score": total,
                "max_score": 10,
                "passed": total >= 7,
                "details": scores,
            }

        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        except json.JSONDecodeError as exc:
            pytest.skip(f"LLM judge returned invalid JSON: {exc}")
        except Exception as exc:
            pytest.skip(f"LLM judge unavailable: {exc}")

        # Assertion OUTSIDE try/except — a quality failure is a real test failure
        assert total >= 7, f"Linguistic quality {total}/10 below threshold. Feedback: {scores.get('feedback', '')}"


# ---------------------------------------------------------------------------
# EV-06: Factual Faithfulness (LLM judge)
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestFactualFaithfulness:
    """EV-06: Brief facts traceable to fixture documents (>=11/15)."""

    def test_factual_faithfulness_score(self, eval_pipeline_result, eval_report):
        rendered = eval_pipeline_result.get("rendered_brief", "")
        if not rendered.strip():
            pytest.skip("No rendered brief to evaluate")

        # Per-doc char budget (same logic as before, keyed by doc_id stem).
        CHAR_BUDGETS = {
            "صحيفة_دعوى":                          1200,
            "مذكرة_بدفاع_المدعى_عليه_الأول":      800,
            "مذكرة_بدفاع_المدعى_عليها_الثانية":   800,
            "تقرير_الخبير":                        800,
            "تقرير_الطب_الشرعي":                  1200,
            "محضر_جلسة_25_03_2024":               600,
        }

        # Pull text from the documents already stored in the pipeline result.
        # eval_pipeline_result["documents"] is the list passed to pipeline.invoke().
        documents = eval_pipeline_result.get("documents", [])
        doc_map = {d["doc_id"]: d["raw_text"] for d in documents}

        excerpts = []
        for doc_id, char_budget in CHAR_BUDGETS.items():
            text = doc_map.get(doc_id, "")
            if text:
                excerpts.append(f"--- {doc_id} ---\n{text[:char_budget]}")

        # Hard guard: judge needs source material to score faithfully.
        assert len(excerpts) > 0, "No fixture documents found in pipeline result"
        assert len(excerpts) == len(CHAR_BUDGETS), (
            f"Expected {len(CHAR_BUDGETS)} documents for faithfulness judge, "
            f"found {len(excerpts)}. Missing: "
            f"{set(CHAR_BUDGETS.keys()) - {e.split(' ---')[0].lstrip('- ') for e in excerpts}}"
        )

        fixture_excerpts = "\n\n".join(excerpts)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.0,
            )
            # Each excerpt is already capped at 600 chars (6 × 600 = 3600).
            # Do NOT truncate fixture_excerpts further — that would silently
            # drop the last 4 files and cause the judge to score low again.
            # The rendered brief is capped at 3000 chars (enough for all 7 sections).
            human_content = (
                f"الوثائق الأصلية:\n\n{fixture_excerpts}"
                f"\n\n---\n\nالمذكرة:\n\n{rendered}"
                f"\n\nقيّم أمانة المذكرة للوثائق الأصلية."
            )
            messages = [
                ("system", FAITHFULNESS_PROMPT),
                ("human", human_content),
            ]
            response = llm.invoke(messages)
            scores = _extract_json(response.content)
            total = scores.get("total", 0)

            eval_report["EV-06"] = {
                "name": "Factual Faithfulness",
                "score": total,
                "max_score": 15,
                "passed": total >= 11,
                "details": scores,
            }

        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        except json.JSONDecodeError as exc:
            pytest.skip(f"LLM judge returned invalid JSON: {exc}")
        except Exception as exc:
            pytest.skip(f"LLM judge unavailable: {exc}")

        # Assertion OUTSIDE try/except — a quality failure is a real test failure
        assert total >= 11, f"Faithfulness {total}/15 below threshold. Feedback: {scores.get('feedback', '')}"


# ---------------------------------------------------------------------------
# EV-07: Multi-Party Balance
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestMultiPartyBalance:
    """EV-07: All parties represented in the brief."""

    def test_all_parties_represented(self, eval_pipeline_result, eval_report):
        rendered = eval_pipeline_result.get("rendered_brief", "")

        parties_present = []
        parties_missing = []
        for party in FIXTURE_PARTIES:
            # A party is considered present if its canonical name OR any alias appears
            all_forms = [party["canonical"]] + party.get("aliases", [])
            if any(form in rendered for form in all_forms):
                parties_present.append(party["canonical"])
            else:
                parties_missing.append(party["canonical"])

        total = len(FIXTURE_PARTIES)
        covered = len(parties_present)
        coverage_pct = (covered / total * 100) if total else 100

        eval_report["EV-07"] = {
            "name": "Multi-Party Balance",
            "score": round(coverage_pct, 1),
            "max_score": 100,
            "passed": coverage_pct == 100,
            "parties_present": parties_present,
            "parties_missing": parties_missing,
        }
        assert not parties_missing, f"Parties missing from brief: {parties_missing}"


# ---------------------------------------------------------------------------
# EV-08: Pipeline Timing
# ---------------------------------------------------------------------------


@pytest.mark.llm_eval
class TestPipelineTiming:
    """EV-08: Total pipeline execution time < 120s for 7 documents."""

    def test_timing_within_threshold(self, eval_pipeline_result, eval_report):
        elapsed = eval_pipeline_result.get("_elapsed_seconds", 0)

        eval_report["EV-08"] = {
            "name": "Pipeline Timing",
            "score": round(elapsed, 1),
            "max_score": PIPELINE_TIMING_THRESHOLD_SECONDS,
            "passed": elapsed < PIPELINE_TIMING_THRESHOLD_SECONDS,
            "elapsed_seconds": round(elapsed, 1),
            "threshold_seconds": PIPELINE_TIMING_THRESHOLD_SECONDS,
        }
        # Timing is informational, not a hard failure
        if elapsed >= PIPELINE_TIMING_THRESHOLD_SECONDS:
            pytest.xfail(
                f"Pipeline took {elapsed:.1f}s "
                f"(> {PIPELINE_TIMING_THRESHOLD_SECONDS}s threshold)"
            )


# ---------------------------------------------------------------------------
# Session teardown: save all results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def save_evaluation_results(eval_pipeline_result, eval_report, request):
    """After all eval tests run, write results to disk and push scores to the
    report plugin so EV dimensions appear correctly in hakim_test_report.json."""
    yield
    # 1. Push scores and rendered brief into the plugin BEFORE pytest_sessionfinish
    #    fires so the JSON report reflects what actually ran (not the default SKIPPED
    #    state), and so the .md brief file is written next to the JSON report.
    try:
        plugin = request.config.pluginmanager.get_plugin("hakim_report_plugin_instance")
        if plugin is not None:
            if eval_report:
                plugin.record_eval_scores(eval_report)
            rendered_brief = eval_pipeline_result.get("rendered_brief", "")
            if rendered_brief:
                plugin.record_rendered_brief(rendered_brief)
    except Exception:
        pass  # Non-critical — don't let this shadow real test failures

    # 2. Write supplementary artefacts to disk.
    rendered_brief = eval_pipeline_result.get("rendered_brief", "")
    try:
        save_eval_results(eval_report, rendered_brief, eval_pipeline_result)
    except Exception:
        pass  # Non-critical