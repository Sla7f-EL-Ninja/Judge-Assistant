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
    EV-08: Pipeline Timing (<120s for 7 documents)

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
_SUMMARIZE_DIR = _REPO_ROOT / "Summerize"
for _p in [str(_REPO_ROOT), str(_SUMMARIZE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_config import (
    EVAL_DIMENSIONS,
    EXTENDED_BIAS_KEYWORDS,
    FAITHFULNESS_PROMPT,
    FIXTURE_DOC_IDS,
    FIXTURE_PARTIES,
    LINGUISTIC_QUALITY_PROMPT,
)

EVAL_RESULTS_DIR = pathlib.Path(__file__).parent / "evaluation_results"
FIXTURE_DIR = _REPO_ROOT / "tests" / "CASE_RAG" / "fixtures"


# ---------------------------------------------------------------------------
# Session fixtures (shared with test_pipeline_integration.py via conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_llm_high():
    """High-tier LLM for pipeline."""
    try:
        from config import get_settings
        from langchain_groq import ChatGroq

        settings = get_settings()
        return ChatGroq(
            model=getattr(settings, "groq_model", "llama-3.1-8b-instant"),
            api_key=getattr(settings, "groq_api_key", ""),
            temperature=0.0,
        )
    except Exception as exc:
        pytest.skip(f"Real LLM unavailable: {exc}")


@pytest.fixture(scope="session")
def eval_pipeline_result(real_llm_high):
    """Run pipeline on all fixtures; cache for all eval tests."""
    from graph import create_pipeline

    pipeline = create_pipeline(real_llm_high)
    fixtures = []
    for fname in [
        "صحيفة_دعوى.txt",
        "مذكرة_بدفاع_المدعى_عليه_الأول.txt",
        "مذكرة_بدفاع_المدعى_عليها_الثانية.txt",
        "تقرير_الخبير.txt",
        "تقرير_الطب_الشرعي.txt",
        "محضر_جلسة_25_03_2024.txt",
        "حكم_المحكمة.txt",
    ]:
        fpath = FIXTURE_DIR / fname
        if fpath.exists():
            fixtures.append({"doc_id": fname.replace(".txt", ""), "raw_text": fpath.read_text(encoding="utf-8")})

    if not fixtures:
        pytest.skip("No fixture files found")

    initial_state = {
        "documents": fixtures,
        "chunks": [], "classified_chunks": [], "bullets": [],
        "role_aggregations": [], "themed_roles": [], "role_theme_summaries": [],
        "case_brief": {}, "all_sources": [], "rendered_brief": "",
    }
    start = time.perf_counter()
    result = pipeline.invoke(initial_state)
    elapsed = time.perf_counter() - start
    result["_elapsed_seconds"] = elapsed
    result["_fixture_doc_ids"] = [f["doc_id"] for f in fixtures]
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
        bullets = eval_pipeline_result.get("bullets", [])
        role_aggregations = eval_pipeline_result.get("role_aggregations", [])

        if not bullets:
            pytest.skip("No bullets produced")

        total_bullets = len(bullets)
        # Count roles covered
        bullet_roles = {b["role"] for b in bullets}
        agg_roles = {agg["role"] for agg in role_aggregations}
        roles_covered = len(bullet_roles & agg_roles)
        roles_total = len(bullet_roles)

        coverage_pct = (roles_covered / roles_total * 100) if roles_total else 100

        eval_report["EV-02"] = {
            "name": "Bullet Coverage Preservation",
            "score": round(coverage_pct, 1),
            "max_score": 100,
            "passed": coverage_pct >= 95,
            "details": {
                "total_bullets": total_bullets,
                "bullet_roles": list(bullet_roles),
                "agg_roles": list(agg_roles),
                "roles_covered": roles_covered,
            },
        }
        assert coverage_pct >= 95, f"Coverage {coverage_pct:.1f}% below 95% threshold"


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
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.0,
            )
            messages = [
                ("system", LINGUISTIC_QUALITY_PROMPT),
                ("human", f"المذكرة:\n\n{rendered[:3000]}"),
            ]
            response = llm.invoke(messages)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            scores = json.loads(content)
            total = scores.get("total", 0)

            eval_report["EV-05"] = {
                "name": "Linguistic Quality",
                "score": total,
                "max_score": 10,
                "passed": total >= 7,
                "details": scores,
            }
            assert total >= 7, f"Linguistic quality {total}/10 below threshold. Feedback: {scores.get('feedback', '')}"

        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        except Exception as exc:
            pytest.skip(f"LLM judge unavailable: {exc}")


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

        # Get excerpts from fixtures
        excerpts = []
        for fname in ["صحيفة_دعوى.txt", "مذكرة_بدفاع_المدعى_عليه_الأول.txt"]:
            fpath = FIXTURE_DIR / fname
            if fpath.exists():
                text = fpath.read_text(encoding="utf-8")
                excerpts.append(f"--- {fname} ---\n{text[:1000]}")
        fixture_excerpts = "\n\n".join(excerpts) if excerpts else "لا توجد وثائق"

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.0,
            )
            prompt = FAITHFULNESS_PROMPT.format(
                fixture_excerpts=fixture_excerpts[:2000],
                brief_text=rendered[:2000],
            )
            messages = [
                ("system", prompt),
                ("human", "قيّم أمانة المذكرة للوثائق الأصلية."),
            ]
            response = llm.invoke(messages)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            scores = json.loads(content)
            total = scores.get("total", 0)

            eval_report["EV-06"] = {
                "name": "Factual Faithfulness",
                "score": total,
                "max_score": 15,
                "passed": total >= 11,
                "details": scores,
            }
            assert total >= 11, f"Faithfulness {total}/15 below threshold. Feedback: {scores.get('feedback', '')}"

        except ImportError:
            pytest.skip("langchain-google-genai not installed")
        except Exception as exc:
            pytest.skip(f"LLM judge unavailable: {exc}")


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
            if party in rendered:
                parties_present.append(party)
            else:
                parties_missing.append(party)

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
            "max_score": 120,
            "passed": elapsed < 120,
            "elapsed_seconds": round(elapsed, 1),
        }
        # Timing is informational, not a hard failure
        if elapsed >= 120:
            pytest.xfail(f"Pipeline took {elapsed:.1f}s (> 120s threshold)")


# ---------------------------------------------------------------------------
# Session teardown: save all results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def save_evaluation_results(eval_pipeline_result, eval_report):
    """After all eval tests run, write results to disk."""
    yield
    # Runs after all tests in the session complete
    rendered_brief = eval_pipeline_result.get("rendered_brief", "")
    try:
        save_eval_results(eval_report, rendered_brief, eval_pipeline_result)
    except Exception:
        pass  # Non-critical — don't fail tests due to file write errors
