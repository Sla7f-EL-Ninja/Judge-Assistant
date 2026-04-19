"""
eval_harness.py
---------------
Evaluation harness for the Civil Law RAG pipeline.

Metrics:
  - retrieval_precision_at_5:  fraction of expected articles in top-5
  - citation_faithfulness:     fraction of cited article numbers that were retrieved
  - off_topic_accuracy:        fraction of off-topic queries correctly rejected
  - avg_llm_calls:             mean LLM calls per query (budget health)
  - avg_latency_ms:            mean end-to-end latency per query

Thresholds (enforced when run as pytest --eval):
  retrieval_precision_at_5  >= 0.80
  citation_faithfulness     >= 0.95
  off_topic_accuracy        = 1.00

Usage::
    pytest RAG/civil_law_rag/tests/eval_harness.py --eval
    python -m RAG.civil_law_rag.tests.eval_harness  # print report only
"""

from __future__ import annotations

import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

import sys
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from RAG.civil_law_rag.graph import build_graph
from RAG.civil_law_rag.state import make_initial_state

_GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
_THRESHOLDS = {
    "retrieval_precision_at_5": 0.80,
    "citation_faithfulness": 0.95,
    "off_topic_accuracy": 1.00,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_query(app, query: str) -> Dict[str, Any]:
    """Run a single query through the graph and return timing + result state."""
    state = make_initial_state()
    state["last_query"] = query

    t0 = time.perf_counter()
    result = app.invoke(state)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "result": result,
        "latency_ms": latency_ms,
    }


def _retrieved_indices(result_state: dict) -> List[int]:
    return [
        d.metadata.get("index")
        for d in result_state.get("last_results", [])
        if d.metadata.get("index") is not None
    ]


def _cited_indices(answer: str) -> List[int]:
    return [int(m) for m in re.findall(r"المادة\s+(\d+)", answer or "")]


def _is_terminal_failure(result_state: dict) -> bool:
    """True if the pipeline ended in off_topic or cannot_answer."""
    answer = result_state.get("final_answer", "")
    classification = result_state.get("classification", "")
    return (
        classification == "off_topic"
        or "تعذر" in answer
        or "خارج نطاق" in answer
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval() -> Dict[str, Any]:
    samples = yaml.safe_load(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))["samples"]
    app = build_graph()

    total = len(samples)
    retrieval_precisions: List[float] = []
    citation_faithfulnesses: List[float] = []
    off_topic_correct = 0
    off_topic_total   = 0
    llm_calls: List[int] = []
    latencies: List[float] = []
    failures: List[Dict] = []

    for sample in samples:
        sid     = sample["id"]
        query   = sample["query"]
        expected_articles = set(sample.get("expected_articles", []))
        should_fail       = sample.get("should_fail", False)

        run = _run_query(app, query)
        rs  = run["result"]
        latencies.append(run["latency_ms"])
        llm_calls.append(rs.get("llm_call_count", 0))

        is_failure = _is_terminal_failure(rs)

        # Off-topic accuracy
        if should_fail:
            off_topic_total += 1
            if is_failure:
                off_topic_correct += 1
            else:
                failures.append({
                    "id": sid,
                    "issue": "expected_failure_but_got_answer",
                    "answer_preview": (rs.get("final_answer") or "")[:200],
                })
            continue

        # Retrieval precision@5
        if expected_articles:
            retrieved = set(_retrieved_indices(rs))
            hit = len(expected_articles & retrieved)
            precision = hit / len(expected_articles)
            retrieval_precisions.append(precision)
            if precision < 1.0:
                failures.append({
                    "id": sid,
                    "issue": "retrieval_miss",
                    "expected": sorted(expected_articles),
                    "retrieved": sorted(retrieved),
                })

        # Citation faithfulness
        retrieved_all = set(_retrieved_indices(rs))
        cited         = set(_cited_indices(rs.get("final_answer", "")))
        if cited:
            faithful_count = len(cited & retrieved_all)
            faithfulness   = faithful_count / len(cited)
            citation_faithfulnesses.append(faithfulness)
            if faithfulness < 1.0:
                invalid = cited - retrieved_all
                failures.append({
                    "id": sid,
                    "issue": "hallucinated_citations",
                    "invalid_articles": sorted(invalid),
                })

    # Aggregate
    metrics: Dict[str, Any] = {
        "total_samples":              total,
        "retrieval_precision_at_5":   (
            sum(retrieval_precisions) / len(retrieval_precisions)
            if retrieval_precisions else None
        ),
        "citation_faithfulness":      (
            sum(citation_faithfulnesses) / len(citation_faithfulnesses)
            if citation_faithfulnesses else None
        ),
        "off_topic_accuracy":         (
            off_topic_correct / off_topic_total if off_topic_total else None
        ),
        "avg_llm_calls":              sum(llm_calls) / len(llm_calls) if llm_calls else 0,
        "avg_latency_ms":             sum(latencies) / len(latencies) if latencies else 0,
        "failures":                   failures,
    }
    return metrics


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--eval", action="store_true", default=False,
                     help="Run the RAG evaluation harness")


@pytest.fixture(scope="session")
def eval_metrics(request):
    if not request.config.getoption("--eval"):
        pytest.skip("Pass --eval to run the evaluation harness")
    return run_eval()


@pytest.mark.eval
def test_retrieval_precision(eval_metrics):
    score = eval_metrics["retrieval_precision_at_5"]
    if score is None:
        pytest.skip("No samples with expected_articles")
    assert score >= _THRESHOLDS["retrieval_precision_at_5"], (
        f"Retrieval precision@5 = {score:.3f} < threshold {_THRESHOLDS['retrieval_precision_at_5']}\n"
        f"Failures: {eval_metrics['failures']}"
    )


@pytest.mark.eval
def test_citation_faithfulness(eval_metrics):
    score = eval_metrics["citation_faithfulness"]
    if score is None:
        pytest.skip("No samples with citations")
    assert score >= _THRESHOLDS["citation_faithfulness"], (
        f"Citation faithfulness = {score:.3f} < threshold {_THRESHOLDS['citation_faithfulness']}\n"
        f"Failures: {eval_metrics['failures']}"
    )


@pytest.mark.eval
def test_off_topic_accuracy(eval_metrics):
    score = eval_metrics["off_topic_accuracy"]
    if score is None:
        pytest.skip("No off-topic samples")
    assert score >= _THRESHOLDS["off_topic_accuracy"], (
        f"Off-topic accuracy = {score:.3f} < threshold {_THRESHOLDS['off_topic_accuracy']}\n"
        f"Failures: {eval_metrics['failures']}"
    )


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    metrics = run_eval()
    failures = metrics.pop("failures")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", json.dumps(f, ensure_ascii=False))
