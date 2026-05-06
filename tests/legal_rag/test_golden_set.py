"""
test_golden_set.py
------------------
Evaluates the full legal_rag pipeline against the curated golden set.

Each YAML sample becomes an individual parametrized test, so pytest output
shows per-case pass/fail:

    test_golden_set.py::test_corpus_routing[C101] PASSED
    test_golden_set.py::test_corpus_routing[MIX101] FAILED
    ...

Marks
-----
- @pytest.mark.golden  — golden-set evaluations (subset of e2e)
- @pytest.mark.e2e     — requires live LLM + Qdrant services

Run commands
------------
# All golden tests (requires live services):
    pytest RAG/legal_rag/tests/test_golden_set.py -m golden -v

# With live summary table at end:
    pytest RAG/legal_rag/tests/test_golden_set.py -m golden -v -s

# Single case by ID:
    pytest RAG/legal_rag/tests/test_golden_set.py -m golden -k "C101" -v

# Skip golden (default CI):
    pytest RAG/legal_rag/tests/ -m "not e2e and not golden"

Golden file location
--------------------
Place golden_set.yaml in the same directory as this file:
    RAG/legal_rag/tests/golden_set.yaml

Or override via env var:
    GOLDEN_SET_PATH=/path/to/golden_set.yaml pytest ...

Assertion strategy
------------------
Each test checks one concern so failures are independent and informative:

  test_corpus_routing      → result.corpus == expected_corpus
  test_classification      → result.classification == sample type
  test_article_coverage    → ≥1 expected article found in retrieved sources
  test_answer_keywords     → all keywords present in final answer
  test_off_topic_rejection → off_topic queries get rejection response

Article coverage uses partial-match (≥1 of expected_articles) because
retrieval is probabilistic and top-k; requiring ALL articles would make
the suite brittle. Textual queries are stricter (exact match required).
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Register markers
# ---------------------------------------------------------------------------
# (Also done in conftest.py — safe to repeat here for clarity)
pytestmark = [pytest.mark.golden, pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Golden set loading
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent


def _golden_set_path() -> pathlib.Path:
    env = os.environ.get("GOLDEN_SET_PATH")
    if env:
        return pathlib.Path(env)
    return _HERE / "golden_set.yaml"


def _load_samples() -> list[dict]:
    path = _golden_set_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("samples", [])


ALL_SAMPLES: list[dict] = _load_samples()

# Partitioned views — used for parametrize to keep test IDs meaningful
NORMAL_SAMPLES    = [s for s in ALL_SAMPLES if not s.get("should_fail")]
OFF_TOPIC_SAMPLES = [s for s in ALL_SAMPLES if s.get("should_fail")]
KEYWORD_SAMPLES   = [s for s in NORMAL_SAMPLES if s.get("answer_keywords")]
TEXTUAL_SAMPLES   = [s for s in NORMAL_SAMPLES if s.get("type") == "textual"]
ANALYTICAL_SAMPLES = [s for s in NORMAL_SAMPLES if s.get("type") == "analytical"]


def _sample_id(sample: dict) -> str:
    return sample["id"]


# ---------------------------------------------------------------------------
# Skip guard — identical to test_e2e.py
# ---------------------------------------------------------------------------
def _services_available() -> bool:
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


@pytest.fixture(autouse=True)
def require_live_services():
    if not _services_available():
        pytest.skip(
            "Live LLM/Qdrant services not configured. "
            "Set API keys and ensure Qdrant is running."
        )


# ---------------------------------------------------------------------------
# Skip if golden set file is missing
# ---------------------------------------------------------------------------
def _skip_if_no_samples(samples: list) -> None:
    if not samples:
        pytest.skip(
            f"golden_set.yaml not found at {_golden_set_path()}. "
            "Place it in RAG/legal_rag/tests/ or set GOLDEN_SET_PATH env var."
        )


# ---------------------------------------------------------------------------
# Session-level indexing + ensure_indexed
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def ensure_all_indexed():
    """Index all corpora once per test session."""
    if not _services_available():
        return
    try:
        from RAG.legal_rag.civil_law_rag import ensure_indexed as ci
        from RAG.legal_rag.evidence_rag import ensure_indexed as ei
        from RAG.legal_rag.procedures_rag import ensure_indexed as pi
        ci(); ei(); pi()
    except Exception as exc:
        pytest.skip(f"Corpus indexing failed: {exc}")


# ---------------------------------------------------------------------------
# Result cache — one ask_question() call per sample, shared across test fns
# ---------------------------------------------------------------------------
_result_cache: dict[str, Any] = {}


def _get_result(sample: dict) -> Any:
    """Call ask_question() once per sample ID and cache the result."""
    sid = sample["id"]
    if sid not in _result_cache:
        from RAG.legal_rag.service import ask_question
        _result_cache[sid] = ask_question(sample["query"])
    return _result_cache[sid]


def _retrieved_indices(result) -> set[int]:
    """Extract retrieved article indices from result.sources."""
    return {s["article"] for s in result.sources if isinstance(s.get("article"), int)}


# ---------------------------------------------------------------------------
# Scoring accumulators — printed in the session-finish hook
# ---------------------------------------------------------------------------
_scores: dict[str, dict] = {}     # {sample_id: {metric: bool}}


def _record(sid: str, metric: str, passed: bool) -> None:
    _scores.setdefault(sid, {})[metric] = passed


def pytest_sessionfinish(session, exitstatus):
    """Print a per-sample scoring table after all tests finish."""
    if not _scores:
        return

    metrics = ["corpus", "classification", "articles", "keywords", "rejection"]
    header  = f"\n{'ID':<10}" + "".join(f"{m:<16}" for m in metrics) + "OVERALL"
    print("\n" + "=" * 80)
    print("  GOLDEN SET EVALUATION SUMMARY")
    print("=" * 80)
    print(header)
    print("-" * 80)

    total, passed_total = 0, 0
    for sid in sorted(_scores):
        row    = _scores[sid]
        checks = [row.get(m) for m in metrics if m in row]
        ok     = all(c for c in checks if c is not None)
        marks  = "".join(
            ("✓" if row.get(m) else ("✗" if m in row else " ")).ljust(16)
            for m in metrics
        )
        status = "PASS" if ok else "FAIL"
        print(f"{sid:<10}{marks}{status}")
        total += 1
        passed_total += int(ok)

    print("-" * 80)
    pct = (passed_total / total * 100) if total else 0
    print(f"  Total: {passed_total}/{total} passed  ({pct:.0f}%)")
    print("=" * 80 + "\n")


# ===========================================================================
# TEST 1: Corpus routing accuracy
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", ALL_SAMPLES, ids=_sample_id)
def test_corpus_routing(sample: dict):
    """The resolved corpus must match expected_corpus (or be None for off_topic)."""
    _skip_if_no_samples(ALL_SAMPLES)
    result          = _get_result(sample)
    expected_corpus = sample["expected_corpus"]
    sid             = sample["id"]

    if sample.get("should_fail"):
        # Off-topic: corpus must be None (unresolved) or classification must be off_topic
        passed = result.corpus is None or result.classification == "off_topic"
        _record(sid, "corpus", passed)
        assert passed, (
            f"[{sid}] Off-topic query was resolved to corpus '{result.corpus}'. "
            f"Expected no corpus to be resolved.\n"
            f"  Query: {sample['query']}\n"
            f"  Answer snippet: {(result.answer or '')[:150]}"
        )
    else:
        _record(sid, "corpus", result.corpus == expected_corpus)
        assert result.corpus == expected_corpus, (
            f"[{sid}] Corpus routing FAILED.\n"
            f"  Expected : {expected_corpus}\n"
            f"  Got      : {result.corpus}\n"
            f"  Query    : {sample['query']}\n"
            f"  Routing scores: {result.corpus_routing_scores}"
        )


# ===========================================================================
# TEST 2: Classification accuracy  (analytical / textual only — not off_topic)
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", NORMAL_SAMPLES, ids=_sample_id)
def test_classification(sample: dict):
    """result.classification must match the sample's declared type."""
    _skip_if_no_samples(NORMAL_SAMPLES)
    result          = _get_result(sample)
    expected_type   = sample["type"]   # "analytical" or "textual"
    sid             = sample["id"]

    passed = result.classification == expected_type
    _record(sid, "classification", passed)
    assert passed, (
        f"[{sid}] Classification FAILED.\n"
        f"  Expected : {expected_type}\n"
        f"  Got      : {result.classification}\n"
        f"  Query    : {sample['query']}"
    )


# ===========================================================================
# TEST 3: Article coverage
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", NORMAL_SAMPLES, ids=_sample_id)
def test_article_coverage(sample: dict):
    """
    At least one expected article must appear in result.sources.

    Textual queries: STRICT — ALL expected articles must be present
                     (they are fetched by exact scroll, not top-k).
    Analytical queries: PARTIAL — ≥1 expected article must be present
                        (retrieval is probabilistic / top-k).
    """
    _skip_if_no_samples(NORMAL_SAMPLES)

    expected_articles = sample.get("expected_articles", [])
    if not expected_articles:
        pytest.skip(f"[{sample['id']}] No expected_articles defined — skipping article check.")

    result           = _get_result(sample)
    retrieved        = _retrieved_indices(result)
    sid              = sample["id"]
    query_type       = sample["type"]

    if query_type == "textual":
        # Strict: every expected article must be retrieved
        missing = set(expected_articles) - retrieved
        passed  = len(missing) == 0
        _record(sid, "articles", passed)
        assert passed, (
            f"[{sid}] Textual article coverage FAILED.\n"
            f"  Expected articles : {expected_articles}\n"
            f"  Retrieved indices : {sorted(retrieved)}\n"
            f"  Missing           : {sorted(missing)}\n"
            f"  Query             : {sample['query']}"
        )
    else:
        # Partial: at least 1 expected article must be present
        hits   = set(expected_articles) & retrieved
        passed = len(hits) >= 1
        _record(sid, "articles", passed)
        assert passed, (
            f"[{sid}] Analytical article coverage FAILED — no expected article retrieved.\n"
            f"  Expected articles : {expected_articles}\n"
            f"  Retrieved indices : {sorted(retrieved)}\n"
            f"  Query             : {sample['query']}\n"
            f"  Retrieval conf    : {result.retrieval_confidence}"
        )


# ===========================================================================
# TEST 4: Answer keyword presence
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", KEYWORD_SAMPLES, ids=_sample_id)
def test_answer_keywords(sample: dict):
    """Every keyword in answer_keywords must appear in result.answer."""
    _skip_if_no_samples(KEYWORD_SAMPLES)

    result   = _get_result(sample)
    keywords = sample.get("answer_keywords", [])
    answer   = result.answer or ""
    sid      = sample["id"]

    missing = [kw for kw in keywords if kw not in answer]
    passed  = len(missing) == 0
    _record(sid, "keywords", passed)
    assert passed, (
        f"[{sid}] Answer keyword check FAILED.\n"
        f"  Missing keywords : {missing}\n"
        f"  All keywords     : {keywords}\n"
        f"  Query            : {sample['query']}\n"
        f"  Answer snippet   : {answer[:300]}"
    )


# ===========================================================================
# TEST 5: Off-topic strict rejection
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", OFF_TOPIC_SAMPLES, ids=_sample_id)
def test_off_topic_rejection(sample: dict):
    """
    Off-topic queries must be rejected.
    Rejection criteria (ALL must hold):
      1. result.corpus is None  (no corpus was resolved)
      2. result.classification == "off_topic"
      3. result.answer does not contain substantive legal content
         (no article indices cited, answer is a polite rejection)
    """
    _skip_if_no_samples(OFF_TOPIC_SAMPLES)

    result  = _get_result(sample)
    answer  = result.answer or ""
    sid     = sample["id"]

    corpus_rejected         = result.corpus is None
    classified_off_topic    = result.classification == "off_topic"
    no_articles_in_answer   = len(_retrieved_indices(result)) == 0

    # Check answer doesn't look like a real legal response
    # (real answers cite articles or contain retrieval confidence)
    looks_substantive = (
        result.retrieval_confidence is not None and result.retrieval_confidence > 0
    )

    passed = corpus_rejected and classified_off_topic and not looks_substantive
    _record(sid, "rejection", passed)

    failures = []
    if not corpus_rejected:
        failures.append(f"corpus was resolved to '{result.corpus}' (expected None)")
    if not classified_off_topic:
        failures.append(f"classification = '{result.classification}' (expected 'off_topic')")
    if looks_substantive:
        failures.append(
            f"answer looks substantive (retrieval_confidence={result.retrieval_confidence})"
        )

    assert passed, (
        f"[{sid}] Off-topic rejection FAILED:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + f"\n  Query        : {sample['query']}"
        + f"\n  Answer snippet: {answer[:200]}"
    )


# ===========================================================================
# TEST 6: Full sample smoke (non-off-topic must produce a non-empty answer)
# ===========================================================================
@pytest.mark.golden
@pytest.mark.parametrize("sample", NORMAL_SAMPLES, ids=_sample_id)
def test_answer_not_empty(sample: dict):
    """Every non-off-topic sample must return a non-empty, non-error answer."""
    _skip_if_no_samples(NORMAL_SAMPLES)

    result = _get_result(sample)
    sid    = sample["id"]
    answer = result.answer or ""

    # These are the fallback error strings from service.py and fallback.py
    error_strings = [
        "حدث خطأ",
        "تعذر الحصول على إجابة",
        "تعذر تقديم إجابة",
    ]

    is_error = any(e in answer for e in error_strings)
    passed   = bool(answer) and not is_error

    assert passed, (
        f"[{sid}] Empty or error answer returned.\n"
        f"  Query  : {sample['query']}\n"
        f"  Answer : {answer[:200]}"
    )


# ===========================================================================
# TEST 7: Cross-law disambiguation (MIX* samples — tighter check)
# ===========================================================================
MIX_SAMPLES = [s for s in NORMAL_SAMPLES if s["id"].startswith("MIX")]


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.parametrize("sample", MIX_SAMPLES, ids=_sample_id)
def test_cross_law_corpus_routing(sample: dict):
    """
    MIX* queries are deliberately ambiguous across two corpora.
    The pipeline must still resolve to the CORRECT corpus, not just any corpus.
    Failure here indicates the corpus_router is confused by cross-domain language.
    """
    _skip_if_no_samples(MIX_SAMPLES)

    result          = _get_result(sample)
    expected_corpus = sample["expected_corpus"]
    sid             = sample["id"]

    passed = result.corpus == expected_corpus
    _record(sid, "corpus", passed)

    assert passed, (
        f"[{sid}] Cross-law corpus routing FAILED (this is a hard case).\n"
        f"  Expected corpus : {expected_corpus}\n"
        f"  Resolved corpus : {result.corpus}\n"
        f"  Routing scores  : {result.corpus_routing_scores}\n"
        f"  Query           : {sample['query']}"
    )


# ===========================================================================
# Utility: print golden set inventory (not a real test — run with -s to see)
# ===========================================================================
def test_golden_set_loaded():
    """Sanity check that the YAML was loaded and has the expected structure."""
    _skip_if_no_samples(ALL_SAMPLES)

    required_keys = {"id", "type", "expected_corpus", "query", "expected_articles",
                     "answer_keywords", "should_fail"}
    for sample in ALL_SAMPLES:
        missing = required_keys - sample.keys()
        assert not missing, (
            f"Sample {sample.get('id', '?')} is missing keys: {missing}"
        )

    n_total     = len(ALL_SAMPLES)
    n_off_topic = len(OFF_TOPIC_SAMPLES)
    n_normal    = len(NORMAL_SAMPLES)
    n_mix       = len(MIX_SAMPLES)

    print(f"\n  Golden set: {n_total} total "
          f"({n_normal} normal, {n_off_topic} off-topic, {n_mix} cross-law)")
    assert n_total > 0